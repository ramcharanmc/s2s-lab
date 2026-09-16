"""
The frozen eval set: 200 utterances in four cohorts of 50 (v2 plan §6.2).

    cohort     source                                   stands for
    clean      LibriSpeech test-clean, read audiobooks  the easy case
    accented   EdAcc conversations, 22 accents          accents other than mainstream US English
    disfluent  DisfluencySpeech: one speaker            fillers, repetitions, restarts --
               re-enacting disfluent phone calls        PERFORMED, not clinical
    slow       the clean 50, slowed to 0.75x            elderly speech -- A RATE-ONLY PROXY

Every rung runs on the same 200 clips, so the one table can finally carry a p90
(about 100 turns needed) and a WER (reference transcripts needed).

FROZEN, LIKE THE PROBE SET. `build` downloads, selects and writes the audio once
and records the SHA-256 of every file in evalset/manifest.json; it will not run
over an existing manifest. Selection is deterministic -- pinned dataset
revisions, a fixed seed, and the rules below -- so the manifest documents
exactly how every clip was chosen. The audio is not committed; the manifest is.
Build once, on the laptop, and ship the WAVs to Colab or Modal as with the
probes: another machine's resampler may differ in the last bit, and the
checksums would rightly call that a different set.

DOWNLOAD DISCIPLINE (v2 §6.1). Parquet files store audio in row groups, and one
clip cannot be read without its whole group. So the build reads only the text
columns of every row first -- a few MB -- chooses from those, and then fetches
just the audio groups it needs. EdAcc is limited to the fewest row groups that
together cover all its accents: 13 groups, about 380 MB, instead of 3.6 GB.

build()
 ├─ for clean, accented, disfluent:
 │   ├─ source.rows()                  text columns of every row, no audio
 │   ├─ candidates(cohort, rows)       the rules below, in a seeded order
 │   └─ choose(candidates, 50, accept) fetches audio; keeps 3-15 s clips,
 │                                     at most 2 per speaker
 ├─ slow: slow_down(each clean clip)   the same 50 utterances, paired
 └─ writes evalset/audio/<id>.wav, then evalset/manifest.json
"""

from __future__ import annotations

import collections
import datetime
import io
import json
import math
import pathlib
import random
import re
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .probes import sha256_file

EVAL_ROOT = pathlib.Path("evalset")
MANIFEST = "manifest.json"
TARGET_SR = 16_000
PER_COHORT = 50
SEED = 20260915               # fixed before anything was selected
MIN_S, MAX_S = 3.0, 15.0      # the length of a turn, not of a paragraph
MIN_WORDS = 6
PER_SPEAKER = 2
SLOW_RATE = 0.75              # the plan's elderly proxy
NOT_ACCENTED = frozenset({"Mainstream US English", "Don't know"})
COHORTS = ("clean", "accented", "disfluent", "slow")


@dataclass(frozen=True)
class SourceSpec:
    """One Hugging Face dataset, pinned to a single revision."""

    repo: str
    revision: str        # a commit: the manifest must describe one fixed version
    pattern: str         # which parquet files hold the test split
    columns: tuple       # the text columns read for every row
    license: str


SOURCES = {
    "clean": SourceSpec("openslr/librispeech_asr", "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
                        r"^clean/test/", ("id", "speaker_id", "text"), "CC BY 4.0"),
    "accented": SourceSpec("edinburghcstr/edacc", "d9ae7bd344f0562b766ec93ee5ce8f2f9568ce66",
                           r"^data/test-", ("speaker", "text", "accent", "l1", "gender"),
                           "CC BY-SA 4.0 (dataset card); CC BY 4.0 (Edinburgh DataShare)"),
    "disfluent": SourceSpec("amaai-lab/DisfluencySpeech", "b7da294fe3a70dd96df6640893f2a5dfc2c87638",
                            r"^data/test-", ("transcript_a", "transcript_c"), "Apache-2.0"),
}

SPEAKER = {"clean": lambda r: r["speaker_id"], "accented": lambda r: r["speaker"],
           "disfluent": None}                       # DisfluencySpeech has one speaker

# --------------------------------------------------------------------------- #
# Reading a dataset without downloading it
# --------------------------------------------------------------------------- #


class ParquetSource:
    """A Hugging Face dataset's parquet files, read lazily over HTTP.

    rows()          the text columns of every row, each tagged with where it
                    lives: {"file", "group", "index", ...columns}
    group_mb(f, g)  compressed size of one row group's audio, from metadata
    audio(row)      the row's audio, fetching its whole row group once
    """

    def __init__(self, spec: SourceSpec):
        from huggingface_hub import HfApi, HfFileSystem
        self.spec = spec
        self._fs = HfFileSystem()
        self.files = sorted(f for f in HfApi().list_repo_files(
            spec.repo, repo_type="dataset", revision=spec.revision) if re.search(spec.pattern, f))
        self._parquets, self._groups = {}, {}
        self.fetched_mb = 0.0

    def _parquet(self, path):
        import pyarrow.parquet as pq
        if path not in self._parquets:
            self._parquets[path] = pq.ParquetFile(
                self._fs.open(f"datasets/{self.spec.repo}@{self.spec.revision}/{path}"))
        return self._parquets[path]

    def rows(self) -> list[dict]:
        out = []
        for path in self.files:
            pf = self._parquet(path)
            for g in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(g, columns=list(self.spec.columns))
                out += [{**r, "file": path, "group": g, "index": i}
                        for i, r in enumerate(table.to_pylist())]
        return out

    def group_mb(self, path: str, group: int) -> float:
        rg = self._parquet(path).metadata.row_group(group)
        return sum(rg.column(c).total_compressed_size for c in range(rg.num_columns)
                   if rg.column(c).path_in_schema.startswith("audio")) / 1e6

    def audio(self, row: dict) -> tuple[np.ndarray, int]:
        key = (row["file"], row["group"])
        if key not in self._groups:
            table = self._parquet(row["file"]).read_row_group(row["group"], columns=["audio"])
            self._groups[key] = table.column("audio").to_pylist()
            self.fetched_mb += self.group_mb(*key)
        blob = self._groups[key][row["index"]]["bytes"]
        x, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
        return x.mean(axis=1), sr

# --------------------------------------------------------------------------- #
# Choosing clips: pure functions, tested without any download
# --------------------------------------------------------------------------- #


def seeded_order(rows: list, seed: int = SEED) -> list:
    """The rows in a shuffled order that depends only on the seed."""
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


def round_robin(rows: list, key) -> list:
    """Interleave rows by key -- one from each group in turn -- keeping each
    group's own order. With accents as the key, the first 22 clips tried cover
    22 accents instead of whichever accent the shuffle happened to favour."""
    queues: dict = {}
    for r in rows:
        queues.setdefault(key(r), []).append(r)
    keys = sorted(queues)
    out = []
    for depth in range(max((len(q) for q in queues.values()), default=0)):
        out += [queues[k][depth] for k in keys if depth < len(queues[k])]
    return out


def cover_groups(groups: dict) -> list:
    """The fewest row groups that cover every label, chosen greedily.

    Takes:   {group key: (set of labels, cost in MB)}.
    Returns: group keys in the order taken. Each step takes the group adding
             the most labels not yet covered, the cheapest one on a tie.
    """
    covered, chosen = set(), []
    while True:
        best = max(groups, key=lambda k: (len(groups[k][0] - covered), -groups[k][1]))
        new = groups[best][0] - covered
        if not new:
            return chosen
        covered |= new
        chosen.append(best)


def candidates(cohort: str, rows: list[dict], seed: int = SEED, group_mb: dict | None = None) -> list[dict]:
    """A cohort's rules, applied to text columns only.

    Returns: the eligible rows, in the order they will be tried.
    """
    if cohort == "clean":
        ok = [r for r in rows if len(r["text"].split()) >= MIN_WORDS]
        return seeded_order(ok, seed)
    if cohort == "accented":
        ok = [r for r in rows if r["accent"] not in NOT_ACCENTED
              and "<" not in r["text"]                    # <OVERLAP> and other tags
              and len(r["text"].split()) >= MIN_WORDS]
        groups: dict = {}
        for r in ok:
            key = (r["file"], r["group"])
            labels, cost = groups.get(key, (set(), (group_mb or {}).get(key, 1.0)))
            groups[key] = (labels | {r["accent"]}, cost)
        keep = set(cover_groups(groups))
        ok = [r for r in ok if (r["file"], r["group"]) in keep]
        return round_robin(seeded_order(ok, seed), key=lambda r: r["accent"])
    if cohort == "disfluent":
        ok = [r for r in rows if r["transcript_a"].strip() != r["transcript_c"].strip()
              and len(r["transcript_a"].split()) >= MIN_WORDS]
        return seeded_order(ok, seed)
    raise ValueError(f"unknown cohort {cohort!r}")


def choose(ordered: list, n: int, accept, speaker=None, per_speaker: int = PER_SPEAKER) -> list:
    """Walk `ordered`, keeping the first n rows that `accept` passes.

    Takes:   rows in priority order; accept(row) -> audio, or None to skip it;
             speaker(row) -> who is talking, or None for a one-speaker source.
    Returns: [(row, audio), ...], at most per_speaker from any one speaker.
             The cap is checked BEFORE accept, so a capped speaker's clips are
             never downloaded.
    """
    kept, counts = [], collections.Counter()
    for row in ordered:
        if len(kept) == n:
            break
        who = speaker(row) if speaker else None
        if who is not None and counts[who] >= per_speaker:
            continue
        clip = accept(row)
        if clip is None:
            continue
        kept.append((row, clip))
        counts[who] += 1
    return kept

# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def to_16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    """Any rate, any channel count -> float32 mono at 16 kHz."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)
    return x


def slow_down(x: np.ndarray, sr: int, rate: float = SLOW_RATE) -> np.ndarray:
    """Slow speech to `rate` of its speed, keeping its pitch.

    Praat's "Lengthen (overlap-add)" is PSOLA: it repeats pitch periods rather
    than stretching them, so a 0.75x version is 1/0.75 as long at the same F0.
    Rate is all it changes -- which is exactly why this cohort is a proxy.
    """
    import parselmouth
    from parselmouth.praat import call
    sound = parselmouth.Sound(np.asarray(x, dtype=np.float64), sampling_frequency=sr)
    return call(sound, "Lengthen (overlap-add)", 75, 600, 1 / rate).values[0].astype(np.float32)

# --------------------------------------------------------------------------- #
# Build, verify, load
# --------------------------------------------------------------------------- #


def _meta(cohort: str, row: dict) -> dict:
    """What the manifest records about a clip's origin."""
    where = f"{row['file']}#{row['group']}/{row['index']}"
    if cohort == "clean":
        return {"source_id": row["id"], "speaker": str(row["speaker_id"]), "text": row["text"]}
    if cohort == "accented":
        return {"source_id": where, "speaker": row["speaker"], "accent": row["accent"],
                "l1": row["l1"], "gender": row["gender"], "text": row["text"]}
    return {"source_id": where, "speaker": "DisfluencySpeech", "text": row["transcript_a"],
            "text_fluent": row["transcript_c"]}


def _write(root: pathlib.Path, item_id: str, cohort: str, x: np.ndarray, meta: dict) -> dict:
    path = root / "audio" / f"{item_id}.wav"
    sf.write(str(path), np.clip(x, -1.0, 1.0), TARGET_SR, subtype="PCM_16")
    return {"id": item_id, "cohort": cohort, **meta,
            "duration_s": round(len(x) / TARGET_SR, 3), "wav_sha256": sha256_file(path)}


def build(root=EVAL_ROOT, sources: dict | None = None, per_cohort: int = PER_COHORT,
          seed: int = SEED) -> pathlib.Path:
    """Download, choose and write the eval set, once.

    Takes:   the output folder; sources {cohort: object with rows(),
             group_mb() and audio()} -- the real datasets by default.
    Returns: the manifest's path.
    Raises:  FileExistsError over an existing manifest (a frozen set never
             changes); RuntimeError if a source cannot supply enough clips.
    """
    root = pathlib.Path(root)
    path = root / MANIFEST
    if path.exists():
        raise FileExistsError(f"{path} exists: a frozen set never changes")
    sources = sources or {c: ParquetSource(spec) for c, spec in SOURCES.items()}
    (root / "audio").mkdir(parents=True, exist_ok=True)
    items, fetched, clean = [], {}, []
    for cohort in ("clean", "accented", "disfluent"):
        src = sources[cohort]
        rows = src.rows()
        sizes = {(r["file"], r["group"]): src.group_mb(r["file"], r["group"])
                 for r in rows} if cohort == "accented" else None

        def accept(row, src=src):
            x = to_16k_mono(*src.audio(row))
            return x if MIN_S <= len(x) / TARGET_SR <= MAX_S else None

        kept = choose(candidates(cohort, rows, seed, sizes), per_cohort, accept, SPEAKER[cohort])
        if len(kept) < per_cohort:
            raise RuntimeError(f"{cohort}: only {len(kept)} of {per_cohort} clips pass the rules")
        for k, (row, x) in enumerate(kept):
            item = _write(root, f"{cohort}-{k:02d}", cohort, x, _meta(cohort, row))
            items.append(item)
            if cohort == "clean":
                clean.append((item, x))
        fetched[cohort] = round(src.fetched_mb, 1)
    for k, (item, x) in enumerate(clean):
        meta = {"derived_from": item["id"], "rate": SLOW_RATE, "source_id": item["source_id"],
                "speaker": item["speaker"], "text": item["text"]}
        items.append(_write(root, f"slow-{k:02d}", "slow", slow_down(x, TARGET_SR), meta))
    manifest = {
        "set": "evalset", "version": 1, "built": datetime.date.today().isoformat(),
        "seed": seed, "per_cohort": per_cohort,
        "rules": {"duration_s": [MIN_S, MAX_S], "min_words": MIN_WORDS, "per_speaker": PER_SPEAKER,
                  "not_accented": sorted(NOT_ACCENTED), "slow_rate": SLOW_RATE},
        "sources": {c: {"repo": s.repo, "revision": s.revision, "license": s.license}
                    for c, s in SOURCES.items()},
        "fetched_mb": fetched,
        "items": items,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path

def verify(root=EVAL_ROOT) -> list[str]:
    """One line per recording that is missing or changed; empty means intact."""
    path = pathlib.Path(root) / MANIFEST
    if not path.exists():
        return [f"no {MANIFEST}: the eval set is not built yet"]
    problems = []
    for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
        wav = pathlib.Path(root) / "audio" / f"{item['id']}.wav"
        if not wav.exists():
            problems.append(f"{item['id']}: recording missing")
        elif sha256_file(wav) != item["wav_sha256"]:
            problems.append(f"{item['id']}: recording changed")
    return problems


@dataclass
class EvalItem:
    id: str
    cohort: str
    text: str             # the reference transcript: every word spoken
    audio: np.ndarray     # float32 mono, 16 kHz
    meta: dict            # the manifest entry


def load(root=EVAL_ROOT, cohorts=None) -> list[EvalItem]:
    """The built set, in manifest order, optionally only some cohorts."""
    root = pathlib.Path(root)
    out = []
    for item in json.loads((root / MANIFEST).read_text(encoding="utf-8"))["items"]:
        if cohorts and item["cohort"] not in cohorts:
            continue
        x, _ = sf.read(str(root / "audio" / f"{item['id']}.wav"), dtype="float32")
        out.append(EvalItem(item["id"], item["cohort"], item["text"], x, item))
    return out


def eval_clips(root=EVAL_ROOT, cohorts=None) -> dict[str, np.ndarray]:
    """{id: audio} -- the shape rungs.run_turns() takes."""
    return {it.id: it.audio for it in load(root, cohorts)}


def summary(root=EVAL_ROOT) -> str:
    """A table of the built set: clips, speakers and durations per cohort."""
    m = json.loads((pathlib.Path(root) / MANIFEST).read_text(encoding="utf-8"))
    lines = [f"eval set v{m['version']}, built {m['built']}, seed {m['seed']}", "",
             "| cohort | clips | speakers | seconds: min / median / max |",
             "|---|---:|---:|---|"]
    for cohort in COHORTS:
        its = [i for i in m["items"] if i["cohort"] == cohort]
        if not its:
            continue
        d = sorted(i["duration_s"] for i in its)
        lines.append(f"| {cohort} | {len(its)} | {len({i['speaker'] for i in its})} | "
                     f"{d[0]:.1f} / {d[len(d) // 2]:.1f} / {d[-1]:.1f} |")
    accents = collections.Counter(i["accent"] for i in m["items"] if i["cohort"] == "accented")
    lines += ["", f"accents ({len(accents)}): " + ", ".join(f"{a} {n}" for a, n in sorted(accents.items())),
              "downloaded: " + ", ".join(f"{c} {mb:.0f} MB" for c, mb in m["fetched_mb"].items())]
    return "\n".join(lines)

if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "build":
        print(f"built -> {build()}\n")
        print(summary())
    elif cmd == "verify":
        problems = verify()
        print("\n".join(problems) or "intact: every recording matches the manifest")
        sys.exit(1 if problems else 0)
    elif cmd == "summary":
        print(summary())
    else:
        print("usage: python -m s2slab.evalset build | verify | summary")