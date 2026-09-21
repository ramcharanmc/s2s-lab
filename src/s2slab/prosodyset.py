"""
The prosody probe set: the same words from the same person, said neutrally and said with feeling.

The v2 plan, section 6.3: "Neutral words, loaded delivery ... R1 cannot distinguish
the pairs by construction -- this is the input bottleneck, demonstrated rather
than argued." A cascade hands its LLM a transcript, and both clips of a pair
have the same transcript.

    one item = one PAIR    actor 1042, "I think I have a doctor's appointment"
                           neutral  1042_ITH_NEU_XX    raters: neutral, 10 of 11
                           loaded   1042_ITH_FEA_XX    raters: fear,     9 of 10

SOURCE: CREMA-D (Cao et al., IEEE Trans. Affective Computing 2014;
github.com/CheyneyComputerScience/CREMA-D), pinned to one commit. 91 actors
speak 12 sentences whose words carry no emotion, each in six acted emotions,
and 2,443 raters said what they HEARD. Open Database License for the database,
Database Contents License for its contents. The author of this repo records
nothing here: a label that is a perception needs listeners, and these clips
come with theirs.

"DISTRESS" IS A PROXY, THREE TIMES OVER. CREMA-D has no such category. Sad,
fearful and angry delivery are taken as three separate stand-ins, reported one
by one and never pooled. It is acted emotion by consenting actors; nothing here
claims clinical validity or diagnoses anyone (v1 plan, ethics boundary). Each
pair keeps the raters' mean intensity for the loaded clip, a number, beside
the label.

SELECTED BY RULE, FROM THE RATINGS ALONE -- nobody listens first.
  1. voice-only ratings (the raters who heard the clip and saw no face)
  2. a pair qualifies if the neutral clip was heard as neutral and the loaded
     clip as its intended emotion, each by at least `threshold` of at least
     MIN_RATERS raters
  3. actors are split into dev and test BEFORE any pair is drawn, so no voice
     is on both sides; the test pairs are locked like the other test sets
  4. each emotion's threshold is the HIGHEST of THRESHOLDS at which it fills
     both sides -- decided by counting, reported, never by ear. One threshold
     for all three was the first rule (f56c55d); sadness is so rarely heard
     from the voice alone that it dragged fear and anger down to 60% when the
     raters agree on them at 80 and 90%. Changed after seeing the counts and
     nothing else: no clip had been fetched, no system run.
  5. a seeded draw: one pair per actor per emotion, no neutral clip twice, at     most PER_SENTENCE pairs of one sentence per emotion and side

Only the chosen clips are fetched (about 50 KB each); the full corpus is 7.5 GB.
Audio and the ratings file stay out of git; pairs.tsv, the rule's output, is committed.

    python -m s2slab.prosodyset ratings      fetch the two ratings files (4.5 MB), once
    python -m s2slab.prosodyset select       count, choose, write probes/prosody/pairs.tsv
    python -m s2slab.prosodyset audio        fetch the chosen clips
    python -m s2slab.prosodyset freeze       write manifest.json: every clip's SHA-256, once
    python -m s2slab.prosodyset verify       has anything changed since?    
"""

from __future__ import annotations

import csv
import datetime
import json
import pathlib
import random
import urllib.request
from typing import NamedTuple

import numpy as np

from .probes import load_wav, sha256_file

PROSODY_ROOT = pathlib.Path("probes/prosody")
REPO = "CheyneyComputerScience/CREMA-D"
REVISION = "1658cd342dff90010aa843eaeebd53610a08b1dc"      # master on 2026-09-21; last changed 2025-03-21
RAW = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/"
MEDIA = f"https://media.githubusercontent.com/media/{REPO}/{REVISION}/"    # the WAVs are git-lfs objects
RATINGS, SUMMARY = "processedResults/tabulatedVotes.csv", "processedResults/summaryTable.csv"
PAIRS, MANIFEST = "pairs.tsv", "manifest.json"
CLIP_SR = 16_000                      # CREMA-D's AudioWAV: 16 kHz mono, as the rungs expect
SEED = 20260921                       # fixed before any rating was read
NEUTRAL = "NEU"
EMOTIONS = ("SAD", "FEA", "ANG")      # the three proxies, always reported separately
VOTE = {"NEU": "N", "SAD": "S", "FEA": "F", "ANG": "A"}
INTENSITY = {"SAD": "meanSadResp", "FEA": "meanFearResp", "ANG": "meanAngerResp"}
VOICE_IDS = range(100_001, 200_000)   # tabulatedVotes numbers voice-only ratings 1xxxxx, face 2xxxxx, both 3xxxxx
THRESHOLDS = (0.9, 0.8, 0.7, 0.6, 0.5)
MIN_RATERS = 8
DEV_ACTORS = 30                       # of 91
PER_EMOTION = {"dev": 7, "test": 13}  # x 3 emotions = 21 dev pairs + 39 test pairs
PER_SENTENCE = {"dev": 2, "test": 3}
PAIR_COLUMNS = ("id", "split", "emotion", "actor", "sentence", "neutral", "loaded",
                "agree_neutral", "agree_loaded", "raters_neutral", "raters_loaded", "intensity", "threshold")


class Rating(NamedTuple):
    """What the voice-only raters said about one clip."""

    clip: str             # 1042_ITH_FEA_XX
    actor: str
    sentence: str
    emotion: str          # what the actor was asked for
    vote: str             # what most raters heard: one letter, or a tie such as "N:S"
    agreement: float      # the share of raters who gave the majority answer
    raters: int
    intensity: float      # mean 0-100 intensity from raters who heard the INTENDED emotion; -1 if none did

    def heard_as_intended(self, threshold: float) -> bool:
        return self.vote == VOTE[self.emotion] and self.agreement >= threshold and self.raters >= MIN_RATERS


class Pair(NamedTuple):
    id: str               # sad-t01: emotion, side, number
    split: str            # dev | test
    emotion: str
    actor: str
    sentence: str
    neutral: str          # clip names, without .wav
    loaded: str
    agree_neutral: float
    agree_loaded: float
    raters_neutral: int
    raters_loaded: int
    intensity: float
    threshold: float      # the agreement both clips had to reach: the emotion's own


# --------------------------------------------------------------------------- #
# Reading the ratings
# --------------------------------------------------------------------------- #


def read_ratings(path: pathlib.Path) -> dict[str, Rating]:
    """{clip name: Rating} for the voice-only ratings of the emotions this set uses.

    Takes:  CREMA-D's processedResults/tabulatedVotes.csv.
    Raises: ValueError if a column the rule reads is missing -- the file is not what it was.
    """
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        missing = {"fileName", "numResponses", "agreement", "emoVote", *INTENSITY.values()} - set(rows.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: no column {sorted(missing)}")
        for row in rows:
            if int(row[""]) not in VOICE_IDS:
                continue
            actor, sentence, emotion, _level = row["fileName"].split("_")
            if emotion not in VOTE:
                continue
            intensity = float(row[INTENSITY[emotion]]) if emotion in INTENSITY else -1.0
            out[row["fileName"]] = Rating(row["fileName"], actor, sentence, emotion, row["emoVote"],
                                          float(row["agreement"]), int(row["numResponses"]), intensity)
    return out


def read_summary(path: pathlib.Path) -> dict[str, str]:
    """{clip name: VoiceVote} from summaryTable.csv: CREMA-D's own statement of
    what the voice-only raters heard, used to cross-check read_ratings()."""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["FileName"]: row["VoiceVote"] for row in csv.DictReader(f)}


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #


def candidates(ratings: dict[str, Rating], threshold: float) -> list[tuple[Rating, Rating]]:
    """Every (neutral, loaded) pair that qualifies at this threshold, sorted.

    Same actor, same sentence. Where an actor said a sentence at several
    intensities (only "It's eleven o'clock" has low / medium / high), the one
    the raters agreed on most is the candidate; ties go to the first by name.
    """
    best: dict[tuple[str, str, str], Rating] = {}
    for r in sorted(ratings.values()):
        if r.emotion in EMOTIONS and r.heard_as_intended(threshold):
            key = (r.actor, r.sentence, r.emotion)
            if key not in best or r.agreement > best[key].agreement:
                best[key] = r
    out = []
    for (actor, sentence, _emotion), loaded in sorted(best.items()):
        neutral = ratings.get(f"{actor}_{sentence}_{NEUTRAL}_XX")
        if neutral is not None and neutral.heard_as_intended(threshold):
            out.append((neutral, loaded))
    return out


def split_actors(actors, seed: int = SEED) -> dict[str, str]:
    """{actor: "dev" | "test"}: a seeded shuffle of ALL actors, the first
    DEV_ACTORS to dev. Done before any pair is drawn and without looking at a
    rating, so which voices are test voices depends on nothing but the seed."""
    order = sorted(actors)
    random.Random(seed).shuffle(order)
    return {a: ("dev" if k < DEV_ACTORS else "test") for k, a in enumerate(order)}


def draw(cands: list[tuple[Rating, Rating]], sides: dict[str, str], emotion: str, threshold: float,
         used_neutral: frozenset = frozenset(), seed: int = SEED) -> list[Pair]:
    """A seeded draw of ONE emotion's pairs, dev then test, as many as the rule
    allows up to PER_EMOTION.

    Takes:   the candidates at `threshold`; {actor: side}; the emotion; the
             neutral clips earlier emotions already took.
    One pair per actor; no neutral clip in two pairs; at most PER_SENTENCE
    pairs of one sentence per side. The result depends only on the candidates,
    the clips already taken, and the seed.
    """
    order = [c for c in cands if c[1].emotion == emotion]
    random.Random(seed).shuffle(order)
    taken = set(used_neutral)
    pairs = []
    for split in ("dev", "test"):
        actors, per_sentence, n = set(), {}, 0
        for neutral, loaded in order:
            if n == PER_EMOTION[split]:
                break
            if (sides[loaded.actor] != split or loaded.actor in actors or neutral.clip in taken
                    or per_sentence.get(loaded.sentence, 0) >= PER_SENTENCE[split]):
                continue
            n += 1
            actors.add(loaded.actor)
            taken.add(neutral.clip)
            per_sentence[loaded.sentence] = per_sentence.get(loaded.sentence, 0) + 1
            pairs.append(Pair(f"{emotion.lower()}-{split[0]}{n:02d}", split, emotion, loaded.actor,
                              loaded.sentence, neutral.clip, loaded.clip, round(neutral.agreement, 3),
                              round(loaded.agreement, 3), neutral.raters, loaded.raters,
                              round(loaded.intensity, 1), threshold))
    return pairs


def filled(pairs: list[Pair]) -> dict[str, int]:
    """{"dev": pairs drawn, "test": pairs drawn}."""
    return {s: sum(1 for p in pairs if p.split == s) for s in ("dev", "test")}


def select(ratings: dict[str, Rating], seed: int = SEED) -> tuple[dict[str, float], list[Pair], dict]:
    """Apply the rule, one emotion after another in EMOTIONS order.

    Returns: {emotion: the threshold it was drawn at}; the pairs; and the
             counting that chose each threshold, {emotion: {threshold tried:
             {"dev": n, "test": n}}}, kept so it can be reported.
    Raises:  ValueError, with the counts, if an emotion fills at no threshold.
    """
    sides = split_actors({r.actor for r in ratings.values()}, seed)
    chosen, pairs, counts = {}, [], {}
    for emotion in EMOTIONS:
        counts[emotion] = {}
        for threshold in THRESHOLDS:
            mine = draw(candidates(ratings, threshold), sides, emotion, threshold,
                        frozenset(p.neutral for p in pairs), seed)
            counts[emotion][threshold] = filled(mine)
            if counts[emotion][threshold] == PER_EMOTION:
                chosen[emotion] = threshold
                pairs += mine
                break
        else:
            raise ValueError(f"{emotion} fills both sides at no threshold:\n" + format_counts(counts, chosen))
    return chosen, pairs, counts


def format_counts(counts: dict, chosen: dict[str, float]) -> str:
    """The counting behind each emotion's threshold, as a table. A dot: not
    tried, because the emotion had already filled at a higher threshold."""
    cells = [(e, s) for e in counts for s in ("dev", "test")]
    lines = ["| raters agreeing | " + " | ".join(f"{e} {s} (of {PER_EMOTION[s]})" for e, s in cells) + " |",
             "|---|" + "---:|" * len(cells)]
    for threshold in THRESHOLDS:
        if not any(threshold in counts[e] for e in counts):
            break
        row = []
        for e, s in cells:
            n = counts[e].get(threshold, {}).get(s)
            row.append("\u00b7" if n is None else f"**{n}**" if chosen.get(e) == threshold else str(n))
        lines.append(f"| at least {threshold:.0%} | " + " | ".join(row) + " |")
    lines.append("chosen (bold): " + ", ".join(f"{e} at least {t:.0%}" for e, t in chosen.items()))
    return "\n".join(lines)


def cross_check(pairs: list[Pair], summary: dict[str, str]) -> None:
    """CREMA-D's summary table must say the same about every chosen clip. If it
    does not, read_ratings() is reading the wrong rows (not the voice-only ones)."""
    for p in pairs:
        for clip, expected in ((p.neutral, VOTE[NEUTRAL]), (p.loaded, VOTE[p.emotion])):
            if summary.get(clip) != expected:
                raise ValueError(f"{clip}: chosen as heard '{expected}', but summaryTable.csv says "
                                 f"VoiceVote = {summary.get(clip)!r}")


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def write_pairs(pairs: list[Pair], path: pathlib.Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists: the selection is made once. Delete it yourself to select again.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(PAIR_COLUMNS)
        w.writerows(pairs)


def read_pairs(path: pathlib.Path) -> list[Pair]:
    with open(path, newline="", encoding="utf-8") as f:
        return [Pair(r["id"], r["split"], r["emotion"], r["actor"], r["sentence"], r["neutral"], r["loaded"],
                     float(r["agree_neutral"]), float(r["agree_loaded"]), int(r["raters_neutral"]),
                     int(r["raters_loaded"]), float(r["intensity"]), float(r["threshold"]))
                for r in csv.DictReader(f, delimiter="\t")]


def fetch(url: str, dest: pathlib.Path) -> bool:
    """Download url to dest unless dest exists. Returns True if it downloaded."""
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    dest.write_bytes(data)
    return True


def fetch_ratings(root: pathlib.Path = PROSODY_ROOT) -> list[pathlib.Path]:
    """The two ratings files, into root/source/ (ignored by git). 4.5 MB, once."""
    paths = []
    for name in (RATINGS, SUMMARY):
        paths.append(root / "source" / pathlib.Path(name).name)
        fetch(RAW + name, paths[-1])
    return paths


def fetch_audio(pairs: list[Pair], root: pathlib.Path = PROSODY_ROOT) -> int:
    """Only the chosen clips, into root/audio/. Returns how many were downloaded now."""
    clips = sorted({c for p in pairs for c in (p.neutral, p.loaded)})
    return sum(fetch(f"{MEDIA}AudioWAV/{clip}.wav", root / "audio" / f"{clip}.wav") for clip in clips)

# --------------------------------------------------------------------------- #
# Freezing, verifying, loading
# --------------------------------------------------------------------------- #


def _fingerprints(root: pathlib.Path) -> dict:
    """SHA-256 of pairs.tsv (line endings normalized) and of every chosen clip."""
    root = pathlib.Path(root)
    clips = sorted({c for p in read_pairs(root / PAIRS) for c in (p.neutral, p.loaded)})
    return {"files": {PAIRS: sha256_file(root / PAIRS, text=True)},
            "clips": {c: sha256_file(root / "audio" / f"{c}.wav") for c in clips}}


def freeze(root: pathlib.Path = PROSODY_ROOT) -> pathlib.Path:
    """Write manifest.json: where the clips came from, and the SHA-256 of each.

    Refuses if a manifest exists, if a chosen clip is missing, or if a file is
    not audio at 16 kHz mono -- a git-lfs pointer saved as .wav is a few lines
    of text, and would otherwise be frozen as if it were a recording.
    """
    import soundfile as sf
    root = pathlib.Path(root)
    out = root / MANIFEST
    if out.exists():
        raise FileExistsError(f"{out} exists: a frozen set is never re-frozen; make a new version")
    pairs = read_pairs(root / PAIRS)
    clips = sorted({c for p in pairs for c in (p.neutral, p.loaded)})
    missing = [c for c in clips if not (root / "audio" / f"{c}.wav").exists()]
    if missing:
        raise FileNotFoundError(f"not fetched yet: {', '.join(missing[:5])}"
                                + (f" and {len(missing) - 5} more" if len(missing) > 5 else ""))
    for clip in clips:
        try:
            info = sf.info(str(root / "audio" / f"{clip}.wav"))
        except Exception as e:                                     # noqa: BLE001 -- whatever soundfile raises
            raise ValueError(f"{clip}.wav is not audio ({e}): a git-lfs pointer?") from None
        if (info.samplerate, info.channels) != (CLIP_SR, 1) or info.duration < 0.5:
            raise ValueError(f"{clip}.wav: {info.samplerate} Hz, {info.channels} channel(s), "
                             f"{info.duration:.2f} s -- expected {CLIP_SR} Hz mono, at least 0.5 s")
    ratings = {name: sha256_file(root / "source" / name) for name in (pathlib.Path(RATINGS).name,
               pathlib.Path(SUMMARY).name) if (root / "source" / name).exists()}
    out.write_text(json.dumps({"set": "prosody", "version": 1, "frozen": datetime.date.today().isoformat(),
                               "source": {"repo": REPO, "revision": REVISION, "ratings_sha256": ratings},
                               "seed": SEED, "pairs": len(pairs), **_fingerprints(root)}, indent=2) + "\n",
                   encoding="utf-8")
    return out


def verify(root: pathlib.Path = PROSODY_ROOT) -> list[str]:
    """One line per file that is missing or changed; empty means intact."""
    root = pathlib.Path(root)
    if not (root / MANIFEST).exists():
        return [f"no {MANIFEST}: the set is not frozen yet"]
    frozen = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    try:
        now = _fingerprints(root)
    except FileNotFoundError as e:
        return [f"missing: {e.filename}"]
    problems = [f"{name}: changed" for name, h in frozen["files"].items() if now["files"].get(name) != h]
    problems += [f"audio/{clip}.wav: {'missing from pairs.tsv' if clip not in now['clips'] else 'changed'}"
                 for clip, h in frozen["clips"].items() if now["clips"].get(clip) != h]
    return problems


class LoadedPair(NamedTuple):
    pair: Pair
    neutral: np.ndarray       # mono float32 at 16 kHz
    loaded: np.ndarray


def load(split: str = "dev", root: pathlib.Path = PROSODY_ROOT, confirm: str = "") -> list[LoadedPair]:
    """The pairs of one side, with their audio.

    Takes:  "dev" or "test". THE TEST PAIRS ARE A TEST SET: they load only for
            a run that names its registration (docs/preregistration.md), like
            the simulator's EVAL sentences and the frozen barge-in recordings.
    Raises: PermissionError for "test" without `confirm`.
    """
    if split not in ("dev", "test"):
        raise ValueError(f"split is 'dev' or 'test', not {split!r}")
    if split == "test" and not confirm:
        raise PermissionError("the prosody test pairs are a test set: loading them is a registered "
                              "confirmation run (docs/preregistration.md); pass confirm='PR-...'. "
                              "Develop on split='dev'.")
    root = pathlib.Path(root)
    return [LoadedPair(p, load_wav(root / "audio" / f"{p.neutral}.wav"), load_wav(root / "audio" / f"{p.loaded}.wav"))
            for p in read_pairs(root / PAIRS) if p.split == split]


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) == 2 else ""
    source = PROSODY_ROOT / "source"
    if cmd == "ratings":
        for path in fetch_ratings():
            print(f"{path}  {path.stat().st_size / 1e6:.1f} MB")
    elif cmd == "select":
        ratings = read_ratings(source / pathlib.Path(RATINGS).name)
        chosen, pairs, counts = select(ratings)
        cross_check(pairs, read_summary(source / pathlib.Path(SUMMARY).name))
        print(f"{len(ratings)} voice-only ratings of neutral, sad, fearful and angry clips; "
              f"{len({r.actor for r in ratings.values()})} actors; seed {SEED}\n")
        print(format_counts(counts, chosen))
        write_pairs(pairs, PROSODY_ROOT / PAIRS)
        print(f"\n{len(pairs)} pairs -> {PROSODY_ROOT / 'pairs.tsv'}; every chosen clip agrees with summaryTable.csv")
        for emotion in EMOTIONS:
            mine = [p for p in pairs if p.emotion == emotion]
            print(f"  {emotion}: loaded clips heard as intended by {min(p.agree_loaded for p in mine):.0%} to "
                  f"{max(p.agree_loaded for p in mine):.0%} of raters; {len({p.sentence for p in mine})} "
                  f"of 12 sentences; intensity {min(p.intensity for p in mine):.0f} to "
                  f"{max(p.intensity for p in mine):.0f} of 100")
    elif cmd == "audio":
        pairs = read_pairs(PROSODY_ROOT / "pairs.tsv")
        print(f"downloaded {fetch_audio(pairs)} clips; {2 * len(pairs)} are in {PROSODY_ROOT / 'audio'}")
    elif cmd == "freeze":
        print(f"frozen -> {freeze()}")
    elif cmd == "verify":
        problems = verify()
        print("\n".join(problems) if problems else "intact")
        sys.exit(1 if problems else 0)
    else:
        sys.exit("usage: python -m s2slab.prosodyset ratings | select | audio | freeze | verify")