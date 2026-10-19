"""
P1b on real disfluent speech (M40): DisfluencySpeech's validation split as DEV and its test split as TEST,
endpointing turns built as M30's real eval clips are (s2slab.realturns), so every policy and
`sweep.evaluate` run on them unchanged.

    PYTHONPATH=src python -m s2slab.dsturns build      the author's download: both splits, once; their manifests
    PYTHONPATH=src python -m s2slab.dsturns dev        the policies on the DEV turns, and the rule written first

WHY NOT SEP-28k, WHICH v1 §8 NAMED. Its audio is not distributed (it is fetched from the podcasts, which keep
their copyright), its labels are CC BY-NC, its clips are fixed 3-second windows whose end is where the window
stopped, not where a speaker finished, and it has no words for T1 to read (docs/m40-log.md, section 1).
DisfluencySpeech (Apache-2.0, the eval set's own pin) is one speaker re-performing Switchboard utterances
with their fillers, repetitions and restarts: real recorded speech, PERFORMED, not clinical; each clip one
whole utterance, so its end is a turn's end.

THE CLIPS. Every row of a split that passes the eval set's disfluent rules (evalset.candidates: the verbatim
transcript differs from the fluent one, at least 6 words; then 3 to 15 s), in file order: all of them, not 50,
so a cut-off rate under 5% can be told from one of 7%. The test split holds the eval set's 50 disfluent clips;
the manifest marks them (in_evalset). The audio stays out of git (probes/disfluencyspeech/audio/); the
manifests, with every file's SHA-256, are committed.

THE TEST TURNS ARE PR-019's: turns("test") is refused without confirm="PR-019", and nothing here reads the
test audio except the build, which only measures each clip's length.
"""

from __future__ import annotations

import datetime
import json
import pathlib

import numpy as np

from .evalset import MAX_S, MIN_S, SOURCES, TARGET_SR, ParquetSource, SourceSpec, candidates, to_16k_mono
from .probes import sha256_file

ROOT = pathlib.Path("probes/disfluencyspeech")
SPLITS = {"dev": "validation", "test": "test"}
REGISTRATION = "PR-019"
DEV_RUN = pathlib.Path("runs/turn-taking/p1b-dev.json")
GO_POLICY, GO_MIN_CUT = "T0_fixed_300ms", 0.20         # the rule: written before the DEV run (docs/m40-log.md)
LONG_PAUSE_S = 0.3                                      # a mid-turn pause a 300 ms threshold could cut at


def spec(split: str) -> SourceSpec:
    """The eval set's DisfluencySpeech pin, for one split."""
    base = SOURCES["disfluent"]
    return SourceSpec(base.repo, base.revision, rf"^data/{SPLITS[split]}-", base.columns, base.license)


def eligible(rows: list[dict]) -> list[dict]:
    """The eval set's disfluent text rules (evalset.candidates), in file order rather than its seeded order."""
    ok = {id(r) for r in candidates("disfluent", rows)}
    return [r for r in rows if id(r) in ok]


def where(row: dict) -> str:
    """A row's place in the dataset, as the eval set's manifest writes source_id."""
    return f"{row['file']}#{row['group']}/{row['index']}"


def evalset_ids(manifest: pathlib.Path = pathlib.Path("evalset/manifest.json")) -> set[str]:
    """The source_ids of the eval set's 50 disfluent clips."""
    items = json.loads(manifest.read_text(encoding="utf-8"))["items"]
    return {i["source_id"] for i in items if i["cohort"] == "disfluent"}


def manifest_path(split: str, root: pathlib.Path = ROOT) -> pathlib.Path:
    return pathlib.Path(root) / f"manifest-{split}.json"


def build(split: str, root: pathlib.Path = ROOT, source=None, in_evalset: set[str] | None = None) -> pathlib.Path:
    """Download one split's eligible clips and write them with a manifest, once. `source` stands in for the
    dataset in tests (rows(), audio(row), fetched_mb). Refuses over an existing manifest."""
    import soundfile as sf
    root = pathlib.Path(root)
    path = manifest_path(split, root)
    if path.exists():
        raise FileExistsError(f"{path} exists: a frozen set never changes")
    src = source or ParquetSource(spec(split))
    marks = in_evalset if in_evalset is not None else (evalset_ids() if split == "test" else set())
    audio_dir = root / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    items, too_short, too_long = [], 0, 0
    for row in eligible(src.rows()):
        x = to_16k_mono(*src.audio(row))
        seconds = len(x) / TARGET_SR
        if seconds < MIN_S or seconds > MAX_S:
            too_short += seconds < MIN_S
            too_long += seconds > MAX_S
            continue
        item_id = f"ds-{split}-{len(items):03d}"
        wav = audio_dir / f"{item_id}.wav"
        sf.write(str(wav), np.clip(x, -1.0, 1.0), TARGET_SR, subtype="PCM_16")
        items.append({"id": item_id, "source_id": where(row), "in_evalset": where(row) in marks,
                      "text": row["transcript_a"], "text_fluent": row["transcript_c"],
                      "duration_s": round(seconds, 3), "wav_sha256": sha256_file(wav)})
    s = spec(split)
    blob = {"set": "disfluencyspeech", "split": split, "dataset_split": SPLITS[split], "built": str(datetime.date.today()),
            "source": {"repo": s.repo, "revision": s.revision, "license": s.license},
            "rules": {"duration_s": [MIN_S, MAX_S], "text": "evalset.candidates('disfluent'), in file order"},
            "skipped": {"under_3_s": int(too_short), "over_15_s": int(too_long)},
            "fetched_mb": round(float(getattr(src, "fetched_mb", 0.0)), 1), "items": items}
    path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    return path


def verify(split: str, root: pathlib.Path = ROOT) -> list[str]:
    """What differs from the manifest: missing files, changed bytes. Empty when the split is as frozen."""
    root = pathlib.Path(root)
    out = []
    for it in json.loads(manifest_path(split, root).read_text(encoding="utf-8"))["items"]:
        wav = root / "audio" / split / f"{it['id']}.wav"
        if not wav.exists():
            out.append(f"{it['id']}: missing")
        elif sha256_file(wav) != it["wav_sha256"]:
            out.append(f"{it['id']}: changed")
    return out


def turns(split: str, confirm: str = "", root: pathlib.Path = ROOT, words_fn=None) -> list:
    """A split's clips as (id, Utterance), cohort "disfluent", in manifest order. The test split is PR-019's
    confirmation's: refused without confirm="PR-019"."""
    import soundfile as sf

    from .realturns import ASR_MODEL, asr_words, turn
    if split == "test" and confirm != REGISTRATION:
        raise PermissionError(f"the test turns are {REGISTRATION}'s confirmation's: pass confirm=\"{REGISTRATION}\"")
    problems = verify(split, root)
    if problems:
        raise RuntimeError(f"the {split} split is not the frozen set:\n  " + "\n  ".join(problems[:10]))
    model = None
    if words_fn is None:
        from faster_whisper import WhisperModel
        model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
        words_fn = lambda audio: asr_words(audio, model)                       # noqa: E731
    out = []
    for it in json.loads(manifest_path(split, root).read_text(encoding="utf-8"))["items"]:
        x, sr = sf.read(str(pathlib.Path(root) / "audio" / split / f"{it['id']}.wav"), dtype="float32")
        assert sr == TARGET_SR, it["id"]
        out.append((it["id"], turn(x, "disfluent", it["id"], words_fn(x))))
    return out


def dev_policies() -> list:
    """The settings swept on DEV: T0 across 150 to 1,500 ms; T1 and T2 from 300 to 900 ms, T2 with three
    slope thresholds (40 Hz/s was set on the synthetic cohorts, page 18 §9); T3 across five cost ratios."""
    from .policies import DecisionTheoretic, FixedThreshold, ProsodicThreshold, SemanticThreshold
    out = [FixedThreshold(ms) for ms in (150, 300, 500, 700, 900, 1100, 1300, 1500)]
    out += [SemanticThreshold(ms) for ms in (300, 500, 700, 900)]
    for ms in (300, 500, 700, 900):
        for slope in (20.0, 40.0, 80.0):
            p = ProsodicThreshold(ms, slope_threshold=slope)
            p.name = f"T2_prosodic_{ms}ms_s{slope:g}"
            out.append(p)
    out += [DecisionTheoretic(r) for r in (2.0, 4.0, 8.0, 16.0, 32.0)]
    return out


def report(items: list, policies: list) -> dict:
    """Every policy on the turns: cut-offs with their Wilson interval, never-fired, waits; and the turns' pauses."""
    from .stats import wilson_interval
    from .sweep import evaluate
    us = [u for _, u in items]
    longest = [max(u.mid_turn_pauses_s, default=0.0) for u in us]
    rows = []
    for p in policies:
        r = evaluate(p, us)
        lo, hi = wilson_interval(r.n_cutoff, r.n)
        rows.append({"policy": p.name, "n": r.n, "cut": r.n_cutoff, "cut_ci": [round(lo, 4), round(hi, 4)],
                     "no_fire": r.n_never_fired, "p50": r.latency.p50, "p90": r.latency.p90,
                     "cut_ids": [i for (i, _), c in zip(items, r.cut_mask) if c]})
    return {"n": len(us), "rows": rows,
            "pauses": {"turns_with_a_pause": sum(bool(u.mid_turn_pauses_s) for u in us),
                       f"turns_with_one_over_{int(LONG_PAUSE_S * 1000)}ms": sum(x > LONG_PAUSE_S for x in longest),
                       "longest_p50_s": round(float(np.median(longest)), 3) if longest else None,
                       "true_end_p50_s": round(float(np.median([u.true_end_s for u in us])), 3) if us else None}}


def gate(blob: dict) -> dict:
    """The rule written before the DEV run: GO to PR-019 if T0 at 300 ms cuts off at least 20% of the DEV turns,
    so the clips' pauses are long enough for the endpointing question to have an answer; else STOP."""
    row = next(r for r in blob["rows"] if r["policy"] == GO_POLICY)
    share = row["cut"] / row["n"] if row["n"] else 0.0
    go = share >= GO_MIN_CUT
    why = (f"{GO_POLICY} cuts off {row['cut']} of {row['n']} DEV turns ({100 * share:.1f}%); needs at least "
           f"{100 * GO_MIN_CUT:.0f}%")
    return {"go": go, "share": round(share, 4), "why": why}


def format_dev(blob: dict) -> str:
    g = gate(blob)
    pz = blob["pauses"]
    lines = [f"P1b, DEV: DisfluencySpeech's validation split, {blob['n']} turns, padded with their own floor noise", "",
             "| policy | cut off | 95% interval | never fired | median wait ms | p90 ms |", "|---|---:|---:|---:|---:|---:|"]
    for r in blob["rows"]:
        f = lambda v: "—" if v is None or v != v else f"{v:.0f}"                  # noqa: E731
        lines.append(f"| {r['policy']} | {r['cut']} of {r['n']} | {100 * r['cut_ci'][0]:.1f} to {100 * r['cut_ci'][1]:.1f}% "
                     f"| {r['no_fire']} | {f(r['p50'])} | {f(r['p90'])} |")
    lines += ["", f"Pauses: {pz['turns_with_a_pause']} of {blob['n']} turns pause mid-turn; "
                  f"{pz[f'turns_with_one_over_{int(LONG_PAUSE_S * 1000)}ms']} for more than {int(LONG_PAUSE_S * 1000)} ms; "
                  f"the longest pause {1000 * pz['longest_p50_s']:.0f} ms at the median; the turn {pz['true_end_p50_s']:.2f} s",
              "", f"The rule (written before the run): {'GO: register PR-019' if g['go'] else 'STOP: nothing registered'}: {g['why']}"]
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse
    from .sweep import _git_commit
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["build", "dev"])
    a = ap.parse_args(argv)
    if a.what == "build":
        for split in SPLITS:
            path = build(split)
            blob = json.loads(path.read_text(encoding="utf-8"))
            marked = sum(i["in_evalset"] for i in blob["items"])
            print(f"{split} ({blob['dataset_split']}): {len(blob['items'])} clips, {blob['skipped']} skipped, "
                  f"{blob['fetched_mb']} MB fetched" + (f"; {marked} are the eval set's" if split == "test" else "")
                  + f" -> {path}")
        return
    if DEV_RUN.exists():
        raise SystemExit(f"{DEV_RUN} exists: the DEV run is recorded once")
    blob = {"run": "M40, P1b on real disfluent speech: the DEV run", "commit": _git_commit(),
            **report(turns("dev"), dev_policies())}
    DEV_RUN.parent.mkdir(parents=True, exist_ok=True)
    DEV_RUN.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    print(format_dev(blob))
    print(f"\nsaved -> {DEV_RUN}")


if __name__ == "__main__":
    main()
