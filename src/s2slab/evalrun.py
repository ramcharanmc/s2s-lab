"""
One rung across the frozen eval set: latency with a p90, and WER by cohort.

The probe set has 30 turns -- a median and nothing more. The eval set has 200,
so this is the first run that can carry a p90 (timing.py's ~10/(1-p) rule wants
100 samples) and the first that can say anything about accuracy at all.

    python -m s2slab.evalrun r0                              laptop CPU, ~20 min
    python -m s2slab.evalrun r0-gpu --machine "Colab T4"
    python -m s2slab.evalrun --score results/r0/evalset.json  re-score, no models

TWO HALVES, ON PURPOSE.
  run()      needs the models and the audio, and writes one JSON per run:
             every turn's transcript, reply and stage times.
  report()   needs neither. It reads that JSON. So a run produced on Colab or
             Modal is scored here, and re-scored whenever the WER rules change,
             without touching a GPU or re-running anything.

The JSON also carries its own provenance -- rung, commit, machine, eval-set
build date and seed -- because a results file that cannot name what produced it
is not evidence (v1 §5.1, and the same rule sweep.py follows).
"""

from __future__ import annotations

import datetime
import json
import math
import pathlib

from .evalset import EVAL_ROOT, MANIFEST, TARGET_SR, load
from .rungs import get
from .sweep import _git_commit
from .timing import format_table as stage_format
from .timing import summarize
from .wer import format_table as wer_format
from .wer import references, score

RESULTS_ROOT = pathlib.Path("results")


def _finite(x: float) -> float | None:
    """NaN is not JSON. A reply that never speaks has no leading silence, and
    None says that, where NaN would either break the file or read as zero."""
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None

def run(rung, root=EVAL_ROOT, cohorts=None, limit: int | None = None, machine: str = "") -> dict:
    """Every eval clip through one rung.

    Takes:   a rung id or a Rung; which cohorts (default all four); a limit for
             a smoke test; a free-text machine note ("laptop CPU", "Modal T4").
    Calls:   evalset.load() for audio and cohort; rung.load() once, outside the
             clock; rung.respond() per clip.
    Returns: the results dict -- provenance plus one entry per turn.
    """
    rung = get(rung) if isinstance(rung, str) else rung
    items = load(root, cohorts)
    if limit:
        items = items[:limit]
    manifest = json.loads((pathlib.Path(root) / MANIFEST).read_text(encoding="utf-8"))
    load_ms = rung.load()
    turns = {}
    for item in items:
        turn = rung.respond(item.audio, TARGET_SR)
        turns[item.id] = {"cohort": item.cohort,
                          "transcript": turn.transcript,
                          "reply_text": turn.reply_text,
                          "stages_ms": {k: round(v, 1) for k, v in turn.stages_ms.items()},
                          "untimed_ms": round(turn.untimed_ms, 1),
                          "total_ms": round(turn.total_ms, 1),
                          "reply_s": round(turn.reply_s, 3),
                          "leading_silence_ms": _finite(turn.leading_silence_ms)}
    return {"rung": rung.rung_id, "description": rung.description,
            "commit": _git_commit(), "machine": machine,
            "ran": datetime.date.today().isoformat(), "load_ms": round(load_ms, 1),
            "evalset": {"built": manifest["built"], "seed": manifest["seed"], "turns": len(turns)},
            "turns": turns}

def save(results: dict, path=None) -> pathlib.Path:
    """Write a run to JSON. Default: results/<rung>/evalset.json (git-ignored)."""
    path = pathlib.Path(path or RESULTS_ROOT / results["rung"] / "evalset.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def latency_table(results: dict) -> str:
    """Processing time per cohort, and over the whole set.

    A cohort has 50 turns: enough for a median, not for a p90. All 200 together
    support a p90; timing.py prints an em-dash for whatever the sample cannot
    carry, so the table never shows a number nobody should act on.
    """
    groups: dict[str, list[float]] = {}
    for turn in results["turns"].values():
        groups.setdefault(turn["cohort"], []).append(turn["total_ms"])
    groups["all"] = [t["total_ms"] for t in results["turns"].values()]
    lines = ["| cohort | turns | p50 ms | p90 ms | p99 ms |", "|---|---:|---:|---:|---:|"]
    for name, values in groups.items():
        s = summarize(values)
        cells = [f"{s.p50:.0f}" if "p50" not in s.unsupported else "—",
                 f"{s.p90:.0f}" if "p90" not in s.unsupported else "—",
                 f"{s.p99:.0f}" if "p99" not in s.unsupported else "—"]
        lines.append(f"| {name} | {s.n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stage_table(results: dict) -> str:
    """Where the time went, over the whole set -- the same table as rungs.report."""
    rows: dict[str, list[float]] = {}
    for turn in results["turns"].values():
        for name, ms in turn["stages_ms"].items():
            rows.setdefault(name, []).append(ms)
        rows.setdefault("(untimed)", []).append(turn["untimed_ms"])
        rows.setdefault("total", []).append(turn["total_ms"])
    return stage_format({name: summarize(v) for name, v in rows.items()})

def wer_tables(results: dict, root=EVAL_ROOT) -> str:
    """WER per cohort, plus the disfluent cohort against its tidied reference."""
    hyps = {clip: turn["transcript"] or "" for clip, turn in results["turns"].items()}
    out = [wer_format(score(hyps, root), title="Word error rate")]
    fluent = {k: v for k, v in hyps.items() if k in references(root, key="text_fluent")}
    if fluent:
        out += ["", wer_format(score(fluent, root, key="text_fluent"),
                               title="disfluent cohort against the tidied reference "
                                     "(fillers and false starts already removed)")]
    return "\n".join(out)


def report(results: dict, root=EVAL_ROOT) -> str:
    """Everything a run says, from the saved JSON alone."""
    head = (f"{results['rung']}: {results['description']}\n"
            f"commit {results['commit']} · {results['machine'] or 'machine not stated'} · "
            f"run {results['ran']} · eval set built {results['evalset']['built']}, "
            f"seed {results['evalset']['seed']}\n"
            f"load {results['load_ms'] / 1000:.1f} s, once, before the first turn")
    silences = [t["leading_silence_ms"] for t in results["turns"].values()
                if t["leading_silence_ms"] is not None]
    tail = f"reply leading silence: {summarize(silences)}" if silences else \
           "no reply ever spoke, so there is no leading silence"
    return "\n\n".join([head, "### Processing time", latency_table(results),
                        "### Where the time went", stage_table(results),
                        wer_tables(results, root), tail])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one rung over the frozen eval set.")
    parser.add_argument("rung", nargs="?", help="a registered rung id, e.g. r0 or r0-gpu")
    parser.add_argument("--cohorts", help="comma-separated, default all four")
    parser.add_argument("--limit", type=int, help="first N clips only, for a smoke test")
    parser.add_argument("--machine", default="", help='free text, e.g. "Modal T4"')
    parser.add_argument("--out", help="where to write the JSON")
    parser.add_argument("--score", help="score a saved run instead of running anything")
    args = parser.parse_args()

    if args.score:
        print(report(json.loads(pathlib.Path(args.score).read_text(encoding="utf-8"))))
    elif args.rung:
        results = run(args.rung, cohorts=set(args.cohorts.split(",")) if args.cohorts else None,
                      limit=args.limit, machine=args.machine)
        print(f"saved -> {save(results, args.out)}\n")
        print(report(results))
    else:
        parser.error("give a rung id, or --score a saved run")

