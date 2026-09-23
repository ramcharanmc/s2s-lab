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

FIRST_KEYS = {"first_audio_ms": "first audio", "first_sound_ms": "first sound"}

def _peak_vram_mb(reset: bool = False) -> int | None:
    """Peak GPU memory this process has held, in MB; None without a CUDA GPU.
    reset=True clears the peak so a run measures its own rung's models only."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    if reset:
        torch.cuda.reset_peak_memory_stats()
        return None
    return round(torch.cuda.max_memory_allocated() / 2 ** 20)


def _finite(x: float) -> float | None:
    """NaN is not JSON. A reply that never speaks has no leading silence, and
    None says that, where NaN would either break the file or read as zero."""
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None

def run(rung, root=EVAL_ROOT, cohorts=None, limit: int | None = None, machine: str = "") -> dict:
    """Every eval clip through one rung.

    Takes:   a rung id or a Rung; which cohorts (default all four); a limit for
             a smoke test; a free-text machine note ("laptop CPU", "Modal T4").
    Calls:   evalset.load() for audio and cohort; rung.load() once, outside the
             clock; rung.respond() per clip; and, for a rung that has hear()
             (R3: no transcript in the loop), hear() per clip after its turn,
             untimed, stored as "heard".
    Returns: the results dict -- provenance plus one entry per turn, and the
             peak GPU memory held during the run (peak_vram_mb; None on a CPU).
    """
    rung = get(rung) if isinstance(rung, str) else rung
    items = load(root, cohorts)
    if limit:
        items = items[:limit]
    manifest = json.loads((pathlib.Path(root) / MANIFEST).read_text(encoding="utf-8"))
    _peak_vram_mb(reset=True)
    load_ms = rung.load()
    hear = getattr(rung, "hear", None)
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
                          "leading_silence_ms": _finite(turn.leading_silence_ms),
                          "first_audio_ms": _finite(round(turn.first_audio_ms, 1)),
                          "first_sound_ms": _finite(round(turn.first_sound_ms, 1)),
                          "stall_ms": round(turn.stall_ms, 1),
                          "chunks": [[round(ms, 1), n] for ms, n in turn.chunks]}
        if hear is not None:
            turns[item.id]["heard"] = hear(item.audio, TARGET_SR)
    return {"rung": rung.rung_id, "description": rung.description,
            "commit": _git_commit(), "machine": machine,
            "ran": datetime.date.today().isoformat(), "load_ms": round(load_ms, 1),
            "peak_vram_mb": _peak_vram_mb(),
            "evalset": {"built": manifest["built"], "seed": manifest["seed"], "turns": len(turns)},
            "turns": turns}

def save(results: dict, path=None) -> pathlib.Path:
    """Write a run to JSON. Default: results/<rung>/evalset.json (git-ignored)."""
    path = pathlib.Path(path or RESULTS_ROOT / results["rung"] / "evalset.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def latency_table(results: dict, key: str = "total_ms") -> str:
    """One timing per cohort, and over the whole set: total_ms by default,
    or first_audio_ms / first_sound_ms. None (a reply that never spoke) is left out.

    A cohort has 50 turns: enough for a median, not for a p90. All 200 together
    support a p90; timing.py prints an em-dash for whatever the sample cannot
    carry, so the table never shows a number nobody should act on.
    """
    groups: dict[str, list[float]] = {}
    for turn in results["turns"].values():
        groups.setdefault(turn["cohort"], [])
        if turn[key] is not None:
            groups[turn["cohort"]].append(turn[key])
    groups["all"] = [v for vs in groups.values() for v in vs]
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
        for key, name in FIRST_KEYS.items():
            if turn.get(key) is not None:
                rows.setdefault(name, []).append(turn[key])
    return stage_format({name: summarize(v) for name, v in rows.items()})


def has_first_audio(results: dict) -> bool:
    """Runs saved before first audio was measured (runs/r0*, 2026-09-15/16) have
    no such field. They are still reported -- just without those tables."""
    return all("first_audio_ms" in t for t in results["turns"].values())

def wer_tables(results: dict, root=EVAL_ROOT) -> str:
    """WER per cohort, plus the disfluent cohort against its tidied reference.

    A rung whose turns have no transcript at all (R3: the audio goes straight
    into the LLM) gets a sentence instead of a table of blanks. If the run
    also holds "heard" -- the same model asked to transcribe, outside the loop
    -- that is scored in its own table, labelled for what it is.
    """
    turns = results["turns"]
    if turns and all(turn["transcript"] is None for turn in turns.values()):
        out = ["### Word error rate", "",
               "no transcript in the loop: the audio goes straight into the LLM, so there is nothing to score"]
    else:
        hyps = {clip: turn["transcript"] or "" for clip, turn in turns.items()}
        out = [wer_format(score(hyps, root), title="Word error rate")]
        fluent = {k: v for k, v in hyps.items() if k in references(root, key="text_fluent")}
        if fluent:
            out += ["", wer_format(score(fluent, root, key="text_fluent"),
                                   title="disfluent cohort against the tidied reference "
                                         "(fillers and false starts already removed)")]
    heard = {clip: turn["heard"] or "" for clip, turn in turns.items() if turn.get("heard") is not None}
    if heard:
        # Scored through the wrapper rule as it stands NOW, not as it stood when the run was
        # saved: the run file keeps what the model said, and the scoring is re-done here --
        # evalrun's own rule for every WER (the r3-gpu run of 2026-09-23 was saved with 34
        # wrappers its code at the time did not strip).
        from .r3 import unwrap_transcript
        heard = {clip: unwrap_transcript(h) for clip, h in heard.items()}
        out += ["", wer_format(score(heard, root), title="asked to transcribe, outside the loop "
                                                          "(untimed; not what the reply was made from)")]
    return "\n".join(out)


def report(results: dict, root=EVAL_ROOT) -> str:
    """Everything a run says, from the saved JSON alone."""
    head = (f"{results['rung']}: {results['description']}\n"
            f"commit {results['commit']} · {results['machine'] or 'machine not stated'} · "
            f"run {results['ran']} · eval set built {results['evalset']['built']}, "
            f"seed {results['evalset']['seed']}\n"
            f"load {results['load_ms'] / 1000:.1f} s, once, before the first turn"
            + (f" · peak GPU memory {results['peak_vram_mb']:,} MB" if results.get("peak_vram_mb") is not None else ""))
    silences = [t["leading_silence_ms"] for t in results["turns"].values()
                if t["leading_silence_ms"] is not None]
    tail = f"reply leading silence: {summarize(silences)}" if silences else \
           "no reply ever spoke, so there is no leading silence"
    parts = [head, "### Processing time", latency_table(results)]
    if has_first_audio(results):
        parts += ["### First audio: when the first piece of the reply existed",
                  latency_table(results, "first_audio_ms"),
                  "### First sound: first audio plus the silence before the first word",
                  latency_table(results, "first_sound_ms")]
    stalls = [t["stall_ms"] for t in results["turns"].values() if "stall_ms" in t]
    if stalls:
        tail += (f"\nplayback stall: {summarize(stalls)}  "
                 f"({sum(s > 0 for s in stalls)} of {len(stalls)} turns ran dry)")
    parts += ["### Where the time went", stage_table(results),
              wer_tables(results, root), tail]
    return "\n\n".join(parts)

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
