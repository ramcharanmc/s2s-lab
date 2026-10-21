"""
P1c's latency criterion (v1 §8): does the affect side-channel add under 30 ms to a cascade's turn? On the laptop,
free, against a rule written before the run (docs/m45-log.md). PR-005 decided P1c's other half and measured no time.

    PYTHONPATH=src python -m s2slab.p1clatency      the eval set's 200 clips; writes runs/prosody/p1c-latency.json once

WHAT THE SIDE-CHANNEL ADDS, per turn, that R1 does not do already:
  a  eGeMAPS: openSMILE's 88 functionals over the turn's audio (affect.egemaps)
  b  the tagger: standardize, multiply, softmax (Tagger.tag)
  c  the longer prompt: R1's LLM's prefill, one forward pass, with " Their tone of voice was <word>." after the
     transcript, less the same pass without it. The word is the tagger's for that clip. Floored at 0: an added
     sentence cannot make the pass faster, so a negative difference is the clock's noise
THE TRANSCRIPTS are R0's on the same laptop (runs/r0/evalset-laptop-cpu.json, base.en, the words R1 hears too), so
no recognizer runs here. Warmed up once; each part timed REPEATS times per clip, the median kept; c's two passes
alternate which goes first.

THE RULE, written before the run: the serial sum a + b + c, as if the side-channel ran after the recognizer. MET if
its p90 over the 200 clips is under BAR_MS; NOT MET otherwise. p50 and the maximum beside it.
EXPLORATORY, outside the rule: the parallel bound, max(0, a - the recognizer's time) + b + c, since eGeMAPS needs
only the audio and could run beside the recognizer; the recognizer's time is R0's on this laptop, from its run file.

WHAT IT CANNOT SHOW: the laptop's CPU is not the T4 container's; c on a CPU overstates a GPU's prefill; a streaming
eGeMAPS, its features taken while the caller speaks, is not built and would change the answer.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import statistics
import time
from typing import Callable

import numpy as np

from .timing import summarize

RUN = pathlib.Path("runs/prosody/p1c-latency.json")
R0_RUN = pathlib.Path("runs/r0/evalset-laptop-cpu.json")
BAR_MS = 30.0
REPEATS = 3
PARTS = ("a_ms", "b_ms", "c_ms", "added_ms", "parallel_ms")


def tagged(transcript: str, word: str) -> str:
    """What R1's LLM is handed with the side-channel: the words, then one sentence about the voice."""
    return f"{transcript} Their tone of voice was {word}."


def _ms(fn: Callable, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return 1000 * (time.perf_counter() - t0), out


def time_clip(audio: np.ndarray, sr: int, transcript: str, extract: Callable, tagger, prefill: Callable,
              repeats: int = REPEATS) -> dict:
    """The three parts for one clip, each the median of `repeats` timings."""
    a, x = [], None
    for _ in range(repeats):
        ms, x = _ms(extract, audio, sr)
        a.append(ms)
    b, tag = [], None
    for _ in range(repeats):
        ms, tag = _ms(tagger.tag, x)
        b.append(ms)
    text = tagged(transcript, tag.word)
    with_, without = [], []
    for r in range(repeats):                        # which pass goes first alternates, so drift falls on both
        order = ((with_, text), (without, transcript)) if r % 2 == 0 else ((without, transcript), (with_, text))
        for sink, t in order:
            sink.append(_ms(prefill, t)[0])
    a_ms, b_ms = statistics.median(a), statistics.median(b)
    c_with, c_without = statistics.median(with_), statistics.median(without)
    c_ms = max(0.0, c_with - c_without)
    return {"word": tag.word, "a_ms": round(a_ms, 3), "b_ms": round(b_ms, 4), "c_with_ms": round(c_with, 3),
            "c_without_ms": round(c_without, 3), "c_ms": round(c_ms, 3), "added_ms": round(a_ms + b_ms + c_ms, 3)}


def run(items, transcripts: dict[str, str], asr_ms: dict[str, float], extract: Callable, tagger, prefill: Callable,
        sr: int = 16_000, repeats: int = REPEATS) -> dict:
    """Every clip timed; items are (id, cohort, audio). The first clip is run once untimed, to warm everything up."""
    items = list(items)
    first_id, _, first_audio = items[0]
    x = extract(first_audio, sr)
    prefill(tagged(transcripts[first_id], tagger.tag(x).word))
    prefill(transcripts[first_id])
    turns = {}
    for cid, cohort, audio in items:
        t = time_clip(audio, sr, transcripts[cid], extract, tagger, prefill, repeats)
        asr = asr_ms.get(cid)
        t["asr_ms"] = asr
        t["parallel_ms"] = (round(max(0.0, t["a_ms"] - asr) + t["b_ms"] + t["c_ms"], 3) if asr is not None else None)
        turns[cid] = {"cohort": cohort, "duration_s": round(len(audio) / sr, 2), **t}
    return {"repeats": repeats, "bar_ms": BAR_MS, "turns": turns, "summary": summary(turns)}


def summary(turns: dict) -> dict:
    out = {}
    for part in PARTS:
        vals = [t[part] for t in turns.values() if t.get(part) is not None]
        s = summarize(vals)
        out[part] = {"n": s.n, "p50": round(s.p50, 3), "p90": round(s.p90, 3),
                     "max": round(max(vals), 3) if vals else None}
    return out


def gate(blob: dict) -> dict:
    s = blob["summary"]["added_ms"]
    return {"met": s["n"] > 0 and s["p90"] < blob["bar_ms"], "p90": s["p90"], "bar_ms": blob["bar_ms"]}


def format_report(blob: dict) -> str:
    s, g = blob["summary"], gate(blob)
    names = {"a_ms": "a  eGeMAPS", "b_ms": "b  the tagger", "c_ms": "c  the longer prompt",
             "added_ms": "a + b + c  (THE RULE)", "parallel_ms": "parallel bound (exploratory)"}
    lines = [f"P1c's latency, {s['added_ms']['n']} clips, {blob.get('machine', '')}, median of {blob['repeats']} each",
             "", f"{'':32}{'p50 ms':>10}{'p90 ms':>10}{'max ms':>10}"]
    for part in PARTS:
        p = s[part]
        if p["n"]:
            lines.append(f"{names[part]:32}{p['p50']:>10.2f}{p['p90']:>10.2f}{p['max']:>10.2f}")
    lines += ["", f"{'MET' if g['met'] else 'NOT MET'}: the serial sum's p90 {g['p90']:.1f} ms against the bar "
                  f"{g['bar_ms']:.0f} ms (v1 §8, P1c)"]
    by = {}
    for t in blob["turns"].values():
        by.setdefault(t["cohort"], []).append(t["added_ms"])
    lines.append("by cohort, a + b + c p50: " + ", ".join(f"{c} {statistics.median(v):.1f}" for c, v in by.items()))
    return "\n".join(lines)


class Prefill:
    """R1's LLM's prefill: one forward pass over the prompt R1 sends it (QwenLLM.inputs), no token generated."""

    def __init__(self, llm=None):
        if llm is None:
            from .r0 import QwenLLM
            llm = QwenLLM()
        self.llm = llm

    def load(self) -> "Prefill":
        self.llm.load()
        return self

    def __call__(self, text: str) -> None:
        import torch
        with torch.inference_mode():
            self.llm.model(**self.llm.inputs(text))


def main(argv=None) -> None:
    import argparse
    import warnings

    from .affect import TAGGER, Tagger, egemaps
    from .evalset import load, verify
    from .sweep import _git_commit
    warnings.filterwarnings("ignore")
    argparse.ArgumentParser(description=__doc__.split("\n\n")[1]).parse_args(argv)
    if RUN.exists():
        raise SystemExit(f"{RUN} exists: the measurement is recorded once")
    problems = verify()
    if problems:
        raise SystemExit("the eval set is not the frozen set:\n  " + "\n  ".join(problems[:10]))
    r0 = json.loads(R0_RUN.read_text(encoding="utf-8"))["turns"]
    items = [(it.id, it.cohort, it.audio) for it in load()]
    head = {"run": "M45, P1c's latency criterion", "commit": _git_commit(),
            "machine": f"laptop CPU ({platform.system()}, {os.cpu_count()} threads)",
            "transcripts": str(R0_RUN)}
    blob = {**head, **run(items, {k: v["transcript"] for k, v in r0.items()},
                          {k: v["stages_ms"]["asr"] for k, v in r0.items()},
                          egemaps, Tagger.load(TAGGER), Prefill().load())}
    RUN.parent.mkdir(parents=True, exist_ok=True)
    RUN.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    print(format_report(blob))


if __name__ == "__main__":
    main()
