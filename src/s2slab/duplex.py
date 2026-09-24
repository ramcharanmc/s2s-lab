"""
R5 -- full duplex: no turns. The caller's audio streams in, the model's audio streams out, 80 ms at a time.

The v2 plan, R5: "What it fixes: turn-taking itself. No VAD, no endpointer, no
barge-in state machine -- the model predicts the user's channel while speaking,
so interruption handling is emergent." And the job for week 6: "Verify the
~200 ms and 5-per-A100 claims with your own numbers."

WHY THIS IS NOT A RUNG. rungs.Rung answers a finished turn -- respond(audio) ->
Turn -- and R0-R4 all do. A full-duplex model has no finished turn to answer;
forcing it into that interface would put back the abstraction the rung
removes. So this file drives a model through a whole conversation instead:
the eval clip, then TAIL_S of silence, one FRAME_S frame at a time, and keeps
every frame's output audio and text token.

    model.begin(seed)                      fresh streaming state, seeded
    model.step(frame) -> (audio | None,    one 80 ms frame in (24 kHz), what it says
                          text  | None)    back in that step out; None while warming up
    model.end()

WHAT IS MEASURED, ALL IN STREAM TIME -- the model's choices, not the GPU's speed:

    caller end     the last voiced 20 ms frame of the input (vad.frame_speech_prob, as elsewhere)
    response gap   caller end -> the start of the model's first voiced run that is still
                   going at or after it. Negative when the model was already talking as the
                   caller stopped; None when it said nothing within the tail.
    overlap        the share of the caller's speech the model spent voicing too
    backchannels   voiced runs shorter than BACKCHANNEL_S that start before the caller ends
    reply text     the model's text tokens emitted after the caller ends

A chunk emitted at step i can be heard from (i + 1) * FRAME_S -- the frame had to
arrive before the model could answer it -- so that is its stream time. A chunk
is VOICED when its RMS is above VOICED_DB; a run needs MIN_RUN voiced chunks in a
row (a single loud chunk is a click, not speech).

AND ONE HARDWARE NUMBER, apart: the real-time factor, wall time per second of
stream. Below 1.0 the model can run live; the plan's "5 conversations per A100"
is roughly 1 / RTF on that card.

Every clip is seeded (the models sample); the first REPEAT clips are driven a
second time to show that a clip's conversation depends on the clip alone.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from typing import NamedTuple

import numpy as np

from .timing import summarize
from .vad import FRAME_MS, frame_speech_prob

SR = 24_000                   # Mimi's rate
FRAME_S = 0.08                # 12.5 Hz
FRAME = int(SR * FRAME_S)     # 1920 samples
TAIL_S = 4.0                  # silence after the clip: time to answer
VOICED_DB = -35.0             # an output chunk is voiced above this RMS, dBFS
MIN_RUN = 2                   # chunks in a row: 160 ms
BACKCHANNEL_S = 1.0
SEED = 0
REPEAT = 2
IN_SR = 16_000                # the eval set's rate


class Conversation(NamedTuple):
    out: list                 # per step: the output chunk (float32, SR) or None
    text: list                # per step: the text piece or None
    wall_s: float             # wall time for all steps
    stream_s: float           # len(steps) * FRAME_S


def to_model_rate(audio: np.ndarray, sr: int = IN_SR) -> np.ndarray:
    from scipy.signal import resample_poly
    audio = np.asarray(audio, dtype=np.float32)
    if sr == SR:
        return audio
    g = math.gcd(sr, SR)
    return resample_poly(audio, SR // g, sr // g).astype(np.float32)


def converse(model, audio: np.ndarray, sr: int = IN_SR, seed: int = SEED, tail_s: float = TAIL_S) -> Conversation:
    """Stream one clip, then tail_s of silence, through the model, frame by frame.

    Takes:   a model with begin/step/end; the caller's audio and its rate.
    Returns: a Conversation; the wall time covers the steps only (no loading).
    """
    x = to_model_rate(audio, sr)
    x = np.concatenate([x, np.zeros(int(tail_s * SR), np.float32)])
    x = np.concatenate([x, np.zeros((-len(x)) % FRAME, np.float32)])
    out, text = [], []
    model.begin(seed)
    try:
        t0 = time.perf_counter()
        for k in range(0, len(x), FRAME):
            chunk, piece = model.step(x[k:k + FRAME])
            out.append(None if chunk is None else np.asarray(chunk, dtype=np.float32).reshape(-1))
            text.append(piece)
        wall = time.perf_counter() - t0
    finally:
        model.end()
    return Conversation(out, text, wall, len(out) * FRAME_S)


def caller_span(audio: np.ndarray, sr: int = IN_SR) -> tuple[float, float] | None:
    """(first, last) voiced moment of the caller's clip, in seconds; None if never voiced."""
    p = np.array(frame_speech_prob(np.asarray(audio, dtype=np.float32), sr))
    on = np.flatnonzero(p >= 0.5)
    if on.size == 0:
        return None
    f = FRAME_MS / 1000
    return float(on[0] * f), float((on[-1] + 1) * f)


def voiced_runs(conv: Conversation) -> list[tuple[float, float]]:
    """(start, end) in stream time of every voiced run of MIN_RUN chunks or more."""
    voiced = []
    for c in conv.out:
        rms = 0.0 if c is None or not len(c) else float(np.sqrt(np.mean(np.square(c, dtype=np.float64))))
        voiced.append(rms > 0 and 20 * math.log10(rms) > VOICED_DB)
    runs, i = [], 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j + 1 < len(voiced) and voiced[j + 1]:
                j += 1
            if j - i + 1 >= MIN_RUN:
                runs.append(((i + 1) * FRAME_S, (j + 2) * FRAME_S))
            i = j + 1
        else:
            i += 1
    return runs


def measure(audio: np.ndarray, conv: Conversation, sr: int = IN_SR) -> dict:
    """Everything one conversation says about turn-taking. See the docstring for the definitions."""
    span = caller_span(audio, sr)
    runs = voiced_runs(conv)
    out = {"stream_s": round(conv.stream_s, 2), "wall_s": round(conv.wall_s, 3),
           "rtf": round(conv.wall_s / conv.stream_s, 3) if conv.stream_s else None,
           "runs": [[round(a, 2), round(b, 2)] for a, b in runs]}
    if span is None:
        return {**out, "caller_end_s": None, "gap_ms": None, "overlap": None, "backchannels": 0,
                "voiced_during_caller": False, "reply_text": ""}
    start, end = span
    after = [r for r in runs if r[1] > end]
    gap = None if not after else round((after[0][0] - end) * 1000, 1)
    talk = end - start
    overlap = sum(max(0.0, min(b, end) - max(a, start)) for a, b in runs) / talk if talk > 0 else 0.0
    during = [r for r in runs if r[0] < end and r[1] > start]
    reply = "".join(t for k, t in enumerate(conv.text) if t and (k + 1) * FRAME_S >= end).strip()
    return {**out, "caller_end_s": round(end, 3), "gap_ms": gap, "overlap": round(overlap, 3),
            "backchannels": sum(1 for a, b in during if b - a < BACKCHANNEL_S and a < end),
            "voiced_during_caller": bool(during), "reply_text": " ".join(reply.split())}


def run_items(model, items, seed: int = SEED, repeat: int = REPEAT) -> dict:
    """Every eval item through the model, one conversation each; the first `repeat`
    items a second time at the end. Takes items with .id, .cohort, .audio (16 kHz).
    Returns {"turns": {id: {"cohort", **measure}}, "repeat": {id: largest output difference, 0 if identical}}."""
    turns, first = {}, {}
    for it in items:
        conv = converse(model, it.audio, IN_SR, seed)
        first[it.id] = conv
        turns[it.id] = {"cohort": it.cohort, **measure(it.audio, conv)}
    again = {}
    for it in items[:repeat]:
        a = np.concatenate([c for c in first[it.id].out if c is not None] or [np.zeros(0, np.float32)])
        b_conv = converse(model, it.audio, IN_SR, seed)
        b = np.concatenate([c for c in b_conv.out if c is not None] or [np.zeros(0, np.float32)])
        again[it.id] = ("length differs" if len(a) != len(b)
                        else round(float(np.max(np.abs(a - b))) if len(a) else 0.0, 6))
    return {"turns": turns, "repeat": again}


def report(results: dict) -> str:
    """Per cohort and over all clips: the response gap, silences, overlap, backchannels, RTF."""
    turns = results["turns"]
    groups: dict[str, list] = {}
    for t in turns.values():
        groups.setdefault(t["cohort"], []).append(t)
    groups["all"] = list(turns.values())
    fmt = lambda s, k: "—" if k in s.unsupported else f"{getattr(s, k):.0f}"
    lines = [f"{results.get('rung', '?')}: {results.get('description', '')}",
             f"{results.get('machine', '')} · commit {results.get('commit', '?')} · load {results.get('load_s', 0):.0f} s"
             + (f" · peak GPU memory {results['peak_vram_mb']:,} MB" if results.get("peak_vram_mb") else ""), "",
             "stream time: what a caller would hear, independent of the GPU's speed", "",
             "| cohort | clips | response gap p50 ms | p90 ms | no answer within the tail | "
             "voiced while the caller spoke | backchannels | median overlap | median RTF |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, ts in groups.items():
        gaps = [t["gap_ms"] for t in ts if t["gap_ms"] is not None]
        s = summarize(gaps) if gaps else None
        ov = [t["overlap"] for t in ts if t["overlap"] is not None]
        rtf = [t["rtf"] for t in ts if t["rtf"] is not None]
        cells = [fmt(s, "p50") if s else "—", fmt(s, "p90") if s else "—",
                 str(sum(t["gap_ms"] is None for t in ts)), str(sum(t["voiced_during_caller"] for t in ts)),
                 str(sum(t["backchannels"] for t in ts)),
                 f"{np.median(ov):.2f}" if ov else "—", f"{np.median(rtf):.2f}" if rtf else "—"]
        lines.append(f"| {name} | {len(ts)} | " + " | ".join(cells) + " |")
    neg = [t for t in turns.values() if t["gap_ms"] is not None and t["gap_ms"] < 0]
    lines += ["", f"already talking as the caller stopped (negative gap): {len(neg)} of {len(turns)}",
              "the same clip driven twice: " + ", ".join(
                  f"{c} {'identical' if v == 0 else v}" for c, v in results.get("repeat", {}).items())]
    shown = [(c, t) for c, t in turns.items() if t["reply_text"]][:5]
    if shown:
        lines += ["", "what it said after the caller stopped (first five clips that got an answer):"]
        lines += [f"  {c}: {t['reply_text'][:100]!r}" for c, t in shown]
    return "\n".join(lines)


def save(results: dict, path: pathlib.Path) -> pathlib.Path:
    """The whole run as JSON; evidence is never overwritten."""
    from .sidechannel import refuse_to_overwrite
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
