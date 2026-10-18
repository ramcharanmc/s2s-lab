"""
M38, step 2: R1 with its first piece's leading silence trimmed, on development clips, before anything is
registered.

Step 1 (runs/tts/leading-silence.json) found Kokoro starting every clause with about 320 ms of silence: the
padding its duration predictor puts before the first phoneme. r1-gpu-trim (s2slab.r1) cuts the reply's first
piece to 20 ms before its first frame within 50 dB, and leaves later pieces alone. R1 plays its pieces back to
back while the next clause renders, so a first piece 300 ms shorter must be followed 300 ms sooner: the risk is
a stall, which step 1 could not see.

One T4 container, the 30 probe clips (development; the eval set is not touched): r1-gpu-trim, every turn's pieces
kept with how much its first lost. R1 as it is, on the very same turn, is computed from those pieces: the same
pieces, the first one longer by what was cut, its leading silence longer by the same (the cut is below 50 dB of
the piece's loudest, so silent by the rungs' 40 dB measure too). A trim changes when the first sound is heard,
never what is said.

THE RULE, written before the run (docs/m38-log.md): GO to PR-018 if the trimmed first sound's p50 is at most
P50_GATE of R1's on the same turns AND the trimmed turns that stall are no more than STALL_SLACK above R1's;
otherwise M38 stops here, nothing registered.

    modal run modal_app/r0_bench.py::trimdev       one T4; saves runs/r1-gpu/trim-dev-t4.json
"""

from __future__ import annotations

import math

import numpy as np

P50_GATE = 0.85
STALL_SLACK = 2
OUT_SR = 24_000                                      # Kokoro's


def run(clips: dict, sr: int, rung=None) -> dict:
    """r1-gpu-trim on every clip: each turn's pieces, first sound, stalls, and what the trim cut."""
    from .rungs import get
    rung = rung or get("r1-gpu-trim")
    load_ms = rung.load()
    turns = {}
    for clip, audio in clips.items():
        t = rung.respond(audio, sr)
        turns[clip] = {"reply_text": t.reply_text, "total_ms": round(t.total_ms, 1),
                       "first_audio_ms": _num(t.first_audio_ms), "first_sound_ms": _num(t.first_sound_ms),
                       "leading_silence_ms": _num(t.leading_silence_ms), "stall_ms": round(t.stall_ms, 1),
                       "chunks": [[round(ms, 1), n] for ms, n in t.chunks], **rung.turn_extra()}
    return {"rung": rung.rung_id, "load_ms": round(load_ms, 1), "clips": list(clips), "sr": OUT_SR, "turns": turns}


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 1)


def as_is(turn: dict, sr: int = OUT_SR) -> dict:
    """R1 as it is on the same turn: the first piece longer by what the trim cut, its leading silence too."""
    from .r4streamdev import heard
    chunks = [list(c) for c in turn["chunks"]]
    cut = int(round((turn.get("trim_ms") or 0.0) / 1000 * sr))
    if chunks:
        chunks[0][1] += cut
    lead = None if turn["leading_silence_ms"] is None else turn["leading_silence_ms"] + (turn.get("trim_ms") or 0.0)
    return heard([tuple(c) for c in chunks], sr, lead)


def paired(blob: dict) -> dict:
    """{clip: {"trimmed": first sound and stall as measured, "as_is": R1's on the same turn}}."""
    out = {}
    for clip, t in blob["turns"].items():
        out[clip] = {"trimmed": {"first_sound_ms": t["first_sound_ms"], "stall_ms": t["stall_ms"]},
                     "as_is": as_is(t, blob.get("sr", OUT_SR)), "trim_ms": t.get("trim_ms")}
    return out


def _p(values, q):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return round(float(np.percentile(v, q)), 1) if v else None


def gate(blob: dict) -> dict:
    """The rule above, applied."""
    p = paired(blob)
    rows = list(p.values())
    trimmed = [r["trimmed"]["first_sound_ms"] for r in rows]
    as_is_fs = [r["as_is"]["first_sound_ms"] for r in rows]
    out = {"trimmed_p50": _p(trimmed, 50), "trimmed_p90": _p(trimmed, 90),
           "as_is_p50": _p(as_is_fs, 50), "as_is_p90": _p(as_is_fs, 90),
           "trim_p50": _p([r["trim_ms"] for r in rows], 50),
           "stalled_trimmed": sum((r["trimmed"]["stall_ms"] or 0) > 0 for r in rows),
           "stalled_as_is": sum((r["as_is"]["stall_ms"] or 0) > 0 for r in rows),
           "longest_stall_trimmed": max((r["trimmed"]["stall_ms"] or 0) for r in rows) if rows else None}
    out["ratio_p50"] = round(out["trimmed_p50"] / out["as_is_p50"], 3) if out["as_is_p50"] else None
    out["ratio_p90"] = round(out["trimmed_p90"] / out["as_is_p90"], 3) if out["as_is_p90"] else None
    ok_speed = out["ratio_p50"] is not None and out["ratio_p50"] <= P50_GATE
    ok_stalls = out["stalled_trimmed"] <= out["stalled_as_is"] + STALL_SLACK
    out["go"] = ok_speed and ok_stalls
    out["why"] = (f"first sound p50 {out['ratio_p50']} of R1's (needs <= {P50_GATE}); stalled turns {out['stalled_trimmed']} "
                  f"against R1's {out['stalled_as_is']} (needs <= {out['stalled_as_is'] + STALL_SLACK})")
    return out


def format_dev(blob: dict) -> str:
    g = gate(blob)
    f = lambda x: "-" if x is None else f"{x:,.0f}"                                     # noqa: E731
    return "\n".join([
        f"R1 with its first piece trimmed ({blob['rung']}), on {len(blob['clips'])} probe clips (development)", "",
        "| on the same turns | first sound p50 ms | p90 | stalled turns |", "|---|---:|---:|---:|",
        f"| R1 as it is | {f(g['as_is_p50'])} | {f(g['as_is_p90'])} | {g['stalled_as_is']} |",
        f"| first piece trimmed | {f(g['trimmed_p50'])} | {f(g['trimmed_p90'])} | {g['stalled_trimmed']} |",
        f"| ratio | {g['ratio_p50']} | {g['ratio_p90']} | |", "",
        f"The trim cut {f(g['trim_p50'])} ms at the median; the longest stall with it {f(g['longest_stall_trimmed'])} ms.",
        f"The rule (written before the run): {'GO: register PR-018' if g['go'] else 'STOP: nothing registered'}: {g['why']}"])
