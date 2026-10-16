"""
M37: R4 streamed, its start timed from the turn.

M34's development turns (runs/r4-gpu/sized-dev-l40s.json) showed the second block arriving a nearly fixed
time after the first: 1,153 to 1,169 ms on the 29 turns on the 2,048-place cache, 1,528 on the one on 6,144.
That is 96 codes of the Talker (6.6 or 10.5 ms each, as M33 measured) and one decode of the second block's
window (about 520 ms). Later blocks are made faster than they play (1.92 s of audio in 1.2 to 1.5 s), so only
the second can run playback dry. The 0.5 s start buffer (the first block is 0.46 s) always waits for the
second block: about 0.46 s later than playback needs to start.

    TimedStreamingSpeechToSpeech    r4-gpu-stream-sized with its start planned (r4stream.planned_start): when
                                    the first block exists, playback starts after the block's codes at this
                                    turn's own ms a code, plus the second window's decode as timed at load,
                                    less the first block's length, plus a margin chosen on the probe clips
                                    (s2slab.r4timeddev) and frozen with PR-017.
    simulate, release               what a planned start would have done to a turn streamed with no hold, from
                                    the times its codes and pieces were made. A start rule changes when audio
                                    is handed out, never what is made, so every rule is scored on the same turns.
"""

from __future__ import annotations

import math

from .r4 import OUT_SR
from .r4sized import SizedStreamingSpeechToSpeech
from .r4stream import TimedStart, planned_start
from .r4streamdev import heard


def second_window(first: int, block: int, left: int, right: int) -> int:
    """The second block's window, in codes (r4stream.windows_for): the one whose decode the start waits on."""
    return min(left, first) + block + right


class TimedStreamingSpeechToSpeech(SizedStreamingSpeechToSpeech):
    """R4 streamed on a cache sized to the turn, its start timed from the turn (see the module's docstring)."""

    rung_id = "r4-gpu-stream-timed"
    description = ("R4 streamed on a cache sized to the turn, its start timed from the turn: playback starts when "
                   "the second block is due, by this turn's own code rate and the DiT's decode as timed at load, "
                   "less the first block's length, plus a margin; bf16, CUDA")

    def __init__(self, model=None, *, margin_s: float, **stream):
        super().__init__(model, **stream)
        self.margin_s = margin_s

    def _start(self, clock, code_ms):
        left, right = self._context()
        window = self.window_ms[second_window(self.first, self.block, left, right)]
        return TimedStart(self.emit, lambda now, samples: planned_start(now, code_ms, self.block, window,
                                                                        samples / OUT_SR, self.margin_s), clock)


def release(made: list, at: float, lead_ms: float | None, sr: int = OUT_SR) -> dict:
    """The pieces of a turn streamed with no hold, as handed out had they been held until `at` (ms): TimedStart's
    timer hands them on then, or the reply's end (its last piece) if that comes first; never before the first
    piece. Returns {"start_ms", "first_sound_ms", "stall_ms"}, scored as rungs.Turn scores a turn
    (r4streamdev.heard)."""
    if not made:
        return {"start_ms": math.nan, "first_sound_ms": math.nan, "stall_ms": 0.0}
    start = max(min(at, made[-1][0]), made[0][0])
    out = [(start, sum(n for ms, n in made if ms <= start))] + [(ms, n) for ms, n in made if ms > start]
    return {"start_ms": start, **heard(out, sr, lead_ms)}


def simulate(made: list, code_ms: list, lead_ms: float | None, margin_s: float, window_ms: float, block: int,
             sr: int = OUT_SR) -> dict:
    """What TimedStreamingSpeechToSpeech would have done to a turn streamed with no hold: the plan made when the
    first piece exists, from the codes written by then (r4stream.planned_start), then release()."""
    if not made:
        return release(made, 0.0, lead_ms, sr)
    t1, n1 = made[0]
    at = planned_start(t1, [t for t in code_ms if t <= t1], block, window_ms, n1 / sr, margin_s)
    return release(made, at, lead_ms, sr)


def needed_start(made: list, sr: int = OUT_SR) -> float:
    """The latest start, in ms, at which pieces played back to back never wait for the next one."""
    start, before = -math.inf, 0
    for ms, n in made:
        start = max(start, ms - before / sr * 1000)
        before += n
    return start
