"""
Turn-taking in two-channel audio: when each speaker talks, and how the floor changes hands.

FT-1 (Notion page 17 §7.1) fine-tunes Moshi with a LoRA and asks whether its turn-taking moves
toward a reference: "measure overlap ratio and turn-gap distribution before and after LoRA".
This module is the measure. It runs the same way on a recorded dialogue -- a human on each
channel -- and on Moshi answering one channel of it, so the two can be compared.

    a, b = channels(stereo)                   the two speakers (DailyTalk: one per channel)
    s = summary(ipus(a, sr), ipus(b, sr))     b's turn-taking against a
    distance(s_model["fto_ms"], s_ref["fto_ms"])

DEFINITIONS, after Heldner & Edlund (2010), "Pauses, gaps and overlaps in conversations":
  voiced frame    a FRAME_S frame whose level is within REL_DB of the channel's loud frames (its
                  95th-percentile frame) and above FLOOR_DBFS. Relative to each channel, because
                  recordings and a model's output sit at different levels.
  IPU             inter-pausal unit: voiced frames joined across silences shorter than MERGE_S;
                  units shorter than MIN_IPU_S dropped (a click, not speech).
  transition      b takes the floor from a: b starts an IPU after a's most recent IPU started,
                  b had not already spoken since that IPU ended, and b is still speaking when a
                  stops. Its floor-transfer offset, FTO = b's start - a's end: positive is a gap,
                  negative an overlap.
  within          a b IPU that starts and ends inside one a IPU: an overlap that does not take the
                  floor (a backchannel, or a failed attempt). Counted, never a transition.
  overlap ratio   time both channels are voiced / time either is.

Only b's transitions are measured -- the listener answering the speaker. For Moshi, b is Moshi's
channel and a the channel fed to it; for the human reference, b is the human who actually
answered. Pure numpy: no model, no GPU.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

FRAME_S = 0.02
REL_DB = 30.0          # voiced: within this of the channel's 95th-percentile frame level
FLOOR_DBFS = -60.0     # and above this, so a silent channel has no speech at all
MERGE_S = 0.2          # silences shorter than this stay inside one IPU (Heldner & Edlund's 200 ms)
MIN_IPU_S = 0.1


class IPU(NamedTuple):
    start: float
    end: float


def channels(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) of a (samples, 2) array, as float32."""
    x = np.asarray(stereo, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"expected (samples, 2), got {x.shape}")
    return x[:, 0], x[:, 1]


def voiced(audio: np.ndarray, sr: int) -> np.ndarray:
    """One bool per FRAME_S frame: is this channel speaking?"""
    n = int(round(FRAME_S * sr))
    a = np.asarray(audio, dtype=np.float64)
    k = len(a) // n
    if k == 0:
        return np.zeros(0, dtype=bool)
    level = 10 * np.log10(np.mean(a[: k * n].reshape(k, n) ** 2, axis=1) + 1e-12)
    loud = np.percentile(level, 95)
    return level > max(loud - REL_DB, FLOOR_DBFS)


def ipus(audio: np.ndarray, sr: int) -> list[IPU]:
    """The channel's inter-pausal units, in seconds, in order."""
    v = voiced(audio, sr)
    runs, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j + 1 < len(v) and v[j + 1]:
                j += 1
            runs.append([i * FRAME_S, (j + 1) * FRAME_S])
            i = j + 1
        else:
            i += 1
    merged: list[list[float]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < MERGE_S - 1e-9:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [IPU(round(s, 3), round(e, 3)) for s, e in merged if e - s >= MIN_IPU_S - 1e-9]


def transitions(a: list[IPU], b: list[IPU]) -> tuple[list[float], int]:
    """b taking the floor from a. Returns (FTOs in seconds, in order; the number of b IPUs within a)."""
    ftos, within = [], 0
    prev_b_end = -np.inf
    for ib in b:
        before = [ia for ia in a if ia.start < ib.start]
        if not before:
            prev_b_end = max(prev_b_end, ib.end)
            continue
        last_a = before[-1]
        if ib.end <= last_a.end:
            within += 1                              # inside a's IPU: does not take the floor
            continue
        if prev_b_end > last_a.end:
            prev_b_end = max(prev_b_end, ib.end)     # b already had the floor after a stopped
            continue
        ftos.append(ib.start - last_a.end)
        prev_b_end = max(prev_b_end, ib.end)
    return ftos, within


def overlap_ratio(a: list[IPU], b: list[IPU]) -> float:
    """Time both speak / time either speaks. 0 when neither speaks."""
    def total(iv):
        return sum(e - s for s, e in iv)
    both = sum(max(0.0, min(x.end, y.end) - max(x.start, y.start)) for x in a for y in b)
    either = total(a) + total(b) - both
    return both / either if either > 0 else 0.0


def summary(a: list[IPU], b: list[IPU]) -> dict:
    """b's turn-taking against a: every FTO (ms), how many are overlaps, the within count, the overlap ratio."""
    ftos, within = transitions(a, b)
    ms = [round(1000 * f, 1) for f in ftos]
    return {"n_transitions": len(ms), "fto_ms": ms, "overlapped": sum(f < 0 for f in ms), "within": within,
            "overlap_ratio": round(overlap_ratio(a, b), 4),
            "speech_s": [round(sum(e - s for s, e in a), 2), round(sum(e - s for s, e in b), 2)]}


def distance(x_ms, y_ms) -> float:
    """How far apart two FTO distributions are: the Wasserstein-1 distance, in ms -- the average
    amount each offset would have to move to turn one distribution into the other."""
    from scipy.stats import wasserstein_distance
    if not len(x_ms) or not len(y_ms):
        return float("nan")
    return float(wasserstein_distance(x_ms, y_ms))
