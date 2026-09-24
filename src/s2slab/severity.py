"""
P2c -- clinical severity to prosody. A score on a rating scale becomes acoustic targets, the
targets are rendered onto neutral speech, and the render is measured to check it did what
was asked.

    page 09 §6          the severity mapping, taken word for word
    page 09 §7, strategy 5
                        neural TTS for a natural base, then exact, auditable edits in Praat
    page 16, weeks 5-6  the monotonicity test in CI: raise the severity, and every rendered
                        feature must move the way the mapping says

RESEARCH CODE. NO CLINICAL VALIDITY IS CLAIMED OR IMPLIED. The formulas are one page's
hand-written anchors, not fitted to clinical speech. The output is synthetic speech with its
prosody edited to a specification, and every file made from it is named and tagged as
synthetic (page 16 §9).

    t = depression(15)                 PHQ-9, 0-27  (or mania(30): YMRS, 0-60)
    r = render(neutral, sr, t)         one Praat pass: pitch tier and duration tier, then
                                       overlap-add resynthesis (PSOLA)
    got = achieved(neutral, r, sr)     the four features, measured again from the audio
    misses(got, t)                     [] when every target is met within TOLERANCE

WHAT IS RENDERED -- page 09's four acoustic outputs:
    f0_offset_hz   every pitch point shifted by this many Hz
    f0_dev_scale   page 09's "f0_variance_scale", applied the way its own formula applies it:
                   each point pulled toward the centre, centre + a * (f0 - centre). So the
                   spread of F0 scales by a, and its variance by a squared. The name says
                   variance; the formula scales deviations; the formula is followed.
    rate           speech (not pauses) sped up or slowed: duration factor 1 / rate
    pause_scale    every pause of prosody.pause_segments' 150 ms or more stretched by this factor
Pitch and durations change in ONE resynthesis, so pitch is kept while durations change --
page 09's separate WSOLA step folded into the same pass.

NOT RENDERED: response latency, elaboration and interruption. They are about the dialogue --
when to answer, how much to say -- not the sound of one utterance. They stay in the targets,
so nothing in page 09's mapping is silently dropped.

DECISIONS MADE ON DEV AUDIO (Kokoro af_heart and am_michael, two sentences, 2026-09-24):
  - The centre of the compression is the MEDIAN of the measured F0, not page 09's mean.
    Praat's pitch track has octave jumps on real speech, and the mean follows them.
  - The measures are robust versions of the same things: the median offset, and the
    middle-half (25th-75th percentile) spread for the range. A linear edit moves both
    exactly, and a stray octave jump moves neither much.
  - Rate is measured from durations -- speech time is the clip minus its pauses -- not from
    counting voiced frames, which drifted by up to 9% at high speed.
  - Pauses are measured where the render's own time map puts them, as silent time: a pause
    shortened to 0.1 s is still measured, where a 150 ms pause detector would lose it.

What dev showed (2 voices x 2 sentences x 2 sweeps of 6 scores, measured again from the
audio): pitch offset, rate and pauses met their targets in 143 of 144 checks; the pitch
RANGE in 34 of 48, and in 2 of 16 sweeps it stepped the wrong way once, where the target
moved by about 0.1. The range is this render's weak feature on real speech. Reported, not
tuned away. Page 09 §7 also warns that beyond about +-30% rate or 40 Hz of shift audio
starts to sound processed; mania at YMRS 60 asks for +45 Hz at 1.8x speed, past both.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from .prosody import pause_segments

PITCH_FLOOR_HZ, PITCH_CEILING_HZ = 75, 600    # as prosodycheck: raised voices go above 400
TIME_STEP_S = 0.01
EDGE_S = 0.005        # duration-tier points this far inside a pause's edges
TRIM_DB = -40.0       # trim(): frames this far below the loudest are the silence around speech
SILENT_DBFS = -80.0   # trim(): a clip whose loudest frame is under this has no speech at all
SILENCE_FRAME_MS = 10 # silent_s(): finer than pause_segments' 20 ms, since a shortened pause is ~0.1 s
TOLERANCE = {"f0_offset_hz": 3.0,   # Hz, absolute
             "f0_dev_scale": 0.10, "rate": 0.10, "pause_scale": 0.10}   # relative to the target
# Provisional, set on dev. PR-009 fixes the tolerances before the held-out run.
FIELDS = ("f0_offset_hz", "f0_dev_scale", "rate", "pause_scale")


class Scale(NamedTuple):
    name: str
    low: float
    high: float


PHQ9 = Scale("PHQ-9", 0, 27)      # page 09: mild 5-9, moderate 10-14, severe 15+
YMRS = Scale("YMRS", 0, 60)       # page 09: mild 12-19, moderate 20-25, severe 26+


def normalize(score: float, scale: Scale) -> float:
    """Page 09: n = (score - min) / (max - min). A score off the scale is an error, not a clip."""
    if not scale.low <= score <= scale.high:
        raise ValueError(f"{scale.name} runs {scale.low}-{scale.high}; got {score}")
    return (score - scale.low) / (scale.high - scale.low)


class Targets(NamedTuple):
    condition: str
    score: float
    n: float
    f0_offset_hz: float
    f0_dev_scale: float
    rate: float
    pause_scale: float
    dialogue: dict          # page 09's non-acoustic outputs: kept, not rendered


def depression(phq9: float) -> Targets:
    """Page 09 §6, anchored to PHQ-9. Note its offset starts at -5 Hz: a score of 0 is not neutral."""
    n = normalize(phq9, PHQ9)
    return Targets("depression", phq9, n, -5 - 20 * n, 1.0 - 0.6 * n, 1.0 - 0.5 * n, 1.0 + 3.0 * n,
                   {"response_latency_ms": 1200 * n, "elaboration_level": 1.0 - 0.7 * n})


def mania(ymrs: float) -> Targets:
    """Page 09 §6, anchored to YMRS."""
    n = normalize(ymrs, YMRS)
    return Targets("mania", ymrs, n, +45 * n, 1.0 + 0.8 * n, 1.0 + 0.8 * n, 1.0 - 0.7 * n,
                   {"interrupt_prob": 0.5 * n})


CONDITIONS = {"depression": depression, "mania": mania}


def trim(audio: np.ndarray, sr: int, db: float = TRIM_DB, frame_s: float = 0.01) -> np.ndarray:
    """The clip without the silence before the first word and after the last (20 ms kept each side).
    A TTS pads its output; left in, the padding would be stretched as a pause."""
    n = int(frame_s * sr)
    a = np.asarray(audio, dtype=np.float32)
    level = 10 * np.log10(np.mean(a[: len(a) // n * n].reshape(-1, n).astype(np.float64) ** 2, axis=1) + 1e-12)
    if level.max() < SILENT_DBFS:
        return a[:0]
    on = np.flatnonzero(level > level.max() + db)
    return a[max(0, on[0] - 2) * n: (on[-1] + 3) * n]


def f0_hz(audio: np.ndarray, sr: int) -> np.ndarray:
    """F0 of the voiced frames, Hz (Praat, PITCH_FLOOR_HZ to PITCH_CEILING_HZ, 10 ms)."""
    import parselmouth
    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=sr)
    f0 = sound.to_pitch(time_step=TIME_STEP_S, pitch_floor=PITCH_FLOOR_HZ,
                        pitch_ceiling=PITCH_CEILING_HZ).selected_array["frequency"]
    return f0[f0 > 0]


def duration_points(audio: np.ndarray, sr: int, t: Targets) -> tuple[list, list]:
    """The duration tier: 1 / rate over speech, pause_scale inside every pause, with the change
    made within EDGE_S of each pause edge. Returns (points [(time, factor)], pauses [(start, dur)])."""
    speech = 1.0 / t.rate
    pauses = pause_segments(np.asarray(audio, dtype=np.float32), sr)
    pts = [(0.0, speech)]
    for start, dur in pauses:
        end = start + dur
        pts += [(start, speech), (start + EDGE_S, t.pause_scale), (end - EDGE_S, t.pause_scale), (end, speech)]
    pts.append((len(audio) / sr, speech))
    return pts, pauses


def warp(time: float, points: list) -> float:
    """Where input time `time` lands in the output: the integral of the piecewise-linear
    duration factor, as Praat applies it."""
    out = 0.0
    for (t0, f0), (t1, f1) in zip(points, points[1:]):
        if time <= t0:
            break
        u = min(time, t1)
        fu = f0 + (f1 - f0) * (u - t0) / (t1 - t0) if t1 > t0 else f0
        out += (u - t0) * (f0 + fu) / 2
    return out


class Render(NamedTuple):
    audio: np.ndarray       # float32, the input's rate
    pauses_in: list         # [(start, end)] in the input, seconds
    pauses_out: list        # the same pauses, where the time map puts them in the output


def render(audio: np.ndarray, sr: int, t: Targets) -> Render:
    """Page 09 strategy 5's edit, in one pass. Takes trimmed mono audio. The pitch tier is
    shifted and pulled toward the median of the measured F0; the duration tier is
    duration_points(); Praat's overlap-add resynthesis applies both."""
    import parselmouth
    from parselmouth.praat import call
    centre = float(np.median(f0_hz(audio, sr)))
    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=sr)
    manipulation = call(sound, "To Manipulation", TIME_STEP_S, PITCH_FLOOR_HZ, PITCH_CEILING_HZ)
    tier = call(manipulation, "Extract pitch tier")
    call(tier, "Formula", f"{centre} + {t.f0_dev_scale} * (self - {centre}) + {t.f0_offset_hz}")
    call([tier, manipulation], "Replace pitch tier")
    durations = call(manipulation, "Extract duration tier")
    points, pauses = duration_points(audio, sr, t)
    for time, factor in points:
        call(durations, "Add point", time, factor)
    call([durations, manipulation], "Replace duration tier")
    out = call(manipulation, "Get resynthesis (overlap-add)").values[0].astype(np.float32)
    return Render(out, [(s, s + d) for s, d in pauses], [(warp(s, points), warp(s + d, points)) for s, d in pauses])


def silent_s(audio: np.ndarray, sr: int, intervals: list, frame_ms: float = SILENCE_FRAME_MS) -> float:
    """Silent time inside the given intervals, by prosody.pause_segments' rule for silence -- a
    frame under the 20th percentile of the clip's frame energy, floored at 5% of its peak -- on
    10 ms frames, counting a frame when its centre lies inside an interval."""
    hop = int(sr * frame_ms / 1000)
    n = len(audio) // hop
    energy = np.sqrt(np.mean(np.asarray(audio[: n * hop], dtype=np.float64).reshape(n, hop) ** 2, axis=1) + 1e-12)
    quiet = energy < max(np.percentile(energy, 20), energy.max() * 0.05)
    centres = (np.arange(n) + 0.5) * frame_ms / 1000
    inside = np.zeros(n, dtype=bool)
    for start, end in intervals:
        inside |= (centres >= start) & (centres < end)
    return float((quiet & inside).sum()) * frame_ms / 1000


class Achieved(NamedTuple):
    f0_offset_hz: float     # median F0, rendered minus neutral
    f0_dev_scale: float     # middle-half F0 spread, rendered over neutral
    rate: float             # speech time, neutral over rendered
    pause_scale: float      # silent time in the pauses, rendered over neutral; NaN without pauses


def achieved(neutral: np.ndarray, r: Render, sr: int) -> Achieved:
    """The four targets measured again, from the two waveforms (and the render's own time map,
    to find the pauses in the output)."""
    fn, fr = f0_hz(neutral, sr), f0_hz(r.audio, sr)
    spread = lambda f: float(np.subtract(*np.percentile(f, [75, 25])))
    pause_n, pause_r = silent_s(neutral, sr, r.pauses_in), silent_s(r.audio, sr, r.pauses_out)
    speech_n, speech_r = len(neutral) / sr - pause_n, len(r.audio) / sr - pause_r
    return Achieved(float(np.median(fr) - np.median(fn)), spread(fr) / spread(fn), speech_n / speech_r,
                    pause_r / pause_n if pause_n else float("nan"))


def misses(got: Achieved, t: Targets, tolerance: dict = TOLERANCE) -> list[str]:
    """Every target missed by more than its tolerance, as readable lines; [] when all are met.
    A pause target is skipped (not failed) when the clip has no pause to stretch."""
    out = []
    if abs(got.f0_offset_hz - t.f0_offset_hz) > tolerance["f0_offset_hz"]:
        out.append(f"f0_offset_hz {got.f0_offset_hz:+.1f} against {t.f0_offset_hz:+.1f}")
    for field in FIELDS[1:]:
        want, have = getattr(t, field), getattr(got, field)
        if field == "pause_scale" and math.isnan(have):
            continue
        if abs(have / want - 1) > tolerance[field]:
            out.append(f"{field} {have:.2f} against {want:.2f}")
    return out


def monotone(rows: list[tuple[Targets, Achieved]]) -> list[str]:
    """Page 09's test: along a severity sweep, each measured feature must move, step by step, in
    the direction its target moves. Takes rows in rising severity. Returns the violations; []
    is a pass. A step where the target does not move is not tested."""
    out = []
    for field in FIELDS:
        for (t0, a0), (t1, a1) in zip(rows, rows[1:]):
            want = getattr(t1, field) - getattr(t0, field)
            have = getattr(a1, field) - getattr(a0, field)
            if want == 0 or (math.isnan(have)):
                continue
            if have * want <= 0:
                out.append(f"{t0.condition} {field}: {t0.score} -> {t1.score} moved {have:+.3f}, should move {want:+.3f}")
    return out


def sweep(neutral: np.ndarray, sr: int, condition: str, scores) -> list[tuple[Targets, Achieved]]:
    """One neutral clip rendered at each score of one condition, each measured. Scores in rising order."""
    rows = []
    for score in scores:
        t = CONDITIONS[condition](score)
        rows.append((t, achieved(neutral, render(neutral, sr, t), sr)))
    return rows
