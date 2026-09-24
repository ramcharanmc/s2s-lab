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

    python -m s2slab.severity dev [--listen]     Kokoro, the dev voices and sentences, both
                                                 sweeps; laptop CPU, a few minutes, $0
    python -m s2slab.severity test --confirm PR-009
                                                 the registered held-out run, once

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


# --------------------------------------------------------------------------- #
# The dev run: real TTS speech, fixed sentences, both sweeps
#
#     python -m s2slab.severity dev             the table; results/severity/dev.json
#     python -m s2slab.severity dev --listen    and every render as a WAV under
#                                               results/severity/listen/, named and tagged synthetic
# --------------------------------------------------------------------------- #

KOKORO_SR = 24_000
DEV_VOICES = ("af_heart", "am_michael")
DEV_SENTENCES = {   # neutral content, so that only the delivery changes (page 09 §4.4); each has a sentence break
    "d1": "I went to the shop this morning. It was closed, so I came back home.",
    "d2": "My sister called yesterday. We talked for a while, and then I made dinner.",
    "d3": "The bus was late again. I waited by the stop, and then it started to rain.",
    "d4": "We moved the table into the kitchen. It fits better there, and the room looks bigger.",
}
SCORES = {"depression": (0, 5, 10, 15, 20, 27),    # both ends and page 09's band edges: mild 5, moderate 10, severe 15
          "mania": (0, 12, 20, 26, 40, 60)}        # mild 12, moderate 20, severe 26
SYNTHETIC_NOTE = ("SYNTHETIC SPEECH: text-to-speech with its prosody edited to page 09's severity mapping. "
                  "Research code; no clinical validity is claimed or implied.")


def synthetic_name(voice: str, sentence_id: str, t: Targets | None = None) -> str:
    """Every file this module writes says what it is in its name (page 16 §9)."""
    what = "neutral" if t is None else f"{t.condition}-{t.score:g}"
    return f"synthetic-{what}-{voice}-{sentence_id}.wav"


def write_synthetic(path, audio: np.ndarray, sr: int, voice: str, text: str, t: Targets | None = None,
                    subtype: str = "PCM_16") -> None:
    """A WAV whose own metadata says it is synthetic: title, comment and software tags. 16-bit
    for listening; "FLOAT" keeps a neutral clip exact, so a run can be re-scored from it."""
    import soundfile as sf
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=1, subtype=subtype) as f:
        f.title = "synthetic " + ("neutral" if t is None else f"{t.condition} {t.score:g}")
        f.comment = f"{SYNTHETIC_NOTE} Voice {voice}. Text: {text}"
        f.software = "s2s-lab severity.py"
        f.write(np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0))


def kokoro_speaker():
    """speak(text, voice) -> float32 at KOKORO_SR, through R0's Kokoro. Kokoro is not
    bit-identical between runs, so each neutral clip is made once and every render reuses it."""
    from .r0 import KokoroTTS
    tts = KokoroTTS()
    tts.load()

    def speak(text: str, voice: str) -> np.ndarray:
        parts = [np.asarray(r.audio, dtype=np.float32) for r in tts.pipeline(text, voice=voice) if r.audio is not None]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return speak


def run(speak, voices=DEV_VOICES, sentences=DEV_SENTENCES, scores=SCORES, sr: int = KOKORO_SR, listen=None) -> dict:
    """Every voice x sentence spoken once, trimmed, then swept through every score of each
    condition and measured again. Takes speak(text, voice) -> audio at sr; `listen`, a folder
    for the WAVs, or None. Returns {"sr", "tolerance", "rows": one per voice x sentence x condition}.
    Each row carries the SHA-256 of its neutral clip's float32 samples; with `listen`, that clip
    is written exactly (32-bit float), so every render can be made again from it -- Kokoro
    itself never speaks a sentence the same way twice."""
    import hashlib
    rows = []
    for voice in voices:
        for sid, text in sentences.items():
            neutral = trim(speak(text, voice), sr)
            digest = hashlib.sha256(np.ascontiguousarray(neutral, dtype=np.float32).tobytes()).hexdigest()
            if listen is not None:
                write_synthetic(listen / synthetic_name(voice, sid), neutral, sr, voice, text, subtype="FLOAT")
            for condition, levels in scores.items():
                steps = []
                for score in levels:
                    t = CONDITIONS[condition](score)
                    r = render(neutral, sr, t)
                    steps.append((t, achieved(neutral, r, sr)))
                    if listen is not None:
                        write_synthetic(listen / synthetic_name(voice, sid, t), r.audio, sr, voice, text, t)
                rows.append({"voice": voice, "sentence": sid, "condition": condition,
                             "neutral_s": round(len(neutral) / sr, 2), "neutral_sha256": digest,
                             "monotone": monotone(steps),
                             "steps": [{"score": t.score, "target": {f: getattr(t, f) for f in FIELDS},
                                        "achieved": dict(a._asdict()), "misses": misses(a, t)} for t, a in steps]})
    return {"sr": sr, "tolerance": TOLERANCE, "rows": rows}


def report(results: dict) -> str:
    """The sweep table: per condition, the target at each score, then what each voice x sentence
    reached; a * marks a value outside its tolerance. Then the two counts page 09 asks for."""
    fmt = {"f0_offset_hz": "{:+.1f}", "f0_dev_scale": "{:.2f}", "rate": "{:.2f}", "pause_scale": "{:.2f}"}
    lines = [SYNTHETIC_NOTE, ""]
    rows = results["rows"]
    for condition in dict.fromkeys(r["condition"] for r in rows):
        mine = [r for r in rows if r["condition"] == condition]
        scores = [s["score"] for s in mine[0]["steps"]]
        lines += [f"## {condition}, scores {' '.join(f'{s:g}' for s in scores)}", ""]
        for field in FIELDS:
            target = " ".join(fmt[field].format(s["target"][field]) for s in mine[0]["steps"])
            lines.append(f"{field:13s} target  {target}")
            for r in mine:
                vals = []
                for s in r["steps"]:
                    v = s["achieved"][field]
                    missed = any(m.startswith(field) for m in s["misses"])
                    vals.append("nan" if v != v else fmt[field].format(v) + ("*" if missed else ""))
                bad = any(f" {field}:" in m for m in r["monotone"])
                lines.append(f"{'':13s} {r['voice']}:{r['sentence']}  {' '.join(vals)}{'   OUT OF ORDER' if bad else ''}")
            lines.append("")
    sweeps = [(r, f) for r in rows for f in FIELDS]
    ordered = sum(not any(f" {f}:" in m for m in r["monotone"]) for r, f in sweeps)
    lines.append(f"monotonicity: {ordered} of {len(sweeps)} feature sweeps in order "
                 "(a sweep = one voice, sentence, condition and feature, every step)")
    for field in FIELDS:
        checks = [(s, field) for r in rows for s in r["steps"]
                  if not (field == "pause_scale" and s["achieved"][field] != s["achieved"][field])]
        met = sum(not any(m.startswith(field) for m in s["misses"]) for s, _ in checks)
        lines.append(f"within tolerance, {field}: {met} of {len(checks)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# PR-009: the held-out test (docs/preregistration.md)
#
#     python -m s2slab.severity test --confirm PR-009
#
# Held-out voices and sentences, fixed with the registration; none of them had been spoken
# by any code here before the registered run. The run refuses without the confirmation and
# refuses to overwrite its evidence, both before Kokoro loads.
# --------------------------------------------------------------------------- #

REGISTRATION = "PR-009"
RUN_PATH = "runs/severity/pr009.json"
AUDIO_DIR = "results/severity/pr009"          # the renders and the exact neutral clips (ignored)
TEST_VOICES = ("af_sarah", "am_adam")          # neither used in the severity dev work
TEST_SENTENCES = {  # neutral content, a sentence break each; written with the registration
    "t1": "The train leaves at nine. I bought a ticket, and then I found a seat by the window.",
    "t2": "I cleaned the windows today. It took most of the morning, but the light is better now.",
    "t3": "Our neighbour painted the fence. It is green now, and it looks much newer.",
    "t4": "I walked to the post office. The line was short, so I was back before lunch.",
    "t5": "The library opens at ten. I returned two books, and I borrowed a new one.",
    "t6": "We planted tomatoes last spring. They grew well, and we had enough for the whole summer.",
    "t7": "The meeting moved to Thursday. I changed my calendar, and I sent a note to the team.",
    "t8": "I made soup for dinner. It needed more salt, so I added some at the table.",
}
IN_ORDER_FIELDS = ("f0_offset_hz", "rate", "pause_scale")    # S1
S2_MIN_RANGE_IN_ORDER = 28          # of the 32 range sweeps
S4_OFFSET_SHARE = 0.90
S5_RANGE_SHARE = 0.80
CRITERION_REL = 0.10                # rate, pauses, range: within 10% of the target
OFFSET_FLOOR_HZ = 3.0               # the pitch offset: within 3 Hz or 10% of the target, whichever is larger


def in_order(row: dict, field: str) -> bool:
    """One sweep (a saved row) and one feature: every step moved the way its target moved. A
    feature that could not be measured (NaN) at any step counts as out of order."""
    steps = row["steps"]
    for s0, s1 in zip(steps, steps[1:]):
        want = s1["target"][field] - s0["target"][field]
        have = s1["achieved"][field] - s0["achieved"][field]
        if want == 0:
            continue
        if have != have or have * want <= 0:
            return False
    return True


def within(field: str, target: float, got: float) -> bool:
    """PR-009's tolerance for one check. An unmeasured value (NaN) is a miss."""
    if got != got:
        return False
    if field == "f0_offset_hz":
        return abs(got - target) <= max(OFFSET_FLOOR_HZ, CRITERION_REL * abs(target))
    return abs(got / target - 1) <= CRITERION_REL


def criteria(results: dict) -> dict:
    """PR-009's five criteria from a saved run. Returns {"S1".."S5": {"met", "of", "pass"}, "supported"}."""
    rows = results["rows"]

    def share(fields):
        checks = [(s, f) for r in rows for s in r["steps"] for f in fields]
        return sum(within(f, s["target"][f], s["achieved"][f]) for s, f in checks), len(checks)

    s1 = sum(in_order(r, f) for r in rows for f in IN_ORDER_FIELDS), len(rows) * len(IN_ORDER_FIELDS)
    s2 = sum(in_order(r, "f0_dev_scale") for r in rows), len(rows)
    s3, s4, s5 = share(("rate", "pause_scale")), share(("f0_offset_hz",)), share(("f0_dev_scale",))
    out = {"S1": {"met": s1[0], "of": s1[1], "pass": s1[0] == s1[1]},
           "S2": {"met": s2[0], "of": s2[1], "pass": s2[0] >= S2_MIN_RANGE_IN_ORDER},
           "S3": {"met": s3[0], "of": s3[1], "pass": s3[0] == s3[1]},
           "S4": {"met": s4[0], "of": s4[1], "pass": s4[0] >= S4_OFFSET_SHARE * s4[1]},
           "S5": {"met": s5[0], "of": s5[1], "pass": s5[0] >= S5_RANGE_SHARE * s5[1]}}
    out["supported"] = all(v["pass"] for v in out.values())
    return out


def validity(results: dict) -> list[str]:
    """What would make a run not the registered one: the wrong set, a tree with uncommitted changes."""
    problems = []
    if tuple(results.get("voices", ())) != TEST_VOICES or results.get("sentences") != TEST_SENTENCES:
        problems.append("not the held-out voices and sentences")
    if {k: tuple(v) for k, v in results.get("scores", {}).items()} != SCORES:
        problems.append("not the registered scores")
    if len(results["rows"]) != len(TEST_VOICES) * len(TEST_SENTENCES) * len(SCORES):
        problems.append(f"{len(results['rows'])} sweeps, not {len(TEST_VOICES) * len(TEST_SENTENCES) * len(SCORES)}")
    if str(results.get("commit", "")).endswith("-dirty") or results.get("commit") in (None, "", "unknown"):
        problems.append(f"commit {results.get('commit')!r}: not a clean committed tree")
    return problems


def format_criteria(results: dict) -> str:
    """The verdict block, as the registration words it."""
    c = criteria(results)
    pct = lambda k: f" ({100 * c[k]['met'] / c[k]['of']:.1f}%)" if c[k]["of"] else ""
    mark = lambda k: "PASS" if c[k]["pass"] else "FAIL"
    lines = [f"{REGISTRATION}, the registered criteria (docs/preregistration.md):",
             f"  S1  page 09's monotonicity test -- pitch offset, rate, pauses in order, every sweep: "
             f"{c['S1']['met']} of {c['S1']['of']}   {mark('S1')}",
             f"  S2  pitch range in order, at least {S2_MIN_RANGE_IN_ORDER} of 32: {c['S2']['met']} of {c['S2']['of']}   {mark('S2')}",
             f"  S3  rate and pauses within 10%, every check: {c['S3']['met']} of {c['S3']['of']}   {mark('S3')}",
             f"  S4  pitch offset within 3 Hz or 10%, at least 90%: {c['S4']['met']} of {c['S4']['of']}{pct('S4')}   {mark('S4')}",
             f"  S5  pitch range within 10%, at least 80%: {c['S5']['met']} of {c['S5']['of']}{pct('S5')}   {mark('S5')}"]
    problems = validity(results)
    if problems:
        lines.append(f"{REGISTRATION}: NOT THE REGISTERED RUN -- " + "; ".join(problems))
    elif c["supported"]:
        lines.append(f"{REGISTRATION}: SUPPORTED")
    else:
        lines.append(f"{REGISTRATION}: NOT SUPPORTED -- failed " + ", ".join(k for k in ("S1", "S2", "S3", "S4", "S5")
                                                                         if not c[k]["pass"]))
    return "\n".join(lines)


def registered_run(confirm: str, speaker=None, run_path=RUN_PATH, audio_dir=AUDIO_DIR) -> dict:
    """The one PR-009 run. Checks the confirmation and the evidence path BEFORE any TTS loads;
    speaker() -> speak (default: Kokoro). Saves the run file (every measure, the neutral clips'
    SHA-256, the criteria) and writes the renders and the exact neutral clips to audio_dir."""
    import datetime
    import json
    import pathlib

    from .sidechannel import refuse_to_overwrite
    from .sweep import _git_commit
    if confirm != REGISTRATION:
        raise PermissionError(f"the held-out voices and sentences are locked: run with --confirm {REGISTRATION}, "
                              "once, after the registration is committed")
    run_path = pathlib.Path(run_path)
    refuse_to_overwrite(run_path)
    audio = pathlib.Path(audio_dir)
    audio.mkdir(parents=True, exist_ok=True)
    speak = (speaker or kokoro_speaker)()
    results = run(speak, voices=TEST_VOICES, sentences=TEST_SENTENCES, listen=audio)
    results.update(registration=REGISTRATION, commit=_git_commit(),
                   ran=datetime.datetime.now().isoformat(timespec="seconds"), voices=list(TEST_VOICES),
                   sentences=dict(TEST_SENTENCES), scores={k: list(v) for k, v in SCORES.items()})
    results["criteria"] = criteria(results)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    import json
    import pathlib
    import sys

    args = sys.argv[1:]
    if args[:1] == ["dev"] and not set(args[1:]) - {"--listen"}:
        out = pathlib.Path("results/severity")
        listen = out / "listen" if "--listen" in args else None
        (listen or out).mkdir(parents=True, exist_ok=True)
        results = run(kokoro_speaker(), listen=listen)
        (out / "dev.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(report(results))
        print(f"\nsaved -> {out / 'dev.json'}" + (f"; listen: {listen}/" if listen else ""))
    elif args[:1] == ["test"] and len(args) == 3 and args[1] == "--confirm":
        results = registered_run(args[2])
        print(report(results))
        print("\n" + format_criteria(results))
        print(f"\nsaved -> {RUN_PATH}; renders and exact neutral clips: {AUDIO_DIR}/")
    else:
        sys.exit("usage: python -m s2slab.severity dev [--listen]\n"
                 f"       python -m s2slab.severity test --confirm {REGISTRATION}")
