"""
Barge-in policies -- deciding WHETHER THE AGENT SHOULD STOP TALKING.

While the agent speaks, the caller makes a sound. A genuine interruption means
stop; "mm-hm", a cough or "okay yeah, that's fine" means keep going. The ladder,
each rung adding exactly one signal, as in policies.py:

    B0  duration             stop after N ms of caller sound -- the default
    B1  + the words so far   hold while every word is an acknowledgement   (next commit)
    B2  + voicing            a sound with no pitch is not speech           (next commit)

Every policy implements the same call:

        decide(ctx: BargeInContext) -> BargeInDecision

NO LOOKING AHEAD, BY CONSTRUCTION. The context holds the caller's channel and
answers questions by time -- how much caller sound by t, what the recognizer
had produced by t, the pitch before t. A decision at t cannot read anything
after t. That includes the VAD's noise floor: vad.frame_speech_prob() takes it
from the whole clip, which is the future. causal_speech_prob() takes it from
the frames heard so far.

THE POLICY NEVER SEES THE LABEL. It has to notice the caller through the VAD,
as a deployed system would. Labels are for scoring (bargein_eval.py).

TWO NUMBERS, ALWAYS TOGETHER. Stop latency alone is meaningless: B0 at 200 ms
is fast and stops for every "mm-hm". Latency and wrong decisions are quoted as
a pair or not at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vad import FRAME_MS, frame_rms

SOUND_PROB = 0.5        # a frame at or above this is caller sound
HANG_MS = 300.0         # this much quiet ends a sound: "wait, wait" is one sound, not two
NOISE_PERCENTILE = 10.0
SNR_DB_FULL = 20.0      # as vad.frame_speech_prob: probability 1.0 this far above the floor



def causal_speech_prob(audio: np.ndarray, sr: int) -> list[float]:
    """Per-frame probability of caller sound, using only the frames heard so far.

    Takes:   mono audio. Returns: one value in [0, 1] per 20 ms frame.

    Same ramp as vad.frame_speech_prob -- 0 at the noise floor, 1 at 20 dB over
    it -- but frame k's floor is the 10th percentile of frames 0..k, not of the
    whole clip. The first frames have next to no history, so they read as quiet:
    a listener that has just been switched on does not know the room yet.
    """
    rms = frame_rms(np.asarray(audio, dtype=np.float32), sr)
    if rms.size == 0:
        return []
    db = 20.0 * np.log10(rms + 1e-12)
    out = []
    for k in range(len(db)):
        floor = np.percentile(db[: k + 1], NOISE_PERCENTILE)
        out.append(float(np.clip((db[k] - floor) / SNR_DB_FULL, 0.0, 1.0)))
    return out


def sound_run_ms(speech_prob: list[float], frame_ms: float = FRAME_MS) -> list[float]:
    """How much caller sound the current event holds, at the end of each frame.

    Takes:   per-frame probabilities. Returns: ms of sound frames in the event
             so far; the event ends, and the count returns to 0, once HANG_MS
             of quiet has passed. Computed left to right, so entry k depends on
             frames 0..k only.
    """
    out, run, quiet = [], 0.0, 0.0
    for p in speech_prob:
        if p >= SOUND_PROB:
            run, quiet = run + frame_ms, 0.0
        else:
            quiet += frame_ms
            if quiet >= HANG_MS:
                run = 0.0
        out.append(run)
    return out



@dataclass
class BargeInDecision:
    """Whether to stop the agent, when, and why. stop_s is None for "keep going"."""

    stop_s: float | None
    reason: str

    @property
    def fired(self) -> bool:
        return self.stop_s is not None

    def __str__(self) -> str:
        return f"stop at {self.stop_s:.2f} s ({self.reason})" if self.fired else f"keep going ({self.reason})"


@dataclass
class BargeInContext:
    """Everything a policy may look at, addressable BY TIME, on the agent's clock.

    start_s and end_s bound the stretch the policy is asked about (the item's
    window). Audio before start_s is history the policy may use -- it has been
    listening all along -- but a decision is only taken inside the window.
    Every policy receives all three signals, including the ones it ignores.
    """

    audio: np.ndarray
    sr: int
    start_s: float
    end_s: float
    partials: list[tuple[float, str]] = field(default_factory=list)   # (available at s, transcript so far)
    f0_track: list[float] = field(default_factory=list)               # Hz per 10 ms, 0 = unvoiced
    frame_ms: float = FRAME_MS
    f0_hop_s: float = 0.010
    speech_prob: list[float] = field(init=False)
    sound_ms: list[float] = field(init=False)

    def __post_init__(self):
        self.speech_prob = causal_speech_prob(self.audio, self.sr)
        self.sound_ms = sound_run_ms(self.speech_prob, self.frame_ms)

    def sound_ms_at(self, t_s: float) -> float:
        """ms of caller sound in the event in progress at t_s (0 if none)."""
        k = int(round(t_s * 1000 / self.frame_ms)) - 1
        return self.sound_ms[k] if 0 <= k < len(self.sound_ms) else 0.0

    def partial_at(self, t_s: float) -> str:
        """The latest transcript the recognizer had DELIVERED by t_s. A partial
        still being computed at t_s does not exist yet."""
        ready = [text for at, text in self.partials if at <= t_s]
        return ready[-1] if ready else ""

    def f0_before(self, t_s: float, ms: float = 400.0) -> list[float]:
        """Pitch over the last `ms` before t_s, unvoiced frames included as 0."""
        end = int(t_s / self.f0_hop_s)
        return list(self.f0_track[max(0, end - int(ms / 1000 / self.f0_hop_s)): end])



class BargeInPolicy:
    """Base class. The scan lives here and runs frame by frame through the
    window; at each frame it asks `_stop_reason` whether to stop GIVEN WHAT IS
    KNOWN AT THAT INSTANT. The first frame that says stop ends the scan."""

    name = "base"
    baseline = False        # only baselines may be scored on the frozen set without a registration

    def _stop_reason(self, ctx: BargeInContext, t_s: float) -> str | None:
        raise NotImplementedError

    def decide(self, ctx: BargeInContext) -> BargeInDecision:
        first = max(1, int(np.ceil(ctx.start_s * 1000 / ctx.frame_ms)))
        last = min(len(ctx.speech_prob), int(ctx.end_s * 1000 / ctx.frame_ms))
        for k in range(first, last + 1):
            t_s = k * ctx.frame_ms / 1000
            reason = self._stop_reason(ctx, t_s)
            if reason:
                return BargeInDecision(t_s, reason)
        return BargeInDecision(None, "no rule fired")


class DurationThreshold(BargeInPolicy):
    """B0: stop once the caller has made `sound_ms` of sound. What most voice
    agents ship with, and what the probe set was built to defeat."""

    baseline = True

    def __init__(self, sound_ms: float = 400):
        self.threshold_ms = sound_ms
        self.name = f"B0_duration_{sound_ms:.0f}ms"

    def _stop_reason(self, ctx, t_s):
        if ctx.sound_ms_at(t_s) >= self.threshold_ms:
            return f"{self.threshold_ms:.0f} ms of caller sound"
        return None


POLICIES: dict[str, BargeInPolicy] = {p.name: p for p in (
    DurationThreshold(200), DurationThreshold(400), DurationThreshold(800), DurationThreshold(1500))}


def get(name: str) -> BargeInPolicy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; have: {sorted(POLICIES)}")
    return POLICIES[name]

