"""
Endpointing policies -- deciding WHEN THE USER STOPPED TALKING.

This decision, not ASR inference, dominates turn-taking latency in a voice
agent. Teams routinely shave 40 ms off a model while leaving 500 ms on the
table here.

The ladder. Each rung adds exactly one signal:

    T0  silence duration          the industry default, and systematically unfair
    T1  + syntactic completeness  hold the floor mid-sentence
    T2  + prosody                 falling F0 means commit EARLY
    T3  + cost asymmetry          make the tradeoff an explicit parameter

Every policy implements the same call:

    decide(speech_prob, partial, f0_tail) -> EndpointDecision

All three signals are passed to every policy, including the ones that ignore
them. That uniformity is what lets the sweep treat the ladder as a single
comparable family. If a policy needs a special case in the sweep, the interface
is wrong -- fix the interface, not the sweep.

TWO NUMBERS, ALWAYS TOGETHER. Endpoint latency on its own is meaningless: you
can drive it to zero by cutting everybody off. Latency and premature-cutoff
rate are quoted as a pair or not at all. A policy at 120 ms with a 33% cutoff
rate is not fast, it is deaf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .vad import FRAME_MS


@dataclass
class EndpointDecision:
    """Where to cut the turn, and why.

    `fired` distinguishes "a rule triggered" from "we ran out of audio". The
    sweep needs that: an endpoint at end-of-buffer is not a decision, it is the
    absence of one, and averaging the two together flatters every policy.
    """

    endpoint_s: float
    fired: bool
    reason: str

    def __str__(self) -> str:
        return f"{self.endpoint_s * 1000:.0f} ms ({self.reason})"


class EndpointPolicy:
    """Base class. Subclasses decide when the user stopped talking."""

    name = "base"

    def decide(self, speech_prob: list[float], partial: str | None = None,
               f0_tail: list[float] | None = None) -> EndpointDecision:
        raise NotImplementedError

class FixedThreshold(EndpointPolicy):
    """T0. Wait for N milliseconds of silence. Simple, and systematically wrong.

    700 ms feels laggy; 200 ms cuts people off mid-thought. Neither value is
    right, because turn-end probability is a function of syntax, prosody and
    pragmatics -- not of elapsed silence.

    The fairness problem is the important part. Disfluent speakers, elderly
    speakers, people searching for a word and non-native speakers all pause
    LONGER MID-TURN. A fixed threshold therefore cuts them off preferentially.
    It is not that the system is worse for them on average; it fails for them
    specifically, in a way aggregate latency hides completely.
    """

    def __init__(self, silence_ms: float = 700, frame_ms: float = FRAME_MS):
        self.silence_ms = silence_ms
        self.frame_ms = frame_ms
        self.name = f"T0_fixed_{int(silence_ms)}ms"

    def decide(self, speech_prob, partial=None, f0_tail=None) -> EndpointDecision:
        need = max(1, int(round(self.silence_ms / self.frame_ms)))
        run = 0
        for i, p in enumerate(speech_prob):
            run = run + 1 if p < 0.5 else 0
            if run >= need:
                # i is the last silent frame; endpoint at the end of it.
                return EndpointDecision((i + 1) * self.frame_ms / 1000, True, self.name)
        # Never fired. Report end-of-audio, but flag it as not a decision.
        return EndpointDecision(len(speech_prob) * self.frame_ms / 1000, False, "eof")

# Words that cannot end an English turn. A speaker who stops after "my" or
# "because" is mid-thought, whatever the silence detector says.
INCOMPLETE_TAILS = frozenset({
    "a", "an", "the", "my", "your", "his", "her", "its", "our", "their",
    "and", "but", "so", "or", "because", "if", "when", "while", "that",
    "to", "of", "for", "with", "at", "in", "on", "from", "by", "about",
    "is", "was", "are", "were", "am", "be", "been", "will", "would",
    "can", "could", "should", "i", "it's", "um", "uh", "like",
})


class SemanticThreshold(FixedThreshold):
    """T1. T0 plus a completion check on the streaming partial.

    Solves the case every voice agent gets wrong:

        "my account number is four seven ... two nine"
                                        ^ 600 ms of thinking

    The transcript is syntactically incomplete at that point, so hold the floor
    even though the silence threshold fired. Because we can hold, the BASE
    threshold can be shorter -- 500 ms instead of 700 -- which makes the common
    case faster and the hard case safer at the same time.

    The word list is a stub for a trained completion classifier over the
    partial. Keep the interface when you swap it; only `_should_hold` changes.
    """

    def __init__(self, silence_ms: float = 500, extension_ms: float = 400, **kw):
        super().__init__(silence_ms=silence_ms, **kw)
        self.extension_ms = extension_ms
        self.name = f"T1_semantic_{int(silence_ms)}ms"

    def _should_hold(self, partial: str | None) -> bool:
        words = (partial or "").strip().lower().split()
        if not words:
            return False
        return words[-1].strip(".,!?;:\"'") in INCOMPLETE_TAILS

    def decide(self, speech_prob, partial=None, f0_tail=None) -> EndpointDecision:
        base = super().decide(speech_prob, partial, f0_tail)
        if not base.fired:
            return base          # nothing to extend; no decision was made
        if self._should_hold(partial):
            return EndpointDecision(base.endpoint_s + self.extension_ms / 1000,
                                    True, f"{self.name}+hold")
        return EndpointDecision(base.endpoint_s, True, self.name)

def f0_slope_hz_per_s(f0_tail: list[float] | None,
                      hop_ms: float = 10.0) -> float | None:
    """Least-squares slope of the pitch tail, in Hz per second.

    Unvoiced frames arrive as 0.0 from Praat and are dropped -- treating them
    as real pitch values would drag the slope toward whatever the voicing
    pattern happened to be. Returns None when there is too little voiced pitch
    to fit a line, and callers must treat None as "no prosodic evidence"
    rather than as "flat".
    """
    if not f0_tail:
        return None
    pts = [(i, f) for i, f in enumerate(f0_tail) if f and f > 0]
    if len(pts) < 4:
        return None
    xs = [i * hop_ms / 1000 for i, _ in pts]
    ys = [f for _, f in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


class ProsodicThreshold(SemanticThreshold):
    """T2. T1 plus the turn-yield prosodic cue.

    Falling F0 with final lengthening predicts turn-yield; sustained or rising
    F0 predicts hold. This is language-general, genuinely underused, and -- the
    part that matters -- AVAILABLE BEFORE THE WORDS ARE DECODED.

    That asymmetry is the whole point. Every other signal here can only make
    you wait longer. Prosody can make you commit EARLIER, because a confident
    fall is evidence the turn is over before ASR has produced a token. It is
    the only rung on this ladder that can reduce latency rather than trade it.
    """

    def __init__(self, silence_ms: float = 500, yield_bonus_ms: float = 150,
                 slope_threshold: float = 8.0, **kw):
        super().__init__(silence_ms=silence_ms, **kw)
        self.yield_bonus_ms = yield_bonus_ms
        self.slope_threshold = slope_threshold
        self.name = f"T2_prosodic_{int(silence_ms)}ms"

    def decide(self, speech_prob, partial=None, f0_tail=None) -> EndpointDecision:
        base = super().decide(speech_prob, partial, f0_tail)
        if not base.fired:
            return base

        slope = f0_slope_hz_per_s(f0_tail)
        if slope is None:
            return EndpointDecision(base.endpoint_s, True, f"{self.name}+noprosody")

        if slope < -self.slope_threshold:          # clearly falling: they are done
            t = base.endpoint_s - self.yield_bonus_ms / 1000
            return EndpointDecision(max(t, 0.05), True, f"{self.name}+yield")
        if slope > self.slope_threshold:           # rising: they are not done
            t = base.endpoint_s + self.yield_bonus_ms / 1000
            return EndpointDecision(t, True, f"{self.name}+hold")
        return EndpointDecision(base.endpoint_s, True, self.name)

class DecisionTheoretic(EndpointPolicy):
    """T3. Endpoint when the expected cost of waiting exceeds that of cutting off.

        endpoint when   p . C_wait  >  (1 - p) . C_cutoff
        equivalently    p  >  R / (1 + R)      where R = C_cutoff / C_wait

    The value of this policy is NOT that it performs best -- it may well lose
    to T2. Its value is that it turns a hidden threshold into an explicit
    product parameter. "700 ms" encodes a cost judgement nobody wrote down;
    `cost_ratio=4` says out loud that interrupting a caller is four times worse
    than a beat of dead air, and that is a decision a product owner can argue
    with. Every fixed threshold is this policy with its assumptions hidden.

    The posterior is a stub: saturating in silence-run length. A calibrated
    turn-end model replaces `_p_end` and nothing else changes.
    """

    def __init__(self, cost_ratio: float = 4.0, frame_ms: float = FRAME_MS,
                 tau_frames: float = 12.0):
        self.cost_ratio = cost_ratio
        self.frame_ms = frame_ms
        self.tau_frames = tau_frames
        self.threshold = cost_ratio / (1.0 + cost_ratio)
        self.name = f"T3_decision_r{cost_ratio:g}"

    def _p_end(self, silence_run: int) -> float:
        """P(turn is over | this many consecutive silent frames).

        1 - exp(-run/tau). At tau=12 and 20 ms frames, 500 ms of silence gives
        p ~ 0.88. Monotone and saturating, which is the right SHAPE; it is not
        calibrated against data, and it should not be reported as if it were.
        """
        return 1.0 - math.exp(-silence_run / self.tau_frames)

    def decide(self, speech_prob, partial=None, f0_tail=None) -> EndpointDecision:
        run = 0
        for i, p in enumerate(speech_prob):
            run = run + 1 if p < 0.5 else 0
            if run and self._p_end(run) > self.threshold:
                return EndpointDecision((i + 1) * self.frame_ms / 1000, True, self.name)
        return EndpointDecision(len(speech_prob) * self.frame_ms / 1000, False, "eof")

POLICIES: dict[str, EndpointPolicy] = {}

# T0 swept across the usual operating range -- this is the curve everything
# else is measured against.
for _ms in (150, 300, 500, 700, 900):
    _p = FixedThreshold(_ms)
    POLICIES[_p.name] = _p

for _p in (SemanticThreshold(500), ProsodicThreshold(500),
           DecisionTheoretic(2.0), DecisionTheoretic(4.0), DecisionTheoretic(8.0)):
    POLICIES[_p.name] = _p

del _ms, _p


def get(name: str) -> EndpointPolicy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; have: {sorted(POLICIES)}")
    return POLICIES[name]

# cd /mnt/c/Users/ramch/Desktop/Ram/Resources/Audio/Code/s2s-lab && source .venv/bin/activate && python -c "
# import sys; sys.path.insert(0,'src')
# from s2slab.policies import POLICIES, get
# sp = [0.9]*30 + [0.1]*15 + [0.9]*20 + [0.1]*40      # speech, 300ms gap, speech, trail
# falling = [180 - 2.0*i for i in range(40)]
# rising  = [180 + 2.0*i for i in range(40)]
# print('registered:', len(POLICIES))
# for n in sorted(POLICIES):
#     print(f'  {n:24s} {get(n).decide(sp, \"my account number is\", falling)}')
# print()
# p = get('T2_prosodic_500ms')
# print('T2 falling F0:', p.decide(sp, 'that is all', falling))
# print('T2 rising  F0:', p.decide(sp, 'that is all', rising))
# "
# registered: 10
#   T0_fixed_150ms           760 ms (T0_fixed_150ms)
#   T0_fixed_300ms           900 ms (T0_fixed_300ms)
#   T0_fixed_500ms           1800 ms (T0_fixed_500ms)
#   T0_fixed_700ms           2000 ms (T0_fixed_700ms)
#   T0_fixed_900ms           2100 ms (eof)
#   T1_semantic_500ms        2200 ms (T1_semantic_500ms+hold)
#   T2_prosodic_500ms        2050 ms (T2_prosodic_500ms+yield)
#   T3_decision_r2           880 ms (T3_decision_r2)
#   T3_decision_r4           1700 ms (T3_decision_r4)
#   T3_decision_r8           1840 ms (T3_decision_r8)

# T2 falling F0: 1650 ms (T2_prosodic_500ms+yield)
# T2 rising  F0: 1950 ms (T2_prosodic_500ms+hold)
