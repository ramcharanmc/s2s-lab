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

        decide(ctx: TurnContext) -> EndpointDecision

Policies never receive pre-sliced signals. They ask the context what was known
at a given time, which is what makes it impossible for a decision at t to read
evidence from after t.

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
from dataclasses import dataclass, field

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

@dataclass
class TurnContext:
    """Everything a policy may look at, addressable BY TIME.

    The invariant this type exists to enforce: a policy deciding at time t must
    not see evidence from after t. Previously the caller pre-sliced a "tail" of
    the buffer, which leaked turn-final pitch into mid-turn decisions and made
    the prosodic policy look considerably better than it is.
    """
    

    speech_prob: list[float]
    words: list[tuple[str, float]] = field(default_factory=list)   # (word, end_s)
    f0_track: list[float] = field(default_factory=list)
    frame_ms: float = FRAME_MS
    f0_hop_s: float = 0.010

    def partial_at(self, t_s: float) -> str:
        """What ASR would have decoded by t_s. Words ending later are unseen."""
        return " ".join(w for w, end in self.words if end <= t_s)

    def f0_before(self, t_s: float, ms: float = 400.0) -> list[float]:
        """The last `ms` of VOICED pitch before t_s. Never reaches past it.

        Trailing unvoiced frames are skipped first. A candidate endpoint is by
        definition several hundred ms into silence, so a window of wall-clock
        time ending there contains no pitch at all and the prosodic policy goes
        blind. The turn-yield cue lives in the last voiced stretch before the
        pause, which is exactly what this returns.

        This still cannot see past t_s -- the slice is taken first, and only
        then trimmed backwards.
        """
        end = int(t_s / self.f0_hop_s)
        seg = list(self.f0_track[:end])
        while seg and seg[-1] <= 0:
            seg.pop()
        n = int(ms / 1000 / self.f0_hop_s)
        return seg[-n:] if n < len(seg) else seg

class EndpointPolicy:
    """Base class. Subclasses decide when the user stopped talking.

    The scan lives here and runs frame by frame. At every frame with an active
    silence run it asks `_requirement` how much silence would justify firing
    GIVEN THE EVIDENCE AVAILABLE AT THAT INSTANT. That is what a streaming
    endpointer actually does, and it is what stops a policy from reading
    evidence that has not happened yet.
    """

    name = "base"

    def _requirement(self, ctx: TurnContext, t_s: float) -> tuple[float, str]:
        """(silent frames required, reason tag) using only evidence before t_s."""
        raise NotImplementedError

    def decide(self, ctx: TurnContext) -> EndpointDecision:
        run = 0
        for i, p in enumerate(ctx.speech_prob):
            run = run + 1 if p < 0.5 else 0
            if not run:
                continue
            t_s = (i + 1) * ctx.frame_ms / 1000
            need, reason = self._requirement(ctx, t_s)
            if run >= need:
                return EndpointDecision(t_s, True, reason)
        n = len(ctx.speech_prob)
        return EndpointDecision(n * ctx.frame_ms / 1000, False, "eof")

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

    def __init__(self, silence_ms: float = 700):
        self.silence_ms = silence_ms
        self.name = f"T0_fixed_{int(silence_ms)}ms"

    def _requirement(self, ctx, t_s):
        return self.silence_ms / ctx.frame_ms, self.name
    
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

    def _requirement(self, ctx, t_s):
        base = self.silence_ms / ctx.frame_ms
        if self._should_hold(ctx.partial_at(t_s)):
            return base + self.extension_ms / ctx.frame_ms, f"{self.name}+hold"
        return base, self.name

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
                 slope_threshold: float = 40.0, **kw):
        # slope_threshold separates DECLINATION from a turn-final fall. Every
        # speaker's pitch drifts down gradually across an utterance; only the
        # last syllable of a completed turn drops sharply. Measured on the
        # synthetic cohorts: declination sits near -14 Hz/s and mid-turn
        # stretches near 0, while the turn-final fall reaches -103 Hz/s. 40 Hz/s
        # sits in that gap.
        #
        # This number is TUNED ON THE SAME DATA IT IS EVALUATED ON, which is a
        # methodological hole. On real speech it must be re-fit on a dev split
        # -- or better, learned, since declination steepness varies by speaker,
        # utterance length and language. Report it as a fitted parameter, never
        # as a constant of nature.
        super().__init__(silence_ms=silence_ms, **kw)
        self.yield_bonus_ms = yield_bonus_ms
        self.slope_threshold = slope_threshold
        self.name = f"T2_prosodic_{int(silence_ms)}ms"

    def _requirement(self, ctx, t_s):
        need, reason = super()._requirement(ctx, t_s)
        slope = f0_slope_hz_per_s(ctx.f0_before(t_s), ctx.f0_hop_s * 1000)
        if slope is None:
            return need, f"{reason}+noprosody"
        bonus = self.yield_bonus_ms / ctx.frame_ms
        if slope < -self.slope_threshold:
            return max(need - bonus, 1.0), f"{reason}+yield"
        if slope > self.slope_threshold:
            return need + bonus, f"{reason}+hold"
        return need, reason

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

    def __init__(self, cost_ratio: float = 4.0, tau_frames: float = 12.0):
        self.cost_ratio = cost_ratio
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

    @property
    def min_silence_frames(self) -> float:
        """Invert the posterior. 1 - exp(-run/tau) > thr  ->  run > -tau.ln(1 - thr)."""
        return -self.tau_frames * math.log(1.0 - self.threshold)

    def _requirement(self, ctx, t_s):
        return self.min_silence_frames, self.name

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
