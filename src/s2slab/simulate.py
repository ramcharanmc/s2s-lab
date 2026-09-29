"""
Synthetic cohort generation with known ground-truth turn boundaries.

WHAT THIS IS FOR. A policy sweep needs to know when the speaker ACTUALLY
finished, and no recorded corpus hands you that for free -- you would need
hand-marked turn offsets. So we synthesize: build the turn, and the boundary is
known by construction rather than annotated.

WHAT THIS IS NOT. These are PROXIES, and every result sliced by cohort must say
so in the same breath. Specifically:

  * `disfluent` is inserted silence, not stuttering. Real disfluency includes
    blocks, prolongations and repetitions with quite different acoustics.
    SEP-28k is the real thing and it is free -- use it before making any
    fairness CLAIM, as opposed to a fairness HYPOTHESIS.
  * `slow` is a time-stretch proxy for elderly speech. Real ageing changes
    voice quality, jitter and shimmer too, not just rate.
  * There is NO `accented` cohort here, deliberately. Accent is a segmental and
    phonotactic phenomenon; a pulse train cannot proxy it, and inventing one
    would be worse than admitting the gap. That cohort requires Common Voice.

The signal is a pulse train, not speech. That is a feature: it isolates the
TIMING behaviour of an endpointer from every acoustic confound, so a failure is
unambiguously the policy's.

The honest framing for a write-up: "fixed thresholds cut off speakers who pause
mid-turn, demonstrated on synthetic pause distributions and pending replication
on SEP-28k." That sentence survives review. "Our system is fair" does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SR = 16_000
F0_HOP_S = 0.010          # pitch track resolution, matching Praat's default


@dataclass
class Utterance:
    """One synthetic turn, with the answer attached."""

    audio: np.ndarray
    sr: int
    true_end_s: float              # GROUND TRUTH: when the speaker finished
    cohort: str
    words: list[tuple[str, float]]  # (word, end_time_s) -- streaming ASR truth
    n_syllables: int
    mid_turn_pauses_s: list[float]
    f0_track: list[float]          # per 10 ms frame, 0.0 where unvoiced

    # Cache slot, not data. Excluded from repr and equality so two utterances
    # compare by content rather than by whether their context was built.
    _ctx: object = field(default=None, repr=False, compare=False)

    @property
    def longest_mid_turn_pause_s(self) -> float:
        """The number that decides whether a fixed threshold survives this turn."""
        return max(self.mid_turn_pauses_s) if self.mid_turn_pauses_s else 0.0

    def context(self):
        """Build the TurnContext a policy consumes.

        Cached: the VAD pass is the expensive part of a sweep, and the audio
        never changes once an utterance is built.
        """
        from .policies import TurnContext
        from .vad import frame_speech_prob
        if self._ctx is None:
            self._ctx = TurnContext(
                speech_prob=frame_speech_prob(self.audio, self.sr),
                words=self.words,
                f0_track=self.f0_track,
                f0_hop_s=F0_HOP_S,
            )
        return self._ctx


@dataclass
class CohortSpec:
    """What makes one population different from another."""

    name: str
    syllable_s: float              # duration of one syllable
    pause_median_s: float          # median mid-turn pause
    pause_sigma: float             # log-normal shape parameter
    n_pauses: tuple[int, int]      # inclusive range of mid-turn pauses
    n_syllables: tuple[int, int]
    replaces: str                  # the real corpus this stands in for
    f0_start: float = 180.0
    f0_end: float = 150.0
    final_fall_hz: float = 25.0    # extra fall on the last syllable

COHORTS: dict[str, CohortSpec] = {
    # Fluent speaker, few and short mid-turn pauses. Most pauses fall BELOW the
    # 150 ms floor, so they are word gaps rather than pauses -- which is why a
    # short threshold looks great on this cohort and only on this cohort.
    "clean": CohortSpec(
        name="clean",
        syllable_s=0.20,
        pause_median_s=0.10,
        pause_sigma=0.35,
        n_pauses=(0, 1),
        n_syllables=(8, 14),
        replaces="LibriSpeech test-clean",
    ),

    # THE COHORT THAT MATTERS. Median mid-turn pause of 500 ms means a 300 ms or
    # 500 ms fixed threshold fires MID-SENTENCE on most of these turns. This is
    # the population that a latency-only benchmark reports as 'fast'.
    "disfluent": CohortSpec(
        name="disfluent",
        syllable_s=0.22,
        pause_median_s=0.50,
        pause_sigma=0.45,
        n_pauses=(2, 4),
        n_syllables=(8, 14),
        replaces="SEP-28k (stuttering event detection)",
    ),

    # Elderly proxy: 0.75x time-stretch, so syllables AND pauses lengthen
    # together. Note this is rate-only -- real ageing also shifts voice quality.
    "slow": CohortSpec(
        name="slow",
        syllable_s=0.28,
        pause_median_s=0.30,
        pause_sigma=0.40,
        n_pauses=(1, 3),
        n_syllables=(8, 14),
        replaces="real elderly speech (no clean free corpus; proxy only)",
    ),
}

def _voiced_burst(duration_s: float, f0: float) -> np.ndarray:
    """One syllable: a voiced tone under an amplitude envelope.

    The envelope starts and ends at zero, so concatenated bursts have no click
    at the seam -- a click is broadband energy that the VAD would read as
    speech exactly where we intended silence.
    """
    n = max(1, int(duration_s * SR))
    k = np.arange(n)
    env = np.sin(np.pi * k / n) ** 0.6
    return (0.3 * env * np.sin(2 * np.pi * f0 * k / SR)).astype("float32")


# One word per syllable, roughly. What matters is that FUNCTION WORDS are
# scattered through each sentence, so whether a pause lands on a syntactically
# incomplete tail depends on WHERE the pause fell -- not on whether the turn
# has a pause at all.
#
# Two splits, required by PR-001 (docs/preregistration.md):
#   dev   the three sentences every published result so far used, in the same
#         order, so seeds 0 and 1000 reproduce exactly. Develop against these.
#   eval  written and committed BEFORE any completeness check beyond T1 exists,
#         and used by no run until a registered confirmation. With only three
#         sentences, a check tuned on them could simply memorise them.
# The share of function words is similar in both (dev ~40%, eval ~36%), so the
# eval set is not easier for T1. Every sentence has at least 14 words, so no
# turn wraps around to the start of its sentence.
SENTENCES: dict[str, list[str]] = {
    "dev": [
        "my account number is four seven two nine and the code is alpha bravo",
        "i wanted to ask about the appointment that was scheduled for next tuesday morning",
        "the thing is i have been feeling tired and it has not really improved",
    ],
    "eval": [
        "could you tell me when the pharmacy opens because i need my refill today",
        "we moved here last month so the bill still goes to the old address",
        "my daughter usually helps me with the phone but she is away this week",
        "the doctor said i should call back if the pain in my knee gets worse",
        "the reference number they gave me is three one eight five and then a letter",
        "can someone explain why my card was declined at the store this morning please",
    ],
}


def synth_utterance(spec: CohortSpec, rng: np.random.Generator,
                    trailing_silence_s: float = 1.5,
                    split: str = "dev") -> Utterance:
    """One turn from a cohort. `true_end_s` is exact by construction.

    Trailing silence is 1.5 s so that even the 900 ms threshold can fire. A
    buffer too short to let a policy trigger produces `fired=False` for that
    policy on every turn, which is a broken benchmark rather than a slow policy.

    Mid-turn pause durations are LOG-NORMAL, not uniform. Pause distributions
    are heavy-tailed in every corpus anyone has measured, and a uniform draw
    would remove exactly the long tail that breaks fixed thresholds.
    """
    lo, hi = spec.n_syllables
    n_syll = int(rng.integers(lo, hi + 1))
    p_lo, p_hi = spec.n_pauses
    n_pause = int(rng.integers(p_lo, p_hi + 1))
    n_pause = min(n_pause, max(0, n_syll - 2))

    # Pauses go between syllables, never at the very start or the very end --
    # a pause at the end is not a mid-turn pause, it is the end of the turn.
    slots = sorted(rng.choice(np.arange(1, n_syll - 1), size=n_pause,
                              replace=False).tolist()) if n_pause else []

    parts: list[np.ndarray] = []
    f0_track: list[float] = []
    pauses: list[float] = []
    t = 0.0
    if split not in SENTENCES:
        raise KeyError(f"unknown sentence split {split!r}; have: {sorted(SENTENCES)}")
    pool = SENTENCES[split]
    sentence = pool[int(rng.integers(0, len(pool)))].split()
    words: list[tuple[str, float]] = []

    for i in range(n_syll):
        frac = i / max(1, n_syll - 1)
        f0 = spec.f0_start - (spec.f0_start - spec.f0_end) * frac
        if i == n_syll - 1:
            f0 -= spec.final_fall_hz        # turn-final fall: the yield cue
        parts.append(_voiced_burst(spec.syllable_s, f0))
        f0_track += [f0] * int(round(spec.syllable_s / F0_HOP_S))
        t += spec.syllable_s
        words.append((sentence[i % len(sentence)], t))

        if i in slots:
            d = float(rng.lognormal(math.log(spec.pause_median_s), spec.pause_sigma))
            d = min(d, 2.0)                 # cap; beyond 2 s it is a new turn
            parts.append(np.zeros(int(d * SR), dtype="float32"))
            f0_track += [0.0] * int(round(d / F0_HOP_S))
            pauses.append(d)
            t += d

    true_end_s = t
    parts.append(np.zeros(int(trailing_silence_s * SR), dtype="float32"))
    f0_track += [0.0] * int(round(trailing_silence_s / F0_HOP_S))

    return Utterance(
        audio=np.concatenate(parts),
        sr=SR,
        true_end_s=true_end_s,
        cohort=spec.name,
        words=words,
        n_syllables=n_syll,
        mid_turn_pauses_s=pauses,
        f0_track=f0_track,
    )

def make_cohort(name: str, n: int = 50, seed: int = 0, split: str = "dev") -> list[Utterance]:
    """`n` utterances from one cohort. Deterministic given `seed`.

    Seeded from an explicit integer, never from `hash()` -- Python salts string
    hashing per process, so a hash-derived seed silently changes between runs
    and makes a benchmark irreproducible without ever failing.
    """
    if name not in COHORTS:
        raise KeyError(f"unknown cohort {name!r}; have: {sorted(COHORTS)}")
    rng = np.random.default_rng(seed)
    return [synth_utterance(COHORTS[name], rng, split=split) for _ in range(n)]


def make_all(n_per_cohort: int = 50, seed: int = 0,
             split: str = "dev") -> dict[str, list[Utterance]]:
    """Every cohort, each with its own derived seed so adding a cohort does not
    change the utterances generated for the existing ones."""
    return {name: make_cohort(name, n_per_cohort, seed + i, split=split)
            for i, name in enumerate(sorted(COHORTS))}


def describe(cohorts: dict[str, list[Utterance]]) -> str:
    """Markdown summary. Print this next to any cohort-sliced result -- a reader
    cannot interpret a fairness number without the pause distribution."""
    lines = ["| cohort | n | dur p50 s | mid-turn pauses | longest pause p50 s | stands in for |",
             "|---|---:|---:|---:|---:|---|"]
    for name, utts in sorted(cohorts.items()):
        dur = np.median([len(u.audio) / u.sr for u in utts])
        npz = np.median([len(u.mid_turn_pauses_s) for u in utts])
        longest = np.median([u.longest_mid_turn_pause_s for u in utts])
        lines.append(f"| {name} | {len(utts)} | {dur:.2f} | {npz:.0f} | "
                     f"{longest:.2f} | {COHORTS[name].replaces} |")
    return "\n".join(lines)

# cd <repo> && source .venv/bin/activate && python -c "
# import sys; sys.path.insert(0,'src')
# import numpy as np
# from s2slab.simulate import make_all, describe
# from s2slab.vad import frame_speech_prob
# from s2slab.policies import get
# c = make_all(40, seed=0)
# print(describe(c)); print()
# print('premature-cutoff rate, T0 fixed thresholds:')
# print(f\"{'policy':>18} | \" + ' | '.join(f'{k:>10}' for k in sorted(c)))
# for ms in (300, 500, 700):
#     p = get(f'T0_fixed_{ms}ms'); row=[]
#     for name in sorted(c):
#         bad = sum(1 for u in c[name]
#                   if p.decide(frame_speech_prob(u.audio, u.sr), u.partial, u.f0_tail()).endpoint_s < u.true_end_s)
#         row.append(f'{bad/len(c[name]):>9.0%}')
#     print(f'{p.name:>18} | ' + ' | '.join(row))
# "

# Endpointing ladder T0-T3
# Fourth module of the CPU-only harness, and the one that carries the
# research contribution. Each rung adds exactly one signal:
#   T0  silence duration          the industry default
#   T1  + syntactic completeness  hold the floor mid-sentence
#   T2  + prosody                 falling F0 commits EARLY
#   T3  + cost asymmetry          p > R/(1+R), R = C_cutoff/C_wait
# All four share decide(speech_prob, partial, f0_tail) -> EndpointDecision,
# EOFAuthored-By: Claude Opus 5 <noreply@anthropic.com>ion written downete
# [main 283991e] Endpointing ladder T0-T3
#  2 files changed, 570 insertions(+)
#  create mode 100644 src/s2slab/policies.py
#  create mode 100644 tests/test_policies.py

