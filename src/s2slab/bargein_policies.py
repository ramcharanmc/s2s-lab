"""
Barge-in policies -- deciding WHETHER THE AGENT SHOULD STOP TALKING.

While the agent speaks, the caller makes a sound. A genuine interruption means
stop; "mm-hm", a cough or "okay yeah, that's fine" means keep going. The ladder,
each rung adding exactly one signal, as in policies.py:

    B0  duration             stop after N ms of caller sound -- the default
    B1  + the words so far   hold while every word is an acknowledgement   (next commit)
    B2  + is it speech?      a cough is not a word, whatever the recognizer writes
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
from typing import NamedTuple

import numpy as np

from .vad import FRAME_MS, frame_rms

SOUND_PROB = 0.5        # a frame at or above this is caller sound
HANG_MS = 300.0         # this much quiet ends a sound: "wait, wait" is one sound, not two
NOISE_PERCENTILE = 10.0
SNR_DB_FULL = 20.0      # as vad.frame_speech_prob: probability 1.0 this far above the floor
FLOOR_TIME_S = 3.0      # B1: a caller still talking after this has taken the floor, whatever the opening
NO_SPEECH_MAX = 0.25    # B2: the recognizer's own "this is not speech" score; dev speech never exceeds 0.20
WORDS_PER_S_MAX = 10.0  # B2: dev speech peaks under 9 words per second of sound; invented text runs at 14-39
VOICED_MIN = 0.25       # B2: at least this share of the sound must have pitch; dev speech never goes under 0.30

# How an acknowledgement OPENS. General English, written before any scoring and
# tuned on the synthetic dev set only. "uh-huh" is split at its hyphen, so
# "uh" and "huh" are here; so is laughter, which the recognizer writes as words.
ACK_OPENERS = frozenset("""
    yeah yes yep yup yah okay ok right sure alright fine great good perfect exactly
    absolutely definitely cool nice thanks lovely brilliant
    uh huh mm hm hmm mhm mmhmm aha ah oh
    ha hah haha hahaha heh hehe ho hee
""".split())
ACK_PHRASES = frozenset({("i", "see"), ("got", "it"), ("go", "ahead"), ("go", "on"), ("carry", "on"),
                         ("no", "problem"), ("no", "worries"), ("of", "course"), ("all", "right"),
                         ("thank", "you"), ("makes", "sense"), ("sounds", "good"), ("fair", "enough")})
# After an acknowledging opening, these mean the caller wants the floor after all.
STOP_CUES = frozenset("""
    no nope not wait stop but sorry actually hold hang excuse pardon repeat wrong however instead
    dont didnt doesnt isnt wasnt cant cannot wont
""".split())
STOP_PHRASES = frozenset({("can", "you"), ("could", "you"), ("would", "you"), ("can", "i"), ("could", "i"),
                          ("can", "we"), ("could", "we")})



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


def sound_events(speech_prob: list[float], frame_ms: float = FRAME_MS) -> tuple[list[float], list[float | None]]:
    """The caller's sound events, frame by frame.

    Takes:   per-frame probabilities.
    Returns: (run_ms, start_s), one entry per frame, as known at the END of that
             frame: ms of sound frames in the event so far, and when that event
             began (None if no event is in progress). An event ends, and both
             reset, once HANG_MS of quiet has passed. Computed left to right, so
             entry k depends on frames 0..k only.
    """
    runs, starts, run, quiet, start = [], [], 0.0, 0.0, None
    for k, p in enumerate(speech_prob):
        if p >= SOUND_PROB:
            if start is None:
                start = k * frame_ms / 1000
            run, quiet = run + frame_ms, 0.0
        else:
            quiet += frame_ms
            if quiet >= HANG_MS:
                run, start = 0.0, None
        runs.append(run)
        starts.append(start)
    return runs, starts


def sound_run_ms(speech_prob: list[float], frame_ms: float = FRAME_MS) -> list[float]:
    """ms of caller sound in the event so far, per frame (see sound_events)."""
    return sound_events(speech_prob, frame_ms)[0]


class Partial(NamedTuple):
    """One transcript from the recognizer, and when the policy could first read it."""

    at_s: float             # delivered: when the audio ended PLUS the time the recognizer took
    text: str
    final: bool = False     # True if the audio ended in quiet, so its last word is a whole word
    event_s: float = -1.0   # when the sound event it transcribes began: partials of one event share it
    no_speech: float = 0.0  # the recognizer's own probability that the audio held no speech    

def words_of(text: str) -> list[str]:
    """ "Uh-huh, yeah -- I'm right" -> [uh, huh, yeah, im, right]: lower case, hyphens
    split, everything that is not a letter or digit dropped."""
    tokens = ("".join(c for c in w if c.isalnum()) for w in text.lower().replace("-", " ").split())
    return [t for t in tokens if t]



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
    partials: list[Partial] = field(default_factory=list)             # in the order they were delivered
    f0_track: list[float] = field(default_factory=list)               # Hz per 10 ms, 0 = unvoiced
    frame_ms: float = FRAME_MS
    f0_hop_s: float = 0.010
    speech_prob: list[float] = field(init=False)
    sound_ms: list[float] = field(init=False)
    event_start: list[float | None] = field(init=False)
    _voicing: dict = field(init=False, default_factory=dict)
    

    def __post_init__(self):
        self.partials = [Partial(*p) for p in self.partials]
        self.speech_prob = causal_speech_prob(self.audio, self.sr)
        self.sound_ms, self.event_start = sound_events(self.speech_prob, self.frame_ms)

    def _frame(self, t_s: float) -> int:
        return int(round(t_s * 1000 / self.frame_ms)) - 1

    def sound_ms_at(self, t_s: float) -> float:
        """ms of caller sound in the event in progress at t_s (0 if none)."""
        k = self._frame(t_s)
        return self.sound_ms[k] if 0 <= k < len(self.sound_ms) else 0.0

    def event_start_at(self, t_s: float) -> float | None:
        """When the sound event in progress at t_s began; None if there is none."""
        k = self._frame(t_s)
        return self.event_start[k] if 0 <= k < len(self.event_start) else None

    def latest_partial(self, t_s: float) -> Partial | None:
        """The latest transcript the recognizer had DELIVERED by t_s. A partial
        still being computed at t_s does not exist yet."""
        ready = [p for p in self.partials if p.at_s <= t_s]
        return ready[-1] if ready else None

    def partial_at(self, t_s: float) -> str:
        p = self.latest_partial(t_s)
        return p.text if p else ""

    def f0_before(self, t_s: float, ms: float = 400.0) -> list[float]:
        """Pitch over the last `ms` before t_s, unvoiced frames included as 0."""
        end = int(t_s / self.f0_hop_s)
        return list(self.f0_track[max(0, end - int(ms / 1000 / self.f0_hop_s)): end])

    def voicing(self, from_s: float, to_s: float) -> tuple[float, float]:
        """(seconds with pitch, seconds of caller sound) between from_s and to_s.

        The pitch tracker is run on that stretch ALONE, so nothing after to_s
        can influence it -- a whole-take pitch track is smoothed across time and
        would let the future in. Costs a few ms; kept, since the scan asks again.
        """
        key = (round(from_s, 3), round(to_s, 3))
        if key not in self._voicing:
            from .probes import _pitch_track
            a, b = max(0, int(from_s * self.sr)), int(to_s * self.sr)
            voiced_s = sum(1 for f in _pitch_track(self.audio[a:b]) if f > 0) * 0.010 if b - a > 1600 else 0.0
            k0, k1 = max(0, int(from_s * 1000 / self.frame_ms)), int(to_s * 1000 / self.frame_ms)
            sound_s = sum(1 for p in self.speech_prob[k0:k1] if p >= SOUND_PROB) * self.frame_ms / 1000
            self._voicing[key] = (voiced_s, sound_s)
        return self._voicing[key]


class BargeInPolicy:
    """Base class. The scan lives here and runs frame by frame through the
    window; at each frame it asks `_stop_reason` whether to stop GIVEN WHAT IS
    KNOWN AT THAT INSTANT. The first frame that says stop ends the scan."""

    name = "base"
    baseline = False        # only baselines may be scored on the frozen set without a registration
    needs_partials = False  # True: the scorer must run the recognizer and fill ctx.partials

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


class WordsSoFar(BargeInPolicy):
    """B1: read what the caller has said so far. Stop for anything that is not
    an acknowledgement; hold while it is one -- unless a stop cue follows, or
    the caller is still talking after FLOOR_TIME_S.

    THE OPENING DECIDES. "Uh-huh, yeah, I can see it on the screen" is full of
    ordinary words; what marks it as an acknowledgement is how it starts.
    "Yeah but that's wrong" starts the same way, so the hold is given up at "but".

    A PARTIAL'S LAST WORD IS NOT TRUSTED while the caller is still making sound:
    the audio may end mid-word, and half of "writing" comes back as "right".
    Nor is a word that may be the first half of a phrase: "sure, no" is
    "sure, no worries" a moment later.

    AN OPENING, ONCE HEARD, STAYS HEARD. The same sound is transcribed again and
    again as it grows, and the recognizer changes its mind: "I see." came back a
    second later as "Ice. Ice.". Later partials of the same event can end a hold
    with a stop cue; they cannot re-decide how the caller began.
    """

    name = "B1_words"
    needs_partials = True

    def __init__(self, floor_time_s: float = FLOOR_TIME_S, min_unsettled_words: int = 2):
        self.floor_time_s = floor_time_s
        self.min_unsettled_words = min_unsettled_words
        if min_unsettled_words != 2:
            self.name = f"B1_words_min{min_unsettled_words}"

    @staticmethod
    def settled(words: list[str], final: bool) -> list[str]:
        """The words that can be trusted. Unless the audio ended in quiet, the
        last word may be half a word, and the one before it half a phrase."""
        if final:
            return words
        words = words[:-1]
        if words and words[-1] in {first for first, _ in ACK_PHRASES | STOP_PHRASES}:
            words = words[:-1]
        return words

    @staticmethod
    def reading(words: list[str], opening_heard: bool = False) -> tuple[str, str]:
        """("hold" | "stop" | "nothing", why) for the words heard so far.
        opening_heard: an earlier partial of this event already began as an acknowledgement."""
        if not words:
            return "nothing", "no words yet"
        pairs = list(zip(words, words[1:]))
        if not opening_heard and tuple(words[:2]) not in ACK_PHRASES and words[0] not in ACK_OPENERS:
            return "stop", f"heard {words[0]!r}, not an acknowledgement"
        k = 0
        while k < len(words):
            pair = pairs[k] if k < len(pairs) else None
            if pair in ACK_PHRASES:                     # "no worries": the "no" is not a refusal
                k += 2
            elif pair in STOP_PHRASES:
                return "stop", f"heard {' '.join(pair)!r} after an acknowledgement"
            elif words[k] in STOP_CUES:
                return "stop", f"heard {words[k]!r} after an acknowledgement"
            else:
                k += 1
        return "hold", f"acknowledging: {' '.join(words[:3])!r}"

    def trusts(self, ctx: BargeInContext, partial: Partial) -> bool:
        """B1 believes every transcript. B2 does not."""
        return True

    def _stop_reason(self, ctx, t_s):
        latest = ctx.latest_partial(t_s)
        if latest is None:
            return None
        holding = False
        for partial in ctx.partials:                    # this event's partials, as they were delivered
            if partial.at_s > t_s or partial.event_s != latest.event_s or not self.trusts(ctx, partial):                continue
            words = self.settled(words_of(partial.text), partial.final)
            verdict, why = self.reading(words, holding)
            if verdict == "stop" and not holding and not partial.final and len(words) < self.min_unsettled_words:
                continue                                # one word from a snippet still in progress: wait
            if verdict == "stop":
                return why
            holding = holding or verdict == "hold"
        started = ctx.event_start_at(t_s)
        if holding and started is not None and t_s - started >= self.floor_time_s:
            return f"still talking after {self.floor_time_s:.0f} s"
        return None


class SpeechLike(WordsSoFar):
    """B2: B1, but a transcript only counts if the sound it came from is speech.

    A recognizer asked to transcribe a cough does not return nothing. It returns
    "You", or "Poop.", or forty fluent words about a video, and B1 stops the
    agent for them: 6 of 10 real coughs and laughs. Three independent signs that
    a transcript was invented, each set with margin on the dev data:

      the recognizer itself doubts it     no_speech at or above NO_SPEECH_MAX
      nobody talks that fast              more than WORDS_PER_S_MAX words per second of sound
      there is hardly any voice in it     less than VOICED_MIN of the sound has pitch

    Any one of them, and the transcript is ignored -- as if it had never been
    delivered. Everything else is B1's: the opening, the stop cues, the floor time.
    """

    name = "B2_speechlike"

    def trusts(self, ctx, partial):
        if partial.no_speech >= NO_SPEECH_MAX:
            return False
        voiced_s, sound_s = ctx.voicing(partial.event_s, partial.at_s)
        if sound_s <= 0:
            return False
        return len(words_of(partial.text)) / sound_s <= WORDS_PER_S_MAX and voiced_s / sound_s >= VOICED_MIN


POLICIES: dict[str, BargeInPolicy] = {p.name: p for p in (
    DurationThreshold(200), DurationThreshold(400), DurationThreshold(800), DurationThreshold(1500),
    WordsSoFar(), WordsSoFar(min_unsettled_words=1), SpeechLike())}

def get(name: str) -> BargeInPolicy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; have: {sorted(POLICIES)}")
    return POLICIES[name]

