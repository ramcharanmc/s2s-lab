"""
Tests for s2slab.bargein_asr and the B1 policy -- with a fake recognizer.

The fake "hears" a fixed sentence at a fixed speaking rate: asked about the
first 0.9 s of a sound, it returns the words that fit into 0.9 s, the last one
cut short if the audio ends inside it. So every partial is known in advance,
including the half-words a real recognizer produces.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from s2slab.bargein_asr import LEAD_S, cache_path, partials_for, stream_partials
from s2slab.bargein_policies import BargeInContext, Partial, WordsSoFar, words_of

SR = 16_000
WORD_S = 0.3


def room(total_s: float, *bursts: tuple[float, float]) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = 1e-4 * rng.standard_normal(int(total_s * SR)).astype(np.float32)
    for start, lasts in bursts:
        n, a = int(lasts * SR), int(start * SR)
        x[a:a + n] += (0.3 * np.sin(2 * np.pi * 200 * np.arange(n) / SR)).astype(np.float32)
    return x


class FakeRecognizer:
    """Hears `sentence`, one word per WORD_S of LOUD audio; a word the audio cuts
    into comes back as its first half plus a dash, as Whisper does."""

    def __init__(self, sentence: str):
        self.words, self.calls = sentence.split(), []

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls.append(len(audio) / SR)
        loud_s = float((np.abs(audio) > 0.05).sum()) / SR / 0.893       # a 0.3 sine is over 0.05 for 89.3% of the time
        whole = int(loud_s / WORD_S + 1e-6)
        out = self.words[:whole]
        if whole < len(self.words) and loud_s - whole * WORD_S > 0.1:
            out = out + [self.words[whole][: max(1, len(self.words[whole]) // 2)] + "-"]
        return " ".join(out)


def ctx(audio, partials=(), start_s=0.0) -> BargeInContext:
    return BargeInContext(audio, SR, start_s, len(audio) / SR, partials=list(partials))


# --------------------------------------------------------------------------- #
# Streaming partials
# --------------------------------------------------------------------------- #


def test_a_partial_is_delivered_after_the_recognizer_has_taken_its_time():
    c = ctx(room(8.0, (2.0, 0.9)))
    partials = stream_partials(c, FakeRecognizer("no wait please"), latency_ms=500)
    first = partials[0]
    assert first.at_s == pytest.approx(2.2 + 0.5, abs=0.06)             # 200 ms of sound, then 500 ms of work
    assert first.event_s == pytest.approx(2.0, abs=0.04) and not first.final


def test_the_recognizer_does_one_call_at_a_time():
    """At 500 ms a call, a 3 s sound gets about six looks; at 70 ms, a look every 200 ms."""
    sound = room(9.0, (2.0, 3.0))
    slow, fast = FakeRecognizer("a " * 10), FakeRecognizer("a " * 10)
    a = stream_partials(ctx(sound), slow, latency_ms=500)
    b = stream_partials(ctx(sound), fast, latency_ms=70)
    gaps = np.diff([p.at_s for p in a if not p.final])
    assert all(g >= 0.5 - 1e-6 for g in gaps)
    assert len(b) > 2 * len(a) and min(np.diff([p.at_s for p in b if not p.final])) >= 0.2 - 1e-6


def test_the_last_partial_of_a_sound_is_final_and_hears_all_of_it():
    rec = FakeRecognizer("no wait please")
    partials = stream_partials(ctx(room(8.0, (2.0, 1.0))), rec, latency_ms=70)     # room for all three words
    assert partials[-1].final and partials[-1].text == "no wait please"
    assert all(not p.final for p in partials if p.at_s < 3.0)            # the caller was still going
    assert any(p.text.endswith("-") for p in partials)                   # and half-words were delivered on the way


def test_the_audio_starts_a_little_before_the_vad_noticed():
    rec = FakeRecognizer("no")
    stream_partials(ctx(room(8.0, (2.0, 0.9))), rec, latency_ms=70)
    assert rec.calls[0] == pytest.approx(0.2 + LEAD_S, abs=0.06)         # 200 ms of sound + the lead


def test_nothing_is_transcribed_outside_the_window_or_without_sound():
    assert stream_partials(ctx(room(6.0)), FakeRecognizer("no"), latency_ms=70) == []
    rec = FakeRecognizer("no")
    assert stream_partials(ctx(room(9.0, (1.0, 0.9)), start_s=4.0), rec, latency_ms=70) == [] and not rec.calls


def test_two_sounds_are_two_events():
    partials = stream_partials(ctx(room(10.0, (2.0, 0.6), (5.0, 0.6))), FakeRecognizer("okay then"), latency_ms=70)
    assert len({p.event_s for p in partials}) == 2


def test_measured_latency_is_what_the_call_really_took():
    class Slow(FakeRecognizer):
        def transcribe(self, audio):
            import time
            time.sleep(0.05)
            return super().transcribe(audio)

    c = ctx(room(6.0, (2.0, 0.6)))
    partials = stream_partials(c, Slow("no wait"), latency_ms=None)
    assert partials[0].at_s >= 2.2 + 0.05 - 0.03


def test_partials_are_cached_and_the_recognizer_is_not_loaded_twice(tmp_path):
    contexts = {"ss01": ctx(room(8.0, (2.0, 0.9)))}
    loads = []

    def factory():
        loads.append(1)
        return FakeRecognizer("no wait please")

    cache = tmp_path / "bargein" / "partials.json"
    first = partials_for(contexts, factory, 70, cache)
    again = partials_for(contexts, factory, 70, cache)
    assert first == again and len(loads) == 1
    assert isinstance(again["ss01"][0], Partial) and json.loads(cache.read_text())["ss01"]
    partials_for(contexts, factory, 70, cache, refresh=True)
    assert len(loads) == 2
    assert cache_path("bargein_dev", None).name == "partials-bargein_dev-measured.json"
    assert cache_path("bargein", 70).name == "partials-bargein-70ms.json"


# --------------------------------------------------------------------------- #
# B1: reading the words
# --------------------------------------------------------------------------- #


def test_words_are_lower_case_without_punctuation_and_uh_huh_is_two():
    assert words_of("Uh-huh, yeah -- I'm right.") == ["uh", "huh", "yeah", "im", "right"]
    assert words_of("") == [] and words_of(" ... ") == []


@pytest.mark.parametrize("heard, verdict", [
    ("no", "stop"), ("wait wait", "stop"), ("i do not need all that", "stop"), ("could you repeat that", "stop"),
    ("yeah", "hold"), ("uh huh", "hold"), ("i see", "hold"), ("got it", "hold"), ("ha ha ha", "hold"),
    ("okay yes i am writing that down now", "hold"),          # ordinary words after an acknowledging opening
    ("uh huh yeah i can see it on the screen", "hold"),       # "can see" is not "can you"
    ("sure no worries take your time", "hold"),               # the "no" of "no worries" is not a refusal
    ("yeah but that is wrong", "stop"), ("okay wait", "stop"), ("right can you say that again", "stop"),
    ("yes sorry i did not catch that", "stop"),
    ("", "nothing"),
])
def test_the_opening_decides_and_a_stop_cue_ends_a_hold(heard, verdict):
    assert WordsSoFar.reading(heard.split())[0] == verdict


def test_an_unsettled_partial_loses_its_last_word_and_a_dangling_phrase_start():
    assert WordsSoFar.settled(["okay", "yes", "im", "right"], final=False) == ["okay", "yes", "im"]
    assert WordsSoFar.settled(["sure", "no", "worries"], final=False) == ["sure"]      # "no" may be "no worries"
    assert WordsSoFar.settled(["all", "right"], final=False) == []
    assert WordsSoFar.settled(["sure", "no", "worries"], final=True) == ["sure", "no", "worries"]


def decide(partials, audio=None, **kw):
    audio = audio if audio is not None else room(10.0, (2.0, 1.5))
    return WordsSoFar(**kw).decide(ctx(audio, partials))


def test_b1_stops_when_the_words_arrive_not_when_they_were_said():
    d = decide([Partial(2.9, "No, that's not-", False, 2.0)])
    assert d.fired and d.stop_s == pytest.approx(2.9, abs=0.02) and "'no'" in d.reason


def test_b1_keeps_going_through_a_long_acknowledgement():
    d = decide([Partial(2.6, "Okay-", False, 2.0), Partial(3.1, "Okay, yes, I'm right", False, 2.0),
                Partial(3.9, "Okay, yes, I am writing that down now.", True, 2.0)])
    assert not d.fired


def test_half_a_word_does_not_trigger_a_stop():
    """ "sure, no wor-" must not be read as "sure, no". """
    assert not decide([Partial(2.8, "Sure, no wor-", False, 2.0), Partial(3.6, "Sure, no worries.", True, 2.0)]).fired


def test_one_word_from_a_snippet_still_in_progress_is_not_enough_to_stop():
    """The dev set's "uh-huh" came back as "If I-" from its first 200 ms."""
    flicker = [Partial(2.5, "If I-", False, 2.0), Partial(3.0, "Uh-huh.", True, 2.0)]
    assert not decide(flicker).fired
    assert decide(flicker, min_unsettled_words=1).fired
    assert decide([Partial(2.5, "No.", True, 2.0)]).fired                 # one SETTLED word is enough


def test_an_opening_once_heard_is_not_re_decided_by_a_later_transcript():
    """The dev set's "I see." was re-transcribed a second later as "Ice. Ice." """
    assert not decide([Partial(2.6, "I see.", True, 2.0), Partial(3.4, "Ice. Ice. Ice", False, 2.0)]).fired
    d = decide([Partial(2.6, "Yeah.", True, 2.0), Partial(3.4, "Yeah, but that is wrong", True, 2.0)])
    assert d.fired and "'but'" in d.reason                                # ...but a stop cue still ends the hold


def test_a_new_sound_is_read_afresh():
    audio = room(12.0, (2.0, 0.6), (6.0, 1.0))
    d = WordsSoFar().decide(ctx(audio, [Partial(2.9, "Mm-hm.", True, 2.0), Partial(7.2, "No, stop.", True, 6.0)]))
    assert d.fired and d.stop_s == pytest.approx(7.2, abs=0.02)


def test_a_caller_still_talking_after_the_floor_time_has_taken_the_floor():
    audio = room(12.0, (2.0, 5.0))
    d = WordsSoFar().decide(ctx(audio, [Partial(2.7, "Yes it is, um-", False, 2.0)]))
    assert d.fired and d.stop_s == pytest.approx(5.0, abs=0.06) and "still talking" in d.reason
    assert not WordsSoFar(floor_time_s=9.0).decide(ctx(audio, [Partial(2.7, "Yes it is, um-", False, 2.0)])).fired


def test_no_words_means_keep_going():
    """A cough the recognizer returns nothing for. (When it invents a word instead,
    B1 stops: that failure is real, and B2 exists because of it.)"""
    assert not decide([Partial(2.8, "", False, 2.0), Partial(3.4, "", True, 2.0)]).fired
    assert decide([Partial(3.4, "You", True, 2.0)]).fired