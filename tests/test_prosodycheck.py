"""
Tests for s2slab.prosodycheck -- on synthetic tones and a fake recognizer.

A tone has a known pitch and level, so every expected number here can be worked
out by hand: 300 Hz against 200 Hz is 12 * log2(1.5) = +7.02 semitones, and
twice the amplitude is +6.02 dB.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from s2slab.prosodycheck import PITCH_CEILING_HZ, SENTENCES, blind, delta, describe, features
from s2slab.prosodyset import LoadedPair, Pair
from s2slab.wer import light

SR = 16_000


def tone(hz, seconds=1.0, amp=0.2, lead=0.3, tail=0.3, wobble_st=0.0) -> np.ndarray:
    """A tone in a quiet room. wobble_st: the pitch swings this many semitones either way."""
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SR)) / SR
    f = hz * 2 ** (wobble_st * np.sin(2 * np.pi * 2 * t) / 12)
    body = amp * np.sin(2 * np.pi * np.cumsum(f) / SR)
    x = np.concatenate([np.zeros(int(lead * SR)), body, np.zeros(int(tail * SR))])
    return (x + 1e-4 * rng.standard_normal(len(x))).astype(np.float32)


def pair(emotion, neutral, loaded, k=1, sentence="DFA") -> LoadedPair:
    p = Pair(f"{emotion.lower()}-d{k:02d}", "dev", emotion, f"10{k:02d}", sentence, f"10{k:02d}_{sentence}_NEU_XX",
             f"10{k:02d}_{sentence}_{emotion}_XX", 0.9, 0.9, 10, 10, 60.0, 0.9)
    return LoadedPair(p, neutral, loaded)


# --------------------------------------------------------------------------- #
# Three numbers from a clip
# --------------------------------------------------------------------------- #


def test_a_steady_tone_has_its_pitch_no_range_and_its_level():
    f = features(tone(200, amp=0.2))
    assert f.f0_median_hz == pytest.approx(200, abs=2) and f.f0_range_st < 0.3
    assert f.level_db == pytest.approx(20 * math.log10(0.2 / math.sqrt(2)), abs=0.5)      # a sine's RMS


def test_the_silence_around_the_speech_does_not_move_the_level_until_it_is_nine_tenths_of_the_clip():
    tight, padded = features(tone(200, lead=0.1, tail=0.1)), features(tone(200, lead=0.5, tail=1.5))
    assert padded.level_db == pytest.approx(tight.level_db, abs=0.1)              # a third of the clip is speech
    assert features(tone(200, lead=0.5, tail=12.0)).level_db < tight.level_db - 20   # 7%: the percentile lands in silence


def test_a_wobbling_tone_has_a_pitch_range():
    assert features(tone(200, wobble_st=3.0)).f0_range_st == pytest.approx(5.4, abs=0.8)   # 10th to 90th of a sine swing


def test_pitch_above_the_endpointers_400_hz_ceiling_is_still_tracked():
    assert PITCH_CEILING_HZ == 600
    assert features(tone(480)).f0_median_hz == pytest.approx(480, abs=5)


def test_a_clip_with_no_pitch_has_no_pitch_statistics_but_still_a_level():
    rng = np.random.default_rng(1)
    noise = np.concatenate([np.zeros(4800), 0.1 * rng.standard_normal(SR), np.zeros(4800)]).astype(np.float32)
    f = features(noise)
    assert math.isnan(f.f0_median_hz) and math.isnan(f.f0_range_st)
    assert f.level_db == pytest.approx(-20, abs=1.0)


# --------------------------------------------------------------------------- #
# Loaded minus neutral
# --------------------------------------------------------------------------- #


def test_the_change_inside_a_pair_is_in_semitones_and_decibels():
    d = delta(pair("ANG", tone(200, amp=0.1), tone(300, amp=0.2, wobble_st=2.0)))
    assert d.pitch_st == pytest.approx(7.02, abs=0.3)             # 12 * log2(300 / 200)
    assert d.range_st > 2.0
    assert d.level_db == pytest.approx(6.02, abs=0.5)             # twice the amplitude


def test_the_table_gives_the_median_change_and_how_many_pairs_moved_that_way():
    up = [pair("ANG", tone(200), tone(250, amp=0.4), k) for k in (1, 2)]
    down = pair("ANG", tone(200), tone(180, amp=0.4), 3)
    quiet = [pair("SAD", tone(200), tone(190, amp=0.1), k) for k in (4, 5)]
    text = describe(up + [down] + quiet)
    assert "| emotion | pairs | median pitch, semitones | pitch range, semitones | level, dB |" in text
    ang = next(line for line in text.splitlines() if line.startswith("| ANG"))
    assert "| ANG | 3 | +3.9 (2 of 3) |" in ang and ang.endswith("(3 of 3) |")    # pitch: two up, one down; level: all up
    assert float(ang.split("|")[5].split()[0]) == pytest.approx(6.0, abs=0.3)     # twice the amplitude
    sad = next(line for line in text.splitlines() if line.startswith("| SAD"))
    assert "| SAD | 2 | -0.9 (2 of 2) |" in sad and sad.endswith("(2 of 2) |")
    assert float(sad.split("|")[5].split()[0]) == pytest.approx(-6.0, abs=0.3)    # half the amplitude
    assert "FEA" not in text                                                       # no pairs, no row


# --------------------------------------------------------------------------- #
# The same words twice
# --------------------------------------------------------------------------- #


class FakeRecognizer:
    """Hears what it is told to, by the clip's length in samples."""

    def __init__(self, heard: dict[int, str]):
        self.heard = heard

    def transcribe(self, audio):
        return self.heard[len(audio)]


def test_pairs_with_the_same_transcript_are_counted_after_normalizing():
    n1, l1, n2, l2 = (np.zeros(k, np.float32) for k in (100, 101, 102, 103))
    asr = FakeRecognizer({100: "Don't forget a jacket.", 101: "don't forget a jacket",
                          102: "Don't forget a jacket.", 103: "Don't forget the jacket!"})
    text = blind([pair("FEA", n1, l1, 1), pair("FEA", n2, l2, 2)], asr, light)
    assert "| FEA | 2 | 1 | 0.0% | 12.5% |" in text                # one wrong word in eight, on the loaded side
    assert ('''fea-d02 "Don't forget a jacket": neutral heard as "Don't forget a jacket.", '''
            '''loaded as "Don't forget the jacket!"''') in text
    assert "fea-d01" not in text


def test_when_no_pair_differs_the_report_says_none():
    clip = np.zeros(100, np.float32)
    text = blind([pair("SAD", clip, clip)], FakeRecognizer({100: "Don't forget a jacket"}), light)
    assert "| SAD | 1 | 1 | 0.0% | 0.0% |" in text and text.endswith("  none")


def test_all_twelve_sentences_are_known():
    assert len(SENTENCES) == 12 and SENTENCES["ITH"] == "I think I have a doctor's appointment"