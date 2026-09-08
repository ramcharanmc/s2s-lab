"""
Tests for s2slab.prosody.

The strategy: SYNTHESIZE the ground truth. `utterance()` emits a known number
of voiced bursts at a known rate with known pauses, so every assertion compares
against a fact rather than a recorded value. A test asserting `== 4.545` only
proves the code still does what it did; a test asserting `== the number of
syllables we built` proves it is correct.

The signal is a pulse train, not speech. That is deliberate -- a failure here
is a bug in the measurement, never a TTS artifact. It is also fully
deterministic: no RNG, no hashing, no seeds to get wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2slab.prosody import (count_syllable_nuclei, extract_prosody,
                            pause_segments)

SR = 16_000


def utterance(n_syllables: int, rate: float = 1.0, pause_scale: float = 1.0,
              f0: float = 150.0, amplitude: float = 0.3) -> np.ndarray:
    """A pulse train with exactly `n_syllables` voiced bursts.

    rate        -- 1.0 is baseline; 0.5 makes each syllable twice as long.
    pause_scale -- multiplies the pause inserted after every third syllable.
    """
    syll_s = 0.22 / rate
    pause_s = 0.18 * pause_scale
    out, phase = [], 0.0
    for i in range(n_syllables):
        n = int(syll_s * SR)
        seg = np.empty(n, dtype="float32")
        for k in range(n):
            phase += 2 * np.pi * f0 / SR
            seg[k] = amplitude * (np.sin(np.pi * k / n) ** 0.6) * np.sin(phase)
        out.append(seg)
        if i % 3 == 2:
            out.append(np.zeros(int(pause_s * SR), dtype="float32"))
    return np.concatenate(out)


def _tone(seconds: float, hz: float = 200.0) -> np.ndarray:
    return (0.3 * np.sin(2 * np.pi * hz * np.arange(int(seconds * SR)) / SR)
            ).astype("float32")


# --------------------------------------------------------------------------- #
# Pause detection
# --------------------------------------------------------------------------- #


def test_finds_a_pause_of_the_right_length():
    sig = np.concatenate([_tone(0.5), np.zeros(int(0.4 * SR), "float32"), _tone(0.5)])
    pauses = pause_segments(sig, SR)
    assert len(pauses) == 1
    start, dur = pauses[0]
    assert start == pytest.approx(0.5, abs=0.05)
    assert dur == pytest.approx(0.4, abs=0.05)


def test_word_gaps_below_the_floor_are_not_pauses():
    """150 ms is what separates a pause from the gap between two words."""
    sig = np.concatenate([_tone(0.5), np.zeros(int(0.10 * SR), "float32"), _tone(0.5)])
    assert pause_segments(sig, SR) == []


def test_pause_detection_is_level_independent():
    """The threshold is adaptive, so recording level must not change the answer.

    Cohorts recorded at different levels have to stay comparable; a fixed dBFS
    threshold would make LibriSpeech and Common Voice incommensurable.
    """
    quiet = utterance(7, amplitude=0.05)
    loud = utterance(7, amplitude=0.8)
    assert len(pause_segments(quiet, SR)) == len(pause_segments(loud, SR))


def test_trailing_silence_is_not_counted():
    """A pause the clip ends inside has unknown duration -- dropping it is
    correct, since guessing would bias the p90."""
    sig = np.concatenate([utterance(3), np.zeros(int(1.0 * SR), "float32")])
    assert len(pause_segments(sig, SR)) == len(pause_segments(utterance(3), SR))


# --------------------------------------------------------------------------- #
# Syllable nuclei
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [3, 5, 7, 11, 13])
def test_nucleus_count_is_exact(n):
    """We built n bursts; the detector must find n. No tolerance."""
    assert count_syllable_nuclei(utterance(n), SR) == n


def test_nucleus_count_survives_a_rate_change():
    """Slower syllables are still one nucleus each."""
    for rate in (0.5, 0.75, 1.0):
        assert count_syllable_nuclei(utterance(9, rate=rate), SR) == 9


def test_silence_has_no_nuclei():
    assert count_syllable_nuclei(np.zeros(SR, dtype="float32"), SR) == 0


def test_too_short_returns_zero_not_an_error():
    """Praat refuses an intensity analysis below 6.4/minimum_pitch seconds.
    Too short to measure is not the same as broken."""
    assert count_syllable_nuclei(np.zeros(8, dtype="float32"), SR) == 0


# --------------------------------------------------------------------------- #
# Rates -- including the regression guard
# --------------------------------------------------------------------------- #


def test_articulation_rate_is_not_constant():
    """THE REGRESSION GUARD. Do not delete this test.

    The reference implementation estimated syllables as `voiced_frames / 12`,
    which is proportional to phonation time. Dividing that by phonation time
    cancels the audio out entirely, pinning articulation rate near 8.3 syll/s
    for every possible input. Any future change that reintroduces a
    phonation-derived count collapses this spread to zero.
    """
    rates = [extract_prosody(utterance(13, rate=r), SR).articulation_rate
             for r in (1.0, 0.75, 0.5)]
    assert rates[0] > rates[1] > rates[2], f"not monotone: {rates}"
    assert rates[0] / rates[2] > 1.6, f"asked 2x, measured {rates[0]/rates[2]:.2f}x"


def test_articulation_rate_hits_the_requested_magnitude():
    """Direction is not enough. Halving the rate must halve the measurement."""
    fast = extract_prosody(utterance(13, rate=1.0), SR).articulation_rate
    slow = extract_prosody(utterance(13, rate=0.5), SR).articulation_rate
    assert slow == pytest.approx(fast / 2.0, rel=0.15)


def test_pauses_lengthen_without_touching_articulation():
    """The dissociation this module exists to measure.

    Longer pauses at unchanged articulation is the early-Alzheimer's profile --
    a retrieval deficit, not a motor one. Speech rate must fall while
    articulation rate holds. A metric that cannot separate these is useless.
    """
    base = extract_prosody(utterance(13, pause_scale=1.0), SR)
    slow = extract_prosody(utterance(13, pause_scale=4.0), SR)
    assert slow.pause_mean_s > base.pause_mean_s * 2
    assert slow.speech_rate < base.speech_rate
    assert slow.articulation_rate == pytest.approx(base.articulation_rate, rel=0.15)


def test_speech_rate_is_never_above_articulation_rate():
    """Definitional: one divides by total time, the other by phonation time,
    and phonation time is always the smaller. Catches a swapped assignment."""
    for pscale in (1.0, 2.0, 4.0):
        f = extract_prosody(utterance(13, pause_scale=pscale), SR)
        assert f.speech_rate <= f.articulation_rate + 1e-9


def test_rates_are_nan_when_nothing_was_measured():
    """NaN, not 0. 'Zero syllables per second' is a claim; we made no
    measurement, and NaN propagates loudly instead of silently averaging in."""
    f = extract_prosody(np.zeros(SR, dtype="float32"), SR)
    assert f.n_syllables == 0
    assert np.isnan(f.speech_rate)
    assert np.isnan(f.articulation_rate)


def test_known_syllable_count_overrides_detection():
    assert extract_prosody(utterance(13), SR, n_syllables=99).n_syllables == 99


# --------------------------------------------------------------------------- #
# Pitch
# --------------------------------------------------------------------------- #


def test_f0_mean_tracks_the_synthesized_pitch():
    """Unvoiced frames come back as 0.0 from Praat, not NaN. Failing to drop
    them drags the mean toward zero -- this catches that."""
    for target in (120.0, 200.0):
        f = extract_prosody(utterance(9, f0=target), SR)
        assert f.f0_mean == pytest.approx(target, rel=0.10)


def test_f0_is_nan_for_silence():
    assert np.isnan(extract_prosody(np.zeros(SR, dtype="float32"), SR).f0_mean)
