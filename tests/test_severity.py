"""
Tests for s2slab.severity (P2c, page 09 §6-7): the mapping taken word for word, the render
hitting its targets, and the monotonicity test -- on a synthetic voice, so they run in CI in
seconds with no model. The same checks on real TTS speech are a slow test (commit 2).
"""

import math

import numpy as np
import pytest

from s2slab import severity
from s2slab.severity import (PHQ9, YMRS, Achieved, depression, mania, misses, monotone, normalize, render, sweep,
                             trim, warp)

SR = 24_000


def voice(groups=3, syll=4, syll_s=0.2, pause_s=0.35, f0=190.0, swing=30.0, sr=SR) -> np.ndarray:
    """A harmonic 'voice': syllable-shaped bursts of 11 harmonics, an intonation swing of
    +-swing Hz, and a pause between groups of syllables."""
    out, phase, t0 = [], 0.0, 0.0
    for g in range(groups):
        for _ in range(syll):
            n = int(syll_s * sr)
            f = f0 + swing * np.sin(2 * np.pi * 0.7 * (t0 + np.arange(n) / sr))
            ph = phase + np.cumsum(2 * np.pi * f / sr)
            phase = ph[-1]
            wave = sum(np.sin(k * ph) / k for k in range(1, 12))
            out.append(0.15 * np.sin(np.pi * np.arange(n) / n) ** 0.5 * wave)
            t0 += syll_s
        if g < groups - 1:
            out.append(np.zeros(int(pause_s * sr)))
            t0 += pause_s
    return np.concatenate(out).astype(np.float32)


@pytest.fixture(scope="module")
def neutral():
    return voice()


# --------------------------------------------------------------------------- #
# The mapping, word for word from page 09 §6
# --------------------------------------------------------------------------- #


def test_the_depression_mapping_is_page_09s_at_both_ends_and_in_the_middle():
    lo, hi, mid = depression(0), depression(27), depression(13.5)
    assert (lo.f0_offset_hz, lo.f0_dev_scale, lo.rate, lo.pause_scale) == (-5, 1.0, 1.0, 1.0)
    assert (hi.f0_offset_hz, hi.f0_dev_scale, hi.rate, hi.pause_scale) == pytest.approx((-25, 0.4, 0.5, 4.0))
    assert (mid.n, mid.f0_offset_hz, mid.pause_scale) == pytest.approx((0.5, -15, 2.5))
    assert hi.dialogue == pytest.approx({"response_latency_ms": 1200, "elaboration_level": 0.3})


def test_the_mania_mapping_is_page_09s_and_scores_off_the_scale_are_refused():
    lo, hi = mania(0), mania(60)
    assert (lo.f0_offset_hz, lo.f0_dev_scale, lo.rate, lo.pause_scale) == (0, 1.0, 1.0, 1.0)
    assert (hi.f0_offset_hz, hi.f0_dev_scale, hi.rate, hi.pause_scale) == pytest.approx((45, 1.8, 1.8, 0.3))
    assert hi.dialogue == pytest.approx({"interrupt_prob": 0.5})
    assert normalize(20, YMRS) == pytest.approx(1 / 3) and normalize(27, PHQ9) == 1.0
    for bad in (lambda: depression(28), lambda: mania(-1)):
        with pytest.raises(ValueError, match="runs"):
            bad()


# --------------------------------------------------------------------------- #
# The pieces
# --------------------------------------------------------------------------- #


def test_the_time_map_integrates_the_duration_factor():
    pts = [(0.0, 2.0), (1.0, 2.0), (1.0, 0.5), (2.0, 0.5)]            # twice as long, then half
    assert warp(0.5, pts) == pytest.approx(1.0) and warp(1.0, pts) == pytest.approx(2.0)
    assert warp(2.0, pts) == pytest.approx(2.5)
    ramp = [(0.0, 1.0), (1.0, 3.0)]                                      # linear 1 -> 3: area 2
    assert warp(1.0, ramp) == pytest.approx(2.0)


def test_trim_drops_the_silence_around_the_speech_and_keeps_20_ms_each_side():
    a = np.concatenate([np.zeros(SR // 2), voice(groups=1), np.zeros(SR)]).astype(np.float32)
    t = trim(a, SR)
    assert abs(len(t) / SR - len(voice(groups=1)) / SR) < 0.06 and trim(np.zeros(SR, np.float32), SR).size == 0


# --------------------------------------------------------------------------- #
# The render hits its targets, and the sweep moves the right way
# --------------------------------------------------------------------------- #


def test_an_identity_render_changes_nothing_measurable(neutral):
    t = mania(0)
    got = severity.achieved(neutral, render(neutral, SR, t), SR)
    assert misses(got, t) == [] and abs(got.f0_offset_hz) < 1.0 and got.rate == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("t", [depression(27), depression(13.5), mania(60), mania(30)], ids=lambda t: f"{t.condition}-{t.score}")
def test_the_render_meets_every_target_within_tolerance_on_the_synthetic_voice(neutral, t):
    r = render(neutral, SR, t)
    assert r.audio.dtype == np.float32 and len(r.pauses_in) == len(r.pauses_out) == 2
    assert misses(severity.achieved(neutral, r, SR), t) == []


def test_severity_monotonicity_depression_and_mania_all_features_pass(neutral):
    """Page 09 §6's unit test for a generative model: 2 conditions x 4 features, every step."""
    assert monotone(sweep(neutral, SR, "depression", [0, 5, 10, 15, 20, 27])) == []
    assert monotone(sweep(neutral, SR, "mania", [0, 12, 20, 26, 40, 60])) == []


def test_monotone_names_a_step_that_goes_the_wrong_way_and_misses_names_a_miss():
    t0, t1 = depression(0), depression(27)
    good, bad = Achieved(-5, 1.0, 1.0, 1.0), Achieved(-25, 0.4, 0.5, 4.0)
    assert monotone([(t0, good), (t1, bad)]) == []
    wrong = monotone([(t0, good), (t1, Achieved(-25, 1.2, 0.5, 4.0))])
    assert len(wrong) == 1 and "f0_dev_scale" in wrong[0]
    assert misses(Achieved(-20, 0.4, 0.5, 4.0), t1) == ["f0_offset_hz -20.0 against -25.0"]
    assert misses(Achieved(-25, 0.4, 0.5, math.nan), t1) == []          # no pause to stretch: skipped
