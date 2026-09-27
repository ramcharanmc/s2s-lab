"""FT-3's control arithmetic (src.ft3.control): pure functions on arrays, no NeMo, no audio."""

import numpy as np
import pytest

from s2slab import severity
from src.ft3 import control

HOP = 256 / 22_050          # FastPitch's frame, 11.6 ms


def _sentence():
    """A made-up token sequence: ' the cat , sat .' with pad_with_space. Durations in frames:
    the comma's gap is 2 + 12 + 6 = 20 frames (232 ms: a pause); the word gaps 2 frames (not)."""
    #        sp  dh  ah  sp  k   ae  t   sp  ,   sp  s   ae  t   sp  .   sp
    durs = [ 2,  5,  4,  2,  6,  7,  5,  2, 12,  6,  7,  8,  6,  2,  9,  3]
    is_pause = [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    return np.array(durs), np.array(is_pause, dtype=bool)


def test_pause_token_ids_are_space_punctuation_pad_and_blank():
    tokens = ["<pad>", "dh", "ah", " ", ",", ".", "k", "<blank>"]
    ids = control.pause_token_ids(tokens, [",", ".", "?"], pad=0, blank=7)
    assert ids == {0, 3, 4, 5, 7}
    assert control.pause_token_ids(tokens, [","]) == {3, 4}


def test_a_pause_is_a_run_of_pause_tokens_long_enough_and_a_word_gap_is_not():
    durs, is_pause = _sentence()
    runs = control.pause_runs(durs, is_pause, HOP)
    assert runs == [(7, 10), (13, 16)]                      # the comma's gap (20 frames) and the final ". "
    assert control.pause_runs(durs, is_pause, HOP, min_s=1.0) == []
    mask = control.in_runs(len(durs), runs)
    assert mask.sum() == 6 and not mask[0] and not mask[3]  # the leading space and the word gaps are speech


def test_durations_follow_rate_on_speech_and_pause_scale_on_pauses():
    durs, is_pause = _sentence()
    runs = control.pause_runs(durs, is_pause, HOP)
    mask = control.in_runs(len(durs), runs)
    fast = control.control_durations(durs, mask, severity.mania(60))          # rate 1.8, pauses x0.3
    slow = control.control_durations(durs, mask, severity.depression(27))     # rate 0.5, pauses x4
    assert fast.dtype.kind == "i" and slow.dtype.kind == "i"
    assert fast[1] == round(5 / 1.8) and slow[1] == 10                        # a speech token
    assert fast[8] == round(12 * 0.3) and slow[8] == 48                       # the comma
    assert fast[0] == round(2 / 1.8) == 1 and slow[0] == 4                    # the leading space: speech
    tiny = control.control_durations(np.array([1, 1]), np.array([False, True]), severity.mania(60))
    assert tiny[0] == 1 and tiny[1] == 0                                      # speech keeps a frame; a pause may vanish
    same = control.control_durations(durs, mask, severity.mania(0))           # rate 1, pauses x1
    assert np.array_equal(same, durs)


def test_pitch_is_scaled_about_the_mean_and_shifted_in_std_units_on_speech_only():
    pitch = np.array([0.0, 0.5, -0.5, 0.0, 1.0], dtype=np.float32)
    mask = np.array([True, False, False, True, False])
    t = severity.depression(27)                                              # -25 Hz, scale 0.4
    got = control.control_pitch(pitch, mask, t, pitch_std=50.0)
    assert got.dtype == np.float32
    assert got[0] == 0.0 and got[3] == 0.0                                    # pauses untouched
    assert got[1] == pytest.approx(0.4 * 0.5 - 0.5) and got[2] == pytest.approx(0.4 * -0.5 - 0.5)
    assert got[4] == pytest.approx(0.4 - 0.5)
    neutral = control.control_pitch(pitch, mask, severity.mania(0), pitch_std=50.0)   # +0 Hz, scale 1
    assert np.array_equal(neutral, pitch)


def test_intervals_come_from_the_durations_in_seconds():
    durs, is_pause = _sentence()
    runs = control.pause_runs(durs, is_pause, HOP)
    iv = control.intervals(durs, runs, HOP)
    assert iv[0] == pytest.approx((sum(durs[:7]) * HOP, sum(durs[:10]) * HOP))
    assert iv[1][1] == pytest.approx(sum(durs) * HOP)


def test_the_plan_moves_the_pauses_and_keeps_the_neutral_one_identical():
    durs, is_pause = _sentence()
    pitch = np.linspace(-1, 1, len(durs), dtype=np.float32)
    neutral = control.plan(durs, pitch, is_pause, HOP, 60.0, None)
    assert np.array_equal(neutral.durs, durs) and np.array_equal(neutral.pitch, pitch)
    assert neutral.pauses_in == neutral.pauses_out
    p = control.plan(durs, pitch, is_pause, HOP, 60.0, severity.depression(27))
    assert p.pauses_in == neutral.pauses_in and len(p.pauses_out) == 2
    a_in, a_out = p.pauses_in[0], p.pauses_out[0]
    assert (a_out[1] - a_out[0]) == pytest.approx(4 * (a_in[1] - a_in[0]), rel=0.05)   # the pause x4
    assert a_out[0] > a_in[0]                                                  # speech before it got slower
    r = control.as_render(np.zeros(100, dtype=np.float32), p)
    assert isinstance(r, severity.Render) and r.pauses_out == p.pauses_out


def test_the_targets_are_page_09s():
    t = severity.mania(60)
    assert (t.f0_offset_hz, t.f0_dev_scale, t.rate, t.pause_scale) == (45, 1.8, 1.8, pytest.approx(0.3))
    assert control.MIN_PAUSE_S == 0.15
