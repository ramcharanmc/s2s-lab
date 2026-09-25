"""
Tests for s2slab.turngaps -- the FT-1 measure -- on synthetic two-channel tracks whose speech is
placed by the test, so every IPU, gap and overlap can be counted by hand.
"""

import numpy as np
import pytest

from s2slab.turngaps import IPU, channels, distance, ipus, overlap_ratio, summary, transitions

SR = 16_000


def track(spans, length=6.0, amp=0.2, sr=SR) -> np.ndarray:
    """A channel that 'speaks' (a 180 Hz buzz) in each (start, end) span and is silent elsewhere."""
    x = np.zeros(int(length * sr), np.float32)
    for s, e in spans:
        i, j = int(s * sr), int(e * sr)
        t = np.arange(j - i) / sr
        x[i:j] = amp * np.sign(np.sin(2 * np.pi * 180 * t))
    return x


def test_ipus_join_short_silences_and_drop_clicks():
    got = ipus(track([(0.5, 1.0), (1.1, 1.6), (2.4, 2.46), (3.0, 3.8)]), SR)
    assert got == [IPU(0.5, 1.6), IPU(3.0, 3.8)]                     # a 100 ms silence is inside one IPU; 60 ms is a click
    assert ipus(track([(1.0, 1.5), (1.72, 2.0)]), SR) == [IPU(1.0, 1.5), IPU(1.72, 2.0)]   # 220 ms: two IPUs
    assert ipus(np.zeros(SR * 2, np.float32), SR) == []


def test_the_level_is_relative_to_each_channel():
    quiet = track([(0.5, 1.5)], amp=0.002)                            # -54 dBFS speech still counts
    assert ipus(quiet, SR) == [IPU(0.5, 1.5)]
    assert ipus(track([(0.5, 1.5)], amp=0.0002), SR) == []            # under FLOOR_DBFS: nothing


def test_a_gap_an_overlap_and_a_backchannel_are_told_apart():
    a = [IPU(0.5, 2.0), IPU(3.4, 5.0)]
    b = [IPU(2.3, 3.0),              # takes the floor 300 ms after a stops: +0.3
         IPU(4.0, 4.4),              # inside a's second IPU: within, not a transition
         IPU(4.8, 5.6)]              # starts 200 ms before a stops, outlasts it: -0.2
    ftos, within = transitions(a, b)
    assert ftos == pytest.approx([0.3, -0.2]) and within == 1


def test_b_continuing_after_its_own_pause_is_not_a_second_transition():
    a = [IPU(0.5, 1.5)]
    b = [IPU(1.8, 2.5), IPU(2.9, 3.6)]                                # b's second IPU follows b, not a
    assert transitions(a, b) == (pytest.approx([0.3]), 0)
    assert transitions([], b) == ([], 0) and transitions(a, []) == ([], 0)


def test_the_overlap_ratio_is_shared_time_over_speaking_time():
    a, b = [IPU(0.0, 2.0)], [IPU(1.5, 3.0)]                          # 0.5 shared of 3.0 spoken
    assert overlap_ratio(a, b) == pytest.approx(0.5 / 3.0)
    assert overlap_ratio([], []) == 0.0


def test_summary_from_a_stereo_file_counts_everything_by_hand():
    left = track([(0.5, 2.0), (3.4, 5.0)])
    right = track([(2.3, 3.0), (4.0, 4.4), (4.8, 5.6)])
    a, b = channels(np.stack([left, right], axis=1))
    s = summary(ipus(a, SR), ipus(b, SR))
    assert s["n_transitions"] == 2 and s["fto_ms"] == pytest.approx([300.0, -200.0], abs=20)
    assert s["overlapped"] == 1 and s["within"] == 1
    assert s["overlap_ratio"] == pytest.approx((0.4 + 0.2) / (3.1 + 1.9 - 0.6), abs=0.01)
    with pytest.raises(ValueError, match="samples, 2"):
        channels(left)


def test_the_distance_between_offset_distributions_is_in_ms():
    assert distance([100, 200, 300], [100, 200, 300]) == 0.0
    assert distance([0, 0], [300, 300]) == pytest.approx(300.0)       # every offset moves 300 ms
    assert np.isnan(distance([], [100]))
