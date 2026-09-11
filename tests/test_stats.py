"""
Tests for s2slab.stats.

Every expected value here is either worked out by hand (and shown in the test)
or a property that must hold for ANY input. No values are copied from a run of
the code under test -- a test that asserts whatever the code produced last time
only proves the code has not changed, not that it is right.
"""

from __future__ import annotations

import math

import pytest

from s2slab.stats import (Comparison, McNemar, adjust, compare, holm,
                          mcnemar_exact, paired_bootstrap_diff, wilson_interval)


def paired(only_a: int, only_b: int, both: int = 0, neither: int = 0):
    """Build two aligned bool lists with a chosen discordance pattern."""
    a = [True] * only_a + [False] * only_b + [True] * both + [False] * neither
    b = [False] * only_a + [True] * only_b + [True] * both + [False] * neither
    return a, b


# --------------------------------------------------------------------------- #
# wilson_interval
# --------------------------------------------------------------------------- #


def test_wilson_known_values():
    """Hand-computed: 0/60 -> upper 0.0602; 38/60 -> [0.507, 0.744]."""
    assert wilson_interval(0, 60) == pytest.approx((0.0, 0.0602), abs=1e-3)
    assert wilson_interval(38, 60) == pytest.approx((0.507, 0.744), abs=1e-3)
    assert wilson_interval(60, 60) == pytest.approx((0.940, 1.0), abs=1e-3)


def test_zero_events_does_not_mean_zero_risk():
    """The reason Wilson is here. The textbook interval returns [0, 0] for 0/n
    and claims certainty exactly where the sample says least."""
    lo, hi = wilson_interval(0, 60)
    assert lo == 0.0
    assert hi > 0.05


def test_interval_stays_inside_zero_and_one():
    for k, n in [(0, 5), (5, 5), (1, 3), (0, 1), (1, 1), (50, 100)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_interval_contains_the_observed_rate():
    for k, n in [(3, 60), (30, 60), (57, 60)]:
        lo, hi = wilson_interval(k, n)
        assert lo <= k / n <= hi


def test_more_turns_means_a_narrower_interval():
    """Same 63% rate, four times the turns: the interval must tighten."""
    lo60, hi60 = wilson_interval(38, 60)
    lo240, hi240 = wilson_interval(152, 240)
    assert (hi240 - lo240) < (hi60 - lo60)


def test_no_turns_is_nan_not_an_error():
    lo, hi = wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


# --------------------------------------------------------------------------- #
# mcnemar_exact
# --------------------------------------------------------------------------- #


def test_all_discordance_one_way():
    """8 turns only A interrupted, 0 only B. Like 8 coin flips landing the same
    way: two-sided p = 2 * 0.5**8 = 2/256."""
    t = mcnemar_exact(*paired(8, 0))
    assert (t.only_a, t.only_b) == (8, 0)
    assert t.p == pytest.approx(2 / 256)


def test_split_discordance_matches_the_binomial_by_hand():
    """4 vs 11, m = 15: P(X <= 4) = (1+15+105+455+1365)/2**15 = 1941/32768.
    Two-sided p doubles it."""
    t = mcnemar_exact(*paired(4, 11))
    assert t.p == pytest.approx(2 * 1941 / 32768)


def test_concordant_turns_do_not_change_the_answer():
    """Turns where both interrupted, or neither did, carry no information about
    which policy is better. Adding a thousand of them must not move p."""
    bare = mcnemar_exact(*paired(8, 0))
    padded = mcnemar_exact(*paired(8, 0, both=500, neither=500))
    assert padded.p == pytest.approx(bare.p)
    assert padded.discordant == bare.discordant == 8


def test_swapping_the_policies_swaps_the_counts_not_the_p():
    ab = mcnemar_exact(*paired(3, 9))
    a, b = paired(3, 9)
    ba = mcnemar_exact(b, a)
    assert (ba.only_a, ba.only_b) == (ab.only_b, ab.only_a)
    assert ba.p == pytest.approx(ab.p)


def test_no_discordant_turns_gives_p_of_one():
    t = mcnemar_exact(*paired(0, 0, both=20, neither=20))
    assert t.discordant == 0
    assert t.p == 1.0


def test_a_perfectly_even_split_is_p_of_one():
    assert mcnemar_exact(*paired(5, 5)).p == pytest.approx(1.0)


def test_mismatched_lengths_raise():
    """Pairing only means something if entry i is the same turn in both."""
    with pytest.raises(ValueError):
        mcnemar_exact([True, False], [True])


def test_returns_the_documented_type():
    assert isinstance(mcnemar_exact(*paired(2, 1)), McNemar)


# --------------------------------------------------------------------------- #
# paired_bootstrap_diff
# --------------------------------------------------------------------------- #


def test_diff_is_exactly_the_rate_difference():
    """46/60 vs 38/60: diff = 8/60, independent of any resampling."""
    diff, lo, hi = paired_bootstrap_diff(*paired(8, 0, both=38, neither=14))
    assert diff == pytest.approx(8 / 60)


def test_interval_contains_the_difference():
    diff, lo, hi = paired_bootstrap_diff(*paired(8, 0, both=38, neither=14))
    assert lo <= diff <= hi


def test_one_sided_discordance_gives_an_interval_clear_of_zero():
    """Every discordant turn favours B, so no resample can make A better."""
    diff, lo, hi = paired_bootstrap_diff(*paired(8, 0, both=38, neither=14))
    assert lo > 0


def test_same_seed_same_interval():
    args = paired(6, 2, both=30, neither=22)
    assert paired_bootstrap_diff(*args, seed=3) == paired_bootstrap_diff(*args, seed=3)


def test_identical_outcomes_give_a_zero_interval():
    assert paired_bootstrap_diff(*paired(0, 0, both=10, neither=10)) == (0.0, 0.0, 0.0)


def test_bootstrap_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([True], [True, False])


def test_bootstrap_empty_input_is_nan():
    assert all(math.isnan(x) for x in paired_bootstrap_diff([], []))


# --------------------------------------------------------------------------- #
# holm
# --------------------------------------------------------------------------- #


def test_holm_worked_example():
    """Smallest x3, next x2, last x1."""
    assert holm([0.0078, 0.118, 1.0]) == pytest.approx([0.0234, 0.236, 1.0])


def test_holm_returns_values_in_the_original_order():
    """Input unsorted; output must line up with input, not with the sort."""
    assert holm([1.0, 0.0078, 0.118]) == pytest.approx([1.0, 0.0234, 0.236])


def test_adjusted_is_never_smaller_than_raw():
    raw = [0.001, 0.02, 0.04, 0.3, 0.9]
    for r, a in zip(raw, holm(raw)):
        assert a >= r


def test_adjusted_is_capped_at_one():
    assert max(holm([0.5, 0.6, 0.7])) <= 1.0


def test_adjusted_values_keep_the_ranking():
    """A smaller raw p can never end up with a larger adjusted p than a bigger
    raw p. That is what the 'running max' in holm() enforces."""
    raw = [0.04, 0.001, 0.03, 0.02]
    adj = holm(raw)
    for i in range(len(raw)):
        for j in range(len(raw)):
            if raw[i] < raw[j]:
                assert adj[i] <= adj[j]


def test_a_single_test_is_not_penalised():
    assert holm([0.0078]) == pytest.approx([0.0078])


def test_holm_of_nothing_is_nothing():
    assert holm([]) == []


# --------------------------------------------------------------------------- #
# compare / adjust / verdict
# --------------------------------------------------------------------------- #


def test_compare_fills_every_field():
    a, b = paired(8, 0, both=38, neither=14)
    c = compare("disfluent", "T0_500", a, "T1_500", b)
    assert isinstance(c, Comparison)
    assert c.n == 60
    assert c.rate_a == pytest.approx(46 / 60)
    assert c.rate_b == pytest.approx(38 / 60)
    assert (c.test.only_a, c.test.only_b) == (8, 0)
    assert c.diff == pytest.approx(8 / 60)
    assert c.ci[0] <= c.diff <= c.ci[1]


def test_p_adjusted_is_nan_until_adjust_runs():
    """Adjustment needs the whole family, so compare() cannot do it alone."""
    c = compare("x", "A", *_ab(8, 0))
    assert math.isnan(c.p_adjusted)
    assert c.verdict() == "no detectable difference"


def test_adjust_fills_in_place_and_returns_the_same_list():
    comps = [compare("x", "A", *_ab(8, 0)), compare("x", "C", *_ab(1, 1))]
    out = adjust(comps)
    assert out is comps
    assert all(not math.isnan(c.p_adjusted) for c in comps)


def test_verdict_names_the_policy_that_interrupts_less():
    better_b = adjust([compare("x", "A", *_ab(10, 0))])[0]
    better_a = adjust([compare("x", "A", *_ab(0, 10))])[0]
    assert better_b.verdict() == "B interrupts less"
    assert better_a.verdict() == "A interrupts less"


def test_identical_outcomes_are_called_identical():
    c = adjust([compare("x", "A", *_ab(0, 0, both=20))])[0]
    assert c.verdict() == "identical: same turns interrupted"


def test_adding_comparisons_can_cost_you_a_result():
    """THE REASON TO KEEP THE LIST OF COMPARISONS SHORT.

    8/0 on its own: p = 0.0078, clearly significant. Report it alongside nine
    uninformative comparisons and Holm multiplies it by ten: 0.078, and the
    result no longer survives. Every comparison you add makes every other one
    harder to pass.
    """
    alone = adjust([compare("x", "A", *_ab(8, 0))])[0]
    assert alone.verdict() == "B interrupts less"

    crowd = adjust([compare("x", "A", *_ab(8, 0))] +
                   [compare("x", "A", *_ab(3, 3)) for _ in range(9)])
    assert crowd[0].p_adjusted == pytest.approx(10 * 2 / 256)
    assert crowd[0].verdict() == "no detectable difference"


def _ab(only_a: int, only_b: int, both: int = 0, neither: int = 0):
    """(a_cut, 'B', b_cut) -- the argument tail compare() expects after name_a."""
    a, b = paired(only_a, only_b, both, neither)
    return a, "B", b
