"""
Tests for s2slab.policies.

The fixture below is a turn with a MID-TURN PAUSE -- a speaker thinking, not
finishing:

    frames   0-29   speech          0 -  600 ms
    frames  30-44   silence  <----  600 -  900 ms   the 300 ms thinking gap
    frames  45-64   speech        900 - 1300 ms
    frames  65-104  silence      1300 - 2100 ms

TRUE_END_MS = 1300. Any policy endpointing before that cut the speaker off
mid-sentence. That single number is what makes premature-cutoff measurable
here, and it is why the fixture has a mid-turn pause at all: a clean
speech-then-silence turn cannot distinguish a good policy from a lucky one.
"""

from __future__ import annotations

import pytest

from s2slab.policies import (POLICIES, DecisionTheoretic, EndpointDecision,
                             EndpointPolicy, FixedThreshold, ProsodicThreshold,
                             SemanticThreshold, f0_slope_hz_per_s, get)

FRAME_MS = 20.0
TRUE_END_MS = 1300.0


def turn() -> list[float]:
    return [0.9] * 30 + [0.1] * 15 + [0.9] * 20 + [0.1] * 40


def falling_f0(n: int = 40) -> list[float]:
    return [180.0 - 2.0 * i for i in range(n)]      # -200 Hz/s at 10 ms hop


def rising_f0(n: int = 40) -> list[float]:
    return [180.0 + 2.0 * i for i in range(n)]


def is_premature(d: EndpointDecision) -> bool:
    return d.endpoint_s * 1000 < TRUE_END_MS


# --------------------------------------------------------------------------- #
# T0
# --------------------------------------------------------------------------- #


def test_fixed_threshold_fires_after_the_required_silence():
    """150 ms / 20 ms = 8 frames. Silence starts at frame 30, so the 8th silent
    frame is index 37 and the endpoint is the end of it."""
    d = FixedThreshold(150).decide(turn())
    assert d.fired
    assert d.endpoint_s * 1000 == pytest.approx(760.0)


def test_longer_thresholds_ride_through_the_mid_turn_gap():
    """The gap is 15 frames. A threshold needing more than that cannot fire in
    it, which is the whole reason longer thresholds are 'safer'."""
    assert FixedThreshold(500).decide(turn()).endpoint_s * 1000 == pytest.approx(1800.0)
    assert FixedThreshold(700).decide(turn()).endpoint_s * 1000 == pytest.approx(2000.0)


def test_short_thresholds_cut_the_speaker_off():
    """THE FAIRNESS RESULT, in one assertion.

    A speaker who pauses 300 ms mid-turn -- disfluent, elderly, non-native, or
    just thinking -- is cut off by any threshold shorter than that pause. The
    fastest policies here are fast precisely because they are wrong.
    """
    assert is_premature(FixedThreshold(150).decide(turn()))
    assert is_premature(FixedThreshold(300).decide(turn()))
    assert not is_premature(FixedThreshold(500).decide(turn()))


def test_running_out_of_audio_is_flagged_as_not_a_decision():
    """900 ms needs 45 silent frames; the trailing silence has 40. Reporting
    2100 ms as a slow decision would flatter the policy -- it made none."""
    d = FixedThreshold(900).decide(turn())
    assert not d.fired
    assert d.reason == "eof"
    assert d.endpoint_s * 1000 == pytest.approx(2100.0)


def test_all_speech_never_fires():
    d = FixedThreshold(300).decide([0.9] * 50)
    assert not d.fired


def test_empty_input_does_not_crash():
    d = FixedThreshold(300).decide([])
    assert not d.fired
    assert d.endpoint_s == 0.0


# --------------------------------------------------------------------------- #
# T1
# --------------------------------------------------------------------------- #


def test_incomplete_tail_holds_the_floor():
    """'my account number is' cannot end a turn, so extend past the threshold."""
    plain = SemanticThreshold(500).decide(turn(), "that is all")
    held = SemanticThreshold(500).decide(turn(), "my account number is")
    assert held.endpoint_s > plain.endpoint_s
    assert held.endpoint_s * 1000 == pytest.approx(plain.endpoint_s * 1000 + 400)
    assert "hold" in held.reason


def test_complete_tail_does_not_extend():
    d = SemanticThreshold(500).decide(turn(), "that is all")
    assert "hold" not in d.reason
    assert d.endpoint_s * 1000 == pytest.approx(1800.0)


def test_trailing_punctuation_is_stripped_before_the_lookup():
    """ASR partials arrive punctuated; 'is,' must still match 'is'."""
    assert "hold" in SemanticThreshold(500).decide(turn(), "my number is,").reason


def test_missing_or_empty_partial_is_not_a_hold():
    for partial in (None, "", "   "):
        assert "hold" not in SemanticThreshold(500).decide(turn(), partial).reason


def test_a_non_decision_is_never_extended():
    """Extending an eof result would invent a decision that was never made."""
    d = SemanticThreshold(900).decide(turn(), "my account number is")
    assert not d.fired
    assert d.reason == "eof"


def test_semantic_can_use_a_shorter_base_threshold_safely():
    """The point of T1: because it can hold, the base threshold drops from 700
    to 500 -- faster in the common case, no less safe in the hard one."""
    d = SemanticThreshold(500).decide(turn(), "my account number is")
    assert not is_premature(d)


# --------------------------------------------------------------------------- #
# F0 slope
# --------------------------------------------------------------------------- #


def test_slope_sign_and_magnitude():
    assert f0_slope_hz_per_s(falling_f0()) == pytest.approx(-200.0, rel=0.01)
    assert f0_slope_hz_per_s(rising_f0()) == pytest.approx(200.0, rel=0.01)


def test_unvoiced_zeros_are_dropped_not_treated_as_pitch():
    """Praat reports unvoiced frames as 0.0. Fitting through them would make
    the slope a function of the voicing pattern instead of the pitch."""
    clean = [180.0 - 2.0 * i for i in range(20)]
    gappy = [v if i % 3 else 0.0 for i, v in enumerate(clean)]
    assert f0_slope_hz_per_s(gappy) == pytest.approx(f0_slope_hz_per_s(clean), rel=0.05)


def test_too_little_voiced_pitch_returns_none():
    """None means 'no prosodic evidence', which callers must not read as flat."""
    assert f0_slope_hz_per_s(None) is None
    assert f0_slope_hz_per_s([]) is None
    assert f0_slope_hz_per_s([0.0] * 30) is None
    assert f0_slope_hz_per_s([150.0, 0.0, 151.0]) is None


# --------------------------------------------------------------------------- #
# T2
# --------------------------------------------------------------------------- #


def test_falling_pitch_commits_earlier_than_rising():
    """The only rung that can REDUCE latency rather than trade it. Identical
    audio, opposite pitch contours, 300 ms apart."""
    p = ProsodicThreshold(500)
    early = p.decide(turn(), "that is all", falling_f0())
    late = p.decide(turn(), "that is all", rising_f0())
    assert early.endpoint_s < late.endpoint_s
    assert (late.endpoint_s - early.endpoint_s) * 1000 == pytest.approx(300.0)
    assert "yield" in early.reason and "hold" in late.reason


def test_prosody_beats_no_prosody_on_latency():
    p = ProsodicThreshold(500)
    with_cue = p.decide(turn(), "that is all", falling_f0())
    without = p.decide(turn(), "that is all", None)
    assert with_cue.endpoint_s < without.endpoint_s
    assert "noprosody" in without.reason


def test_flat_pitch_leaves_the_decision_alone():
    d = ProsodicThreshold(500).decide(turn(), "that is all", [150.0] * 40)
    assert d.endpoint_s * 1000 == pytest.approx(1800.0)


def test_prosodic_inherits_the_semantic_hold():
    """T2 is T1 plus prosody -- the syntactic hold must still apply."""
    d = ProsodicThreshold(500).decide(turn(), "my account number is", falling_f0())
    assert d.endpoint_s * 1000 == pytest.approx(1800.0 + 400 - 150)


def test_yield_bonus_cannot_produce_a_negative_endpoint():
    d = ProsodicThreshold(500).decide([0.1] * 30, "that is all", falling_f0())
    assert d.endpoint_s >= 0.05


# --------------------------------------------------------------------------- #
# T3
# --------------------------------------------------------------------------- #


def test_threshold_follows_the_cost_ratio():
    """p > R/(1+R). The arithmetic that turns a hidden threshold into a
    parameter somebody can argue with."""
    assert DecisionTheoretic(4.0).threshold == pytest.approx(0.8)
    assert DecisionTheoretic(1.0).threshold == pytest.approx(0.5)


def test_higher_cutoff_cost_endpoints_later():
    """Monotone by construction: the more you fear interrupting, the longer you
    wait. If this ever inverts, the posterior or the algebra is wrong."""
    ts = [DecisionTheoretic(r).decide(turn()).endpoint_s for r in (2.0, 4.0, 8.0)]
    assert ts[0] < ts[1] < ts[2]


def test_a_cheap_cutoff_gambles_and_loses():
    """r=2 needs only 14 silent frames, so it fires inside the 15-frame gap.
    That is the cost parameter doing visible work, not a bug."""
    assert is_premature(DecisionTheoretic(2.0).decide(turn()))
    assert not is_premature(DecisionTheoretic(4.0).decide(turn()))


def test_posterior_is_monotone_and_bounded():
    p = DecisionTheoretic(4.0)
    vals = [p._p_end(n) for n in range(0, 60)]
    assert vals[0] == pytest.approx(0.0)
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert max(vals) < 1.0


# --------------------------------------------------------------------------- #
# The registry and the shared interface
# --------------------------------------------------------------------------- #


def test_registry_is_populated_and_keys_match_names():
    assert len(POLICIES) == 10
    for name, policy in POLICIES.items():
        assert policy.name == name


def test_get_raises_helpfully_on_an_unknown_name():
    with pytest.raises(KeyError, match="unknown policy"):
        get("T9_nonexistent")


def test_every_policy_accepts_the_full_signal_set():
    """The uniformity that lets the sweep treat the ladder as one family. A
    policy needing a special case here means the interface is wrong."""
    for name, policy in POLICIES.items():
        d = policy.decide(turn(), "my account number is", falling_f0())
        assert isinstance(d, EndpointDecision), name
        assert d.endpoint_s >= 0.0, name
        assert isinstance(d.fired, bool), name
        assert d.reason, name


def test_every_policy_survives_degenerate_input():
    for name, policy in POLICIES.items():
        assert policy.decide([], None, None).fired is False, name
        assert policy.decide([0.9] * 10, None, None).fired is False, name


def test_base_class_refuses_to_decide():
    with pytest.raises(NotImplementedError):
        EndpointPolicy().decide([0.1] * 50)
