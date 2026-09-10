"""
Tests for s2slab.policies.

The fixture is a turn with a MID-TURN PAUSE -- a speaker thinking, not
finishing:

    frames   0-29   speech          0 -  600 ms   "my account number is"
    frames  30-44   silence  <----  600 -  900 ms   the 300 ms thinking gap
    frames  45-64   speech        900 - 1300 ms   "four seven two nine"
    frames  65-104  silence      1300 - 2100 ms

TRUE_END_MS = 1300. Any policy endpointing before that cut the speaker off.

The words carry END TIMES and the pitch track is a full contour, so a policy
sees only what was knowable at its candidate endpoint. That is the property
these tests exist to protect: an earlier version handed policies a pre-sliced
buffer tail, which let a mid-turn decision read turn-final pitch.
"""

from __future__ import annotations

import pytest

from s2slab.policies import (POLICIES, DecisionTheoretic, EndpointDecision,
                             EndpointPolicy, FixedThreshold, ProsodicThreshold,
                             SemanticThreshold, TurnContext, f0_slope_hz_per_s,
                             get)

FRAME_MS = 20.0
HOP_S = 0.010
TRUE_END_MS = 1300.0

# Speech occupies f0 frames 0-59 (0-600 ms) and 90-129 (900-1300 ms).
_SEG1 = range(0, 60)
_SEG2 = range(90, 130)

WORDS = [("my", 0.15), ("account", 0.30), ("number", 0.45), ("is", 0.60),
         ("four", 1.00), ("seven", 1.10), ("two", 1.20), ("nine", 1.30)]


def speech_prob() -> list[float]:
    return [0.9] * 30 + [0.1] * 15 + [0.9] * 20 + [0.1] * 40


def _track(seg1_fn, seg2_fn) -> list[float]:
    out = [0.0] * 210
    for i in _SEG1:
        out[i] = seg1_fn(i)
    for i in _SEG2:
        out[i] = seg2_fn(i)
    return out


def f0_falling() -> list[float]:
    """Steep fall everywhere voiced: about -50 Hz/s, well past the threshold."""
    return _track(lambda i: 200.0 - 0.5 * i, lambda i: 200.0 - 0.5 * i)


def f0_rising() -> list[float]:
    return _track(lambda i: 120.0 + 0.5 * i, lambda i: 120.0 + 0.5 * i)


def f0_flat() -> list[float]:
    return _track(lambda i: 170.0, lambda i: 170.0)


def f0_declination_then_fall() -> list[float]:
    """Realistic: gentle drift mid-turn, sharp drop only at the very end.

    This is the contour that separates a working prosodic policy from one that
    fires on ordinary declination.
    """
    return _track(lambda i: 180.0 - 0.05 * i,          # ~ -5 Hz/s
                  lambda i: 175.0 - 0.05 * i - (8.0 * max(0, i - 120)))


def ctx(words=None, f0=None, probs=None) -> TurnContext:
    return TurnContext(
        speech_prob=probs if probs is not None else speech_prob(),
        words=WORDS if words is None else words,
        f0_track=f0 if f0 is not None else f0_flat(),
        frame_ms=FRAME_MS,
        f0_hop_s=HOP_S,
    )


def is_premature(d: EndpointDecision) -> bool:
    return d.endpoint_s * 1000 < TRUE_END_MS


# --------------------------------------------------------------------------- #
# The context: no evidence from the future
# --------------------------------------------------------------------------- #


def test_partial_grows_with_time():
    c = ctx()
    assert c.partial_at(0.5) == "my account number"
    assert c.partial_at(0.9) == "my account number is"
    assert c.partial_at(2.0) == "my account number is four seven two nine"


def test_partial_is_empty_before_any_word_finishes():
    assert ctx().partial_at(0.05) == ""


def test_f0_window_cannot_reach_past_the_candidate():
    """THE INVARIANT. A decision at t must not see pitch from after t."""
    c = ctx(f0=f0_declination_then_fall())
    mid = f0_slope_hz_per_s(c.f0_before(0.9), HOP_S * 1000)
    end = f0_slope_hz_per_s(c.f0_before(2.0), HOP_S * 1000)
    assert mid is not None and end is not None
    assert end < mid - 20, "the turn-final fall leaked into the mid-turn window"


def test_f0_window_skips_trailing_silence():
    """A candidate is several hundred ms into silence, so a wall-clock window
    ending there holds no pitch at all and the policy goes blind."""
    c = ctx(f0=f0_falling())
    window = c.f0_before(2.0)
    assert sum(1 for v in window if v > 0) > 0


def test_f0_window_is_empty_when_nothing_voiced_precedes():
    assert ctx(f0=[0.0] * 210).f0_before(1.0) == []


# --------------------------------------------------------------------------- #
# T0
# --------------------------------------------------------------------------- #


def test_fixed_threshold_fires_after_the_required_silence():
    """150 ms / 20 ms = 7.5 frames, so the 8th silent frame (index 37) fires."""
    d = FixedThreshold(150).decide(ctx())
    assert d.fired
    assert d.endpoint_s * 1000 == pytest.approx(760.0)


def test_longer_thresholds_ride_through_the_mid_turn_gap():
    assert FixedThreshold(500).decide(ctx()).endpoint_s * 1000 == pytest.approx(1800.0)
    assert FixedThreshold(700).decide(ctx()).endpoint_s * 1000 == pytest.approx(2000.0)


def test_short_thresholds_cut_the_speaker_off():
    """THE FAIRNESS RESULT. Any threshold shorter than the mid-turn pause cuts
    off a speaker who was still talking -- and it is fast because it is wrong."""
    assert is_premature(FixedThreshold(150).decide(ctx()))
    assert is_premature(FixedThreshold(300).decide(ctx()))
    assert not is_premature(FixedThreshold(500).decide(ctx()))


def test_running_out_of_audio_is_flagged_as_not_a_decision():
    d = FixedThreshold(900).decide(ctx())
    assert not d.fired
    assert d.reason == "eof"


def test_empty_input_does_not_crash():
    d = FixedThreshold(300).decide(ctx(probs=[]))
    assert not d.fired
    assert d.endpoint_s == 0.0


# --------------------------------------------------------------------------- #
# T1
# --------------------------------------------------------------------------- #


def test_the_hold_rescues_an_aggressive_threshold():
    """T1'S WHOLE VALUE PROPOSITION, in two lines.

    A 150 ms threshold cuts this speaker off mid-word. The same threshold plus
    a completeness check does not -- at 760 ms the partial reads '...number is',
    which cannot end a turn, so the requirement rises and the 15-frame gap can
    never satisfy it.

    Note the hold's effect is a decision that did NOT happen, so the reason on
    the decision that eventually fires says nothing about it. Assert the
    behaviour, not the label.
    """
    assert is_premature(FixedThreshold(150).decide(ctx()))
    assert not is_premature(SemanticThreshold(150).decide(ctx()))


def test_does_not_hold_when_the_partial_is_complete():
    """By 1800 ms the partial ends on 'nine', which can end a turn."""
    d = SemanticThreshold(500).decide(ctx())
    assert "hold" not in d.reason
    assert d.endpoint_s * 1000 == pytest.approx(1800.0)


def test_completeness_is_evaluated_at_the_candidate_not_the_buffer_end():
    """THE LEAK THIS REFACTOR CLOSED.

    One utterance, one policy, two candidate times: the requirement must differ,
    because what ASR had decoded differed. Previously the policy was handed a
    single partial computed for the whole buffer, so both candidates saw the
    same text and T1 could not distinguish them.
    """
    p = SemanticThreshold(150)
    c = ctx()
    need_mid, reason_mid = p._requirement(c, 0.80)    # "...number is" -> hold
    need_late, reason_late = p._requirement(c, 1.80)  # "...nine"      -> release
    assert need_mid > need_late
    assert "hold" in reason_mid
    assert "hold" not in reason_late


def test_trailing_punctuation_is_stripped():
    words = [("my", 0.15), ("number", 0.45), ("is,", 0.60)]
    assert "hold" in SemanticThreshold(150).decide(ctx(words=words)).reason


def test_no_words_is_not_a_hold():
    assert "hold" not in SemanticThreshold(500).decide(ctx(words=[])).reason


# --------------------------------------------------------------------------- #
# F0 slope
# --------------------------------------------------------------------------- #


def test_slope_sign_and_magnitude():
    ramp = [180.0 - 2.0 * i for i in range(40)]
    assert f0_slope_hz_per_s(ramp) == pytest.approx(-200.0, rel=0.01)
    assert f0_slope_hz_per_s([v + 4.0 * i for i, v in enumerate(ramp)]) > 0


def test_unvoiced_zeros_are_dropped_not_treated_as_pitch():
    clean = [180.0 - 2.0 * i for i in range(20)]
    gappy = [v if i % 3 else 0.0 for i, v in enumerate(clean)]
    assert f0_slope_hz_per_s(gappy) == pytest.approx(f0_slope_hz_per_s(clean), rel=0.05)


def test_too_little_voiced_pitch_returns_none():
    assert f0_slope_hz_per_s(None) is None
    assert f0_slope_hz_per_s([]) is None
    assert f0_slope_hz_per_s([0.0] * 30) is None
    assert f0_slope_hz_per_s([150.0, 0.0, 151.0]) is None


# --------------------------------------------------------------------------- #
# T2
# --------------------------------------------------------------------------- #


def test_falling_pitch_commits_earlier_than_rising():
    """The only rung that can REDUCE latency rather than trade it."""
    p = ProsodicThreshold(500)
    early = p.decide(ctx(f0=f0_falling()))
    late = p.decide(ctx(f0=f0_rising()))
    assert early.endpoint_s < late.endpoint_s
    assert "yield" in early.reason and "hold" in late.reason


def test_flat_pitch_leaves_the_decision_alone():
    d = ProsodicThreshold(500).decide(ctx(f0=f0_flat()))
    assert d.endpoint_s * 1000 == pytest.approx(1800.0)


def test_declination_alone_does_not_trigger_a_yield():
    """CALIBRATION GUARD.

    Every speaker's pitch drifts down across an utterance. A threshold tuned on
    a steep synthetic contour fires on that drift and yields mid-turn. The
    threshold must sit above ordinary declination and below a real final fall.
    """
    c = ctx(f0=f0_declination_then_fall())
    slope = f0_slope_hz_per_s(c.f0_before(0.9), HOP_S * 1000)
    assert slope is not None and slope < 0, "fixture should be gently declining"
    assert abs(slope) < ProsodicThreshold(500).slope_threshold, \
        "declination now trips the yield threshold -- re-tune it"


def test_no_voiced_pitch_reports_no_prosody():
    d = ProsodicThreshold(500).decide(ctx(f0=[0.0] * 210))
    assert "noprosody" in d.reason


def test_prosodic_inherits_the_semantic_hold():
    """T2 is T1 plus prosody, so the syntactic hold must survive the override.

    Checked at the requirement level: with an incomplete partial and flat pitch,
    T2's requirement must equal T1's. If T2's _requirement stopped calling
    super(), the ladder would silently lose a rung.
    """
    c = ctx(f0=f0_flat())
    t1_need, _ = SemanticThreshold(150)._requirement(c, 0.80)
    t2_need, t2_reason = ProsodicThreshold(150)._requirement(c, 0.80)
    assert t2_need == pytest.approx(t1_need)
    assert "hold" in t2_reason
    assert not is_premature(ProsodicThreshold(150).decide(c))


# --------------------------------------------------------------------------- #
# T3
# --------------------------------------------------------------------------- #


def test_threshold_follows_the_cost_ratio():
    assert DecisionTheoretic(4.0).threshold == pytest.approx(0.8)
    assert DecisionTheoretic(1.0).threshold == pytest.approx(0.5)


def test_higher_cutoff_cost_endpoints_later():
    ts = [DecisionTheoretic(r).decide(ctx()).endpoint_s for r in (2.0, 4.0, 8.0)]
    assert ts[0] < ts[1] < ts[2]


def test_a_cheap_cutoff_gambles_and_loses():
    """r=2 needs ~14 silent frames and the gap has 15, so it fires mid-turn.
    The cost parameter doing visible work, not a bug."""
    assert is_premature(DecisionTheoretic(2.0).decide(ctx()))
    assert not is_premature(DecisionTheoretic(4.0).decide(ctx()))


def test_min_silence_frames_inverts_the_posterior():
    p = DecisionTheoretic(4.0)
    assert p._p_end(p.min_silence_frames) == pytest.approx(p.threshold, rel=1e-6)


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


def test_every_policy_takes_one_context_and_returns_one_decision():
    """The uniformity that lets the sweep treat the ladder as one family."""
    c = ctx(f0=f0_falling())
    for name, policy in POLICIES.items():
        d = policy.decide(c)
        assert isinstance(d, EndpointDecision), name
        assert d.endpoint_s >= 0.0, name
        assert isinstance(d.fired, bool), name
        assert d.reason, name


def test_every_policy_survives_degenerate_input():
    for name, policy in POLICIES.items():
        assert policy.decide(ctx(probs=[], words=[], f0=[])).fired is False, name
        assert policy.decide(ctx(probs=[0.9] * 10)).fired is False, name


def test_only_the_base_class_defines_the_scan():
    """Subclasses answer `_requirement`; the loop lives in exactly one place.
    A subclass that overrides `decide` can reintroduce the look-ahead bug."""
    for cls in (FixedThreshold, SemanticThreshold, ProsodicThreshold,
                DecisionTheoretic):
        assert "decide" not in cls.__dict__, f"{cls.__name__} overrides decide"


def test_base_class_refuses_to_decide():
    with pytest.raises(NotImplementedError):
        EndpointPolicy().decide(ctx())
