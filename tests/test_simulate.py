"""
Tests for s2slab.simulate.

Two obligations. First, the ground truth must be exactly right -- `true_end_s`
is the reference every premature-cutoff number is measured against, so an error
here silently corrupts the headline result rather than failing loudly. Second,
the cohorts must actually differ in the way the fairness argument claims: if
`disfluent` does not really have longer mid-turn pauses than `clean`, the whole
finding is an artifact of the generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2slab.policies import get
from s2slab.simulate import (COHORTS, SR, CohortSpec, Utterance, describe,
                             make_all, make_cohort, synth_utterance)
from s2slab.vad import frame_speech_prob


# --------------------------------------------------------------------------- #
# Ground truth -- the number everything else is scored against
# --------------------------------------------------------------------------- #


def test_everything_after_true_end_is_silence():
    """If audio continues past true_end_s, the ground truth is a lie and every
    premature-cutoff rate computed from it is wrong."""
    for u in make_cohort("disfluent", 12, seed=1):
        tail = u.audio[int(u.true_end_s * u.sr):]
        assert tail.size > 0
        assert np.abs(tail).max() == 0.0


def test_speech_is_present_just_before_true_end():
    """The turn must actually end AT true_end_s, not before it."""
    for u in make_cohort("clean", 12, seed=2):
        i = int(u.true_end_s * u.sr)
        window = u.audio[max(0, i - int(0.05 * u.sr)): i]
        assert np.sqrt((window.astype("float64") ** 2).mean()) > 1e-3


def test_trailing_silence_lets_every_policy_fire():
    """A buffer too short for the longest threshold produces fired=False on
    every turn -- a broken benchmark, not a slow policy."""
    policy = get("T0_fixed_900ms")
    for u in make_cohort("clean", 10, seed=3):
        d = policy.decide(u.context())
        assert d.fired


def test_true_end_equals_speech_plus_mid_turn_pauses():
    """Cross-check the bookkeeping against the spec, independently of audio."""
    spec = COHORTS["disfluent"]
    rng = np.random.default_rng(4)
    u = synth_utterance(spec, rng)
    expected = u.n_syllables * spec.syllable_s + sum(u.mid_turn_pauses_s)
    assert u.true_end_s == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_same_seed_reproduces_the_cohort_exactly():
    a = make_cohort("disfluent", 8, seed=7)
    b = make_cohort("disfluent", 8, seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x.audio, y.audio)
        assert x.true_end_s == y.true_end_s
        assert x.mid_turn_pauses_s == y.mid_turn_pauses_s


def test_different_seeds_produce_different_cohorts():
    a = make_cohort("disfluent", 8, seed=7)
    b = make_cohort("disfluent", 8, seed=8)
    assert [u.true_end_s for u in a] != [u.true_end_s for u in b]


def test_each_cohort_gets_its_own_derived_seed():
    """Adding a cohort must not change the utterances generated for existing
    ones, or every previously reported number silently shifts."""
    all_c = make_all(6, seed=0)
    assert [u.true_end_s for u in all_c["clean"]] != \
           [u.true_end_s for u in all_c["disfluent"]]


# --------------------------------------------------------------------------- #
# The cohorts must actually differ
# --------------------------------------------------------------------------- #


def test_disfluent_has_longer_mid_turn_pauses_than_clean():
    """The premise of the entire fairness result. If this fails, the finding is
    an artifact of the generator rather than a property of the policies."""
    clean = make_cohort("clean", 40, seed=10)
    disfl = make_cohort("disfluent", 40, seed=11)
    c_longest = np.median([u.longest_mid_turn_pause_s for u in clean])
    d_longest = np.median([u.longest_mid_turn_pause_s for u in disfl])
    assert d_longest > c_longest + 0.2


def test_disfluent_has_more_mid_turn_pauses_than_clean():
    clean = make_cohort("clean", 40, seed=12)
    disfl = make_cohort("disfluent", 40, seed=13)
    assert np.mean([len(u.mid_turn_pauses_s) for u in disfl]) > \
           np.mean([len(u.mid_turn_pauses_s) for u in clean]) + 1.0


def test_slow_cohort_is_slower_than_clean():
    """Elderly proxy: syllables lengthen, so turns take longer for the same
    number of syllables."""
    clean = COHORTS["clean"]
    slow = COHORTS["slow"]
    assert slow.syllable_s > clean.syllable_s


def test_pause_durations_are_heavy_tailed():
    """Log-normal, not uniform. A uniform draw removes exactly the long tail
    that breaks fixed thresholds, which would make the result meaningless."""
    pauses = [p for u in make_cohort("disfluent", 60, seed=14)
              for p in u.mid_turn_pauses_s]
    assert len(pauses) > 50
    p50, p90 = np.percentile(pauses, [50, 90])
    assert p90 > p50 * 1.6, "tail too thin to be log-normal"


def test_pause_durations_are_capped():
    """Beyond ~2 s it is a new turn, not a pause."""
    for u in make_cohort("disfluent", 40, seed=15):
        assert all(p <= 2.0 for p in u.mid_turn_pauses_s)


# --------------------------------------------------------------------------- #
# Pause placement
# --------------------------------------------------------------------------- #


def test_no_pause_at_the_start_of_a_turn():
    """Leading silence is not a mid-turn pause, and would make the first
    syllable's onset ambiguous."""
    for u in make_cohort("disfluent", 15, seed=16):
        head = u.audio[: int(0.05 * u.sr)]
        assert np.abs(head).max() > 0.0


def test_no_pause_immediately_before_true_end():
    """A pause at the end IS the end of the turn, not a mid-turn pause."""
    for u in make_cohort("disfluent", 15, seed=17):
        i = int(u.true_end_s * u.sr)
        window = u.audio[max(0, i - int(0.1 * u.sr)): i]
        assert np.abs(window).max() > 0.0


def test_clean_cohort_pauses_mostly_fall_below_the_pause_floor():
    """Clean pauses are word gaps, not pauses -- which is why short thresholds
    look flawless on this cohort and only on this cohort."""
    pauses = [p for u in make_cohort("clean", 60, seed=18)
              for p in u.mid_turn_pauses_s]
    if pauses:
        assert np.median(pauses) < 0.2


# --------------------------------------------------------------------------- #
# Pitch track
# --------------------------------------------------------------------------- #


def test_f0_track_covers_the_whole_utterance():
    for u in make_cohort("clean", 8, seed=19):
        expected = len(u.audio) / u.sr / 0.010
        assert len(u.f0_track) == pytest.approx(expected, rel=0.02)


def test_f0_is_zero_during_pauses_and_trailing_silence():
    """Praat reports unvoiced as 0.0; the simulator must match that convention
    or the slope fit sees pitch that a real tracker would not."""
    u = make_cohort("disfluent", 1, seed=20)[0]
    assert u.f0_track[-1] == 0.0
    assert any(v == 0.0 for v in u.f0_track)


def test_turn_final_fall_is_present():
    """The yield cue T2 depends on. Without it T2 has nothing to read."""
    u = make_cohort("clean", 1, seed=21)[0]
    voiced = [v for v in u.f0_track if v > 0]
    assert voiced[-1] < voiced[0]
    assert voiced[-1] < np.median(voiced)


def test_context_exposes_words_and_pitch():
    u = make_cohort("clean", 1, seed=22)[0]
    c = u.context()
    assert c.words == u.words
    assert c.f0_track == u.f0_track
    assert len(c.speech_prob) > 0


def test_context_is_cached():
    """The VAD pass is the expensive part of a sweep; the audio never changes."""
    u = make_cohort("clean", 1, seed=23)[0]
    assert u.context() is u.context()


def test_partial_grows_monotonically_with_time():
    """Words are timestamped, so ASR truth truncates honestly at any candidate."""
    u = make_cohort("disfluent", 1, seed=24)[0]
    c = u.context()
    seen = [c.partial_at(t) for t in (0.3, 1.0, 2.0, u.true_end_s)]
    for a, b in zip(seen, seen[1:]):
        assert b.startswith(a)
    assert len(seen[-1]) > len(seen[0])


def test_incomplete_tails_depend_on_where_the_pause_fell():
    """THE LEAK THAT WAS REMOVED.

    The partial must encode only what had been said by a given moment -- never
    whether the turn contains a pause at all. A previous version set the tail to
    an incomplete phrase iff the utterance had any mid-turn pause, which handed
    T1 most of the answer.

    Sampling partials across a cohort must therefore produce BOTH outcomes: some
    land on a function word, some do not.
    """
    from s2slab.policies import INCOMPLETE_TAILS

    verdicts = set()
    for u in make_cohort("disfluent", 30, seed=25):
        c = u.context()
        for frac in (0.25, 0.4, 0.55, 0.7, 0.85):
            words = c.partial_at(u.true_end_s * frac).split()
            if words:
                verdicts.add(words[-1] in INCOMPLETE_TAILS)
    assert verdicts == {True, False}, (
        "every sampled partial resolves the same way -- the generator is "
        "leaking pause existence into the text")


# --------------------------------------------------------------------------- #
# The API
# --------------------------------------------------------------------------- #


def test_unknown_cohort_raises_helpfully():
    with pytest.raises(KeyError, match="unknown cohort"):
        make_cohort("elderly", 5)


def test_make_all_returns_every_registered_cohort():
    assert set(make_all(3)) == set(COHORTS)


def test_every_cohort_declares_what_it_stands_in_for():
    """A proxy that does not say what it proxies is just an unlabelled fake."""
    for spec in COHORTS.values():
        assert spec.replaces.strip()


def test_describe_names_the_real_corpora():
    """The pause distribution has to travel with any cohort-sliced number, or a
    reader cannot interpret it."""
    table = describe(make_all(5))
    assert "SEP-28k" in table
    assert "proxy only" in table
    for name in COHORTS:
        assert name in table


# --------------------------------------------------------------------------- #
# The headline result
# --------------------------------------------------------------------------- #


def test_short_thresholds_fail_disfluent_speakers_and_not_clean_ones():
    """THE FAIRNESS RESULT.

    A 300 ms threshold is flawless on clean speech and cuts off nearly every
    disfluent speaker. Any benchmark that reports one aggregate number would
    select this policy as optimal.
    """
    cohorts = make_all(40, seed=0)
    policy = get("T0_fixed_300ms")

    def cutoff_rate(utts):
        bad = sum(1 for u in utts
                  if policy.decide(u.context()).endpoint_s < u.true_end_s)
        return bad / len(utts)

    assert cutoff_rate(cohorts["clean"]) < 0.05
    assert cutoff_rate(cohorts["disfluent"]) > 0.80


def test_no_fixed_threshold_is_both_fast_and_fair():
    """The motivation for the whole T1-T3 ladder.

    Even at 900 ms -- laggy enough that users think the system is broken --
    disfluent speakers are still cut off. The fix cannot come from a bigger
    number; it has to come from a different signal.
    """
    disfl = make_all(40, seed=0)["disfluent"]
    policy = get("T0_fixed_900ms")
    bad = sum(1 for u in disfl
              if policy.decide(u.context()).endpoint_s < u.true_end_s)
    assert bad > 0, "if this passes cleanly the cohort is too easy — lengthen the tail"


def test_cutoff_rate_falls_monotonically_as_the_threshold_grows():
    """Sanity check on the sweep's core axis. A non-monotone curve means the
    VAD or the policy is misbehaving, not that the finding is subtle."""
    disfl = make_all(40, seed=0)["disfluent"]
    rates = []
    for ms in (300, 500, 700, 900):
        policy = get(f"T0_fixed_{ms}ms")
        bad = sum(1 for u in disfl
                  if policy.decide(u.context()).endpoint_s < u.true_end_s)
        rates.append(bad / len(disfl))
    assert all(b <= a for a, b in zip(rates, rates[1:])), rates


# --------------------------------------------------------------------------- #
# DEV / EVAL sentence split (PR-001, requirement V1)
# --------------------------------------------------------------------------- #


def _spoken(u) -> str:
    return " ".join(w for w, _ in u.words)


def test_default_cohorts_use_only_dev_sentences():
    """Every result published so far used the three DEV sentences. A default
    cohort must never contain an EVAL sentence."""
    from s2slab.simulate import SENTENCES
    for name in COHORTS:
        for u in make_cohort(name, 30, seed=50):
            assert any(s.startswith(_spoken(u)) for s in SENTENCES["dev"]), _spoken(u)


def test_dev_split_is_the_default():
    a = make_all(20, seed=51)
    b = make_all(20, seed=51, split="dev")
    for name in COHORTS:
        assert [_spoken(u) for u in a[name]] == [_spoken(u) for u in b[name]]


def test_eval_split_draws_only_eval_sentences():
    """Plumbing check only: builds EVAL turns but scores NO policy on them, so
    it does not use the EVAL set in the PR-001 sense."""
    from s2slab.simulate import SENTENCES
    for u in make_cohort("disfluent", 20, seed=52, split="eval"):
        assert any(s.startswith(_spoken(u)) for s in SENTENCES["eval"]), _spoken(u)


def test_the_splits_share_no_sentence():
    from s2slab.simulate import SENTENCES
    assert not set(SENTENCES["dev"]) & set(SENTENCES["eval"])


def test_no_sentence_is_short_enough_to_wrap():
    """synth_utterance takes word i % len(sentence); a sentence shorter than the
    longest turn would silently repeat its opening words."""
    from s2slab.simulate import SENTENCES
    longest_turn = max(spec.n_syllables[1] for spec in COHORTS.values())
    for split, pool in SENTENCES.items():
        for s in pool:
            assert len(s.split()) >= longest_turn, (split, s)


def test_eval_is_not_easier_for_t1_than_dev():
    """If EVAL had far more function words, T1 would hold more often there and
    the confirmation run would flatter it. The shares must stay close."""
    from s2slab.policies import INCOMPLETE_TAILS
    from s2slab.simulate import SENTENCES

    def share(pool):
        words = [w for s in pool for w in s.split()]
        return sum(w in INCOMPLETE_TAILS for w in words) / len(words)

    assert abs(share(SENTENCES["dev"]) - share(SENTENCES["eval"])) < 0.10


def test_unknown_split_raises_helpfully():
    with pytest.raises(KeyError, match="unknown sentence split"):
        make_cohort("clean", 1, seed=0, split="test")
