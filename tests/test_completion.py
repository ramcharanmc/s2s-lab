"""
Tests for T1b (PR-001): the policy, its scorer's cache, the DEV tuning, and the requirement V4 --
T1b never cuts off a turn T1 spared. A deterministic fake scorer stands in for the language model;
the one test that loads the real model is marked slow.
"""

from __future__ import annotations

import hashlib

import pytest

from s2slab.completion import GRID, LMEnd, MARGIN_P50_MS, MARGIN_P90_MS, dev, t1b
from s2slab.policies import POLICIES, CompletionThreshold, get
from s2slab.simulate import make_all


def fake(partial: str) -> float:
    """A score in [0, 1) that depends on the text alone, so a test is repeatable."""
    return int(hashlib.sha256(partial.encode()).hexdigest()[:8], 16) / 16 ** 8


def _cut(d, u) -> bool:
    return d.fired and d.endpoint_s < u.true_end_s


@pytest.mark.parametrize("turns", ["prefix", "sentence"])
def test_t1b_never_cuts_off_a_turn_t1_spared(turns):
    """V4. On every DEV turn of every cohort, both turn modes and both development seeds, T1b decides
    no earlier than T1, so a turn T1b cuts off is one T1 cut off too."""
    t1, t1b_ = get("T1_semantic_500ms"), CompletionThreshold(fake, 0.5)
    for seed in (0, 1000):
        for utts in make_all(60, seed, split="dev", turns=turns).values():
            for u in utts:
                a, b = t1.decide(u.context()), t1b_.decide(u.context())
                assert b.endpoint_s >= a.endpoint_s
                if _cut(b, u):
                    assert _cut(a, u)


def test_with_a_scorer_that_never_holds_t1b_decides_exactly_as_t1():
    t1, same = get("T1_semantic_500ms"), CompletionThreshold(lambda p: 1.0, 0.5)
    for utts in make_all(20, 0, split="dev", turns="sentence").values():
        for u in utts:
            assert t1.decide(u.context()).endpoint_s == same.decide(u.context()).endpoint_s


def test_t1b_holds_where_t1_does_and_where_the_model_says_unfinished():
    p = CompletionThreshold(lambda text: 0.9 if text.endswith("floor") else 0.001, 0.01)
    assert p._should_hold("could you show me the")                      # T1's word list
    assert p._should_hold("could you show me the apartment")            # the model: unfinished
    assert not p._should_hold("could you show me the apartment on the first floor")
    assert not p._should_hold("")


def test_t1b_is_not_in_the_default_sweep():
    """The default sweep must not load a language model."""
    assert not any(isinstance(p, CompletionThreshold) for p in POLICIES.values())


def test_the_model_is_asked_once_a_text():
    class Counting(LMEnd):
        calls = 0

        def _score(self, partial):
            Counting.calls += 1
            return 0.5
    s = Counting()
    for _ in range(3):
        s("my account number")
    assert (Counting.calls, s.cache) == (1, {"my account number": 0.5})


def test_dev_picks_the_largest_cutoff_that_keeps_the_margin():
    blob = dev(scorer=fake, n=20)
    assert (blob["split"], blob["turns"], blob["grid"]) == ("dev", "sentence", list(GRID))
    keeps = [th for th in GRID
             if all(r["p50_t1b"] - r["p50_t1"] <= MARGIN_P50_MS and r["p90_t1b"] - r["p90_t1"] <= MARGIN_P90_MS
                    for r in blob["rows"] if r["theta"] == th and r["cohort"] == "clean")]
    assert blob["chosen"] == (max(keeps) if keeps else None)
    assert len(blob["rows"]) == len(GRID) * 2 * 3


def test_t1b_with_a_given_cutoff_and_scorer():
    p = t1b(0.02, fake)
    assert (p.name, p.theta, p.scorer) == ("T1b_complete_500ms", 0.02, fake)


@pytest.mark.slow
def test_the_real_model_scores_a_finished_dev_sentence_above_its_unfinished_prefix():
    s = LMEnd()
    assert s("could you show me the apartment on the first floor") > s("could you show me the apartment on the")


def test_the_cutoff_is_the_one_the_dev_run_chose_and_is_frozen():
    """PR-001, V2: T1b is frozen before the confirmation; the cut-off is the DEV run's, by the rule."""
    import json
    import pathlib

    from s2slab.completion import DEV_OUT, RULE, T1B_THETA
    blob = json.loads((pathlib.Path(__file__).resolve().parents[1] / DEV_OUT).read_text(encoding="utf-8"))
    assert T1B_THETA == blob["chosen"] == 0.01 and T1B_THETA in GRID
    assert (blob["split"], blob["turns"], blob["seeds"], blob["rule"]) == ("dev", "sentence", [0, 1000], RULE)
    assert t1b(scorer=fake).theta == 0.01
