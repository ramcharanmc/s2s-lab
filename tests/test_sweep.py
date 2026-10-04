"""
Tests for s2slab.sweep.

Most assertions build PolicyResult objects directly rather than running a whole
sweep, so the suite stays fast and each rule is tested in isolation. Two
integration tests run the real thing.

The property under most scrutiny here is the FRONTIER. It is the output a reader
acts on -- "these are your real options" -- so a bug that admits a bad policy to
it is worse than a wrong number in a cell, which at least invites checking.
"""

from __future__ import annotations

import json

import pytest

from s2slab.policies import POLICIES, get
from s2slab.simulate import make_cohort
from s2slab.sweep import (COMPARISONS, PolicyResult, evaluate, format_cohort,
                          format_comparisons, pareto_frontier, run_comparisons,
                          run_sweep, write_results)
from s2slab.timing import summarize


def result(policy="p", cohort="c", n=60, n_cutoff=0, n_never_fired=0,
           lat_ms=500.0, n_scored=None) -> PolicyResult:
    """A PolicyResult with a latency sample big enough to be supported."""
    scored = n - n_cutoff - n_never_fired if n_scored is None else n_scored
    return PolicyResult(policy=policy, cohort=cohort, n=n, n_cutoff=n_cutoff,
                        n_never_fired=n_never_fired,
                        latency=summarize([lat_ms] * scored))


# --------------------------------------------------------------------------- #
# The three buckets
# --------------------------------------------------------------------------- #


def test_every_turn_lands_in_exactly_one_bucket():
    """Interrupted, never-decided, or correctly endpointed. No double counting,
    no turn quietly dropped."""
    utts = make_cohort("disfluent", 30, seed=1)
    r = evaluate(get("T0_fixed_500ms"), utts)
    assert r.n == 30
    assert r.n_cutoff + r.n_never_fired + r.latency.n == r.n


def test_a_cutoff_contributes_no_latency():
    """THE MEASUREMENT BUG THIS MODULE EXISTS TO AVOID.

    An interruption is not negative latency. Mixing them produces a median that
    describes neither -- which is how a policy that interrupts most speakers
    gets reported as fast.
    """
    utts = make_cohort("disfluent", 40, seed=2)
    r = evaluate(get("T0_fixed_300ms"), utts)
    assert r.n_cutoff > 0
    assert r.latency.n == r.n - r.n_cutoff - r.n_never_fired
    if r.latency.n:
        assert r.latency.p50 >= 0, "a latency over correct turns cannot be negative"


def test_latency_is_never_negative_on_any_cohort_or_policy():
    """Sweeping the whole grid: no cell may report a negative latency, because
    every negative case was routed to the cutoff bucket instead."""
    for r in run_sweep(n_per_cohort=15, seed=3):
        if r.latency.n:
            assert r.latency.p50 >= 0, f"{r.policy}/{r.cohort}"


def test_cutoff_rate_denominator_is_every_turn():
    """Excluding the turns we failed on would flatter the rate."""
    r = result(n=100, n_cutoff=25)
    assert r.cutoff_rate == pytest.approx(0.25)


def test_no_fire_is_tracked_separately_from_cutoff():
    """Running out of audio is not an interruption and not a latency."""
    r = result(n=100, n_cutoff=10, n_never_fired=20)
    assert r.cutoff_rate == pytest.approx(0.10)
    assert r.no_fire_rate == pytest.approx(0.20)
    assert r.latency.n == 70


def test_empty_cohort_does_not_divide_by_zero():
    import math
    r = PolicyResult("p", "c", 0, 0, 0, summarize([]))
    assert math.isnan(r.cutoff_rate)


# --------------------------------------------------------------------------- #
# The frontier
# --------------------------------------------------------------------------- #


def test_a_strictly_worse_policy_is_dominated():
    good = result("good", lat_ms=300.0, n_cutoff=5)
    bad = result("bad", lat_ms=600.0, n_cutoff=20)
    assert [r.policy for r in pareto_frontier([good, bad])] == ["good"]


def test_a_tradeoff_keeps_both():
    """Faster but riskier versus slower but safer: both are real options."""
    fast = result("fast", lat_ms=200.0, n_cutoff=20)
    safe = result("safe", lat_ms=800.0, n_cutoff=2)
    assert len(pareto_frontier([fast, safe])) == 2


def test_a_tie_on_one_axis_is_broken_by_the_other():
    same_cut_slow = result("slow", lat_ms=600.0, n_cutoff=10)
    same_cut_fast = result("fast", lat_ms=400.0, n_cutoff=10)
    assert [r.policy for r in pareto_frontier([same_cut_slow, same_cut_fast])] == ["fast"]


def test_frontier_is_sorted_by_latency():
    rows = [result("c", lat_ms=900.0, n_cutoff=1),
            result("a", lat_ms=200.0, n_cutoff=30),
            result("b", lat_ms=500.0, n_cutoff=10)]
    assert [r.policy for r in pareto_frontier(rows)] == ["a", "b", "c"]


def test_a_policy_with_no_scored_turns_is_excluded():
    """100% cutoff means no latency sample at all. Admitting it on a NaN would
    make the worst possible policy look unbeatable on the latency axis."""
    dire = PolicyResult("dire", "c", 60, 60, 0, summarize([]))
    fine = result("fine", lat_ms=500.0, n_cutoff=5)
    assert [r.policy for r in pareto_frontier([dire, fine])] == ["fine"]


def test_an_underpowered_latency_is_excluded():
    """THE REGRESSION GUARD.

    A policy that interrupted 97% of speakers has a p50 built from two turns. It
    is a real number, so nothing dominates it, so it lands on the frontier
    looking like a fast option. The sample-size rule from timing.py is what
    stops the frontier recommending a policy that mostly fails.
    """
    barely = result("barely", n=60, n_cutoff=58, lat_ms=100.0)   # 2 scored
    honest = result("honest", n=60, n_cutoff=5, lat_ms=500.0)    # 55 scored
    assert "p50" in barely.latency.unsupported
    assert [r.policy for r in pareto_frontier([barely, honest])] == ["honest"]


def test_exactly_at_the_support_threshold_is_admitted():
    """20 scored turns is the documented minimum for p50; 19 is not."""
    at = result("at", n=100, n_cutoff=80, lat_ms=100.0)       # 20 scored
    below = result("below", n=100, n_cutoff=81, lat_ms=100.0)  # 19 scored
    assert "p50" not in at.latency.unsupported
    assert "p50" in below.latency.unsupported


def test_frontier_of_an_empty_list_is_empty():
    assert pareto_frontier([]) == []


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #


def test_sweep_covers_every_policy_and_cohort():
    results = run_sweep(n_per_cohort=10, seed=4)
    assert len(results) == 3 * len(POLICIES)
    assert {r.cohort for r in results} == {"clean", "disfluent", "slow"}
    assert {r.policy for r in results} == set(POLICIES)


def _comparable(d: dict) -> dict:
    """NaN never equals itself, so a dict holding one never equals its own copy.

    That is a direct consequence of reporting NaN rather than 0 for "not
    measured" -- the right call, but it means any structural comparison has to
    normalise first. Doing it here rather than weakening the metric.
    """
    import math
    return {k: ("nan" if isinstance(v, float) and math.isnan(v) else v)
            for k, v in d.items()}


def test_sweep_is_deterministic():
    """Same seed, same numbers -- or no result is reproducible."""
    a = run_sweep(n_per_cohort=10, seed=5)
    b = run_sweep(n_per_cohort=10, seed=5)
    assert [_comparable(r.as_dict()) for r in a] == \
           [_comparable(r.as_dict()) for r in b]


def test_a_different_seed_changes_the_numbers():
    """Guards against a sweep that ignores its seed and silently returns the
    same cohort every time, which would make determinism trivially true."""
    a = run_sweep(n_per_cohort=10, seed=5)
    b = run_sweep(n_per_cohort=10, seed=6)
    assert [_comparable(r.as_dict()) for r in a] != \
           [_comparable(r.as_dict()) for r in b]


def test_the_headline_finding_holds():
    """Short thresholds are flawless on fluent speakers and catastrophic on
    disfluent ones. If this ever stops being true, something broke upstream."""
    results = {(r.policy, r.cohort): r for r in run_sweep(n_per_cohort=40, seed=0)}
    assert results[("T0_fixed_300ms", "clean")].cutoff_rate < 0.05
    assert results[("T0_fixed_300ms", "disfluent")].cutoff_rate > 0.80


# --------------------------------------------------------------------------- #
# Reporting and provenance
# --------------------------------------------------------------------------- #


def test_table_marks_the_frontier_and_hides_unsupported_numbers():
    rows = [result("winner", cohort="x", lat_ms=300.0, n_cutoff=2),
            result("loser", cohort="x", lat_ms=900.0, n_cutoff=30)]
    table = format_cohort(rows, "x")
    assert "### x" in table
    assert "**frontier**" in table
    assert "—" in table, "p90 is unsupported at this sample size"


def test_table_shows_a_dash_when_there_is_no_latency_at_all():
    rows = [PolicyResult("dire", "x", 60, 60, 0, summarize([]))]
    assert "| — |" in format_cohort(rows, "x")


def test_results_file_records_what_produced_it(tmp_path):
    """Silent changes to a benchmark are how people accidentally claim
    improvements. A results file that cannot name its own commit, seed and
    sample size is not evidence of anything."""
    results = run_sweep(n_per_cohort=10, seed=7)
    path = write_results(results, 10, 7, str(tmp_path / "sweep.json"))
    blob = json.loads(path.read_text())
    assert blob["seed"] == 7
    assert blob["n_per_cohort"] == 10
    assert blob["commit"]
    assert len(blob["results"]) == len(results)
    assert "cutoff_rate" in blob["results"][0]


def test_results_file_creates_its_directory(tmp_path):
    path = write_results(run_sweep(5, 8), 5, 8, str(tmp_path / "a" / "b.json"))
    assert path.exists()


# --------------------------------------------------------------------------- #
# cut_mask -- the per-turn record the paired tests depend on
# --------------------------------------------------------------------------- #


def test_cut_mask_has_one_entry_per_turn_and_agrees_with_the_count():
    utts = make_cohort("disfluent", 30, seed=31)
    r = evaluate(get("T0_fixed_500ms"), utts)
    assert len(r.cut_mask) == r.n == 30
    assert sum(r.cut_mask) == r.n_cutoff


def test_cut_mask_follows_the_order_of_the_turns():
    """The pairing contract: entry i must be turn i. Reverse the turns and the
    mask must reverse with them."""
    utts = make_cohort("disfluent", 20, seed=32)
    fwd = evaluate(get("T0_fixed_500ms"), utts).cut_mask
    rev = evaluate(get("T0_fixed_500ms"), list(reversed(utts))).cut_mask
    assert rev == tuple(reversed(fwd))


def test_cut_mask_stays_out_of_the_printout():
    """250 bools in every repr would bury everything else."""
    r = evaluate(get("T0_fixed_500ms"), make_cohort("clean", 5, seed=33))
    assert "cut_mask" not in repr(r)


def test_t1_can_never_interrupt_a_turn_t0_spared():
    """WHY THE '0' IN '41 / 0' IS NOT EVIDENCE.

    At the same base threshold T1 can only ever require MORE silence than T0,
    never less, so it can only fire later. A turn T0 did not interrupt, T1
    cannot interrupt either -- the zero is guaranteed by the code. What the
    comparison actually measures is the other number: how many of T0's
    interruptions T1 rescues. If this test ever fails, the hold has started
    subtracting silence and T1 is no longer a strict extension of T0.
    """
    for cohort in ("disfluent", "slow"):
        utts = make_cohort(cohort, 40, seed=34)
        t0 = evaluate(get("T0_fixed_500ms"), utts).cut_mask
        t1 = evaluate(get("T1_semantic_500ms"), utts).cut_mask
        assert all(t0_i or not t1_i for t0_i, t1_i in zip(t0, t1)), cohort


def test_known_simulator_flattery_t2_never_changes_who_is_interrupted():
    """Documents a limitation of the SIMULATOR, not a virtue of T2.

    In synthetic turns the only sharp pitch fall is at the true end of the
    turn, so T2's early commit can never land mid-turn and T2 interrupts
    exactly the turns T1 does. Real speech falls at clause boundaries in the
    middle of a turn too. On real audio this assertion should FAIL -- and the
    size of that failure is the real cost of committing early on pitch.
    """
    for cohort in ("disfluent", "slow"):
        utts = make_cohort(cohort, 40, seed=35)
        t1 = evaluate(get("T1_semantic_500ms"), utts).cut_mask
        t2 = evaluate(get("T2_prosodic_500ms"), utts).cut_mask
        assert t1 == t2, f"{cohort}: T2 now changes who is interrupted -- update this test"


# --------------------------------------------------------------------------- #
# run_comparisons / format_comparisons
# --------------------------------------------------------------------------- #


def test_one_comparison_per_declared_pair_per_cohort():
    comps = run_comparisons(run_sweep(n_per_cohort=15, seed=36))
    assert len(comps) == 3 * len(COMPARISONS)
    assert {c.cohort for c in comps} == {"clean", "disfluent", "slow"}


def test_every_comparison_is_adjusted_as_one_family():
    """Holm runs once over all nine, so every adjusted p is filled in and none
    is smaller than its raw p."""
    import math
    comps = run_comparisons(run_sweep(n_per_cohort=15, seed=37))
    for c in comps:
        assert not math.isnan(c.p_adjusted)
        assert c.p_adjusted >= c.test.p - 1e-12


def test_comparisons_compare_the_same_turns():
    comps = run_comparisons(run_sweep(n_per_cohort=15, seed=38))
    assert all(c.n == 15 for c in comps)


def test_a_pair_naming_an_unswept_policy_is_skipped_not_crashed():
    results = run_sweep(n_per_cohort=5, seed=39)
    assert run_comparisons(results, pairs=[("T0_fixed_500ms", "T9_nope")]) == []


def test_comparison_table_has_one_row_per_comparison():
    comps = run_comparisons(run_sweep(n_per_cohort=15, seed=40))
    table = format_comparisons(comps)
    rows = [ln for ln in table.splitlines()
            if ln.startswith("| ") and not ln.startswith("| cohort")]
    assert len(rows) == len(comps)
    assert f"Holm-adjusted across {len(comps)}" in table


def test_comparisons_are_saved_with_the_results(tmp_path):
    results = run_sweep(n_per_cohort=10, seed=41)
    comps = run_comparisons(results)
    blob = json.loads(write_results(results, 10, 41, str(tmp_path / "s.json"),
                                    comparisons=comps).read_text())
    assert len(blob["comparisons"]) == len(comps)
    row = blob["comparisons"][0]
    for key in ("cohort", "a", "b", "only_a", "only_b", "p", "p_holm", "ci", "verdict"):
        assert key in row


def test_results_without_comparisons_still_write(tmp_path):
    """The old four-argument call must keep working."""
    blob = json.loads(write_results(run_sweep(5, 42), 5, 42,
                                    str(tmp_path / "s.json")).read_text())
    assert blob["comparisons"] == []


# --------------------------------------------------------------------------- #
# Sentence split: reproduction, provenance, and the EVAL lock
# --------------------------------------------------------------------------- #


def test_default_split_reproduces_the_published_seed0_numbers():
    """Notion page 18 section 7 reports 206 and 158 of 250 disfluent turns at
    seed 0. The DEV split keeps the same sentences and the same random draw, so
    those numbers must not move."""
    from s2slab.simulate import make_all
    utts = make_all(250, seed=0)["disfluent"]
    assert evaluate(get("T0_fixed_500ms"), utts).n_cutoff == 206
    assert evaluate(get("T1_semantic_500ms"), utts).n_cutoff == 158


def test_results_file_records_the_split(tmp_path):
    blob = json.loads(write_results(run_sweep(5, 60), 5, 60,
                                    str(tmp_path / "s.json")).read_text())
    assert blob["split"] == "dev"
    assert blob["registration"] is None


def test_results_file_records_a_registration(tmp_path):
    blob = json.loads(write_results(run_sweep(5, 61), 5, 61,
                                    str(tmp_path / "s.json"),
                                    split="dev", registration="PR-001").read_text())
    assert blob["registration"] == "PR-001"


def test_results_file_records_the_turn_mode(tmp_path):
    blob = json.loads(write_results(run_sweep(5, 62, turns="sentence"), 5, 62, str(tmp_path / "s.json"),
                                    turns="sentence").read_text())
    assert blob["turns"] == "sentence"
    blob = json.loads(write_results(run_sweep(5, 62), 5, 62, str(tmp_path / "p.json")).read_text())
    assert blob["turns"] == "prefix"


def test_command_line_refuses_eval_without_a_registration():
    """The EVAL lock. Runs the real command; it must exit before generating any
    audio, so no policy is ever scored on the EVAL sentences by accident."""
    import os
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", "s2slab.sweep", "--split", "eval", "--n", "1"],
        cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    assert "reserved for a registered confirmation run" in proc.stderr


def test_command_line_refuses_pr001s_confirmation_on_the_old_turns():
    """Seed 2000 is spent once: a PR-001 confirmation on prefix turns must stop before any audio."""
    import os
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", "s2slab.sweep", "--split", "eval", "--confirm", "PR-001", "--n", "1"],
        cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    assert "add --turns sentence" in proc.stderr
