"""
Tests for s2slab.pr019, PR-019's analysis (P1b on real disfluent speech): the criteria on hand-made test turns --
DEV-like turns that fail S3, a T2 that is truly faster, cut-offs over 5%, a T2 that cuts more -- a turn neither
policy ends, the registered setting, and the run refused without its confirmation.
"""

from __future__ import annotations

import pytest

from s2slab import pr019


def side(cut=False, wait=900.0, reason="T2_prosodic_900ms_s20"):
    return {"fired": True, "endpoint_s": 1.0, "cut": cut, "reason": reason, "wait_ms": None if cut else wait}


def blob(n=207, t2=None, t0=None):
    t2, t0 = t2 or {}, t0 or {}
    turns = {f"ds-test-{i:03d}": {"true_end_s": 5.0, "pauses_s": [0.36],
                                  "T2": side(**t2.get(i, {})), "T0": side(reason="T0_fixed_900ms", **t0.get(i, {}))}
             for i in range(n)}
    return {"registration": "PR-019", "commit": "abc1234", "split": "test", "n": n, "turns": turns,
            "policies": {"T2": "T2_prosodic_900ms_s20", "T0": "T0_fixed_900ms"},
            "sweep": {"rows": [{"policy": "T0_fixed_900ms", "cut": 2, "n": n, "p50": 900.0, "p90": 900.0}]}}


def test_dev_like_turns_pass_s1_and_s2_and_fail_s3():
    b = blob(t2={0: {"cut": True}}, t0={0: {"cut": True}, 1: {"cut": True}})       # DEV: T2 cut 1, T0 2, the same wait
    r = pr019.criteria(b, n_boot=300)
    assert r["checks"]["S1 under 5%"][0] and r["checks"]["S2 no more cut-offs"][0]
    assert not r["checks"]["S3 strictly better"][0] and r["verdict"] == "NOT SUPPORTED: failed S3 strictly better"
    assert r["describe"]["only"] == {"T2": 0, "T0": 1} and r["describe"]["cut_ids"]["T0"] == ["ds-test-000", "ds-test-001"]
    text = pr019.format_pr019(b, r)
    assert "S3 strictly better     FAIL" in text and "T0_fixed_900ms" in text and "longest pause, the median: 360 ms" in text


def test_a_faster_t2_with_no_more_cut_offs_is_supported():
    b = blob(t2={i: {"wait": 750.0, "reason": "T2_prosodic_900ms_s20+yield"} for i in range(207)})
    r = pr019.criteria(b, n_boot=300)
    assert r["verdict"] == "SUPPORTED: all three criteria hold" and r["values"]["wait_ratio"][2] < 1.0
    assert r["describe"]["t2_at_the_end"] == {"yield": 207}


def test_cut_offs_near_5_percent_fail_s1_and_more_cut_offs_fail_s2():
    r = pr019.criteria(blob(t2={i: {"cut": True} for i in range(5)}, t0={i: {"cut": True} for i in range(5)}), n_boot=300)
    assert not r["checks"]["S1 under 5%"][0] and "5 of 207" in r["checks"]["S1 under 5%"][1]
    assert pr019.criteria(blob(t2={i: {"cut": True} for i in range(4)}), n_boot=300)["checks"]["S1 under 5%"][0]
    r = pr019.criteria(blob(t2={0: {"cut": True}}), n_boot=300)
    assert not r["checks"]["S2 no more cut-offs"][0]


def test_many_fewer_cut_offs_pass_s3_by_mcnemar_alone():
    r = pr019.criteria(blob(t0={i: {"cut": True} for i in range(8)}), n_boot=300)
    assert r["checks"]["S3 strictly better"][0] and r["values"]["mcnemar_p"] < 0.01


def test_a_turn_no_policy_ends_is_a_deviation_and_the_setting_is_deVs():
    b = blob()
    b["turns"]["ds-test-003"]["T0"]["fired"] = False
    with pytest.raises(ValueError, match="drops no turn"):
        pr019.criteria(b, n_boot=10)
    pols = pr019.policies()
    assert pols["T2"].name == "T2_prosodic_900ms_s20" and pols["T2"].slope_threshold == 20.0
    assert pols["T2"].silence_ms == 900 and pols["T0"].name == "T0_fixed_900ms"


def test_the_run_is_refused_without_its_confirmation():
    with pytest.raises(SystemExit, match="--confirm PR-019"):
        pr019.main([])
