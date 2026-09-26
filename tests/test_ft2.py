"""
Tests for FT-2's criterion (s2slab.ft2) on made-up transcripts over the real eval-set manifest:
no audio, no model, no network.
"""

import re

import pytest

from s2slab import ft2, wer


@pytest.fixture(scope="module")
def refs():
    return wer.references()


def _drop_fillers(text: str) -> str:
    return re.sub(r"\b(uh|um|hmm|mm|mhm|mmm)\b[,.]?\s*", "", text, flags=re.I).strip()


def test_the_eval_set_has_the_registered_shape(refs):
    cohorts = [c for c, _ in refs.values()]
    assert len(refs) == 200 and cohorts.count(ft2.TARGET_COHORT) == 50
    assert sum(cohorts.count(c) for c in ft2.GUARD_COHORTS) == 150
    fillers_in_target = sum(sum(ft2.fillers(t).values()) for c, t in refs.values() if c == ft2.TARGET_COHORT)
    assert fillers_in_target > 0, "the disfluent references carry fillers, or S2 measures nothing"


def test_a_lora_that_keeps_the_fillers_and_changes_nothing_else_is_supported(refs):
    base = {i: _drop_fillers(t) for i, (c, t) in refs.items()}       # small.en's habit: fillers gone
    lora = {i: t for i, (c, t) in refs.items()}                       # verbatim, perfect
    c = ft2.pr011(base, lora, n=500)
    assert c["n_clips"] == 200 and c["n_target"] == 50 and c["n_guard"] == 150
    assert c["s1"]["lora_raw_wer"] == 0.0 and c["s1"]["base_raw_wer"] > 0 and c["s1"]["ci95_pp"][1] < 0
    assert c["s2"]["lora_recall"] == 1.0 and c["s2"]["base_kept"] == 0 and c["s2"]["pass"]
    assert c["s3"]["diff_pp"] == pytest.approx(0.0) and c["s3"]["pass"]      # normalized: fillers never counted
    assert c["supported"]


def test_each_criterion_can_fail_on_its_own(refs):
    verbatim = {i: t for i, (c, t) in refs.items()}
    dropped = {i: _drop_fillers(t) for i, (c, t) in refs.items()}
    # S1 and S2: the LoRA behaves like the base -- no gain, no fillers
    same = ft2.pr011(dropped, dropped, n=200)
    assert not same["s1"]["pass"] and same["s1"]["diff_pp"] == 0.0 and not same["supported"]
    # S3: the LoRA keeps fillers but garbles every other speaker's clip
    broken = {i: (t if c == ft2.TARGET_COHORT else "nothing like it at all") for i, (c, t) in refs.items()}
    guard_fail = ft2.pr011(dropped, broken, n=200)
    assert guard_fail["s1"]["pass"] and guard_fail["s2"]["pass"] and not guard_fail["s3"]["pass"]
    assert not guard_fail["supported"]
    # S2 alone: fillers kept only in part
    ids = [i for i, (c, _) in refs.items() if c == ft2.TARGET_COHORT]
    half = dict(verbatim)
    for i in ids[::2]:
        half[i] = dropped[i]
    partial = ft2.pr011(dropped, half, n=200)
    assert 0 < partial["s2"]["lora_recall"] < 1


def test_the_bootstrap_is_seeded_and_the_difference_is_in_points(refs):
    ids = [i for i, (c, _) in refs.items() if c == ft2.TARGET_COHORT]
    base = ft2.per_clip({i: _drop_fillers(refs[i][1]) for i in ids}, refs, wer.light)
    lora = ft2.per_clip({i: refs[i][1] for i in ids}, refs, wer.light)
    a = ft2.bootstrap_points(base, lora, ids, n=300, seed=0)
    b = ft2.bootstrap_points(base, lora, ids, n=300, seed=0)
    assert a == b and a[1] <= a[0] <= a[2] and a[0] == pytest.approx(-100 * ft2.corpus_rate(base, ids))


def test_the_report_names_the_verdict_and_the_validity_problems(refs):
    base = {i: _drop_fillers(t) for i, (c, t) in refs.items()}
    lora = {i: t for i, (c, t) in refs.items()}
    results = {"criteria": ft2.pr011(base, lora, n=200), "commit": "abc1234", "export_sha256": "ff",
               "base_model": ft2.BASE_MODEL}
    text = ft2.format_pr011(results)
    assert text.endswith("PR-011: SUPPORTED") and "S1" in text and "S3" in text
    results["commit"] = "abc1234-dirty"
    assert "NOT THE REGISTERED RUN" in ft2.format_pr011(results)
    results["commit"] = "abc1234"
    results["criteria"] = ft2.pr011(base, base, n=200)
    assert ft2.format_pr011(results).endswith("NOT SUPPORTED -- failed S1, S2")
    half = {k: v for k, v in base.items() if refs[k][0] != "slow"}
    short = {"criteria": ft2.pr011(half, half, n=100), "commit": "abc1234", "export_sha256": "ff", "base_model": "small.en"}
    assert any("not the frozen 200" in p for p in ft2.pr011_validity(short))
    assert ft2.FILLER_RECALL_BAR == 0.80 and ft2.GUARD_PP == 1.0 and ft2.BASE_MODEL == "small.en"
