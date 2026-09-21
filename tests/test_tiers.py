"""
Tests for s2slab.tiers -- PR-004's analysis, on made-up runs only.

The registered run does not exist yet when these are written, and the script
must not be shaped by it. Every run here is built by make_run(): 40 clips, ten
per cohort, each with the reference "the quick brown fox jumps" (five words).
"""

from __future__ import annotations

import numpy as np
import pytest

from s2slab.compare import resamples
from s2slab.tiers import analyse, format_report, percentile_ci, reproduces, wer_difference
from s2slab.wer import light

COHORTS = ("clean", "accented", "disfluent", "slow")
REF = "the quick brown fox jumps"
IDS = [f"{c}-{k:02d}" for c in COHORTS for k in range(10)]
REFS = {i: (i.split("-")[0], REF) for i in IDS}
MACHINE = "Modal Tesla T4, 580.95.05, one container: r1-gpu then r2-gpu then r2-gpu-turbo"


def make_run(rung, heard=None, said=None, first_audio=500.0, asr=70.0, commit="abc1234", machine=MACHINE):
    """heard / said: {clip id: text} for the clips that differ from the reference / from "Okay."."""
    heard, said = heard or {}, said or {}
    turns = {}
    for i in IDS:
        fa = first_audio[i] if isinstance(first_audio, dict) else first_audio
        turns[i] = {"cohort": REFS[i][0], "transcript": heard.get(i, REF.capitalize() + "."),
                    "reply_text": said.get(i, "Okay."), "stages_ms": {"asr": asr, "llm": 300.0, "tts": 130.0},
                    "total_ms": fa + 200, "first_audio_ms": fa, "first_sound_ms": fa + 320,
                    "stall_ms": 0.0, "reply_s": 1.0, "chunks": [[fa, 24000]]}
    return {"rung": rung, "commit": commit, "machine": machine, "ran": "2026-09-21", "turns": turns}


# R1 mishears one word in every accented clip and in two clean ones: 12 errors in 200 words = 6.0%.
FAST_HEARD = {**{f"accented-{k:02d}": "the quick brown fox jump" for k in range(10)},
              "clean-00": "the quick brown box jumps", "clean-01": "a quick brown fox jumps"}
# The slow tier still mishears two accented clips, and one slow clip R1 got right: 3 errors = 1.5%.
SLOW_HEARD = {"accented-00": "the quick brown fox jump", "accented-01": "the quick brown socks jumps",
              "slow-00": "the quick brown fox jumped"}


def pair(first_audio=880.0, **kw):
    said = {i: "Different words." for i in ("accented-02", "accented-03", "clean-00")}
    return (make_run("r1-gpu", FAST_HEARD),
            make_run("r2-gpu-turbo", SLOW_HEARD, said, first_audio=first_audio, asr=380.0, **kw))


# --------------------------------------------------------------------------- #
# The two statistics
# --------------------------------------------------------------------------- #


def test_the_wer_difference_is_corpus_level_and_in_points():
    d = wer_difference([1, 0, 2, 0], [0, 0, 1, 0], [4, 4, 4, 4], resamples(4, 2000))
    assert d.value == pytest.approx(-12.5)                       # 1/16 - 3/16
    assert d.lo <= d.value <= d.hi <= 0


def test_long_clips_weigh_more_than_short_ones():
    """Corpus WER, not the mean of clip WERs: one error in a 2-word clip and none in a 98-word clip is 1%."""
    d = wer_difference([0, 0], [1, 0], [2, 98], resamples(2, 100))
    assert d.value == pytest.approx(1.0)


def test_identical_tiers_differ_by_nothing():
    d = wer_difference([1, 2, 0], [1, 2, 0], [5, 5, 5], resamples(3, 500))
    assert (d.value, d.lo, d.hi) == (0.0, 0.0, 0.0)


def test_a_percentile_comes_with_its_interval():
    values = np.arange(1.0, 202.0)                               # 1..201: the median is 101
    e = percentile_ci(values, 50, resamples(201, 2000))
    assert e.value == 101.0 and e.lo < 101.0 < e.hi and e.hi - e.lo < 40


# --------------------------------------------------------------------------- #
# The registered criteria
# --------------------------------------------------------------------------- #


def test_a_slow_tier_that_hears_more_and_starts_within_a_second_is_supported():
    fast, slow = pair(first_audio=880.0)
    r = analyse(fast, slow, REFS, light, n_boot=2000)
    assert r["describe"]["wer_overall"] == pytest.approx((6.0, 1.5))
    assert all(ok for ok, _ in r["checks"].values()) and all(ok for ok, _ in r["valid"].values())
    assert r["verdict"] == "SUPPORTED: all three criteria hold"
    assert "-4.500" in r["checks"]["S1 hears more"][1]           # 1.5% - 6.0%
    assert "-16.000" in r["checks"]["S2 accented"][1] and "(10 clips)" in r["checks"]["S2 accented"][1]


def test_too_slow_a_start_fails_s3_alone():
    fast, slow = pair(first_audio=1100.0)
    r = analyse(fast, slow, REFS, light, n_boot=2000)
    assert r["verdict"] == "NOT SUPPORTED: failed S3 within a second"


def test_s3_is_judged_on_the_intervals_upper_end_not_the_median():
    first = {i: (990.0 if k % 2 else 1030.0) for k, i in enumerate(IDS)}     # p50 1010, straddling the bar
    fast, slow = pair(first_audio=first)
    ok, detail = analyse(fast, slow, REFS, light, n_boot=2000)["checks"]["S3 within a second"]
    assert not ok and "1010.000" in detail


def test_a_gain_that_misses_the_accented_cohort_fails_s2():
    slow_heard = {f"accented-{k:02d}": "the quick brown fox jump" for k in range(10)}     # accent no better
    fast = make_run("r1-gpu", {**FAST_HEARD, **{f"slow-{k:02d}": "quick fox" for k in range(10)}})
    r = analyse(fast, make_run("r2-gpu-turbo", slow_heard, first_audio=880.0), REFS, light, n_boot=2000)
    assert r["checks"]["S1 hears more"][0] and not r["checks"]["S2 accented"][0]
    assert r["verdict"] == "NOT SUPPORTED: failed S2 accented"


def test_a_tier_that_hears_no_better_fails_s1():
    fast, _ = pair()
    r = analyse(fast, make_run("r2-gpu-turbo", FAST_HEARD, first_audio=880.0), REFS, light, n_boot=500)
    assert not r["checks"]["S1 hears more"][0] and "0.000" in r["checks"]["S1 hears more"][1]


# --------------------------------------------------------------------------- #
# Validity: the same container, a clean tree, the same clips
# --------------------------------------------------------------------------- #


def test_a_run_from_a_dirty_tree_is_not_the_registered_run():
    fast, slow = pair(commit="abc1234-dirty")
    fast["commit"] = "abc1234-dirty"
    r = analyse(fast, slow, REFS, light, n_boot=500)
    assert r["verdict"] == "NOT THE REGISTERED RUN: failed V3 clean tree"


def test_runs_from_two_commits_or_two_containers_are_not_the_registered_run():
    fast, slow = pair(commit="fffffff")
    assert "V3 clean tree" in analyse(fast, slow, REFS, light, n_boot=500)["verdict"]
    fast, slow = pair(machine="Modal Tesla T4, 580.95.05")
    assert "V1 same run" in analyse(fast, slow, REFS, light, n_boot=500)["verdict"]


def test_runs_over_different_clips_are_refused():
    fast, slow = pair()
    del slow["turns"]["slow-09"]
    with pytest.raises(ValueError, match="same clips"):
        analyse(fast, slow, REFS, light, n_boot=100)


def test_a_turn_with_no_reply_audio_is_refused_not_dropped():
    fast, slow = pair()
    slow["turns"]["clean-03"]["first_audio_ms"] = None
    with pytest.raises(ValueError, match="clean-03"):
        analyse(fast, slow, REFS, light, n_boot=100)


# --------------------------------------------------------------------------- #
# What is described
# --------------------------------------------------------------------------- #


def test_error_overlap_is_clips_wrong_for_both_over_clips_wrong_for_either():
    d = analyse(*pair(), REFS, light, n_boot=500)["describe"]
    assert d["overlap"] == {"both": 2, "only_fast": 10, "only_slow": 1, "neither": 27,
                            "share": pytest.approx(2 / 13)}


def test_agreement_is_word_for_word_after_normalizing():
    d = analyse(*pair(), REFS, light, n_boot=500)["describe"]
    # they differ on accented-01..09 (9), clean-00, clean-01, slow-00; accented-00 is misheard identically
    assert d["agree"] == 40 - 12


def test_reply_changes_are_split_by_whether_the_transcript_differed():
    d = analyse(*pair(), REFS, light, n_boot=500)["describe"]
    assert d["reply_changed"] == {"all": 3, "transcript_differs": (12, 3), "transcript_same": (28, 0)}


def test_the_added_wait_is_split_into_the_recognizer_and_everything_after_it():
    first = {i: 880.0 for i in IDS}
    first.update({"accented-02": 1000.0, "accented-03": 1000.0, "clean-00": 1000.0})     # the changed replies
    fast, slow = pair(first_audio=first)
    w = analyse(fast, slow, REFS, light, n_boot=500)["describe"]["added_wait"]
    assert (w["first_audio"], w["asr"], w["rest"]) == (380.0, 310.0, 70.0)
    assert w["rest_same_reply"] == (37, 70.0) and w["rest_new_reply"] == (3, 190.0)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def test_the_report_gives_the_verdict_for_the_registered_tier_only():
    fast, slow = pair()
    text = format_report(fast, slow, analyse(fast, slow, REFS, light, n_boot=500))
    assert text.startswith("PR-004: r2-gpu-turbo against r1-gpu")
    assert "V3 clean tree        PASS  commit abc1234" in text
    assert "S1 hears more        PASS" in text and "SUPPORTED: all three criteria hold" in text
    assert "error overlap 15% (both / either)" in text
    assert "first audio +380 ms = recognizer +310 ms + everything after it +70 ms" in text

    small = make_run("r2-gpu", SLOW_HEARD, first_audio=600.0, asr=125.0)
    text = format_report(fast, small, analyse(fast, small, REFS, light, n_boot=500), tested=False)
    assert text.startswith("Described only, not tested: r2-gpu against r1-gpu")
    assert "PASS" not in text and "FAIL" not in text and "SUPPORTED" not in text
    assert "normalized WER, slow minus fast: -4.500" in text


def test_a_rerun_is_compared_with_the_run_it_should_reproduce():
    then = make_run("r1-gpu", FAST_HEARD, first_audio=491.0, commit="b107927")
    now = make_run("r1-gpu", {**FAST_HEARD, "slow-05": "the quick brown fox"}, first_audio=501.0)
    text = reproduces(now, then)
    assert "identical transcripts 39, identical replies 40" in text
    assert "first-audio p50 501 vs 491 ms, median per-turn difference +10 ms" in text