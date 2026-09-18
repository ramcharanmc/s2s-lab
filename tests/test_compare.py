"""
Tests for s2slab.compare -- PR-002's analysis, on made-up runs.

Each fake run is built so the right answer is known: a streaming rung that
really is faster, one that is not, one that says different words, one that
stalls. No models, no audio, no eval set.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2slab.compare import (Estimate, compare, correlation, format_report, percentile_ratio,
                            resamples)

N = 200


def run(rung, first, total, n_words, stall=None, reply=None, pieces=None):
    """A results dict in evalrun's shape. `first`, `total`, `n_words` are per-clip lists."""
    turns = {}
    for k in range(N):
        text = reply[k] if reply else " ".join(["word"] * n_words[k])
        n_pieces = pieces[k] if pieces else 1
        turns[f"clip-{k:03d}"] = {
            "cohort": ("clean", "accented", "disfluent", "slow")[k % 4],
            "transcript": "heard", "reply_text": text,
            "total_ms": float(total[k]), "first_audio_ms": float(first[k]),
            "first_sound_ms": float(first[k]) + 320.0, "reply_s": 3.0 + 0.4 * (n_pieces - 1),
            "stall_ms": float(stall[k]) if stall else 0.0,
            "chunks": [[float(first[k]), 24_000]] * n_pieces}
    return {"rung": rung, "commit": "abc1234", "machine": "test", "ran": "2026-09-18", "turns": turns}


def pair(stream_first_factor=0.6, stream_total_factor=1.05, **stream_kwargs):
    """A batch rung whose time is set by reply length, and a streaming rung whose
    first audio is not: about 300 ms whatever the reply."""
    rng = np.random.default_rng(1)
    n_words = rng.integers(5, 40, N)
    total = 250 + 25.0 * n_words + rng.normal(0, 20, N)
    batch = run("r0-gpu", total, total, n_words)
    first = stream_first_factor * np.median(total) + rng.normal(0, 25, N)
    stream = run("r1-gpu", first, stream_total_factor * total, n_words,
                 pieces=[1 + w // 12 for w in n_words], **stream_kwargs)
    return batch, stream


# --------------------------------------------------------------------------- #
# The statistics
# --------------------------------------------------------------------------- #


def test_a_percentile_ratio_and_its_interval():
    idx = resamples(N, 2000)
    a = np.linspace(60, 140, N)
    est = percentile_ratio(a, 2 * a, 50, idx)
    assert est.value == pytest.approx(0.5)
    assert est.lo == pytest.approx(0.5) and est.hi == pytest.approx(0.5)   # paired: the clip cancels


def test_the_bootstrap_is_paired():
    """Two rungs that differ by 10% on every clip, on clips that vary 10-fold.
    Unpaired, the interval would be wide; paired, it is a point."""
    rng = np.random.default_rng(0)
    b = rng.uniform(100, 1000, N)
    est = percentile_ratio(0.9 * b, b, 90, resamples(N, 2000))
    assert est.hi - est.lo < 1e-9


def test_correlation_matches_numpy_and_brackets_it():
    rng = np.random.default_rng(0)
    x = rng.normal(size=N)
    y = x + rng.normal(size=N)
    est = correlation(x, y, resamples(N, 2000))
    assert est.value == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert est.lo < est.value < est.hi


def test_resamples_are_reproducible():
    assert np.array_equal(resamples(50, 10), resamples(50, 10))


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


def test_a_streaming_rung_that_delivers_is_supported():
    result = compare(*pair(), n_boot=2000)
    assert result["valid"] and all(ok for ok, _ in result["checks"].values())
    assert result["verdict"].startswith("SUPPORTED")


def test_too_small_a_gain_fails_s1_by_its_upper_bound():
    """A true ratio of 0.74 is under the 0.75 bar, but its interval is not."""
    result = compare(*pair(stream_first_factor=0.74), n_boot=2000)
    ok, detail = result["checks"]["S1 median"]
    assert not ok and "0.75" in detail
    assert result["verdict"].startswith("NOT SUPPORTED") and "S1" in result["verdict"]


def test_first_audio_that_still_tracks_reply_length_fails_s3():
    batch, _ = pair()
    n_words = [len(t["reply_text"].split()) for t in batch["turns"].values()]
    first = [0.5 * t["total_ms"] for t in batch["turns"].values()]      # faster, but still reply-bound
    stream = run("r1-gpu", first, [t["total_ms"] for t in batch["turns"].values()], n_words)
    result = compare(batch, stream, n_boot=2000)
    assert result["checks"]["S1 median"][0] and not result["checks"]["S3 reply length"][0]


def test_a_slower_turn_fails_the_speed_guardrail():
    result = compare(*pair(stream_total_factor=1.3), n_boot=2000)
    assert not result["checks"]["S4 speed guardrail"][0]


@pytest.mark.parametrize("stalls, passes", [
    ([50.0] * 20 + [0.0] * 180, True),          # 20 short stalls: allowed
    ([50.0] * 21 + [0.0] * 179, False),         # 21: too many turns stall
    ([250.0] * 10 + [0.0] * 190, True),         # 10 long stalls: allowed
    ([250.0] * 11 + [0.0] * 189, False),        # 11: too many long ones
])
def test_the_stall_guardrail_counts_turns(stalls, passes):
    result = compare(*pair(stall=stalls), n_boot=500)
    assert result["checks"]["S5 stall guardrail"][0] is passes


def test_different_words_make_it_not_a_valid_test():
    batch, stream = pair()
    for k, turn in enumerate(stream["turns"].values()):
        if k < 11:                                   # 189 of 200 the same: one short
            turn["reply_text"] += " extra"
    result = compare(batch, stream, n_boot=500)
    assert not result["valid"] and result["verdict"].startswith("NOT A VALID TEST")
    assert "189 of 200" in result["v3"]


def test_whitespace_does_not_count_as_different_words():
    batch, stream = pair()
    for turn in stream["turns"].values():
        turn["reply_text"] = "  " + turn["reply_text"].replace(" ", "  ")
    assert compare(batch, stream, n_boot=500)["describe"]["same_words"] == N


def test_runs_over_different_clips_are_refused():
    batch, stream = pair()
    stream["turns"]["clip-999"] = stream["turns"].pop("clip-000")
    with pytest.raises(ValueError, match="same clips"):
        compare(batch, stream, n_boot=500)


def test_a_turn_with_no_reply_audio_is_refused_not_dropped():
    batch, stream = pair()
    stream["turns"]["clip-007"]["first_audio_ms"] = None
    with pytest.raises(ValueError, match="clip-007"):
        compare(batch, stream, n_boot=500)


def test_the_report_states_every_check_and_the_verdict():
    batch, stream = pair()
    text = format_report(batch, stream, compare(batch, stream, n_boot=500))
    for needle in ("V3 same words", "S1 median", "S2 tail", "S3 reply length",
                   "S4 speed guardrail", "S5 stall guardrail", "SUPPORTED",
                   "r1-gpu", "r0-gpu", "abc1234", "pieces per reply", "extra reply audio per seam"):
        assert needle in text, needle
    assert str(Estimate(0.7, 0.65, 0.74)) == "0.700 [95% CI 0.650, 0.740]"