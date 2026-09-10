"""
Tests for s2slab.timing.

Most assertions here feed in durations directly rather than sleeping, so the
suite is fast and deterministic. Only the `stage()` tests need a real clock,
and those assert lower bounds -- a sleep can overrun but never finish early.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from s2slab.timing import (LatencySummary, format_table, per_stage_ms, stage,
                           summarize, summarize_stages)


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #


def test_percentiles_of_a_known_ramp():
    s = summarize(range(1, 101))          # 1..100
    assert s.n == 100
    assert s.p50 == pytest.approx(50.5)
    assert s.p90 == pytest.approx(90.1)


def test_empty_input_is_nan_not_zero():
    """No samples is not a measurement of zero latency."""
    s = summarize([])
    assert s.n == 0
    assert np.isnan(s.p50) and np.isnan(s.p90) and np.isnan(s.p99)


def test_nan_inputs_are_dropped_but_still_reduce_n():
    """A failed turn must not poison every percentile -- but it must not be
    hidden either, or a benchmark with 40% failures looks like a clean run."""
    s = summarize([1.0, 2.0, float("nan"), 3.0, float("inf")])
    assert s.n == 3
    assert s.p50 == pytest.approx(2.0)


def test_a_single_sample_still_reports_something():
    s = summarize([42.0])
    assert s.n == 1
    assert s.p50 == pytest.approx(42.0)


# --------------------------------------------------------------------------- #
# Sample-size honesty -- the point of this module
# --------------------------------------------------------------------------- #


def test_small_samples_cannot_support_high_percentiles():
    s = summarize([1.0] * 30)
    assert "p99" in s.unsupported
    assert "p90" in s.unsupported
    assert "p50" not in s.unsupported


def test_large_samples_support_everything():
    assert summarize([1.0] * 1000).unsupported == []


def test_p50_needs_only_a_handful():
    assert "p50" in summarize([1.0] * 19).unsupported
    assert "p50" not in summarize([1.0] * 20).unsupported


def test_the_underpowered_p99_is_a_real_distortion():
    """Not a pedantic warning -- the number is genuinely wrong.

    With 30 samples numpy places p99 at index 0.99*29 = 28.71, i.e. 71% of the
    way from the second-largest sample to the largest. It describes one
    outlier, not the system. With enough samples the same shape of data reports
    the true value.
    """
    few = summarize([1.0] * 29 + [900.0])
    many = summarize([1.0] * 1999 + [900.0])
    assert few.p99 > 500        # nonsense, driven entirely by the outlier
    assert many.p99 == pytest.approx(1.0)
    assert "p99" in few.unsupported
    assert "p99" not in many.unsupported


def test_str_shows_the_sample_count():
    assert "n=30" in str(summarize([1.0] * 30))
    assert "unsupported" in str(summarize([1.0] * 30))
    assert "unsupported" not in str(summarize([1.0] * 1000))


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #


def test_stage_records_name_and_a_plausible_duration():
    sink: list = []
    with stage("asr", sink):
        time.sleep(0.02)
    assert len(sink) == 1
    name, t0, t1 = sink[0]
    assert name == "asr"
    assert (t1 - t0) * 1000 >= 19.0      # a sleep may overrun, never undershoot


def test_stage_records_even_when_the_body_raises():
    """A crashed stage with no timing is invisible in the report -- the worst
    failure mode there is for a latency investigation."""
    sink: list = []
    with pytest.raises(ValueError):
        with stage("llm", sink):
            raise ValueError("boom")
    assert len(sink) == 1
    assert sink[0][0] == "llm"


def test_one_sink_threads_through_many_stages_in_order():
    sink: list = []
    for name in ("vad", "asr", "llm", "tts"):
        with stage(name, sink):
            pass
    assert [row[0] for row in sink] == ["vad", "asr", "llm", "tts"]


# --------------------------------------------------------------------------- #
# per_stage_ms / summarize_stages
# --------------------------------------------------------------------------- #


def test_repeated_stage_names_are_summed_not_overwritten():
    """Three TTS chunks in one turn cost the sum of three chunks. Overwriting
    would report only the last and understate the stage."""
    sink = [("tts", 0.0, 0.010), ("tts", 0.010, 0.020), ("tts", 0.020, 0.035)]
    assert per_stage_ms(sink)["tts"] == pytest.approx(35.0)


def test_per_stage_ms_converts_seconds_to_milliseconds():
    assert per_stage_ms([("asr", 1.0, 1.25)])["asr"] == pytest.approx(250.0)


def test_absent_stages_are_not_diluted_by_zeros():
    """A conditionally-run stage reports the sample size it actually has."""
    sinks = [
        [("asr", 0.0, 0.1), ("affect", 0.1, 0.2)],
        [("asr", 0.0, 0.1)],
        [("asr", 0.0, 0.1)],
    ]
    out = summarize_stages(sinks)
    assert out["asr"].n == 3
    assert out["affect"].n == 1
    assert out["affect"].p50 == pytest.approx(100.0)


def test_summarize_stages_returns_one_entry_per_distinct_stage():
    sinks = [[("a", 0.0, 0.1), ("b", 0.1, 0.3)] for _ in range(5)]
    out = summarize_stages(sinks)
    assert set(out) == {"a", "b"}
    assert out["b"].p50 == pytest.approx(200.0)


# --------------------------------------------------------------------------- #
# format_table
# --------------------------------------------------------------------------- #


def test_table_hides_percentiles_it_cannot_support():
    """The table refuses to print a number you have not earned."""
    table = format_table({"asr": summarize([10.0] * 30)})
    assert "—" in table
    assert "| 30 |" in table


def test_table_shows_percentiles_when_supported():
    table = format_table({"asr": summarize([10.0] * 1000)})
    assert "—" not in table


def test_table_sorts_slowest_first():
    """The row you should optimize is always at the top."""
    table = format_table({
        "fast": summarize([1.0] * 30),
        "slow": summarize([500.0] * 30),
        "mid": summarize([50.0] * 30),
    })
    body = [ln for ln in table.splitlines()
            if ln.startswith("| ") and "p50 ms" not in ln]
    assert [ln.split("|")[1].strip() for ln in body] == ["slow", "mid", "fast"]


def test_table_includes_a_title_when_given():
    assert "### Cascade" in format_table({"a": summarize([1.0])}, "Cascade")
