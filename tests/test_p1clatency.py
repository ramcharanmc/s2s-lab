"""
Tests for s2slab.p1clatency (M45): P1c's latency criterion, with a fake extractor, tagger and LLM whose times are set,
so the parts, the floor on c, the rule and the parallel bound can be checked without openSMILE or a model.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from s2slab import p1clatency
from s2slab.affect import Tag

SR = 16_000


def sleeper(ms: float):
    def f(*_):
        time.sleep(ms / 1000)
        return np.zeros(88)
    return f


class FakeTagger:
    def tag(self, x):
        return Tag("FEA", "fearful", 0.9)


class FakePrefill:
    """A prefill whose time grows with the prompt's length: `per_char` ms a character."""

    def __init__(self, per_char: float):
        self.per_char, self.calls = per_char, []

    def __call__(self, text):
        self.calls.append(text)
        time.sleep(self.per_char * len(text) / 1000)


def items(n=3):
    return [(f"clean-{i:02d}", "clean" if i % 2 else "accented", np.zeros(SR)) for i in range(n)]


def test_the_three_parts_their_sum_and_the_longer_prompt():
    pre = FakePrefill(0.0)
    t = p1clatency.time_clip(np.zeros(SR), SR, "hello there", sleeper(8), FakeTagger(), pre, repeats=3)
    assert t["word"] == "fearful" and 7 < t["a_ms"] < 30 and t["b_ms"] < 1
    assert t["added_ms"] == pytest.approx(t["a_ms"] + t["b_ms"] + t["c_ms"], abs=0.01)
    assert pre.calls.count("hello there") == 3 and pre.calls.count("hello there Their tone of voice was fearful.") == 3
    assert pre.calls[0] != pre.calls[2] or pre.calls[1] != pre.calls[3]      # the order alternates
    assert p1clatency.tagged("I am fine.", "sad") == "I am fine. Their tone of voice was sad."


def test_a_longer_prompt_costs_its_time_and_a_negative_difference_is_floored():
    t = p1clatency.time_clip(np.zeros(SR), SR, "hi", sleeper(0), FakeTagger(), FakePrefill(0.5), repeats=3)
    assert t["c_ms"] > 10 and t["c_with_ms"] > t["c_without_ms"]           # 33 characters more, 0.5 ms each

    class Backwards:                                                         # the tagged prompt runs faster: noise
        def __call__(self, text):
            time.sleep(0.010 if "tone" not in text else 0.0)
    t = p1clatency.time_clip(np.zeros(SR), SR, "hi", sleeper(0), FakeTagger(), Backwards(), repeats=3)
    assert t["c_with_ms"] < t["c_without_ms"] and t["c_ms"] == 0.0


def test_the_run_warms_up_and_the_rule_and_the_parallel_bound():
    ids = items(4)
    transcripts = {cid: "a sentence" for cid, _, _ in ids}
    asr = {cid: 5.0 for cid, _, _ in ids}
    pre = FakePrefill(0.0)
    blob = p1clatency.run(ids, transcripts, asr, sleeper(12), FakeTagger(), pre, sr=SR, repeats=1)
    assert len(pre.calls) == 2 + 2 * 4                                       # the warm-up's two, then two a clip
    t = blob["turns"]["clean-01"]
    assert t["duration_s"] == 1.0 and t["asr_ms"] == 5.0
    assert t["parallel_ms"] == pytest.approx(max(0, t["a_ms"] - 5) + t["b_ms"] + t["c_ms"], abs=0.01)
    assert blob["summary"]["added_ms"]["n"] == 4 and blob["bar_ms"] == 30.0
    assert p1clatency.gate(blob)["met"]                                      # about 12 ms: under the bar
    text = p1clatency.format_report({**blob, "machine": "test"})
    assert "MET: the serial sum's p90" in text and "parallel bound" in text and "by cohort" in text

    slow = p1clatency.run(ids, transcripts, {}, sleeper(35), FakeTagger(), pre, sr=SR, repeats=1)
    assert not p1clatency.gate(slow)["met"] and "NOT MET" in p1clatency.format_report(slow)
    assert slow["turns"]["clean-00"]["parallel_ms"] is None and slow["summary"]["parallel_ms"]["n"] == 0


def test_it_is_recorded_once(monkeypatch, tmp_path):
    path = tmp_path / "p1c-latency.json"
    path.write_text("{}")
    monkeypatch.setattr(p1clatency, "RUN", path)
    with pytest.raises(SystemExit, match="recorded once"):
        p1clatency.main([])
