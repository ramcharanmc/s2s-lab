"""
Tests for s2slab.pr015 -- PR-015's run and analysis -- with the fake Omni model of test_r4stream and
hand-made run files whose numbers the criteria are checked against.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from s2slab.pr015 import (P50_MAX, REGISTRATION, RUN, STREAM, criteria, eval_clips, format_pr015, run, save)
from s2slab.r4 import OUT_SR
from s2slab.r4stream import FastOmni
from test_r4stream import FakeASR, FakeBase, FakeCache, Recorder

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_eval_clips_are_refused_without_the_registration():
    with pytest.raises(PermissionError, match="PR-015"):
        eval_clips()
    with pytest.raises(PermissionError):
        eval_clips(confirm="PR-014")


def test_the_run_answers_with_all_three_rungs_and_the_stream_as_frozen():
    clips = {f"c{i}": np.zeros(16000, dtype=np.float32) for i in range(3)}
    blob = run(clips, 16000, {c: "clean" for c in clips}, asr=FakeASR(), base=FakeBase(n_codes=300),
               make_fast=lambda b: FastOmni(b, compile_fn=Recorder(), cache_factory=FakeCache))
    assert list(blob["rungs"]) == ["r4-gpu", "r4-gpu-fast", STREAM] and "listen" not in blob
    assert blob["setting"] == {"first_codes": 24, "block_codes": 96, "start_buffer_s": 0.5}
    first = blob["rungs"][STREAM]["turns"]["c0"]["chunks"][0][1]
    assert first >= 0.5 * OUT_SR                                 # the start buffer held the first block back
    assert "buffers" not in blob["rungs"][STREAM]                # no buffer computed on top of a buffered run
    assert all(t["cohort"] == "clean" for r in blob["rungs"].values() for t in r["turns"].values())
    assert all(t["codes_as_fast"] for t in blob["rungs"][STREAM]["turns"].values())


def turn(sound, total, stall=0.0, text="sure here it is", errors=0, codes_as_fast=None):
    t = {"first_sound_ms": sound, "total_ms": total, "stall_ms": stall, "reply_s": total / 1000, "reply_text": text,
         "words": 4, "word_errors": errors, "cohort": "clean"}
    if codes_as_fast is not None:
        t["codes_as_fast"] = codes_as_fast
    return t


def blob_of(stream_sound=3000.0, stream_total=7000.0, stalls=0, text_diff=0, codes_diff=0, errors=0, n=200):
    ids = [f"c{i}" for i in range(n)]
    rng = np.random.default_rng(1)
    r4 = {i: turn(6000 + rng.normal(0, 300), 6000 + rng.normal(0, 300)) for i in ids}
    fast = {i: turn(6000.0, 6000.0) for i in ids}
    stream = {i: turn(stream_sound + rng.normal(0, 100), stream_total, stall=(500.0 if k < stalls else 0.0),
                      text=("other" if k < text_diff else "sure here it is"), errors=(1 if k < errors else 0),
                      codes_as_fast=k >= codes_diff)
              for k, i in enumerate(ids)}
    return {"clips": ids, "commit": "abc1234", "machine": "Modal L40S", "versions": "torch",
            "setting": {"first_codes": 24, "block_codes": 96, "start_buffer_s": 0.5},
            "rungs": {"r4-gpu": {"turns": r4}, "r4-gpu-fast": {"turns": fast}, STREAM: {"turns": stream}}}


def test_a_stream_that_halves_the_wait_and_keeps_up_is_supported():
    res = criteria(blob_of(), n_boot=500)
    assert res["valid"] and res["verdict"] == "SUPPORTED: all five criteria hold"
    text = format_pr015(blob_of(), res)
    assert "S1 median" in text and "PASS" in text and "r4-gpu-fast's total p50 over R4's" in text


def test_each_criterion_can_fail():
    assert not criteria(blob_of(stream_sound=6000 * P50_MAX * 1.1), n_boot=500)["checks"]["S1 median"][0]
    assert not criteria(blob_of(stream_total=9000.0), n_boot=500)["checks"]["S3 speed guardrail"][0]
    assert not criteria(blob_of(stalls=11), n_boot=500)["checks"]["S4 stall guardrail"][0]     # 11 over 200 ms
    assert criteria(blob_of(stalls=10), n_boot=500)["checks"]["S4 stall guardrail"][0]
    assert not criteria(blob_of(errors=40), n_boot=500)["checks"]["S5 words"][0]               # 40 of 800 words: +5 points


def test_a_stream_that_says_other_words_or_other_codes_is_not_a_valid_test():
    assert criteria(blob_of(text_diff=11), n_boot=500)["verdict"].startswith("NOT A VALID TEST: V1")
    assert criteria(blob_of(codes_diff=11), n_boot=500)["verdict"].startswith("NOT A VALID TEST: V2")
    assert criteria(blob_of(text_diff=10), n_boot=500)["valid"]


def test_a_silent_turn_is_not_dropped():
    b = blob_of()
    b["rungs"][STREAM]["turns"]["c3"]["first_sound_ms"] = None
    with pytest.raises(ValueError, match="drops no clip"):
        criteria(b, n_boot=100)


def test_the_run_file_is_written_once(tmp_path):
    p = save({"x": 1}, tmp_path / "pr015.json")
    with pytest.raises(FileExistsError):
        save({"x": 2}, p)
    assert RUN.parts == ("runs", "r4-gpu", "pr015.json") and REGISTRATION == "PR-015"


def test_the_l40s_app_has_the_confirmation_entry_point():
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod.pr015_run, "remote") and hasattr(mod.pr015, "info")
