"""
Tests for s2slab.pr016 -- PR-016's run and analysis -- with the tiny Talker and fake DiT of test_r4sized,
and hand-made run files whose numbers the criteria are checked against.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_r4sized import TinyBase, same  # noqa: E402
from test_r4stream import FakeASR  # noqa: E402

from s2slab.pr016 import (P50_MAX, REGISTRATION, SIZED, STREAM, criteria, eval_clips, format_pr016, run,  # noqa: E402
                          save)
from s2slab.r4stream import START_BUFFER_S, FastOmni  # noqa: E402


def test_the_eval_clips_are_refused_without_the_registration():
    with pytest.raises(PermissionError, match="PR-016"):
        eval_clips()
    with pytest.raises(PermissionError):
        eval_clips(confirm="PR-015")


def test_the_run_answers_with_three_rungs_both_streams_holding_the_frozen_buffer():
    clips = {f"c{i}": np.full(16_000, float(i), dtype=np.float32) for i in range(3)}
    cohorts = {"c0": "clean", "c1": "accented", "c2": "slow"}
    blob = run(clips, 16_000, cohorts, asr=FakeASR(), base=TinyBase(), make_fast=lambda b: FastOmni(b, compile_fn=same))
    assert list(blob["rungs"]) == ["r4-gpu-fast", STREAM, SIZED] and "listen" not in blob
    assert all("buffers" not in r for r in blob["rungs"].values())             # the buffer is held, not computed
    assert blob["setting"]["start_buffer_s"] == START_BUFFER_S == 0.5 and blob["setting"]["sizes"] == [2048, 6144]
    for name in (STREAM, SIZED):
        assert all(t["codes_as_fast"] and t["cohort"] == cohorts[c] for c, t in blob["rungs"][name]["turns"].items())
    assert all(t["places"] == 2048 and not t["cut"] for t in blob["rungs"][SIZED]["turns"].values())


def made(n=200, sized_fs=2700.0, stream_fs=3400.0, cut=0, stall=0.0, errors=(1, 1)):
    """A run file of n clips: the sized stream's and the stream's first sound around the given values."""
    rng = np.random.default_rng(0)
    ids = [f"c{i:03d}" for i in range(n)]

    def turn(fs, total, i, extra):
        return dict({"first_sound_ms": fs + rng.normal(0, 60), "total_ms": total + rng.normal(0, 200), "stall_ms": 0.0,
                     "reply_s": 6.0, "words": 20, "word_errors": 0, "reply_text": f"reply {i}", "cohort": "clean",
                     "codes_as_fast": True}, **extra)
    rungs = {"r4-gpu-fast": {"turns": {c: turn(6000, 6000, i, {}) for i, c in enumerate(ids)}},
             STREAM: {"turns": {c: turn(stream_fs, 7700, i, {"word_errors": errors[0] if i == 0 else 0})
                                for i, c in enumerate(ids)}},
             SIZED: {"turns": {c: turn(sized_fs, 5800, i, {"places": 2048, "cut": i < cut, "stall_ms": stall if i == 1 else 0.0,
                                                           "word_errors": errors[1] if i == 0 else 0})
                               for i, c in enumerate(ids)}}}
    return {"clips": ids, "rungs": rungs, "commit": "abc1234", "machine": "Modal NVIDIA L40S", "versions": "torch",
            "setting": {"first_codes": 24, "block_codes": 96, "start_buffer_s": 0.5, "sizes": [2048, 6144], "room_codes": 1024}}


def test_the_criteria_hold_on_a_run_like_development():
    r = criteria(made(), n_boot=300)
    assert r["valid"] and r["verdict"] == "SUPPORTED: all five criteria hold"
    assert all(ok for ok, _ in r["checks"].values())
    text = format_pr016(made(), r)
    assert "S1 median            PASS" in text and "2048 places on 200 turns" in text and "stalls, the stream: 0 turns" in text


def test_a_sized_stream_no_sooner_fails_s1_and_a_cut_reply_makes_it_not_valid():
    r = criteria(made(sized_fs=3300.0), n_boot=300)
    assert r["valid"] and not r["checks"]["S1 median"][0] and r["verdict"].startswith("NOT SUPPORTED")
    assert f"<= {P50_MAX}" in r["checks"]["S1 median"][1]
    r = criteria(made(cut=3), n_boot=300)
    assert not r["valid"] and "V3" in r["verdict"]
    r = criteria(made(stall=500.0), n_boot=300)
    assert r["checks"]["S4 stall guardrail"][0] and r["describe"]["stalls"] == [500.0]      # one stall is allowed


def test_a_turn_with_no_first_sound_is_a_deviation_not_a_drop(tmp_path):
    blob = made()
    blob["rungs"][SIZED]["turns"]["c005"]["first_sound_ms"] = None
    with pytest.raises(ValueError, match="drops no clip"):
        criteria(blob, n_boot=10)
    path = save(made(), tmp_path / "pr016.json")
    with pytest.raises(Exception):
        save(made(), path)                                                    # never overwritten
    assert REGISTRATION == "PR-016"
