"""
Tests for s2slab.pr017 -- PR-017's run and analysis -- with the tiny Talker and fake DiT of test_r4sized, and
hand-made run files, shaped like M37's development turns, whose numbers the criteria are checked against.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2slab.pr017 import (P50_MAX, REGISTRATION, TIMED, criteria, eval_clips, format_pr017, run, save)
from s2slab.r4 import OUT_SR
from s2slab.r4stream import SAMPLES_PER_CODE, START_BUFFER_S, FastOmni
from s2slab.r4streamdev import buffered
from s2slab.r4timed import MARGIN_S, simulate

FIRST, BLOCK = 23 * SAMPLES_PER_CODE, 96 * SAMPLES_PER_CODE


def test_the_eval_clips_are_refused_without_the_registration():
    with pytest.raises(PermissionError, match="PR-017"):
        eval_clips()
    with pytest.raises(PermissionError):
        eval_clips(confirm="PR-016")


def test_the_margin_is_development_s_and_the_rung_is_registered():
    from s2slab.rungs import RUNGS
    assert MARGIN_S == 0.05 and RUNGS[TIMED] == "s2slab.r4timed:TimedStreamingSpeechToSpeech"


def test_the_run_answers_with_the_timed_rung_and_keeps_the_buffers_turn_beside_it():
    pytest.importorskip("transformers")
    from test_r4sized import TinyBase, same
    from test_r4stream import FakeASR
    clips = {f"c{i}": np.full(16_000, float(i), dtype=np.float32) for i in range(3)}
    cohorts = {"c0": "clean", "c1": "accented", "c2": "slow"}
    blob = run(clips, 16_000, cohorts, asr=FakeASR(), base=TinyBase(), make_fast=lambda b: FastOmni(b, compile_fn=same))
    assert blob["rung"] == TIMED and blob["setting"]["margin_s"] == MARGIN_S and blob["setting"]["buffer_s"] == START_BUFFER_S
    for c, t in blob["turns"].items():
        assert t["cohort"] == cohorts[c] and t["planned_ms"] is not None and not t["cut"]
        assert sum(n for _, n in t["chunks"]) == sum(n for _, n in t["made"])
        assert set(t["buffer"]) == {"first_audio_ms", "first_sound_ms", "stall_ms"}


def turn(i, gain=True, stall=0.0, cut=False, per=6.5, decode=525.0, window=513.5):
    """One eval turn as the timed rung would record it: codes, pieces as made (the second window's decode a
    little over the load's, as in development), the live first sound as the plan makes it."""
    codes, made, t = [], [], 1000.0 + 37 * (i % 20)                  # the Thinker's time: 1.0 to 1.7 s
    for _ in range(36):
        codes.append(t)
        t += per
    t = codes[-1] + 251.0
    made.append([t, FIRST])
    for _ in range(3):
        for _ in range(96):
            t += per
            codes.append(t)
        t += decode
        made.append([t, BLOCK])
    lead = 200.0
    live = simulate(made, codes, lead, MARGIN_S, window, 96)
    buf = buffered(made, OUT_SR, START_BUFFER_S, lead)
    return {"first_sound_ms": live["first_sound_ms"] if gain else buf["first_sound_ms"], "first_audio_ms": live["start_ms"],
            "planned_ms": live["start_ms"], "stall_ms": stall, "total_ms": t - 900, "reply_s": 6.0, "words": 20,
            "word_errors": 0, "cohort": ("clean", "accented", "disfluent", "slow")[i % 4], "places": 2048, "cut": cut,
            "leading_silence_ms": lead, "chunks": made, "made": made, "code_ms": codes, "decode_ms": [251.0, decode, decode, decode],
            "buffer": buf}


def made_run(n=200, **kw):
    ids = [f"c{i:03d}" for i in range(n)]
    special = kw.pop("special", {})
    return {"clips": ids, "rung": TIMED, "commit": "abc1234", "machine": "Modal NVIDIA L40S", "versions": "torch",
            "first": 24, "block": 96, "left": 24, "right": 12, "window_ms": {"36": 251.0, "132": 513.5},
            "turns": {c: turn(i, **special.get(i, kw)) for i, c in enumerate(ids)},
            "setting": {"first_codes": 24, "block_codes": 96, "margin_s": 0.05, "buffer_s": 0.5, "sizes": [2048, 6144],
                        "room_codes": 1024}}


def test_the_criteria_hold_on_a_run_like_development():
    blob = made_run()
    r = criteria(blob, n_boot=300)
    assert r["valid"] and r["verdict"] == "SUPPORTED: all three criteria hold"
    d = r["describe"]
    assert d["buffer_stalled"] == 0 and d["simulated_off_ms"][1] == pytest.approx(0) and d["slack_ms"][1] > 0
    text = format_pr017(blob, r)
    assert "S1 median            PASS" in text and "2048 places on 200 turns" in text and "stalls, timed: 0 turns" in text


def test_no_gain_fails_s1_and_stalls_fail_s3():
    r = criteria(made_run(gain=False), n_boot=300)
    assert r["valid"] and not r["checks"]["S1 median"][0] and r["verdict"].startswith("NOT SUPPORTED")
    assert f"<= {P50_MAX}" in r["checks"]["S1 median"][1]
    r = criteria(made_run(special={1: {"stall": 30.0}, 2: {"stall": 40.0}}), n_boot=300)
    assert r["checks"]["S3 stalls"][0]                                     # two short stalls are allowed
    r = criteria(made_run(special={1: {"stall": 150.0}}), n_boot=300)
    assert not r["checks"]["S3 stalls"][0]                                 # one over 100 ms is not
    r = criteria(made_run(special={i: {"stall": 10.0} for i in range(3)}), n_boot=300)
    assert not r["checks"]["S3 stalls"][0]                                 # nor three of any length


def test_cut_replies_or_other_audio_make_it_not_valid():
    r = criteria(made_run(special={i: {"cut": True} for i in range(3)}), n_boot=300)
    assert not r["valid"] and "V2" in r["verdict"]
    blob = made_run()
    blob["turns"]["c004"]["chunks"] = blob["turns"]["c004"]["made"][:-1]
    assert "V1" in criteria(blob, n_boot=300)["verdict"]


def test_a_turn_with_no_first_sound_is_a_deviation_not_a_drop(tmp_path):
    blob = made_run()
    blob["turns"]["c005"]["first_sound_ms"] = None
    with pytest.raises(ValueError, match="drops no clip"):
        criteria(blob, n_boot=10)
    path = save(made_run(n=4), tmp_path / "pr017.json")
    with pytest.raises(Exception):
        save(made_run(n=4), path)                                             # never overwritten
    assert REGISTRATION == "PR-017"
