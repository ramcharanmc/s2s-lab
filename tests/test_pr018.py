"""
Tests for s2slab.pr018 -- PR-018's analysis -- on hand-made run files shaped like the eval harness's for r1-gpu-trim:
the criteria against development-like turns, a trim that gains nothing, a cut that took sound, stalls, a turn with no
first sound, and a run file never overwritten.
"""

import pytest

from s2slab.pr018 import P50_MAX, REGISTRATION, criteria, format_pr018, save

SR = 24_000


def turn(i, trim_ms=300.0, gap_ms=600.0, stall=0.0, cut_db=-60.0):
    """One eval turn as r1-gpu-trim records it: a first piece of 2 s after its cut, the second gap_ms after it;
    first audio 450 to 640 ms (R1's on the eval set: about 490 at the median)."""
    first_audio = 450.0 + 19 * (i % 11)
    return {"cohort": ("clean", "accented", "disfluent", "slow")[i % 4], "first_audio_ms": first_audio,
            "first_sound_ms": first_audio + 20.0, "leading_silence_ms": 20.0, "stall_ms": stall, "trim_ms": trim_ms,
            "cut_peak_db": cut_db, "chunks": [[first_audio, 2 * SR], [first_audio + gap_ms, SR]], "total_ms": 3000.0,
            "reply_text": "x"}


def run(n=200, special=None, **kw):
    special = special or {}
    return {"rung": "r1-gpu-trim", "commit": "abc1234", "machine": "Modal Tesla T4",
            "turns": {f"c{i:03d}": turn(i, **special.get(i, kw)) for i in range(n)}}


def test_the_criteria_hold_on_turns_like_development():
    blob = run()
    r = criteria(blob, n_boot=300)
    assert r["valid"] and r["verdict"] == "SUPPORTED: all three criteria hold"
    d = r["describe"]
    assert d["first_sound"]["as_is"][0] - d["first_sound"]["trimmed"][0] == pytest.approx(300.0)
    assert d["second_piece_ahead_ms"][1] == pytest.approx(1400.0) and d["single_piece"] == 0
    text = format_pr018(blob, r)
    assert "S1 median              PASS" in text and "stalls: trimmed 0 turns, as is 0" in text


def test_no_gain_fails_s1_and_a_loud_cut_makes_it_not_valid():
    r = criteria(run(trim_ms=0.0), n_boot=300)
    assert r["valid"] and not r["checks"]["S1 median"][0] and f"<= {P50_MAX}" in r["checks"]["S1 median"][1]
    r = criteria(run(special={3: {"cut_db": -30.0}}), n_boot=300)
    assert not r["valid"] and "V1" in r["verdict"] and "199 of 200" in r["v"]["V1 the cut was silent"][1]


def test_stalls_fail_s3_when_there_are_more_than_four_or_one_is_long():
    assert criteria(run(special={i: {"stall": 50.0} for i in range(4)}), n_boot=300)["checks"]["S3 stalls"][0]
    assert not criteria(run(special={i: {"stall": 50.0} for i in range(5)}), n_boot=300)["checks"]["S3 stalls"][0]
    assert not criteria(run(special={1: {"stall": 250.0}}), n_boot=300)["checks"]["S3 stalls"][0]


def test_a_turn_with_no_first_sound_is_a_deviation_not_a_drop(tmp_path):
    blob = run()
    blob["turns"]["c007"]["first_sound_ms"] = None
    with pytest.raises(ValueError, match="drops no clip"):
        criteria(blob, n_boot=10)
    path = save(run(n=4), tmp_path / "pr018.json")
    with pytest.raises(Exception):
        save(run(n=4), path)                                                  # never overwritten
    assert REGISTRATION == "PR-018"
