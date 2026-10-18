"""
M38, step 2: R1 with its first piece's leading silence trimmed (s2slab.r1's FirstTrimmed, r1-trim, r1-gpu-trim) and
its development run (s2slab.r1trim), with R1's own fakes: only the first piece is cut, what was cut is recorded and
was silent, R1 as it is comes back from the same turn's pieces, and the rule.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from test_r1 import ONE_SECOND, FakeASR, FakeStreamingLLM

from s2slab import evalrun, r1trim
from s2slab.r1 import StreamingCascade, TrimmedStreamingCascade
from s2slab.rungs import get

SR = 24_000
REPLY = "Your appointment is on Tuesday, and I have noted the change. Anything else?"


class PaddedTTS:
    """Kokoro's habit: 320 ms of silence, then half a second of tone per word."""

    SR = SR

    def load(self):
        pass

    def speak(self, text):
        tone = 0.3 * np.sin(2 * np.pi * 200 * np.arange(len(text.split()) * SR // 2) / SR)
        return np.concatenate([np.zeros(int(0.32 * SR)), tone]).astype(np.float32)


def test_both_rungs_are_registered_and_build_without_a_model_library():
    assert get("r1-trim").rung_id == "r1-trim" and get("r1-gpu-trim").rung_id == "r1-gpu-trim"
    assert get("r1-gpu-trim").tts.device == "cuda" and get("r1-gpu-trim").asr.model_name == get("r1-gpu").asr.model_name


def test_only_the_first_piece_is_cut_and_what_was_cut_was_silent():
    plain = StreamingCascade(FakeASR(), FakeStreamingLLM(REPLY), PaddedTTS())
    cut = TrimmedStreamingCascade(FakeASR(), FakeStreamingLLM(REPLY), PaddedTTS())
    a, b = plain.respond(ONE_SECOND, 16_000), cut.respond(ONE_SECOND, 16_000)
    assert len(a.chunks) == len(b.chunks) == 3 and a.reply_text == b.reply_text
    assert b.chunks[0][1] == a.chunks[0][1] - int(0.30 * SR) and [n for _, n in b.chunks[1:]] == [n for _, n in a.chunks[1:]]
    assert a.leading_silence_ms == 320.0 and b.leading_silence_ms == 20.0
    extra = cut.turn_extra()
    assert extra["trim_ms"] == pytest.approx(300.0) and extra["cut_peak_db"] < -100      # digital silence
    assert not hasattr(plain, "turn_extra")
    cut.respond(ONE_SECOND, 16_000)
    assert cut.turn_extra()["trim_ms"] == pytest.approx(300.0)                           # reset each turn, then set


def test_r1_as_it_is_comes_back_from_the_same_turns_pieces():
    cut = TrimmedStreamingCascade(FakeASR(), FakeStreamingLLM(REPLY), PaddedTTS())
    t = cut.respond(ONE_SECOND, 16_000)
    row = {"first_sound_ms": t.first_sound_ms, "stall_ms": t.stall_ms, "leading_silence_ms": t.leading_silence_ms,
           "chunks": [list(c) for c in t.chunks], **cut.turn_extra()}
    back = r1trim.as_is(row)
    plain = StreamingCascade(FakeASR(), FakeStreamingLLM(REPLY), PaddedTTS()).respond(ONE_SECOND, 16_000)
    assert back["first_sound_ms"] == pytest.approx(t.first_audio_ms + 320.0)
    assert back["first_sound_ms"] - t.first_sound_ms == pytest.approx(300.0) and plain.leading_silence_ms == 320.0


def turn(first_audio, gap_ms, trim_ms=300.0, lead=20.0):
    """A turn as r1-gpu-trim would record it: a first piece of 1 s (after its cut), the second gap_ms after it."""
    return {"first_audio_ms": first_audio, "first_sound_ms": first_audio + lead, "leading_silence_ms": lead,
            "stall_ms": max(0.0, first_audio + gap_ms - (first_audio + 1000.0)), "trim_ms": trim_ms,
            "chunks": [[first_audio, SR], [first_audio + gap_ms, SR]], "reply_text": "x", "total_ms": 3000.0}


def test_the_rule_goes_on_when_the_gain_holds_and_stops_when_stalls_eat_it():
    calm = {"rung": "r1-gpu-trim", "clips": ["a", "b", "c"], "sr": SR,
            "turns": {"a": turn(500.0, 800.0), "b": turn(600.0, 900.0), "c": turn(550.0, 950.0)}}
    g = r1trim.gate(calm)
    assert g["go"] and g["stalled_trimmed"] == 0 and g["stalled_as_is"] == 0 and g["trim_p50"] == 300.0
    assert g["ratio_p50"] == pytest.approx((550.0 + 20.0) / (550.0 + 320.0), abs=1e-3)
    tight = {**calm, "turns": {k: turn(500.0, 1150.0) for k in ("a", "b", "c")}}         # the second piece late for the cut
    g = r1trim.gate(tight)
    assert not g["go"] and g["stalled_trimmed"] == 3 and g["stalled_as_is"] == 0 and "needs <= 2" in g["why"]
    assert "STOP" in r1trim.format_dev(tight) and "GO: register PR-018" in r1trim.format_dev(calm)


def test_a_rung_can_add_to_its_turns_record_in_the_eval_run(tmp_path, monkeypatch):
    (tmp_path / evalrun.MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / evalrun.MANIFEST).write_text('{"built": "test", "seed": 0, "items": []}', encoding="utf-8")
    items = [SimpleNamespace(id="x", cohort="clean", audio=ONE_SECOND)]
    monkeypatch.setattr(evalrun, "load", lambda root, cohorts=None: items)
    rung = TrimmedStreamingCascade(FakeASR(), FakeStreamingLLM(REPLY), PaddedTTS())
    out = evalrun.run(rung, root=tmp_path)
    assert out["turns"]["x"]["trim_ms"] == pytest.approx(300.0) and out["turns"]["x"]["cut_peak_db"] < -100
