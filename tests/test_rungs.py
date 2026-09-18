"""
Tests for s2slab.rungs -- the interface every rung on the ladder implements.

No models here. Rungs built from sleeps stand in for real stages, so each test
knows how long every stage took. Sleeps only ever run long, never short, so
every timing assertion is a lower bound plus a generous ceiling.
"""

from __future__ import annotations

import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pytest

from s2slab.probes import PROBE_ROOT, read_script
from s2slab.rungs import (ENDPOINT_WAIT_S, RUNGS, Reply, Rung, StubRung, get, report,
                          run_turns, silence_before_speech_ms, summarize_turns)
from s2slab.timing import stage

SR = 16_000
OUT_SR = 24_000
ONE_SECOND = np.zeros(SR, dtype=np.float32)


class SleepyRung(Rung):
    """Two stages of known length, and a load step that can be slow on purpose."""

    rung_id = "sleepy"

    def __init__(self, asr_s=0.03, llm_s=0.02, load_s=0.0):
        self.asr_s, self.llm_s, self.load_s = asr_s, llm_s, load_s
        self.loads = 0

    def _load(self):
        self.loads += 1
        time.sleep(self.load_s)

    def _respond(self, audio, sr, sink):
        with stage("asr", sink):
            time.sleep(self.asr_s)
        with stage("llm", sink):
            time.sleep(self.llm_s)
        return Reply(audio, sr, "heard", "said")

class StreamyRung(Rung):
    """Hands its reply out in two half-second pieces: the first after first_s,
    the second rest_s later -- the shape of R1."""

    rung_id = "streamy"

    def __init__(self, first_s=0.02, rest_s=0.03):
        self.first_s, self.rest_s = first_s, rest_s

    def _respond(self, audio, sr, sink):
        with stage("tts", sink):
            time.sleep(self.first_s)
        self.emit(_tone(0.5))
        with stage("tts", sink):
            time.sleep(self.rest_s)
        self.emit(_tone(0.5))
        return Reply(None, OUT_SR, "heard", "said")


def _tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * OUT_SR)) / OUT_SR
    return (amp * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)


def _silence_then_tone(lead_s: float) -> np.ndarray:
    return np.concatenate([np.zeros(int(lead_s * OUT_SR), np.float32), _tone(1.0)])


# --------------------------------------------------------------------------- #
# The stub: the harness on a pipeline that does nothing
# --------------------------------------------------------------------------- #


def test_stub_runs_three_stages_in_order():
    turn = StubRung().respond(ONE_SECOND, SR)
    assert list(turn.stages_ms) == ["asr", "llm", "tts"]


def test_stub_costs_next_to_nothing():
    """Zeros are the correct answer. The ceiling is generous for a busy machine."""
    turn = StubRung().respond(ONE_SECOND, SR)
    assert turn.total_ms < 50
    assert all(ms >= 0 for ms in turn.stages_ms.values())


def test_stub_replies_with_silence_at_its_own_rate():
    turn = StubRung().respond(ONE_SECOND, SR)
    assert turn.sr == OUT_SR and turn.audio.dtype == np.float32
    assert turn.reply_s == pytest.approx(0.5)
    assert math.isnan(turn.leading_silence_ms)      # it never speaks


# --------------------------------------------------------------------------- #
# Timing: every millisecond lands in a stage, or visibly in none
# --------------------------------------------------------------------------- #


def test_each_stage_is_timed_on_its_own():
    turn = SleepyRung(asr_s=0.03, llm_s=0.02).respond(ONE_SECOND, SR)
    assert 30 <= turn.stages_ms["asr"] < 200
    assert 20 <= turn.stages_ms["llm"] < 200


def test_the_stages_add_up_to_the_total():
    turn = SleepyRung().respond(ONE_SECOND, SR)
    assert turn.total_ms >= sum(turn.stages_ms.values())
    assert turn.untimed_ms < 5        # nothing outside the stages but two function calls


def test_work_outside_every_stage_shows_up_as_untimed():
    """The failure this guards: a slow step nobody wrapped in stage() would
    otherwise vanish from the table while the user still waits for it."""

    class Leaky(SleepyRung):
        def _respond(self, audio, sr, sink):
            time.sleep(0.03)                        # work no stage claims
            return super()._respond(audio, sr, sink)

    assert Leaky().respond(ONE_SECOND, SR).untimed_ms >= 30


# --------------------------------------------------------------------------- #
# Loading is not latency
# --------------------------------------------------------------------------- #


def test_loading_happens_before_the_clock_starts():
    rung = SleepyRung(load_s=0.2)
    turn = rung.respond(ONE_SECOND, SR)    # nobody called load(): respond does, first
    assert rung.loads == 1
    assert turn.total_ms < 150             # ~50 ms of stages, not 250


def test_load_reports_its_time_and_runs_once():
    rung = SleepyRung(load_s=0.05)
    assert rung.load() >= 50
    assert rung.load() == 0.0
    rung.respond(ONE_SECOND, SR)
    assert rung.loads == 1


def test_respond_refuses_stereo():
    with pytest.raises(ValueError, match="mono"):
        StubRung().respond(np.zeros((SR, 2), dtype=np.float32), SR)


# --------------------------------------------------------------------------- #
# Leading silence: latency the user hears but no stage timer sees
# --------------------------------------------------------------------------- #


def test_leading_silence_finds_the_first_sound():
    assert silence_before_speech_ms(_silence_then_tone(0.3), OUT_SR) == pytest.approx(300, abs=20)


def test_leading_silence_ignores_a_faint_noise_floor():
    """Some engines dither their padding. 1e-4 is about 66 dB below the tone."""
    rng = np.random.default_rng(0)
    reply = _silence_then_tone(0.3)
    reply[: int(0.3 * OUT_SR)] = 1e-4 * rng.standard_normal(int(0.3 * OUT_SR))
    assert silence_before_speech_ms(reply, OUT_SR) == pytest.approx(300, abs=20)


def test_leading_silence_does_not_depend_on_output_level():
    reply = _silence_then_tone(0.3)
    assert silence_before_speech_ms(reply * 0.01, OUT_SR) == silence_before_speech_ms(reply, OUT_SR)


def test_a_reply_that_never_speaks_has_no_leading_silence():
    assert math.isnan(silence_before_speech_ms(np.zeros(OUT_SR, np.float32), OUT_SR))
    assert math.isnan(silence_before_speech_ms(np.zeros(0, np.float32), OUT_SR))

# --------------------------------------------------------------------------- #
# First audio and first sound: when the caller starts hearing the reply
# --------------------------------------------------------------------------- #


def test_a_batch_reply_first_exists_when_the_turn_ends():
    """R0's case, with no change to R0: returned audio existed only at the end."""
    turn = SleepyRung().respond(ONE_SECOND, SR)
    assert turn.first_audio_ms == turn.total_ms
    assert turn.chunks == [(turn.total_ms, SR)]


def test_a_streamed_reply_first_exists_when_its_first_piece_does():
    turn = StreamyRung(first_s=0.02, rest_s=0.08).respond(ONE_SECOND, SR)
    (first_ms, n1), (second_ms, n2) = turn.chunks
    assert 20 <= turn.first_audio_ms == first_ms < second_ms <= turn.total_ms
    assert second_ms - first_ms >= 80
    assert n1 == n2 == OUT_SR // 2
    assert len(turn.audio) == OUT_SR and turn.sr == OUT_SR    # the pieces, joined in order


def test_first_sound_of_a_batch_reply_adds_its_leading_silence():
    class Padded(SleepyRung):
        def _respond(self, audio, sr, sink):
            super()._respond(audio, sr, sink)
            return Reply(_silence_then_tone(0.3), OUT_SR, "heard", "said")

    turn = Padded().respond(ONE_SECOND, SR)
    assert turn.first_sound_ms == pytest.approx(turn.total_ms + 300, abs=20)


def test_first_sound_waits_for_the_chunk_that_holds_the_first_word():
    """First piece: 0.1 s of pure padding, handed out at once. The voice is in
    the second piece, 150 ms later. Playback would reach the voice after 100 ms,
    but it cannot play what does not exist yet -- so the caller hears it when
    the second piece arrives, not at first audio + 100 ms."""
    class PadThenVoice(Rung):
        rung_id = "pad-then-voice"

        def _respond(self, audio, sr, sink):
            self.emit(np.zeros(int(0.1 * OUT_SR), np.float32))
            time.sleep(0.15)
            self.emit(_tone(0.5))
            return Reply(None, OUT_SR, "heard", "said")

    turn = PadThenVoice().respond(ONE_SECOND, SR)
    (first_ms, _), (second_ms, _) = turn.chunks
    assert turn.first_sound_ms == second_ms
    assert turn.first_sound_ms - turn.first_audio_ms >= 150

def test_a_batch_reply_never_stalls():
    assert SleepyRung().respond(ONE_SECOND, SR).stall_ms == 0.0


def test_a_piece_that_arrives_while_the_last_is_still_playing_does_not_stall():
    """Half a second of audio, and the next piece 80 ms later: playback never runs dry."""
    assert StreamyRung(rest_s=0.08).respond(ONE_SECOND, SR).stall_ms == 0.0


def test_a_piece_that_arrives_after_the_last_finished_playing_stalls():
    """c12 on the laptop: 1.6 s of audio at 2165 ms, the next piece at 4668 ms --
    the caller hears 903 ms of silence that is in no WAV file."""
    class Slow(Rung):
        rung_id = "slow-second-piece"

        def _respond(self, audio, sr, sink):
            self.emit(_tone(0.1))                   # plays for 100 ms
            time.sleep(0.25)                        # the next piece is 250 ms away
            self.emit(_tone(0.1))
            return Reply(None, OUT_SR, "heard", "said")

    turn = Slow().respond(ONE_SECOND, SR)
    (first_ms, _), (second_ms, _) = turn.chunks
    assert turn.stall_ms == pytest.approx(second_ms - first_ms - 100, abs=1)
    assert turn.stall_ms >= 150


def test_report_prints_the_playback_stall():
    assert "playback stall:" in report("stub", {"c1": ONE_SECOND}, SR)


def test_a_reply_that_never_speaks_has_no_first_sound():
    assert math.isnan(StubRung().respond(ONE_SECOND, SR).first_sound_ms)


def test_empty_pieces_are_not_first_audio():
    class LateStart(StreamyRung):
        def _respond(self, audio, sr, sink):
            self.emit(np.zeros(0, np.float32))          # nothing yet
            return super()._respond(audio, sr, sink)

    turn = LateStart(first_s=0.03).respond(ONE_SECOND, SR)
    assert len(turn.chunks) == 2 and turn.first_audio_ms >= 30


def test_a_rung_that_emits_nothing_has_no_first_audio():
    class Mute(Rung):
        rung_id = "mute"

        def _respond(self, audio, sr, sink):
            return Reply(None, OUT_SR, "heard", "")

    turn = Mute().respond(ONE_SECOND, SR)
    assert len(turn.audio) == 0 and math.isnan(turn.first_audio_ms)


def test_emitting_and_returning_audio_is_refused():
    """Two sources of the same reply: which one did the caller hear?"""
    class Both(StreamyRung):
        def _respond(self, audio, sr, sink):
            super()._respond(audio, sr, sink)
            return Reply(_tone(0.1), OUT_SR, "heard", "said")

    with pytest.raises(ValueError, match="emitted audio AND returned it"):
        Both().respond(ONE_SECOND, SR)


def test_emit_outside_a_turn_is_refused_even_after_a_turn_failed():
    class Broken(StreamyRung):
        def _respond(self, audio, sr, sink):
            self.emit(_tone(0.1))
            raise RuntimeError("model crashed")

    rung = Broken()
    with pytest.raises(RuntimeError, match="only works inside respond"):
        rung.emit(_tone(0.1))
    with pytest.raises(RuntimeError, match="model crashed"):
        rung.respond(ONE_SECOND, SR)
    with pytest.raises(RuntimeError, match="only works inside respond"):
        rung.emit(_tone(0.1))


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


def test_get_returns_a_fresh_unloaded_rung_under_its_own_id():
    a, b = get("stub"), get("stub")
    assert isinstance(a, StubRung) and a is not b
    assert a.rung_id == "stub" and not a._loaded


def test_an_unknown_rung_lists_the_known_ones():
    with pytest.raises(KeyError, match="stub"):
        get("r9")


def test_importing_the_registry_loads_no_model_library():
    """The CPU track must never pay for torch. A fresh interpreter, because
    another test in this session may already have imported one."""
    root = pathlib.Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    code = ("import sys, s2slab.rungs; "
            "print([m for m in ('torch', 'transformers', 'faster_whisper', 'kokoro') "
            "if m in sys.modules])")
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.stdout.strip() == "[]", proc.stderr
    assert set(RUNGS) >= {"stub"}


# --------------------------------------------------------------------------- #
# Running many turns
# --------------------------------------------------------------------------- #


def test_run_turns_keeps_clip_ids_and_order():
    turns = run_turns(StubRung(), {"b": ONE_SECOND, "a": ONE_SECOND[:8000]}, SR)
    assert list(turns) == ["b", "a"]


def test_summary_has_each_stage_plus_untimed_total_and_first_audio():
    """The stub replies with silence, so it has a first audio but no first sound
    -- and that row is left out rather than printed as nan."""
    turns = run_turns(StubRung(), {f"c{i}": ONE_SECOND for i in range(5)}, SR)
    summary = summarize_turns(turns)
    assert set(summary) == {"asr", "llm", "tts", "(untimed)", "total", "first audio"}
    assert all(s.n == 5 for s in summary.values())


def test_summary_includes_first_sound_once_a_reply_speaks():
    turns = run_turns(StreamyRung(), {f"c{i}": ONE_SECOND for i in range(3)}, SR)
    assert summarize_turns(turns)["first sound"].n == 3


def test_report_prints_the_table_load_time_and_leading_silence():
    text = report("stub", {"c1": ONE_SECOND}, SR)
    assert "| asr |" in text and "| total |" in text
    assert "load:" in text and "reply leading silence:" in text


@pytest.mark.skipif(not all((PROBE_ROOT / d / f"{i}.{ext}").exists()
                            for i in read_script()
                            for d, ext in (("audio", "wav"), ("labels", "txt"))),
                    reason="the probe recordings are not on this machine")
def test_probe_clips_end_one_endpoint_wait_after_the_true_end():
    from s2slab.probes import load_item
    from s2slab.rungs import probe_clips
    clips = probe_clips()
    assert len(clips) == 30
    u = load_item("f01")
    assert len(clips["f01"]) / SR == pytest.approx(u.true_end_s + ENDPOINT_WAIT_S, abs=1 / SR)