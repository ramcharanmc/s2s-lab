"""
Tests for s2slab.duplex (the full-duplex driver and its turn-taking measures) and the
shape of s2slab.r5's Moshi adapter -- with a fake model whose speech is scheduled
by the test, so every gap and overlap can be counted by hand. No model, no GPU.

Stream time: a chunk emitted at step i is heard from (i + 1) * 0.08 s.
"""

from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np
import pytest

from s2slab import duplex
from s2slab.duplex import (FRAME, FRAME_S, SR, Conversation, caller_span, converse, measure, report, run_items, save,
                           voiced_runs)

IN_SR = 16_000


def clip(speech=(0.5, 2.0), length=3.0, seed=0) -> np.ndarray:
    """A caller: silence with Gaussian noise as speech between the given seconds."""
    rng = np.random.default_rng(seed)
    a = np.zeros(int(length * IN_SR), np.float32)
    s, e = int(speech[0] * IN_SR), int(speech[1] * IN_SR)
    a[s:e] = 0.2 * rng.standard_normal(e - s).astype(np.float32)
    return a


class FakeDuplex:
    """Speaks (loud noise) on the scheduled steps, says nothing otherwise; None for the first `delay` steps."""

    def __init__(self, speak=(), text=None, delay=0, seeded=True):
        self.speak, self.text, self.delay, self.seeded = set(speak), dict(text or {}), delay, seeded
        self.began, self.ended, self.frames = 0, 0, []

    def begin(self, seed):
        self.began += 1
        self.k = 0
        self.rng = np.random.default_rng(seed if self.seeded else None)

    def step(self, frame):
        self.frames.append(len(frame))
        k, self.k = self.k, self.k + 1
        if k < self.delay:
            return None, None
        amp = 0.3 if k in self.speak else 0.0
        return (amp * self.rng.standard_normal(FRAME)).astype(np.float32), self.text.get(k)

    def end(self):
        self.ended += 1


def conv_of(chunks, text=None) -> Conversation:
    return Conversation(chunks, text or [None] * len(chunks), 1.0, len(chunks) * FRAME_S)


LOUD, QUIET = np.full(FRAME, 0.3, np.float32), np.zeros(FRAME, np.float32)

# --------------------------------------------------------------------------- #
# The pieces
# --------------------------------------------------------------------------- #


def test_the_caller_span_is_the_first_and_last_voiced_moment():
    assert caller_span(clip()) == pytest.approx((0.5, 2.0))
    assert caller_span(np.zeros(IN_SR, np.float32)) is None


def test_a_run_needs_two_voiced_chunks_and_is_heard_one_frame_after_its_step():
    runs = voiced_runs(conv_of([None, None, LOUD, QUIET, LOUD, LOUD, LOUD, QUIET]))
    assert runs == [pytest.approx((0.40, 0.64))]                                # the lone step-2 chunk is a click
    assert voiced_runs(conv_of([LOUD * 0.01] * 5)) == []                         # -43 dBFS: below VOICED_DB


def test_runs_a_word_pause_apart_are_one_utterance():
    three_quiet = [LOUD, LOUD, QUIET, QUIET, QUIET, LOUD, LOUD]                  # 240 ms apart: one
    assert voiced_runs(conv_of(three_quiet)) == [pytest.approx((0.08, 0.64))]
    four_quiet = [LOUD, LOUD, QUIET, QUIET, QUIET, QUIET, LOUD, LOUD]            # 320 ms apart: two
    assert voiced_runs(conv_of(four_quiet)) == [pytest.approx((0.08, 0.24)), pytest.approx((0.56, 0.72))]


def test_the_clip_is_streamed_at_24k_in_80ms_frames_after_the_lead_and_before_the_tail():
    m = FakeDuplex()
    conv = converse(m, np.zeros(IN_SR, np.float32))                              # 4.8 s + 1 s + 8 s = 13.8 s at 24 kHz
    assert m.frames == [FRAME] * 173 and len(conv.out) == 173 and conv.stream_s == pytest.approx(13.84)
    assert (m.began, m.ended) == (1, 1) and conv.wall_s >= 0 and conv.begin_s >= 0 and conv.lead_s == duplex.LEAD_S


# --------------------------------------------------------------------------- #
# The measures, counted by hand
# --------------------------------------------------------------------------- #


def test_an_answer_after_the_caller_stops_has_a_positive_gap_and_its_words():
    # the caller stops at 2.0 s; the model speaks from step 29, heard from 30 * 0.08 = 2.4 s
    m = FakeDuplex(speak=range(29, 41), text={30: " Sure", 31: ",", 32: " I can", 10: " hm"})
    t = measure(clip(), converse(m, clip(), lead_s=0, tail_s=4.0))
    assert t["caller_end_s"] == pytest.approx(2.0) and t["gap_ms"] == pytest.approx(400.0)
    assert t["overlap"] == 0 and t["voiced_during_caller"] is False and t["backchannels"] == 0
    assert t["reply_text"] == "Sure, I can"                                     # the step-10 piece came before the end
    assert t["rtf"] >= 0 and t["stream_s"] == pytest.approx(7.04)              # a fake is fast: its RTF can round to 0.000
    assert t["opened"] is False and t["pieces"] == [[10, " hm"], [30, " Sure"], [31, ","], [32, " I can"]]
    assert t["interruptions"] == 0 and t["setup_s"] >= 0


def test_talking_over_the_end_is_a_negative_gap_with_overlap_and_a_backchannel():
    m = FakeDuplex(speak=range(20, 31))                                          # heard 1.68 -> 2.56 s
    t = measure(clip(), converse(m, clip(), lead_s=0))
    assert t["gap_ms"] == pytest.approx(-320.0)
    assert t["overlap"] == pytest.approx(0.32 / 1.5, abs=0.001)                  # 1.68 -> 2.0 of the caller's 0.5 -> 2.0
    assert t["voiced_during_caller"] is True and t["backchannels"] == 1          # 0.88 s, shorter than 1 s
    assert t["interruptions"] == 0


def test_silence_through_the_tail_is_no_answer():
    t = measure(clip(), converse(FakeDuplex(), clip()))
    assert t["gap_ms"] is None and t["overlap"] == 0 and t["runs"] == [] and t["opened"] is False


def test_a_greeting_in_the_lead_in_is_the_opening_not_overlap_and_the_caller_is_measured_after_it():
    # lead 4.8 s: the caller speaks 5.3 -> 6.8 s of stream time. The model greets at steps 5-20
    # (heard 0.48 -> 1.76 s) and answers from step 89 (heard 7.2 s): gap 400 ms, no overlap.
    m = FakeDuplex(speak=list(range(5, 21)) + list(range(89, 100)))
    conv = converse(m, clip())
    t = measure(clip(), conv)
    assert t["opened"] is True and t["caller_end_s"] == pytest.approx(6.8) and t["gap_ms"] == pytest.approx(400.0)
    assert t["overlap"] == 0 and t["voiced_during_caller"] is False and t["backchannels"] == 0
    a = duplex.stereo(clip(), conv)                                              # the caller comes in after the lead
    k = int(duplex.LEAD_S * SR)
    assert not a[:k, 0].any() and np.abs(a[k + int(0.6 * SR):k + int(1.9 * SR), 0]).mean() > 0.05


def test_a_phrase_in_the_callers_pause_is_one_interruption_and_a_greeting_still_going_is_only_overlap():
    # lead 0.8 s: the caller speaks 1.3 -> 2.8 s. The greeting (steps 5-15, heard 0.48 -> 1.36 s) runs
    # 0.06 s into the caller. Steps 20-24 and 27-33 (heard 1.68 -> 2.08 and 2.24 -> 2.8 s) are 160 ms
    # apart: one utterance of 1.12 s, begun while the caller talks -- an interruption, not two backchannels.
    m = FakeDuplex(speak=list(range(5, 16)) + list(range(20, 25)) + list(range(27, 34)))
    t = measure(clip(), converse(m, clip(), lead_s=0.8))
    assert t["runs"] == [[0.48, 1.36], [1.68, 2.8]]
    assert t["opened"] is True and t["interruptions"] == 1 and t["backchannels"] == 0
    assert t["overlap"] == pytest.approx((0.06 + 1.12) / 1.5, abs=0.002) and t["voiced_during_caller"] is True


# --------------------------------------------------------------------------- #
# A set of items, the seeding check, the report
# --------------------------------------------------------------------------- #


class Item(NamedTuple):
    id: str
    cohort: str
    audio: np.ndarray


ITEMS = [Item("clean-00", "clean", clip()), Item("clean-01", "clean", clip(seed=1)),
         Item("accented-00", "accented", clip(speech=(0.5, 1.0), length=2.0))]


def test_a_seeded_model_answers_a_clip_the_same_twice_and_an_unseeded_one_does_not():
    got = run_items(FakeDuplex(speak=range(29, 41)), ITEMS)
    assert set(got["turns"]) == {i.id for i in ITEMS} and got["repeat"] == {"clean-00": 0.0, "clean-01": 0.0}
    assert got["turns"]["accented-00"]["cohort"] == "accented"
    loose = run_items(FakeDuplex(speak=range(29, 41), seeded=False), ITEMS, repeat=1)
    assert loose["repeat"]["clean-00"] > 0


def test_the_report_gives_the_gap_the_silences_and_the_overlap_per_cohort(tmp_path):
    # after the 4.8 s lead the clean callers speak 5.3 -> 6.8 s: a greeting at steps 5-20, an answer from step 89
    greet_then_answer = FakeDuplex(speak=list(range(5, 21)) + list(range(89, 101)), text={2: " Hi", 90: " Sure"})
    results = {"rung": "r5-fake", "description": "a fake", "machine": "laptop", "commit": "abc", "load_s": 1.0,
               **run_items(greet_then_answer, ITEMS)}
    text = report(results)
    assert "| clean | 2 | 2 | — | — | 0 | 0 | 0 | 0 | 0.00 |" in text           # opened both; two clips carry no percentile
    assert "median set-up s" in text
    assert results["turns"]["clean-00"]["pieces"] == [[2, " Hi"], [90, " Sure"]]
    assert "| accented | 1 |" in text and "| all | 3 |" in text
    assert "already talking as the caller stopped (negative gap): 0 of 3" in text
    assert "the same clip driven twice: clean-00 identical, clean-01 identical" in text
    assert "clean-00: 'Sure'" in text
    path = save(results, tmp_path / "r5.json")
    assert json.loads(path.read_text(encoding="utf-8"))["turns"]["clean-00"]["gap_ms"] == pytest.approx(400.0)
    with pytest.raises(FileExistsError):
        save(results, tmp_path / "r5.json")


# --------------------------------------------------------------------------- #
# The Moshi adapter's shape, without the model
# --------------------------------------------------------------------------- #


def test_the_moshi_adapter_is_registered_and_builds_without_the_package():
    from s2slab import r5
    m = r5.get("r5-moshi")
    assert m.repo == "kyutai/moshiko-pytorch-bf16" and m.device == "cuda" and (m.temp, m.temp_text) == (0.8, 0.7)
    assert r5.PAD_PIECES == (0, 3) and duplex.SR == SR == 24_000 and FRAME == 1920
    m.end()                                                                      # nothing open: a no-op
    with pytest.raises(KeyError, match="r5-moshi"):
        r5.get("r5-nothing")


def test_the_listening_file_puts_the_caller_left_and_the_model_right_where_it_is_heard():
    from s2slab.duplex import stereo
    conv = conv_of([None, LOUD, QUIET])
    a = stereo(np.full(IN_SR // 10, 0.1, np.float32), conv)                     # 0.1 s of caller
    assert a.shape == (4 * FRAME, 2) and a.dtype == np.float32
    assert abs(float(a[200:2000, 0].mean()) - 0.1) < 0.005 and not a[:, 0][3 * FRAME:].any()   # resampling rings at the edges
    assert not a[:2 * FRAME, 1].any() and np.allclose(a[2 * FRAME:3 * FRAME, 1], 0.3)   # step 1 heard from 2 * 80 ms
