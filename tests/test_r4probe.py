"""
Tests for s2slab.r4probe -- the reply-prosody measure and the container loop,
with synthetic voices and a fake rung. No model.

A "voice" here is a harmonic tone whose pitch glides between two frequencies,
so its F0 range is known in advance: 150 -> 200 Hz is about 5 semitones,
150 -> 300 Hz about 12.
"""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib

import numpy as np
import pytest
import soundfile as sf

from s2slab.prosodyset import Pair
from s2slab.r4probe import (analyse, answer_clips, format_report, measure, pair_deltas, save, to_16k, wav_bytes,
                            write_replies)
from s2slab.rungs import Reply, Rung

pytest.importorskip("parselmouth")
ROOT = pathlib.Path(__file__).resolve().parents[1]
SR = 24_000


def glide(f0_from: float, f0_to: float, seconds: float = 1.5, sr: int = SR, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    f0 = np.linspace(f0_from, f0_to, len(t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    return (amp * sum(np.sin(k * phase) / k for k in range(1, 6))).astype(np.float32)


def pair(emotion: str, k: int) -> Pair:
    a = f"10{k:02d}"
    return Pair(f"{emotion.lower()}-d{k:02d}", "dev", emotion, a, "DFA", f"{a}_DFA_NEU_XX", f"{a}_DFA_{emotion}_XX",
                0.9, 0.9, 10, 10, 60.0, 0.9)


def test_a_wider_glide_measures_as_a_wider_range_and_24k_is_resampled():
    narrow, wide = measure(wav_bytes(glide(150, 200), SR)), measure(wav_bytes(glide(150, 300), SR))
    assert narrow["seconds"] == 1.5 and 3.0 < narrow["range_st"] < 6.0 and 9.0 < wide["range_st"] < 13.0
    assert 160 < narrow["pitch_hz"] < 190                                    # not 2/3 of it: the rate was honoured
    assert len(to_16k(np.zeros(SR, np.float32), SR)) == 16_000


def test_silence_has_no_pitch_and_the_pair_does_not_move():
    silent = measure(wav_bytes(np.zeros(SR, np.float32), SR))
    assert silent["range_st"] is None and silent["pitch_hz"] is None
    p = pair("ANG", 1)
    rows = pair_deltas([p], {p.neutral: silent, p.loaded: measure(wav_bytes(glide(150, 300), SR))},
                       {p.neutral: "", p.loaded: "Oh no!"})
    assert rows[0]["d_range_st"] is None and rows[0]["moves"] is False and rows[0]["words_upset"] == 2


def test_deltas_are_upset_reply_minus_calm_reply():
    p = pair("SAD", 2)
    m = {p.neutral: {"pitch_hz": 200.0, "range_st": 4.0, "level_db": -20.0, "seconds": 2.0},
         p.loaded: {"pitch_hz": 400.0, "range_st": 7.5, "level_db": -17.0, "seconds": 3.0}}
    row = pair_deltas([p], m, {p.neutral: "Sure.", p.loaded: "I'm so sorry to hear that."})[0]
    assert (row["d_range_st"], row["d_pitch_st"], row["d_level_db"], row["moves"]) == (3.5, 12.0, 3.0, True)
    assert (row["words_calm"], row["words_upset"]) == (1, 6)


class GlideRung(Rung):
    """Answers a loud clip with a wide glide and a quiet one with a narrow glide."""

    rung_id, description = "glide", "a test double"

    def __init__(self):
        self.loads = 0

    def _load(self):
        self.loads += 1

    def _respond(self, audio, sr, sink):
        loud = float(np.abs(audio).max()) > 0.1
        return Reply(glide(150, 300 if loud else 200), SR, None, "Oh dear!" if loud else "Okay.")


def clip(amp: float) -> bytes:
    return wav_bytes(np.full(16_000, amp, np.float32), 16_000)


def test_the_container_loop_answers_every_clip_and_checks_a_repeat():
    p = pair("FEA", 3)
    clips = {p.neutral: clip(0.01), p.loaded: clip(0.5)}
    got = answer_clips(GlideRung(), clips, repeat=2)
    assert set(got["replies"]) == set(clips) and got["repeat"] == {p.neutral: 0.0, p.loaded: 0.0}
    assert got["replies"][p.loaded]["text"] == "Oh dear!"
    back, sr = sf.read(io.BytesIO(got["replies"][p.loaded]["wav"]), dtype="float32")
    assert sr == SR and len(back) == int(1.5 * SR)
    measures, deltas = analyse([p], {"glide": got})
    assert deltas["glide"][0]["moves"] is True and deltas["glide"][0]["d_range_st"] > 4


def test_the_report_counts_moving_pairs_per_rung_and_emotion(tmp_path):
    pairs = [pair("SAD", 1), pair("ANG", 2)]
    clips = {c: clip(a) for p in pairs for c, a in ((p.neutral, 0.01), (p.loaded, 0.5))}
    answers = {"r4-gpu": answer_clips(GlideRung(), clips, repeat=1)}
    answers["r3-gpu"] = {"replies": {c: {**r, "wav": answers["r4-gpu"]["replies"][pairs[0].neutral]["wav"]}
                                     for c, r in answers["r4-gpu"]["replies"].items()}, "repeat": {}}
    measures, deltas = analyse(pairs, answers)
    text = format_report(pairs, deltas, {r: a["repeat"] for r, a in answers.items()})
    assert "| R4 (Qwen2.5-Omni speaks) | all | 2 | 2 of 2 |" in text
    assert "| R3 (Qwen2-Audio -> Kokoro) | all | 2 | 0 of 2 |" in text              # every reply the same narrow glide
    assert "the same clip answered twice: 1001_DFA_NEU_XX identical" in text
    assert "| sad-d01 | +0.0 | +" in text or "| sad-d01 |" in text
    write_replies(answers, tmp_path / "replies")
    assert len(list((tmp_path / "replies" / "r4-gpu").glob("*.wav"))) == 4
    save(tmp_path / "run.json", "2 dev pairs", pairs, answers, measures, deltas)
    kept = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert "wav" not in kept["replies"]["r4-gpu"][pairs[0].neutral] and kept["deltas"]["r4-gpu"][1]["moves"] is True
    with pytest.raises(FileExistsError):
        save(tmp_path / "run.json", "again", pairs, answers, measures, deltas)


def test_the_modal_module_has_the_pairs_entrypoint_and_its_function():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module.answer, "remote") and hasattr(module.pairs, "info")
