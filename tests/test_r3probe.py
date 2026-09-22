"""
Tests for s2slab.r3probe (the laptop side of R3's probe) and for the container-side
loop in modal_app/r3_probe.py, run here with a fake speech-LLM.

No GPU, no model: the fake answers Yes to any clip whose audio is louder than a
threshold, so every expected number can be counted by hand.
"""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib

import numpy as np
import pytest
import soundfile as sf

from s2slab import r3probe
from s2slab.prosodyset import Pair
from s2slab.r3probe import SYSTEM, WORDINGS, clips_for, format_report, save, score_pairs

ROOT = pathlib.Path(__file__).resolve().parents[1]


def pair(emotion: str, k: int, split: str = "dev") -> Pair:
    actor = f"10{k:02d}"
    return Pair(f"{emotion.lower()}-{split[0]}{k:02d}", split, emotion, actor, "DFA", f"{actor}_DFA_NEU_XX",
                f"{actor}_DFA_{emotion}_XX", 0.9, 0.9, 10, 10, 60.0, 0.9)


PAIRS = [pair("SAD", 1), pair("FEA", 2), pair("ANG", 3), pair("ANG", 4)]


def wav_bytes(amp: float, n: int = 8000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.full(n, amp, np.float32), 16_000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# the loaded clips are loud except sad's; actor 1004's NEUTRAL clip is loud too
CLIPS = {"1001_DFA_NEU_XX": wav_bytes(0.01), "1001_DFA_SAD_XX": wav_bytes(0.02),
         "1002_DFA_NEU_XX": wav_bytes(0.01), "1002_DFA_FEA_XX": wav_bytes(0.5),
         "1003_DFA_NEU_XX": wav_bytes(0.01), "1003_DFA_ANG_XX": wav_bytes(0.5),
         "1004_DFA_NEU_XX": wav_bytes(0.5), "1004_DFA_ANG_XX": wav_bytes(0.6)}


class FakeSpeechLLM:
    """Yes (0.9) if the clip is loud, No (0.1) if quiet; a second wording is always on the fence."""

    def __init__(self):
        self.asked = []

    def p_yes(self, audio, sr, system, question):
        self.asked.append((sr, system, question))
        if question.startswith("fence"):
            return 0.5, 1.0
        return (0.9, 1.0) if float(np.abs(audio).max()) > 0.1 else (0.1, 1.0)

    def transcribe(self, audio, sr, prompt):
        return f"heard {len(audio)} samples"


@pytest.fixture(scope="module")
def r3_probe():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("r3_probe", ROOT / "modal_app" / "r3_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# The container-side loop, with a fake model
# --------------------------------------------------------------------------- #


def test_every_clip_is_asked_every_wording_once_and_transcribed_once(r3_probe):
    llm = FakeSpeechLLM()
    questions = {"loud": "Is it loud?", "fence": "fence?"}
    answers = r3_probe.run_probe(llm, CLIPS, SYSTEM, questions, "Transcribe.")
    assert set(answers) == set(CLIPS) and len(llm.asked) == 8 * 2
    assert all(sr == 16_000 and system == SYSTEM for sr, system, _ in llm.asked)
    assert answers["1002_DFA_FEA_XX"] == {"p_yes": {"loud": 0.9, "fence": 0.5}, "mass": {"loud": 1.0, "fence": 1.0},
                                          "transcript": "heard 8000 samples"}
    assert r3_probe.run_probe(llm, CLIPS, SYSTEM, questions, None)["1001_DFA_NEU_XX"]["transcript"] == ""


def test_the_image_holds_the_one_model_the_probe_loads_and_the_guardrails_are_set(r3_probe):
    assert r3_probe.MODEL in r3_probe.MODELS
    assert r3_probe.SpeechLLM().model_name == r3_probe.MODEL
    assert r3_probe.A10_DOLLARS_PER_HOUR == 1.10


# --------------------------------------------------------------------------- #
# The laptop side: scoring and the report
# --------------------------------------------------------------------------- #


def test_pairs_are_scored_like_the_side_channels():
    p = {"1001_DFA_NEU_XX": 0.1, "1001_DFA_SAD_XX": 0.1, "1002_DFA_NEU_XX": 0.1, "1002_DFA_FEA_XX": 0.9,
         "1003_DFA_NEU_XX": 0.1, "1003_DFA_ANG_XX": 0.9, "1004_DFA_NEU_XX": 0.9, "1004_DFA_ANG_XX": 0.9}
    assert score_pairs(PAIRS, p) == {"SAD": (0, 0, 1), "FEA": (1, 1, 1), "ANG": (1, 1, 2)}


def test_the_report_has_a_row_per_wording_and_a_per_pair_table_for_the_first(r3_probe):
    answers = r3_probe.run_probe(FakeSpeechLLM(), CLIPS, SYSTEM, {"loud": "Is it loud?", "fence": "fence?"}, "T.")
    text = format_report(PAIRS, answers, {"loud": "Is it loud?", "fence": "fence?"})
    assert "| loud | 0 · 0 | 1 · 1 | 1 · 1 |" in text and "| fence | 0 · 0 | 0 · 0 | 0 · 0 |" in text
    assert 'loud: "Is it loud?" -- Yes to 4 of 8 clips, median p(Yes) 0.50' in text
    assert "per pair, wording 'loud':" in text
    assert "| ang-d04 | 0.90 | 0.90 | **wrong** | wrong | 'heard 8000 samples' |" in text
    assert "| fea-d02 | 0.10 | 0.90 | right | right |" in text


def test_a_saved_run_holds_the_answers_and_scores_and_is_never_overwritten(tmp_path, r3_probe):
    answers = r3_probe.run_probe(FakeSpeechLLM(), CLIPS, SYSTEM, {"loud": "Is it loud?"}, None)
    path = tmp_path / "prosody" / "r3.json"
    save(path, "4 dev pairs", PAIRS, answers, {"loud": "Is it loud?"})
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert kept["run"] == "4 dev pairs" and kept["system"] == SYSTEM and len(kept["pairs"]) == 4
    assert kept["scores"]["loud"] == {"SAD": [0, 0, 1], "FEA": [1, 1, 1], "ANG": [1, 1, 2]}
    assert kept["answers"]["1004_DFA_NEU_XX"]["p_yes"]["loud"] == 0.9
    with pytest.raises(FileExistsError, match="never overwritten"):
        save(path, "again", PAIRS, answers, {"loud": "Is it loud?"})


def test_the_wordings_are_four_and_the_first_is_the_side_channels_question():
    from s2slab.sidechannel import QUESTION
    assert len(WORDINGS) == 4 and WORDINGS["tone"] == QUESTION


# --------------------------------------------------------------------------- #
# The lock: test clips are never read without a registration
# --------------------------------------------------------------------------- #


def test_the_test_clips_cannot_be_prepared_for_sending_without_a_registration():
    with pytest.raises(PermissionError, match="preregistration"):
        clips_for("test")


@pytest.mark.skipif(not (r3probe.PROSODY_ROOT / "manifest.json").exists()
                    or not any((r3probe.PROSODY_ROOT / "audio").glob("*.wav")),
                    reason="the prosody set is not on this machine")
def test_the_dev_clips_are_sent_as_the_frozen_wav_bytes():
    pairs, clips = clips_for("dev")
    assert len(pairs) == 21 and len(clips) == 42
    data = clips[pairs[0].neutral]
    assert data[:4] == b"RIFF" and data == (r3probe.PROSODY_ROOT / "audio" / f"{pairs[0].neutral}.wav").read_bytes()
