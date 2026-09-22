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


def test_the_wordings_are_four_the_first_is_the_side_channels_question_and_one_is_registered():
    from s2slab.sidechannel import QUESTION
    from s2slab.r3probe import REGISTERED_WORDING
    assert len(WORDINGS) == 4 and WORDINGS["tone"] == QUESTION
    assert REGISTERED_WORDING in WORDINGS and WORDINGS[REGISTERED_WORDING] == "Is the person speaking upset?"


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


# --------------------------------------------------------------------------- #
# PR-006's criteria: R3 against the side-channel
# --------------------------------------------------------------------------- #

from s2slab.r3probe import compare, format_criteria, per_pair, sidechannel_outcomes  # noqa: E402


def thirteen(emotion, k0):
    return [pair(emotion, k0 + j, "test") for j in range(13)]


TEST_PAIRS = thirteen("SAD", 10) + thirteen("FEA", 30) + thirteen("ANG", 50)


def fake_sidechannel_run(right_ids):
    """A PR-005 run file where the tagger answered exactly right_ids right: the
    loaded clip tagged angry with p(Yes) 0.9, the neutral clip tagged neutral
    with 0.1 -- or, when wrong, the neutral clip tagged sad with 0.9."""
    run = {"pairs": [pr._asdict() for pr in TEST_PAIRS], "tags": {}, "p_yes": {}}
    for pr in TEST_PAIRS:
        run["tags"][pr.loaded] = "angry"
        run["p_yes"][pr.loaded] = {"none": 0.5, "neutral": 0.1, "sad": 0.9, "fearful": 0.9, "angry": 0.9}
        run["tags"][pr.neutral] = "neutral" if pr.id in right_ids else "sad"
        run["p_yes"][pr.neutral] = {"none": 0.5, "neutral": 0.1, "sad": 0.9, "fearful": 0.9, "angry": 0.9}
    return run


def r3_scores(answered_ids, ranked_ids):
    """R3's {clip: p(Yes)}: loaded 0.95 everywhere; neutral 0.2 if answered right,
    0.7 if only ranked right, 0.99 if neither."""
    p = {}
    for pr in TEST_PAIRS:
        p[pr.loaded] = 0.95
        p[pr.neutral] = 0.2 if pr.id in answered_ids else 0.7 if pr.id in ranked_ids else 0.99
    return p


def test_the_side_channels_outcomes_are_read_from_its_run_file():
    right = {pr.id for pr in TEST_PAIRS[:7]}
    out = sidechannel_outcomes(fake_sidechannel_run(right))
    assert len(out) == 39 and {i for i, ok in out.items() if ok} == right


def test_claim_a_needs_twelve_ranked_right_per_emotion_and_claim_b_the_tags_counts_and_a_sign_test():
    ids = [pr.id for pr in TEST_PAIRS]
    sad, fea, ang = ids[:13], ids[13:26], ids[26:]
    side = sidechannel_outcomes(fake_sidechannel_run(set(sad[:7]) | set(fea[:7]) | set(ang[:11])))   # 7 · 7 · 11
    # R3: ranks everything right except one fear pair; answers 11 · 9 · 9 right
    p = r3_scores(set(sad[:11]) | set(fea[:9]) | set(ang[:9]), set(ids) - {fea[12]})
    r = compare(TEST_PAIRS, p, side)
    assert r["r3"] == {"SAD": (11, 13, 13), "FEA": (9, 12, 13), "ANG": (9, 13, 13)}
    assert r["tag"] == {"SAD": 7, "FEA": 7, "ANG": 11} and r["A"] == {"SAD": True, "FEA": True, "ANG": True}
    assert r["B1"] == {"SAD": True, "FEA": True, "ANG": False}
    assert r["fixed"] == sorted(sad[7:11] + fea[7:9]) and r["broken"] == sorted(ang[9:11])
    assert r["p"] == pytest.approx(0.289, abs=0.001) and r["B2"] is False        # 6 fixed, 2 broken
    text = format_criteria(r, "PR-006")
    assert "| A2 fear | 12 of 13 | at least 12 | PASS |" in text
    assert "| B1 anger | 9 of 13 | 11 of 13 | at least 11 | FAIL |" in text
    assert "| B2 overall | 29 of 39 | 25 of 39 | more, and sign test p < 0.05 | FAIL |" in text
    assert "PR-006 Claim A: SUPPORTED" in text and "PR-006 Claim B: NOT SUPPORTED: failed B1 anger, B2" in text
    assert "PR-006" not in format_criteria(r)                                  # no verdict without a registration


def test_claim_b_can_hold_and_claim_a_can_fail():
    ids = [pr.id for pr in TEST_PAIRS]
    side = sidechannel_outcomes(fake_sidechannel_run(set(ids[:7]) | set(ids[13:20]) | set(ids[26:37])))
    p = r3_scores(set(ids), set(ids))                                          # 13 · 13 · 13 answered right
    for i in ids[:3]:                                                          # ...but three sad pairs ranked wrong
        p[TEST_PAIRS[ids.index(i)].neutral] = 0.99
    r = compare(TEST_PAIRS, p, side)
    assert r["A"]["SAD"] is False and r["A"]["FEA"] and r["r3"]["SAD"] == (10, 10, 13)
    assert r["B1"] == {"SAD": True, "FEA": True, "ANG": True} and r["B2"] is True
    assert r["p"] == pytest.approx(0.0127, abs=0.001)                       # 14 fixed, 3 broken: 2 * P(X <= 3 | n = 17)
    text = format_criteria(r, "PR-006")
    assert "Claim A: NOT SUPPORTED: failed sad" in text and "Claim B: SUPPORTED" in text


def test_the_per_pair_table_shows_both_systems_and_whether_r3_heard_the_words():
    ids = [pr.id for pr in TEST_PAIRS]
    side = sidechannel_outcomes(fake_sidechannel_run(set(ids[:1])))
    p = r3_scores(set(ids[1:2]), set(ids[:2]))
    transcripts = {TEST_PAIRS[0].neutral: "The transcription is: 'Don't forget a jacket.'",
                   TEST_PAIRS[1].neutral: "Don't forget the jacket"}
    text = per_pair(TEST_PAIRS[:2], p, side, transcripts)
    assert f"| {ids[0]} | 0.70 | 0.95 | **wrong** | right | right | yes |" in text
    assert f"| {ids[1]} | 0.20 | 0.95 | right | right | wrong | no |" in text
    assert text.endswith("word for word after normalizing: 1 of 2")
