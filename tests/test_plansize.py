"""
Tests for the cascade at the plans' sizes (M45): the rung's models, the A10 image that holds them, where its runs are
saved, the refusal pattern and the side-by-side comparison, on made-up turns. Nothing here loads a model.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from s2slab import plansize
from s2slab.rungs import get

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_rung_is_r1_with_the_plans_recognizer_and_llm():
    plan, base = get("r1-gpu-plan"), get("r1-gpu")
    assert (plan.asr.model_name, plan.asr.device, plan.asr.compute_type) == ("large-v3-turbo", "cuda", "int8_float16")
    assert (plan.llm.model_name, plan.llm.device) == ("Qwen/Qwen2.5-7B-Instruct", "cuda")
    assert type(plan.llm) is type(base.llm) and type(plan.tts) is type(base.tts) and plan.tts.voice == base.tts.voice
    assert plan.llm.generate_kwargs() == base.llm.generate_kwargs()             # the same greedy decoding


@pytest.fixture(scope="module")
def bench():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("plansize_bench", ROOT / "modal_app" / "plansize_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_a10_image_holds_both_rungs_models_and_the_runs_are_saved_once(bench):
    r0_spec = importlib.util.spec_from_file_location("r0_bench", ROOT / "modal_app" / "r0_bench.py")
    r0_bench = importlib.util.module_from_spec(r0_spec)
    r0_spec.loader.exec_module(r0_bench)
    for rung_id in bench.RUNGS:
        rung = get(rung_id)
        assert r0_bench.whisper_repo(rung.asr.model_name) in bench.MODELS
        assert rung.llm.model_name in bench.MODELS
        assert f"voices/{rung.tts.voice}.pt" in bench.MODELS["hexgrad/Kokoro-82M"]
    assert bench.results_path("r1-gpu-plan").relative_to(ROOT).parts == ("runs", "r1-gpu-plan", "evalset-modal-a10.json")
    assert bench.results_path("r1-gpu").name == "evalset-modal-a10-plan.json"           # M19's A10 run keeps its name
    assert not bench.results_path("r1-gpu").exists() and (ROOT / "runs" / "r1-gpu" / "evalset-modal-a10.json").exists()
    assert bench.results_path("r1-gpu-plan", limit=8).relative_to(ROOT).parts[0] == "results"
    assert plansize.PLAN == bench.results_path("r1-gpu-plan").relative_to(ROOT)
    assert plansize.BASE == bench.results_path("r1-gpu").relative_to(ROOT)


@pytest.mark.parametrize("reply", ["I'm sorry, but I can't assist with that request.", "I'm sorry, but I can't assist with that.",
                                   "I cannot help with that.", "I won't help with GPS.", "I am unable to provide that."])
def test_a_refusal_is_found(reply):
    assert plansize.REFUSAL.search(reply)


@pytest.mark.parametrize("reply", ["I'm sorry to hear that. How can I help?", "I can help with that.",
                                   "That sounds lovely; I can't wait to hear more."])
def test_a_reply_that_is_not_a_refusal_is_not(reply):
    assert not plansize.REFUSAL.search(reply)


def turn(transcript, reply, first_audio, asr=50.0):
    return {"cohort": "clean", "transcript": transcript, "reply_text": reply, "first_audio_ms": first_audio,
            "first_sound_ms": first_audio + 300, "total_ms": first_audio + 400,
            "stages_ms": {"asr": asr, "llm": 200.0, "tts": 100.0}}


def test_the_comparison_reads_both_runs_the_same_way():
    from s2slab.wer import references
    refs = references()
    ids = sorted(refs)[:3]
    base = {"machine": "m", "peak_vram_mb": 1700,
            "turns": {c: turn("", "I'm sorry, but I can't assist with that request.", 400.0) for c in ids}}
    plan = {"machine": "m", "peak_vram_mb": 17000, "turns": {c: turn(refs[c][1], "Sure.", 600.0, 120.0) for c in ids}}
    c = plansize.compare(plan, base)
    assert c["r1-gpu"]["refusals"] == ids and c["r1-gpu-plan"]["refusals"] == []
    assert c["r1-gpu-plan"]["wer_raw"]["all"] == 0.0 and c["r1-gpu"]["wer_raw"]["all"] == 100.0
    assert c["r1-gpu-plan"]["first_audio_ms"]["p50"] == 600.0 and c["r1-gpu-plan"]["stages_ms"]["asr"]["p50"] == 120.0
    text = plansize.format_report(c)
    assert "refusals, of 200" in text and "refused by r1-gpu-plan: none" in text and "17000" in text
    with pytest.raises(ValueError, match="same clips"):
        plansize.compare(plan, {**base, "turns": {ids[0]: base["turns"][ids[0]]}})
