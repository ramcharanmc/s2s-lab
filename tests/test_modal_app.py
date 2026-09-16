"""
Tests for modal_app/ -- checked on the laptop, without running anything on Modal.

Defining a Modal app builds nothing and contacts nobody, so the module can be
imported and inspected here. Skipped when the modal package is not installed.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from s2slab.rungs import get

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def r0_bench():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("r0_bench", ROOT / "modal_app" / "r0_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rung_id", ["r0-gpu", "r0-gpu-bigger"])
def test_the_image_holds_every_model_the_gpu_rungs_load(r0_bench, rung_id):
    """The container runs with HF_HUB_OFFLINE=1, so a model missing from MODELS
    fails at load instead of downloading on the GPU's meter. This catches it
    here, for free, whenever a rung changes its models."""
    rung = get(rung_id)
    assert f"Systran/faster-whisper-{rung.asr.model_name}" in r0_bench.MODELS
    assert rung.llm.model_name in r0_bench.MODELS
    assert f"voices/{rung.tts.voice}.pt" in r0_bench.MODELS["hexgrad/Kokoro-82M"]

def test_a_full_eval_run_is_saved_as_evidence_and_a_smoke_test_is_not(r0_bench):
    """The full run lands in runs/, which is committed; a --limit run lands in
    the ignored results/, so a smoke test can never pass for the real row."""
    full = r0_bench.results_path("r0-gpu")
    smoke = r0_bench.results_path("r0-gpu", limit=8)
    assert full.relative_to(ROOT).parts == ("runs", "r0-gpu", "evalset-modal-t4.json")
    assert smoke.relative_to(ROOT).parts[0] == "results"
    assert "limit8" in smoke.name
