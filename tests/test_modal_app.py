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


GPU_RUNGS = ["r0-gpu", "r0-gpu-bigger", "r1-gpu", "r2-gpu", "r2-gpu-turbo"]


@pytest.mark.parametrize("rung_id", GPU_RUNGS)
def test_the_image_holds_every_model_the_gpu_rungs_load(r0_bench, rung_id):
    """The container runs with HF_HUB_OFFLINE=1, so a model missing from MODELS
    fails at load instead of downloading on the GPU's meter. This catches it
    here, for free, whenever a rung changes its models."""
    rung = get(rung_id)
    assert r0_bench.whisper_repo(rung.asr.model_name) in r0_bench.MODELS
    assert rung.llm.model_name in r0_bench.MODELS
    assert f"voices/{rung.tts.voice}.pt" in r0_bench.MODELS["hexgrad/Kokoro-82M"]

def test_the_image_looks_for_each_recognizer_where_faster_whisper_will(r0_bench):
    """whisper_repo() is our copy of faster-whisper's own table. If the two
    disagree, the image holds a model the container will not find."""
    utils = pytest.importorskip("faster_whisper.utils")
    for rung_id in GPU_RUNGS:
        name = get(rung_id).asr.model_name
        assert r0_bench.whisper_repo(name) == utils._MODELS[name], name


def test_a_full_eval_run_is_saved_as_evidence_and_a_smoke_test_is_not(r0_bench):
    """The full run lands in runs/, which is committed; a --limit run lands in
    the ignored results/, so a smoke test can never pass for the real row."""
    full = r0_bench.results_path("r0-gpu")
    smoke = r0_bench.results_path("r0-gpu", limit=8)
    assert full.relative_to(ROOT).parts == ("runs", "r0-gpu", "evalset-modal-t4.json")
    assert smoke.relative_to(ROOT).parts[0] == "results"
    assert "limit8" in smoke.name

def test_a_tag_names_the_run_and_keeps_the_untagged_name_unchanged(r0_bench):
    assert r0_bench.results_path("r0-gpu").name == "evalset-modal-t4.json"
    assert r0_bench.results_path("r1-gpu", tag="pr002").relative_to(ROOT).parts == \
        ("runs", "r1-gpu", "evalset-modal-t4-pr002.json")
    assert r0_bench.results_path("r1-gpu", limit=8, tag="pr002").name == \
        "evalset-modal-t4-pr002-limit8.json"


def test_recorded_evidence_is_never_overwritten(r0_bench):
    """runs/r0-gpu/evalset-modal-t4.json is committed. A second run under the
    same name must stop before it costs anything; scratch may be overwritten."""
    recorded = r0_bench.results_path("r0-gpu")
    assert recorded.exists(), "the committed R0 run is missing"
    with pytest.raises(SystemExit, match="never overwritten"):
        r0_bench.refuse_to_overwrite([recorded])
    r0_bench.refuse_to_overwrite([r0_bench.results_path("r0-gpu", tag="not-run-yet")])
    r0_bench.refuse_to_overwrite([r0_bench.results_path("r0-gpu", limit=8)])


# --------------------------------------------------------------------------- #
# The R3 image: the rung's eval-set run on the A10, with R0 and R1 beside it
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def r3_probe():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("r3_probe", ROOT / "modal_app" / "r3_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_r3_image_holds_every_model_the_rungs_it_runs_load(r0_bench, r3_probe):
    from s2slab import r3
    r3_rung = get("r3-gpu")
    assert r3_rung.llm.model_name == r3.MODEL == r3_probe.MODEL and r3.MODEL in r3_probe.MODELS
    assert f"voices/{r3_rung.tts.voice}.pt" in r3_probe.MODELS["hexgrad/Kokoro-82M"]
    for rung_id in ("r0-gpu", "r1-gpu"):                                   # the same-container comparators
        rung = get(rung_id)
        assert r0_bench.whisper_repo(rung.asr.model_name) in r3_probe.MODELS
        assert rung.llm.model_name in r3_probe.MODELS
        assert f"voices/{rung.tts.voice}.pt" in r3_probe.MODELS["hexgrad/Kokoro-82M"]


def test_the_a10_run_is_saved_as_evidence_and_its_smoke_test_is_not(r3_probe):
    assert r3_probe.results_path("r3-gpu").relative_to(ROOT).parts == ("runs", "r3-gpu", "evalset-modal-a10.json")
    assert r3_probe.results_path("r0-gpu", tag="m19").name == "evalset-modal-a10-m19.json"
    smoke = r3_probe.results_path("r3-gpu", limit=8)
    assert smoke.relative_to(ROOT).parts[0] == "results" and "limit8" in smoke.name
    assert hasattr(r3_probe.evaluate, "remote") and hasattr(r3_probe.evalset, "info")   # a Modal Function and a LocalEntrypoint


# --------------------------------------------------------------------------- #
# The R4 image: the S2S rung on the L40S, with R3 beside it
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def r4_bench():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_r4_image_holds_every_model_the_rungs_it_runs_load(r4_bench):
    from s2slab import r3, r4
    r4_rung, r3_rung = get("r4-gpu"), get("r3-gpu")
    assert r4_rung.model.model_name == r4.MODEL and r4.MODEL in r4_bench.MODELS
    assert r3_rung.llm.model_name == r3.MODEL and r3.MODEL in r4_bench.MODELS
    assert f"voices/{r3_rung.tts.voice}.pt" in r4_bench.MODELS["hexgrad/Kokoro-82M"]
    assert r4_bench.L40S_DOLLARS_PER_HOUR == 1.95


def test_the_l40s_run_is_saved_as_evidence_and_its_smoke_test_is_not(r4_bench):
    assert r4_bench.results_path("r4-gpu").relative_to(ROOT).parts == ("runs", "r4-gpu", "evalset-modal-l40s.json")
    assert r4_bench.results_path("r3-gpu", tag="m20").name == "evalset-modal-l40s-m20.json"
    smoke = r4_bench.results_path("r4-gpu", limit=8)
    assert smoke.relative_to(ROOT).parts[0] == "results" and "limit8" in smoke.name
    assert hasattr(r4_bench.evaluate, "remote") and hasattr(r4_bench.bench, "remote") and hasattr(r4_bench.main, "info")
