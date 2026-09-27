"""
Tests for modal_app/ -- checked on the laptop, without running anything on Modal.

Defining a Modal app builds nothing and contacts nobody, so the module can be
imported and inspected here. Skipped when the modal package is not installed.
"""

from __future__ import annotations

import importlib.util
import json
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


# --------------------------------------------------------------------------- #
# The R5 image: full-duplex Moshi on the A10
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def r5_bench():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("r5_bench", ROOT / "modal_app" / "r5_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_r5_image_holds_the_model_the_adapter_loads_and_saves_evidence_only_for_the_full_eval_set(r5_bench):
    from s2slab import r5
    assert r5.get("r5-moshi").repo in r5_bench.MODELS and r5_bench.A10_DOLLARS_PER_HOUR == 1.10
    assert r5_bench.results_path("r5-moshi", True).relative_to(ROOT).parts == ("runs", "r5-moshi", "evalset-modal-a10.json")
    assert r5_bench.results_path("r5-moshi", True, limit=8).relative_to(ROOT).parts[0] == "results"
    assert r5_bench.results_path("r5-moshi", False).relative_to(ROOT).parts[0] == "results"
    assert hasattr(r5_bench.smoke, "remote") and hasattr(r5_bench.evaluate, "remote") and hasattr(r5_bench.main, "info")


def test_the_personaplex_image_pins_the_adapters_weights_and_runs_on_functions_of_its_own(r5_bench):
    from s2slab import r5
    m = r5.get("r5-personaplex")
    assert (r5_bench.PERSONAPLEX_REPO, r5_bench.PERSONAPLEX_REVISION) == (m.repo, m.revision)
    assert "voices.tgz" in r5_bench.PERSONAPLEX_FILES and r5_bench.HF_SECRET == "huggingface-secret"
    assert r5_bench.functions("r5-moshi") == (r5_bench.smoke, r5_bench.evaluate)
    assert r5_bench.functions("r5-personaplex") == (r5_bench.smoke_personaplex, r5_bench.evaluate_personaplex)
    assert hasattr(r5_bench.smoke_personaplex, "remote") and hasattr(r5_bench.evaluate_personaplex, "remote")
    with pytest.raises(KeyError):
        r5_bench.functions("r5-nothing")



def test_the_ft1_dialogues_run_is_moshi_on_the_a10_and_reads_only_the_dev_fold(r5_bench):
    assert hasattr(r5_bench.dialogues_remote, "remote") and hasattr(r5_bench.dialogues, "info")
    assert r5_bench.DIALOGUE_TAIL_S == 4.0



@pytest.fixture(scope="module")
def ft2_train():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("ft2_train", ROOT / "modal_app" / "ft2_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ft2_image_pins_what_the_laptop_env_resolved(ft2_train):
    """ft/uv.lock is the laptop env; the GPU image installs the same versions (Python 3.12's, where
    the lock resolves two), with CUDA torch of the same version."""
    import re
    lock = (ROOT / "ft" / "uv.lock").read_text(encoding="utf-8")
    versions = {}
    for m in re.finditer(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"\nsource = \{[^}]*\}\n'
                         r'(resolution-markers = \[[^\]]*\])?', lock):
        markers = m.group(3) or ""
        if "python_full_version < '3.12'" in markers and "python_full_version >= '3.12'" not in markers:
            continue
        versions.setdefault(m.group(1), set()).add(m.group(2).replace("+cpu", ""))
    for pin in ft2_train.PINS + [ft2_train.TORCH]:
        name, version = pin.split("==")
        assert version in versions.get(name, set()), f"{pin} is not what ft/uv.lock resolved: {versions.get(name)}"


def test_the_ft2_image_holds_the_model_the_experiment_trains(ft2_train):
    import yaml
    model_cfg = yaml.safe_load((ROOT / "ft" / "configs" / "model" / "whisper_lora.yaml").read_text(encoding="utf-8"))
    assert model_cfg["model_name"] in ft2_train.MODELS
    assert (ROOT / "ft" / "configs" / "experiment" / f"{ft2_train.EXPERIMENT}.yaml").exists()
    # Hydra refuses an override of a key the config does not declare: the first dev run died on this.
    trainer_cfg = yaml.safe_load((ROOT / "ft" / "configs" / "trainer" / "default.yaml").read_text(encoding="utf-8"))
    assert "max_time" in trainer_cfg


def test_the_ft2_caps_are_dollars_and_the_trainer_stops_inside_them(ft2_train):
    assert ft2_train.seconds_for(1.10) == 3600 and ft2_train.seconds_for(ft2_train.CAP_DOLLARS) == 4909
    assert ft2_train.max_time(4909) == "00:01:16:49" and ft2_train.max_time(100) == "00:00:01:00"
    assert ft2_train.DEV_CAP_DOLLARS < ft2_train.EVAL_CAP_DOLLARS < ft2_train.CAP_DOLLARS <= 1.5
    assert "trainer.max_steps=30" in ft2_train.DEV_OVERRIDES and "data.val_limit=8" in ft2_train.DEV_OVERRIDES
    for fn in (ft2_train.train_remote, ft2_train.evaluate_remote, ft2_train.export_remote, ft2_train.fetch_data,
               ft2_train.bench_remote):
        assert hasattr(fn, "remote")
    for ep in (ft2_train.dev, ft2_train.train, ft2_train.evaluate, ft2_train.export, ft2_train.bench):
        assert hasattr(ep, "info")


def test_the_ft2_bench_holds_the_comparator_and_lets_the_lora_meet_the_eval_set_once(ft2_train, r0_bench, tmp_path):
    from s2slab import ft2
    assert ft2_train.BASE_ASR == ft2.BASE_MODEL
    assert r0_bench.whisper_repo(ft2_train.BASE_ASR) in ft2_train.MODELS      # offline: the weights must be baked in
    assert ft2_train.FASTER_WHISPER == "faster-whisper==1.2.1"
    recorded = tmp_path / "pr011.json"
    ft2_train.bench_guard("", 8, "", "abc1234", recorded)                    # base alone, any way you like
    ft2_train.bench_guard("", 0, "", "abc1234-dirty", recorded)
    with pytest.raises(SystemExit, match="PR-011"):
        ft2_train.bench_guard("", 0, "export-x", "abc1234", recorded)         # the LoRA needs the confirm
    with pytest.raises(SystemExit, match="limit"):
        ft2_train.bench_guard("PR-011", 8, "export-x", "abc1234", recorded)
    with pytest.raises(SystemExit, match="clean"):
        ft2_train.bench_guard("PR-011", 0, "export-x", "abc1234-dirty", recorded)
    ft2_train.bench_guard("PR-011", 0, "export-x", "abc1234", recorded)     # the registered run
    recorded.write_text("{}")
    with pytest.raises(SystemExit, match="deviation"):
        ft2_train.bench_guard("PR-011", 0, "export-x", "abc1234", recorded)


def test_ft2_validations_are_read_from_what_the_module_writes(ft2_train, tmp_path):
    run = tmp_path / "runs" / "x"
    (run / "sub").mkdir(parents=True)
    for step in (150, 300):
        metrics = {"wer_raw": 0.1 * step / 150, "wer_normalized": 0.05, "filler_recall": 0.5,
                   "fillers_in_refs": 10, "n_clips": 8, "n_words": 100, "loss": 0.3}
        (run / "sub" / f"val-step-{step:06d}.jsonl").write_text(
            json.dumps({"metrics": metrics}) + "\n" + json.dumps({"ref": "a", "hyp": "a"}) + "\n", encoding="utf-8")
    rows = ft2_train._validations(run, root=str(tmp_path))
    assert [r["step"] for r in rows] == [150, 300] and rows[1]["wer_raw"] == pytest.approx(0.2)
    assert rows[0]["loss"] == 0.3
    assert rows[0]["file"] == "runs/x/sub/val-step-000150.jsonl"



@pytest.fixture(scope="module")
def ft3_train():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("ft3_train", ROOT / "modal_app" / "ft3_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ft3_image_pins_what_the_laptop_env_installs(ft3_train):
    """ft/requirements-ft3.txt is the laptop env; the image installs the same NeMo, with CUDA torch of the
    version the env's install command names."""
    req = (ROOT / "ft" / "requirements-ft3.txt").read_text(encoding="utf-8")
    assert ft3_train.NEMO in req and "torch==2.8.0" in req and ft3_train.TORCH == "torch==2.8.0"
    assert ft3_train.SPEECH_TAG == "v" + ft3_train.NEMO.split("==")[1]        # the scripts at the wheel's version


def test_the_ft3_stages_are_the_tutorials_recipe_with_caps(ft3_train):
    for o in ("+trainer.max_steps=1000", "model.optim.lr=2e-4", "model.train_ds.dataloader_params.batch_size=24",
              "model.n_speakers=1", "+model.text_tokenizer.add_blank_at=true"):
        assert o in ft3_train.FASTPITCH_OVERRIDES
    for o in ("model.max_steps=1000", "model.optim.lr=0.00001", "model.train_ds.dataloader_params.batch_size=32"):
        assert o in ft3_train.HIFIGAN_OVERRIDES
    assert not any(o.startswith("model.pitch_mean") or o.startswith("model.pitch_std") for o in ft3_train.FASTPITCH_OVERRIDES), \
        "the pitch statistics come from extract_sup_data.py on the speaker's own data, not from the tutorial's numbers"
    assert ft3_train.seconds_for(ft3_train.TRAIN_CAP_DOLLARS) <= 3600 and ft3_train.TRAIN_CAP_DOLLARS + ft3_train.VOCODER_CAP_DOLLARS <= 2.0
    for fn in (ft3_train.data_remote, ft3_train.train_remote, ft3_train.vocoder_remote):
        assert hasattr(fn, "remote")
    for ep in (ft3_train.data, ft3_train.train, ft3_train.vocoder):
        assert hasattr(ep, "info")
    assert ft3_train.SPEAKER == "9017" and ft3_train.DATA_URL.endswith("9017_5_mins.tar.gz")
