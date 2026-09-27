"""The configs compose and the whole pipeline runs one batch on the CPU with the fake data."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.train import train
from tests.conftest import _compose


def test_the_experiment_config_is_the_registered_one():
    cfg = _compose("train.yaml", ["experiment=ft2_whisper_verbatim"])
    assert cfg.model.model_name == "openai/whisper-small.en" and cfg.data.text_column == "transcript_a"
    assert (cfg.model.lora_r, cfg.model.lora_alpha, list(cfg.model.target_modules)) == (32, 64, ["q_proj", "v_proj"])
    assert cfg.trainer.max_steps == 900 and cfg.trainer.val_check_interval == 150 and cfg.trainer.accelerator == "gpu"
    assert cfg.data.batch_size == 16 and cfg.data.val_limit == 64
    assert cfg.callbacks.model_checkpoint.save_top_k == -1 and "early_stopping" not in cfg.callbacks
    assert cfg.test is False and cfg.data.model_name == cfg.model.model_name
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()


def test_the_eval_config_reads_the_whole_validation_split():
    cfg = _compose("eval.yaml", [])
    assert cfg.data.val_limit is None and cfg.adapter_dir is None
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()


def test_one_batch_end_to_end(cfg_train: DictConfig, tmp_path):
    HydraConfig().set_config(cfg_train)
    hydra.utils.instantiate(cfg_train.data)
    metrics, objects = train(cfg_train)
    assert "train/loss" in metrics and "val/wer_raw" in metrics and "val/filler_recall" in metrics
    assert 0 <= float(metrics["val/wer_raw"])
    adapters = list((tmp_path).rglob("adapters/step-*"))
    assert adapters, "every validation saves its adapter"
    assert all(a.parent == tmp_path / "adapters" for a in adapters), "at the run's root, not under the logger's folder"
    assert (adapters[0] / "adapter_model.safetensors").exists()
    jsonl = list(tmp_path.rglob("val-step-*.jsonl"))
    assert jsonl
    import json
    first = json.loads(jsonl[0].read_text(encoding="utf-8").splitlines()[0])["metrics"]
    assert {"loss", "wer_raw", "wer_normalized", "filler_recall", "n_clips"} <= set(first) and first["loss"] > 0
