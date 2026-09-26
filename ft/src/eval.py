"""The dev look: one adapter (or the base model alone) over the WHOLE validation split.

    uv run src/eval.py adapter_dir=null                                    # base whisper-small.en
    uv run src/eval.py adapter_dir=logs/train/runs/.../adapters/step-000900

Validation, not test: the split is DisfluencySpeech's validation split (250 clips of the same
speaker); the registered test is s2s-lab's bench on the merged model. Writes metrics.json and the
per-clip val-*.jsonl into the run's output directory.
"""

import json
import pathlib
from typing import Any

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import Logger

from src.utils import RankedLogger, extras, instantiate_loggers, log_hyperparameters, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    if cfg.get("adapter_dir"):
        log.info(f"Loading the adapter from {cfg.adapter_dir}")
        model.load_adapter(cfg.adapter_dir)
    else:
        log.info("No adapter: the base model, the comparator")

    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    object_dict = {"cfg": cfg, "datamodule": datamodule, "model": model, "logger": logger, "trainer": trainer}
    if logger:
        log_hyperparameters(object_dict)

    log.info("Validating")
    trainer.validate(model=model, datamodule=datamodule)
    metric_dict = {k: float(v) for k, v in trainer.callback_metrics.items()}
    out = pathlib.Path(cfg.paths.output_dir) / "metrics.json"
    out.write_text(json.dumps({"adapter_dir": cfg.get("adapter_dir"), "model": cfg.model.model_name,
                               "data": datamodule.describe(), **metric_dict}, indent=2), encoding="utf-8")
    log.info(f"{metric_dict}\n-> {out}")
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
