"""Train: the template's entry point, with one addition -- `adapter_dir` warm-starts the LoRA.

    uv run src/train.py experiment=ft2_whisper_verbatim trainer=gpu
    uv run src/train.py experiment=ft2_whisper_verbatim debug=fast         # one batch, CPU
"""

from typing import Any

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import Logger

from src.utils import (RankedLogger, extras, get_metric_value, instantiate_callbacks, instantiate_loggers,
                       log_hyperparameters, task_wrapper)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    if cfg.get("adapter_dir"):
        log.info(f"Loading the adapter from {cfg.adapter_dir}")
        model.load_adapter(cfg.adapter_dir)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {"cfg": cfg, "datamodule": datamodule, "model": model, "callbacks": callbacks,
                   "logger": logger, "trainer": trainer}
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    metric_dict = dict(trainer.callback_metrics)
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)
    metric_dict, _ = train(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
