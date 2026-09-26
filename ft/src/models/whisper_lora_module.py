"""
FT-2: a LoRA on Whisper, trained to write down what was said -- fillers, repetitions and all.

The base model is frozen; PEFT's LoRA sits on the attention's q and v projections (PEFT's own
Whisper recipe: rank 32, alpha 64, dropout 0.05), about 1.5% of whisper-small.en's parameters.
Validation generates, then scores with s2s-lab's WER (raw and normalized) and the filler recall
(components/transcripts.py). Every validation also saves the adapter to
<output_dir>/adapters/step-NNNNNN -- all of them are kept (FT-1's lesson), and each is a few MB.
Lightning's .ckpt files hold only the adapter's weights too (on_save_checkpoint), never the 1 GB
base.

`model_config_overrides` builds a randomly initialised model from the base's config with those
fields changed: for the tests, a Whisper a few layers wide that still uses the real tokenizer.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor

from src.models.components.transcripts import summarize


def build_base(model_name: str, model_config_overrides: dict | None = None) -> WhisperForConditionalGeneration:
    """The pretrained model, or (overrides given) a random one of that shape, for tests."""
    if model_config_overrides:
        config = WhisperConfig.from_pretrained(model_name, **model_config_overrides)
        model = WhisperForConditionalGeneration(config)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None            # the English-only models force nothing
    model.generation_config.forced_decoder_ids = None
    return model


class WhisperLoRAModule(LightningModule):
    def __init__(self, model_name: str = "openai/whisper-small.en", lora_r: int = 32, lora_alpha: int = 64,
                 lora_dropout: float = 0.05, target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
                 optimizer: Any = None, warmup_steps: int = 50, max_new_tokens: int = 200,
                 model_config_overrides: dict | None = None, compile: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.strict_loading = False                    # the checkpoints hold the adapter only
        base = build_base(model_name, model_config_overrides)
        self.net = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                                   target_modules=list(target_modules), bias="none"))
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.train_loss, self.val_loss = MeanMetric(), MeanMetric()
        self._val_rows: list[dict] = []

    # -- what is trained --------------------------------------------------------------------------
    def trainable_parameters(self) -> tuple[int, int, float]:
        """(trainable, total, fraction)."""
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.net.parameters())
        return trainable, total, trainable / total

    def on_fit_start(self) -> None:
        trainable, total, frac = self.trainable_parameters()
        self.print(f"LoRA: {trainable:,} trainable of {total:,} parameters ({100 * frac:.2f}%)")
        if self.trainer.datamodule is not None and hasattr(self.trainer.datamodule, "describe"):
            self.print(self.trainer.datamodule.describe())

    # -- steps ------------------------------------------------------------------------------------
    def forward(self, input_features: torch.Tensor, labels: torch.Tensor):
        return self.net(input_features=input_features, labels=labels)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self(batch["input_features"], batch["labels"]).loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def transcribe(self, input_features: torch.Tensor) -> list[str]:
        """Greedy, up to max_new_tokens, special tokens stripped."""
        ids = self.net.generate(input_features=input_features, max_new_tokens=self.hparams.max_new_tokens,
                                do_sample=False, num_beams=1)
        return [t.strip() for t in self.processor.tokenizer.batch_decode(ids, skip_special_tokens=True)]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss = self(batch["input_features"], batch["labels"]).loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        hyps = self.transcribe(batch["input_features"])
        self._val_rows += [{"source": s, "ref": r, "hyp": h} for s, r, h in zip(batch["source"], batch["text"], hyps)]

    def on_validation_epoch_start(self) -> None:
        self._val_rows = []

    def on_validation_epoch_end(self) -> None:
        rows = self._val_rows
        if not rows:
            return
        metrics = summarize((r["ref"], r["hyp"]) for r in rows)
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=name in ("wer_raw", "filler_recall"), sync_dist=True)
        metrics["loss"] = float(self.val_loss.compute())     # logged already (validation_step); kept with the rows
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            out = self.output_dir
            out.mkdir(parents=True, exist_ok=True)
            tag = f"step-{self.global_step:06d}"
            with open(out / f"val-{tag}.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps({"metrics": metrics}) + "\n")
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            self.save_adapter(out / "adapters" / tag)

    # -- the adapter on disk ----------------------------------------------------------------------
    @property
    def output_dir(self) -> pathlib.Path:
        return pathlib.Path(self.trainer.log_dir or self.trainer.default_root_dir)

    def save_adapter(self, path: pathlib.Path | str) -> pathlib.Path:
        """PEFT's format: adapter_model.safetensors + adapter_config.json, a few MB."""
        path = pathlib.Path(path)
        self.net.save_pretrained(str(path))
        return path

    def load_adapter(self, path: pathlib.Path | str) -> None:
        """The adapter's weights into this module's LoRA (same rank and targets)."""
        from safetensors.torch import load_file
        state = load_file(str(pathlib.Path(path) / "adapter_model.safetensors"))
        result = set_peft_model_state_dict(self.net, state)
        unexpected = getattr(result, "unexpected_keys", [])
        if unexpected:
            raise ValueError(f"{path}: {len(unexpected)} adapter keys do not fit this model, e.g. {unexpected[0]}")

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Lightning's .ckpt keeps the trainable weights only: the base is the published one."""
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k in trainable}

    # -- optimisation -----------------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        from transformers import get_linear_schedule_with_warmup
        params = [p for p in self.net.parameters() if p.requires_grad]
        optimizer = (self.hparams.optimizer(params=params) if self.hparams.optimizer is not None
                     else torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01))
        total = int(self.trainer.estimated_stepping_batches)
        scheduler = get_linear_schedule_with_warmup(optimizer, min(self.hparams.warmup_steps, total), total)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
