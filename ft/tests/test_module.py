"""The LoRA module: what trains, what a checkpoint holds, the metrics, the adapter round trip, the export."""

import math

import pytest
import torch
from safetensors.torch import load_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from s2slab import wer
from src import export
from src.models.components import transcripts
from src.models.whisper_lora_module import WhisperLoRAModule
from tests.conftest import TINY, TINY_OVERRIDES


def test_only_the_lora_trains_and_it_is_a_small_fraction(module):
    trainable, total, frac = module.trainable_parameters()
    assert 0 < frac < 0.2 and trainable < total
    names = {n for n, p in module.net.named_parameters() if p.requires_grad}
    assert names and all("lora_" in n for n in names)
    assert not any(p.requires_grad for n, p in module.net.named_parameters() if "lora_" not in n)


def test_a_training_step_moves_only_the_adapter(module, datamodule):
    batch = next(iter(datamodule.train_dataloader()))
    loss = module(batch["input_features"], batch["labels"]).loss
    assert torch.isfinite(loss)
    loss.backward()
    with_grad = {n for n, p in module.net.named_parameters() if p.grad is not None}
    assert with_grad and all("lora_" in n for n in with_grad)


def test_the_checkpoint_holds_the_adapter_only(module):
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    keys = list(checkpoint["state_dict"])
    assert keys and all("lora_" in k for k in keys)
    assert sum(v.numel() for v in checkpoint["state_dict"].values()) == module.trainable_parameters()[0]


def test_the_adapter_round_trips_into_a_fresh_module(module, tmp_path):
    torch.manual_seed(0)
    for p in module.net.parameters():                     # a random adapter that is not zero
        if p.requires_grad:
            p.data.normal_(0, 0.1)
    adapter = module.save_adapter(tmp_path / "adapter")
    assert (adapter / "adapter_model.safetensors").exists() and (adapter / "adapter_config.json").exists()
    trainable = {n: p.detach().clone() for n, p in module.net.named_parameters() if p.requires_grad}
    assert len(load_file(str(adapter / "adapter_model.safetensors"))) == len(trainable)

    fresh = WhisperLoRAModule(model_name=TINY, model_config_overrides=TINY_OVERRIDES, lora_r=4, lora_alpha=8)
    before = {n: p.detach().clone() for n, p in fresh.net.named_parameters() if p.requires_grad}
    fresh.load_adapter(adapter)
    after = dict(fresh.net.named_parameters())
    assert all(torch.equal(after[n], trainable[n]) for n in trainable)
    assert any(not torch.equal(before[n], trainable[n]) for n in trainable)     # it really changed something
    wrong = WhisperLoRAModule(model_name=TINY, model_config_overrides=TINY_OVERRIDES, lora_r=8, lora_alpha=8)
    with pytest.raises((ValueError, RuntimeError)):
        wrong.load_adapter(adapter)                                              # rank 4 weights into rank 8


def _saved_random_base(tmp_path):
    """A random tiny Whisper saved to disk, so a module can be built ON it and a merge compared TO it."""
    from src.models.whisper_lora_module import build_base
    base_dir = tmp_path / "base"
    build_base(TINY, TINY_OVERRIDES).save_pretrained(str(base_dir))
    WhisperProcessor.from_pretrained(TINY).save_pretrained(str(base_dir))
    return base_dir


def test_merge_equals_base_plus_adapter_on_one_base(datamodule, tmp_path):
    base_dir = _saved_random_base(tmp_path)
    module = WhisperLoRAModule(model_name=str(base_dir), lora_r=4, lora_alpha=8)
    torch.manual_seed(1)
    for p in module.net.parameters():
        if p.requires_grad:
            p.data.normal_(0, 0.1)
    module.eval()
    batch = next(iter(datamodule.val_dataloader()))
    with torch.no_grad():
        want = module(batch["input_features"], batch["labels"]).logits
    adapter = module.save_adapter(tmp_path / "adapter")
    merged_dir = export.merge(adapter, tmp_path / "export", model_name=str(base_dir))
    merged = WhisperForConditionalGeneration.from_pretrained(str(merged_dir)).eval()
    assert not any("lora" in k for k in merged.state_dict())
    with torch.no_grad():
        got = merged(input_features=batch["input_features"], labels=batch["labels"]).logits
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)
    base = WhisperForConditionalGeneration.from_pretrained(str(base_dir)).eval()
    with torch.no_grad():
        plain = base(input_features=batch["input_features"], labels=batch["labels"]).logits
    assert not torch.allclose(plain, want, atol=1e-4)                      # the adapter did change the model


def test_the_export_converts_to_ctranslate2(tmp_path):
    base_dir = _saved_random_base(tmp_path)
    module = WhisperLoRAModule(model_name=str(base_dir), lora_r=4, lora_alpha=8)
    adapter = module.save_adapter(tmp_path / "adapter")
    hf = export.merge(adapter, tmp_path / "export", model_name=str(base_dir))
    ct2 = export.to_ctranslate2(hf, tmp_path / "export", quantization="float32")
    assert (ct2 / "model.bin").exists() and (ct2 / "tokenizer.json").exists()
    assert (ct2 / "preprocessor_config.json").exists()


def test_transcribe_returns_one_string_per_clip(module, datamodule):
    module.eval()
    batch = next(iter(datamodule.val_dataloader()))
    with torch.no_grad():
        hyps = module.transcribe(batch["input_features"])
    assert len(hyps) == batch["input_features"].shape[0] and all(isinstance(h, str) for h in hyps)


def test_the_validation_numbers_are_s2slabs_wer_plus_the_filler_recall():
    pairs = [("uh, the, the cat sat.", "the cat sat"),                  # raw: 2 of 5 words deleted; normalized: "uh" gone
             ("Um, I think so.", "um i think so"),                      # raw 0; normalized 0 (both drop um)
             ("Mm, that is it.", "That is it.")]
    m = transcripts.summarize(pairs)
    assert m["wer_raw"] == pytest.approx(wer.corpus_wer(pairs, wer.light).rate)
    assert m["wer_normalized"] == pytest.approx(wer.corpus_wer(pairs, wer.standard).rate)
    assert m["wer_raw"] == pytest.approx(3 / 13) and m["wer_normalized"] == pytest.approx(1 / 10)
    assert m["fillers_in_refs"] == 3 and m["filler_recall"] == pytest.approx(1 / 3)
    assert m["n_clips"] == 3 and m["n_words"] == 13
    assert math.isnan(transcripts.summarize([("no fillers here", "no fillers here")])["filler_recall"])
    assert transcripts.filler_recall([("uh uh", "uh")]) == (1, 2)       # word for word, up to the reference's count
    assert transcripts.filler_recall([("uh", "um")]) == (0, 1)          # a different filler does not count
