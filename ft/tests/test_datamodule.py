"""The datamodule: Whisper's batch format, what is dropped, the held-out split, the eval set's pin."""

import pytest
import torch

from s2slab import evalset
from src.data import disfluencyspeech_datamodule as dsm
from src.data.disfluencyspeech_datamodule import DisfluencySpeechDataModule


def test_the_revision_is_the_eval_sets_and_the_test_split_is_not_listed():
    spec = next(s for s in evalset.SOURCES.values() if s.repo == dsm.REPO)
    assert dsm.REVISION == spec.revision
    assert set(dsm.FILES) == {"train", "validation"} and all("test" not in f for fs in dsm.FILES.values() for f in fs)


def test_a_batch_is_in_whispers_training_format(datamodule):
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["input_features"].shape == (2, 80, 3000) and batch["input_features"].dtype == torch.float32
    labels = batch["labels"]
    tok = datamodule.processor.tokenizer
    sot, eot = tok.convert_tokens_to_ids("<|startoftranscript|>"), tok.eos_token_id
    assert labels.dtype == torch.long and labels.ndim == 2
    assert (labels[:, 0] != sot).all() and (labels[:, 0] == tok.convert_tokens_to_ids("<|notimestamps|>")).all()
    for row, text in zip(labels, batch["text"]):
        real = row[row != -100]
        assert real[-1].item() == eot
        assert tok.decode(real, skip_special_tokens=True).strip() == text
    assert len(batch["source"]) == 2 and all(":" in s for s in batch["source"])


def test_what_whisper_cannot_train_on_is_dropped_and_counted(datamodule):
    assert len(datamodule.data_train) == 8                       # 10 written: the 31 s clip and the empty text go
    assert datamodule.dropped["train"] == {"too_long": 1, "no_text": 1, "too_many_tokens": 0}
    assert "train 8 clips, val 3" in datamodule.describe()


def test_every_clip_comes_out_as_16k_mono(datamodule):
    stereo = next(it for it in datamodule.data_train if it["source"].endswith(":3"))     # the 22.05 kHz stereo one
    assert stereo["audio"].ndim == 1 and abs(len(stereo["audio"]) - 0.9 * 16_000) < 16    # 0.9 s at 16 kHz


def test_val_limit_takes_the_same_clips_every_time(fake_data):
    a = DisfluencySpeechDataModule(data_dir=str(fake_data), model_name=dsm_tiny(), val_limit=3); a.setup()
    b = DisfluencySpeechDataModule(data_dir=str(fake_data), model_name=dsm_tiny(), val_limit=3); b.setup()
    full = DisfluencySpeechDataModule(data_dir=str(fake_data), model_name=dsm_tiny(), val_limit=None); full.setup()
    assert [r["source"] for r in a.data_val.rows] == [r["source"] for r in b.data_val.rows]
    assert len(a.data_val) == 3 and len(full.data_val) == 4
    with pytest.raises(ValueError):
        DisfluencySpeechDataModule(val_limit=0)
    with pytest.raises(ValueError):
        DisfluencySpeechDataModule(text_column="transcript_z")


def test_there_is_no_test_loader(datamodule):
    with pytest.raises(RuntimeError, match="PR-011"):
        datamodule.test_dataloader()


def test_setup_refuses_to_run_before_the_files_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        DisfluencySpeechDataModule(data_dir=str(tmp_path), model_name=dsm_tiny()).setup()


def dsm_tiny() -> str:
    from tests.conftest import TINY
    return TINY
