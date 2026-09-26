"""
Fixtures: a fake DisfluencySpeech (four parquet files of synthetic audio with made-up verbatim
transcripts, the dataset's exact schema) and a tiny random Whisper that uses the real tokenizer of
openai/whisper-tiny.en. Nothing here needs a GPU or the real data; the tokenizer files (a few MB)
are fetched from the Hub on the first run and cached.
"""

from __future__ import annotations

import io
import pathlib
from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.data.disfluencyspeech_datamodule import FILES, DisfluencySpeechDataModule
from src.models.whisper_lora_module import WhisperLoRAModule

TINY = "openai/whisper-tiny.en"
# A Whisper a few units wide, built from tiny.en's config (so the vocabulary and the special tokens
# are the real ones) with random weights: tests run in seconds on a CPU.
TINY_OVERRIDES = {"d_model": 32, "encoder_layers": 1, "decoder_layers": 1, "encoder_attention_heads": 2,
                  "decoder_attention_heads": 2, "encoder_ffn_dim": 64, "decoder_ffn_dim": 64}

TRAIN_TEXTS = ["You know, it's someone who had, uh, killed two teenage boys.",
               "Did you see the, the C N N, the, the dog show, uh, we don't have cable.",
               "Um, I think the attitude is a lot better now.",
               "Well, hmm, we went there last, last summer.",
               "Right.", "So, uh, what do you, what do you do for a living?",
               "I have three, uh, three cats and a dog.", "Mm, that is about it, I guess."]
VAL_TEXTS = ["Uh, no, not really.", "We, we moved here in, um, ninety two.",
             "That's, that's what I heard.", "Yeah."]


def _wav_bytes(seconds: float, sr: int = 16_000, channels: int = 1, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    x = 0.2 * np.sin(2 * np.pi * (120 + 40 * seed) * t) + 0.02 * rng.standard_normal(t.size)
    x = np.stack([x] * channels, axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _table(rows: list[tuple[bytes, str]]) -> pa.Table:
    audio = pa.array([{"bytes": b, "path": f"clip{i}.wav"} for i, (b, _) in enumerate(rows)],
                     type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    texts = [t for _, t in rows]
    return pa.table({"audio": audio, "transcript_annotated": texts, "transcript_a": texts,
                     "transcript_b": texts, "transcript_c": [t.replace("uh, ", "").replace("um, ", "") for t in texts]})


@pytest.fixture(scope="session")
def fake_data(tmp_path_factory) -> pathlib.Path:
    """The dataset's layout: three train files, one validation file. Train holds 8 usable clips
    (one of them 22.05 kHz stereo) plus a 31-second clip and an empty transcript, both to be dropped."""
    root = tmp_path_factory.mktemp("disfluencyspeech")
    train = [(_wav_bytes(0.6 + 0.1 * i, seed=i), t) for i, t in enumerate(TRAIN_TEXTS)]
    train[3] = (_wav_bytes(0.9, sr=22_050, channels=2, seed=3), TRAIN_TEXTS[3])
    train += [(_wav_bytes(31.0, seed=90), "This clip is longer than Whisper's window."),
              (_wav_bytes(0.5, seed=91), "")]
    thirds = [train[0:4], train[4:8], train[8:]]
    for path, rows in zip(FILES["train"], thirds):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_table(rows), root / path)
    pq.write_table(_table([(_wav_bytes(0.7 + 0.1 * i, seed=20 + i), t) for i, t in enumerate(VAL_TEXTS)]),
                   root / FILES["validation"][0])
    return root


@pytest.fixture
def datamodule(fake_data) -> DisfluencySpeechDataModule:
    dm = DisfluencySpeechDataModule(data_dir=str(fake_data), model_name=TINY, batch_size=2, val_limit=3)
    dm.prepare_data()          # every file exists: nothing is fetched
    dm.setup()
    return dm


@pytest.fixture
def module() -> WhisperLoRAModule:
    return WhisperLoRAModule(model_name=TINY, model_config_overrides=TINY_OVERRIDES, lora_r=4, lora_alpha=8,
                             max_new_tokens=6)


def _compose(name: str, overrides: list[str]) -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        return compose(config_name=name, return_hydra_config=True, overrides=overrides)


@pytest.fixture
def cfg_train(fake_data, tmp_path) -> Iterator[DictConfig]:
    """The experiment config, pointed at the fake data and the tiny model, one batch on the CPU."""
    tiny = ",".join(f"{k}:{v}" for k, v in TINY_OVERRIDES.items())
    cfg = _compose("train.yaml", ["experiment=ft2_whisper_verbatim", "trainer=cpu", "debug=fast",
                                  f"data.data_dir={fake_data.as_posix()}", "data.num_workers=0",
                                  "data.persistent_workers=False", "data.pin_memory=False", "data.batch_size=2",
                                  "data.val_limit=3", f"model.model_name={TINY}",
                                  f"model.model_config_overrides={{{tiny}}}", "model.lora_r=4", "model.lora_alpha=8",
                                  "model.max_new_tokens=6", f"paths.root_dir={tmp_path.as_posix()}"])
    with open_dict(cfg):
        cfg.paths.output_dir = tmp_path.as_posix()
        cfg.paths.log_dir = tmp_path.as_posix()
    yield cfg
    GlobalHydra.instance().clear()
