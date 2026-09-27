"""HiFi-GAN's training mels: the fine-tuned FastPitch's own spectrogram of every clip, with the
ground-truth alignment (the tutorial's step 'Finetuning HiFi-GAN'), so that the vocoder learns the
mels it will be given. Runs wherever NeMo does -- the laptop's CPU included, which is where it is
tested (the first Modal run of this loop failed on its first clip).

    records = mels.write(spec_model, records, mel_dir)      # adds "mel_filepath" to each record
"""
from __future__ import annotations

import pathlib

import numpy as np


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    """Mono float32 at the model's rate."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio


def prior_frames(spect_frames: int, spect_len: int) -> int:
    """The number of frames the alignment prior is built for: the spectrogram's SHAPE. NeMo 3.0.0's
    preprocessor returns one frame more than the length it states (90 and 89 for a 22,784-sample
    clip), and the aligner adds the prior to an attention map of the shape. The tutorial builds the
    prior from the stated length, which this version refuses: 'tensor a (90) must match tensor b (89)'.
    The training does as here: TTSDataset takes the prior's length from the mel's shape."""
    if not 0 <= spect_frames - spect_len <= 1:
        raise ValueError(f"spectrogram of {spect_frames} frames with a stated length of {spect_len}")
    return spect_frames


def mel_for(spec_model, interp, text: str, audio: np.ndarray):
    """One clip's mel, (80, frames), on the CPU; frames = the preprocessor's stated length."""
    import torch
    device = next(spec_model.parameters()).device
    wav = torch.from_numpy(audio).unsqueeze(0).to(device)
    wav_len = torch.tensor([wav.shape[1]], dtype=torch.long, device=device)
    with torch.no_grad():
        tokens = spec_model.parse(text)
        tokens_len = torch.tensor([tokens.shape[-1]], dtype=torch.long, device=device)
        spect, spect_len = spec_model.preprocessor(input_signal=wav, length=wav_len)
        frames = prior_frames(spect.shape[2], int(spect_len.item()))
        prior = torch.from_numpy(interp(frames, int(tokens_len.item()))).unsqueeze(0).to(device)
        mel = spec_model.forward(text=tokens, input_lens=tokens_len, spec=spect, mel_lens=spect_len, attn_prior=prior)[0]
    return mel[0].cpu().numpy()


def write(spec_model, records: list[dict], mel_dir: pathlib.Path) -> list[dict]:
    """A mel per record under mel_dir; the records back with "mel_filepath" and "mel_frames"."""
    from nemo.collections.tts.parts.utils.tts_dataset_utils import BetaBinomialInterpolator
    interp = BetaBinomialInterpolator()
    sample_rate = int(spec_model.cfg.sample_rate)
    mel_dir = pathlib.Path(mel_dir)
    mel_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, r in enumerate(records):
        mel = mel_for(spec_model, interp, r["text"], load_audio(r["audio_filepath"], sample_rate))
        path = mel_dir / f"mel_{i}.npy"
        np.save(path, mel)
        out.append({**r, "mel_filepath": str(path), "mel_frames": int(mel.shape[1])})
    return out
