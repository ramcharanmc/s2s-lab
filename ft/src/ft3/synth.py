"""
FT-3's renderer: NeMo FastPitch + HiFi-GAN with the severity targets as inputs (src.ft3.control).

    PYTHONPATH=ft:src python -m src.ft3.synth "The train leaves at nine. I bought a ticket." \
        --condition depression --score 15 --out results/ft3/dev
    -> results/ft3/dev/<name>-neutral.wav, <name>-depression-15.wav, and the four measures against
       their targets (s2slab.severity.achieved / misses), printed

Needs the FT-3 environment (ft/requirements-ft3.txt): nemo-toolkit pins hydra-core, omegaconf
and the `lightning` package, so it does not share ft/'s env. CPU is enough for inference.
Models: `tts_en_fastpitch` (LJSpeech, CC-BY-4.0) and `tts_en_hifigan`, from NGC on first use,
or a local .nemo path for a fine-tuned voice (FT-3's training, later).
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import NamedTuple

import numpy as np

from s2slab import severity
from src.ft3 import control


class Voice(NamedTuple):
    fastpitch: object
    vocoder: object
    sr: int
    hop_s: float
    pitch_mean: float
    pitch_std: float
    pause_ids: set
    name: str


def load(fastpitch: str = "tts_en_fastpitch", hifigan: str = "tts_en_hifigan", device: str = "cpu") -> Voice:
    """Both models in eval mode on `device`; `fastpitch` / `hifigan` are NGC names or .nemo paths."""
    import torch
    from nemo.collections.tts.models import FastPitchModel, HifiGanModel
    fp = (FastPitchModel.restore_from(fastpitch) if fastpitch.endswith(".nemo") else FastPitchModel.from_pretrained(fastpitch))
    voc = (HifiGanModel.restore_from(hifigan) if hifigan.endswith(".nemo") else HifiGanModel.from_pretrained(hifigan))
    fp, voc = fp.to(device).eval(), voc.to(device).eval()
    torch.set_grad_enabled(False)
    cfg = fp.cfg
    sr = int(cfg.sample_rate)
    hop = int(cfg.preprocessor.n_window_stride)
    tok = fp.vocab
    pause_ids = control.pause_token_ids(list(tok.tokens), list(tok.PUNCT_LIST), space=" ",
                                        pad=getattr(tok, "pad", None), blank=getattr(tok, "blank", None))
    return Voice(fp, voc, sr, hop / sr, float(cfg.pitch_mean), float(cfg.pitch_std), pause_ids,
                 pathlib.Path(fastpitch).stem if fastpitch.endswith(".nemo") else fastpitch)


class Predicted(NamedTuple):
    tokens: object          # (1, T) long tensor
    durs: np.ndarray        # the model's predicted frames per token
    pitch: np.ndarray       # the model's predicted pitch per token, normalized units
    is_pause: np.ndarray


def predict(v: Voice, text: str) -> Predicted:
    """The model's own durations and pitch for the text: the neutral plan's inputs."""
    tokens = v.fastpitch.parse(text)
    spect, dec_lens, durs, log_durs, pitch, *_ = v.fastpitch.fastpitch.infer(text=tokens)
    ids = tokens[0].cpu().numpy()
    return Predicted(tokens, durs[0].cpu().numpy().astype(int), pitch[0].cpu().numpy().astype(np.float32),
                     np.array([int(i) in v.pause_ids for i in ids]))


def render(v: Voice, pred: Predicted, t: severity.Targets | None) -> tuple[np.ndarray, control.Plan]:
    """One render through forward() with the plan's durations and pitch; the audio at v.sr."""
    import torch
    p = control.plan(pred.durs, pred.pitch, pred.is_pause, v.hop_s, v.pitch_std, t)
    durs = torch.tensor(p.durs, dtype=torch.long, device=pred.tokens.device).unsqueeze(0)
    pitch = torch.tensor(p.pitch, dtype=torch.float32, device=pred.tokens.device).unsqueeze(0)
    spect, *_ = v.fastpitch(text=pred.tokens, durs=durs, pitch=pitch, pace=1.0)
    audio = v.vocoder.convert_spectrogram_to_audio(spec=spect)
    return audio[0].cpu().numpy().astype(np.float32), p


def measure(v: Voice, text: str, t: severity.Targets):
    """Neutral and controlled renders of one sentence, and severity's four measures of the pair."""
    pred = predict(v, text)
    neutral, _ = render(v, pred, None)
    audio, p = render(v, pred, t)
    got = severity.achieved(neutral, control.as_render(audio, p), v.sr)
    return neutral, audio, p, got


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("--condition", choices=sorted(severity.CONDITIONS), default="depression")
    ap.add_argument("--score", type=float, default=15)
    ap.add_argument("--fastpitch", default="tts_en_fastpitch")
    ap.add_argument("--hifigan", default="tts_en_hifigan")
    ap.add_argument("--out", default="results/ft3/dev")
    args = ap.parse_args(argv)
    t0 = time.time()
    v = load(args.fastpitch, args.hifigan)
    print(f"loaded {v.name}: {v.sr} Hz, hop {1000 * v.hop_s:.2f} ms, pitch mean {v.pitch_mean:.1f} Hz, std {v.pitch_std:.1f} Hz, "
          f"{len(v.pause_ids)} pause token ids, {time.time() - t0:.0f} s")
    t = severity.CONDITIONS[args.condition](args.score)
    neutral, audio, p, got = measure(v, args.text, t)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{v.name}-{abs(hash(args.text)) % 10_000:04d}"
    severity.write_synthetic(out / f"{stem}-neutral.wav", neutral, v.sr, v.name, args.text)
    severity.write_synthetic(out / f"{stem}-{t.condition}-{t.score:g}.wav", audio, v.sr, v.name, args.text, t)
    print(f"tokens {len(p.durs)}, pauses {len(p.runs)}: in {[(round(a, 2), round(b, 2)) for a, b in p.pauses_in]} "
          f"-> out {[(round(a, 2), round(b, 2)) for a, b in p.pauses_out]}")
    print(f"neutral {len(neutral) / v.sr:.2f} s, rendered {len(audio) / v.sr:.2f} s")
    for field in severity.FIELDS:
        print(f"  {field:14s} target {getattr(t, field):+7.2f}   got {getattr(got, field):+7.2f}")
    m = severity.misses(got, t)
    print("misses:", m if m else "none")
    print(f"-> {out}/{stem}-*.wav")


if __name__ == "__main__":
    main()
