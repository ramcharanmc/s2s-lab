"""
VAP over prepared turns (PR-014). Run in VAP's own environment, not s2slab's:

    ~/.venvs/vap/bin/python src/s2slab/vaprun.py results/pr014/dev              p_now for every turn -> vap.json
    ~/.venvs/vap/bin/python src/s2slab/vaprun.py results/pr014/dev --causality  how far VAP reads ahead

STANDALONE ON PURPOSE. It imports numpy, soundfile, torch 2.4 and `vap` (ErikEkstedt/VoiceActivityProjection
at VAP_COMMIT, installed in ~/.venvs/vap), and nothing from s2slab. The turns it reads were written by
`s2slab.vapturns export`, with each WAV's SHA-256; a WAV that does not match is refused.

THE TWO CHANNELS. The caller is on the first channel. The second is the system: when the export names a
prompt (PR-014, Deviations), the system's spoken prompt, a short gap, then digital silence, while the
caller's channel holds the caller's own floor noise until the turn begins. Without a prompt, the second
channel is silent throughout, as the repo does for a mono file.

WHAT IT WRITES. For each turn, VAP's `p_now` for the second channel: the probability that the system
speaks next within about 0.6 s. One value per 50 Hz frame (20 ms), the frame VAP computed from the audio
up to that frame's end, counted from the start of the caller's turn: the prompt's frames are dropped.

THE CAUSALITY CHECK (PR-014, V3). The same turn, run whole and cut short at several points in the
caller's turn: a model that reads no audio after time t gives the same value at every frame up to t
either way. The frames before each cut point that differ are how far it reads ahead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

VAP_COMMIT = "f39a78b"
STATE_DICT = "example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"
SR = 16_000
FRAME_SAMPLES = 320                     # 20 ms at 16 kHz: VAP's 50 Hz frame, and the VAD's
TOL = 1e-4                              # two values closer than this are the same value


def sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_ahead_frames(whole: list[float], cut: list[float]) -> int:
    """How many frames at the end of `cut` (the turn run only up to a cut point) differ from the same
    frames of the turn run whole. 0 means the model read nothing after the cut."""
    n = min(len(cut), len(whole))
    k = 0
    while k < n and abs(cut[n - 1 - k] - whole[n - 1 - k]) > TOL:
        k += 1
    return k


def floor_noise(x, n: int, key: str):
    """n samples of Gaussian noise at x's floor (its 10th-percentile 20 ms frame RMS), seeded by key:
    the caller's room, before the caller speaks."""
    import numpy as np
    frames = x[: len(x) // FRAME_SAMPLES * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES).astype("float64")
    level = float(np.percentile(np.sqrt((frames ** 2).mean(axis=1) + 1e-12), 10)) if len(frames) else 0.0
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(0.0, level, n).astype("float32")


def lead(prompt, gap_s: float):
    """The system's channel before the caller's turn: the prompt, then silence for gap_s, padded to a whole
    number of 20 ms frames so the caller's turn starts on a frame boundary. None: no lead."""
    import numpy as np
    if prompt is None:
        return None
    x = np.concatenate([prompt, np.zeros(int(round(gap_s * SR)), dtype="float32")])
    return np.concatenate([x, np.zeros(-len(x) % FRAME_SAMPLES, dtype="float32")])


def stereo(x, lead_audio, key: str, floor_from=None):
    """(2, n): the caller on the first channel, the system on the second. The caller's noise before the
    turn is at the floor of `floor_from` (default x): the causality check cuts x short, and its lead-in
    must stay the same as the whole turn's."""
    import numpy as np
    if lead_audio is None:
        return np.stack([x, np.zeros_like(x)])
    ch1 = np.concatenate([floor_noise(x if floor_from is None else floor_from, len(lead_audio), key), x])
    ch2 = np.concatenate([lead_audio, np.zeros_like(x)])
    return np.stack([ch1, ch2])


def load_model(vap_root: pathlib.Path):
    import torch
    from vap.model import VapConfig, VapGPT
    m = VapGPT(VapConfig())
    m.load_state_dict(torch.load(vap_root / STATE_DICT, map_location="cpu"))
    return m.eval()


def p_system(model, audio2) -> list[float]:
    """p_now for the second channel, one value a 20 ms frame, for a (2, n) waveform.

    Computed as the repo's VapGPT.probs computes p_now (the softmax of the model's logits, then
    objective.probs_next_speaker_aggregate over bins 0 to 1), without the rest of probs(): at f39a78b it
    overwrites its `vad` argument with the model's own estimate and then always computes a training
    loss, which fails on audio shorter than the 2 s projection window."""
    import torch
    w = torch.from_numpy(audio2).float()[None]
    with torch.no_grad():
        probs = model(w)["logits"].softmax(dim=-1)
        p_now = model.objective.probs_next_speaker_aggregate(probs, from_bin=0, to_bin=1)
    return [float(v) for v in p_now[0, :, 1]]


def _meta(d: pathlib.Path):
    import soundfile as sf
    meta = json.loads((d / "turns.json").read_text(encoding="utf-8"))
    pr = meta.get("prompt")
    prompt = None
    if pr:
        path = (d / pr["file"]).resolve()
        if sha256(path) != pr["sha256"]:
            raise SystemExit(f"{path} is not the prompt the export named")
        prompt, sr = sf.read(str(path), dtype="float32")
        if sr != SR:
            raise SystemExit(f"{path}: {sr} Hz, not {SR}")
    return meta, lead(prompt, pr["gap_s"] if pr else 0.0)


def _turns(d: pathlib.Path, meta):
    import soundfile as sf
    for tid, t in meta["turns"].items():
        wav = d / f"{tid}.wav"
        if sha256(wav) != t["wav_sha256"]:
            raise SystemExit(f"{wav} is not the file vapturns exported")
        x, sr = sf.read(str(wav), dtype="float32")
        if sr != SR:
            raise SystemExit(f"{wav}: {sr} Hz, not {SR}")
        yield tid, x


def main(argv=None) -> None:
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", type=pathlib.Path)
    ap.add_argument("--vap", type=pathlib.Path, default=pathlib.Path.home() / "vap")
    ap.add_argument("--causality", action="store_true")
    a = ap.parse_args(argv)
    commit = subprocess.run(["git", "-C", str(a.vap), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    if commit != VAP_COMMIT:
        raise SystemExit(f"VAP at {a.vap} is {commit or 'not a git checkout'}, not {VAP_COMMIT}")
    torch.manual_seed(0)
    model = load_model(a.vap)
    meta, lead_audio = _meta(a.dir)
    skip = 0 if lead_audio is None else len(lead_audio) // FRAME_SAMPLES
    info = {"vap_commit": commit, "state_dict": STATE_DICT, "state_dict_sha256": sha256(a.vap / STATE_DICT),
            "torch": torch.__version__, "frame_ms": 20, "channel": "second (the system): p_now",
            "prompt_sha256": (meta.get("prompt") or {}).get("sha256"), "lead_frames": skip}
    if a.causality:
        worst, rows = 0, []
        for tid, x in _turns(a.dir, meta):
            whole = p_system(model, stereo(x, lead_audio, tid))[skip:]
            for frac in (0.25, 0.5, 0.75):
                n = int(len(x) * frac) // FRAME_SAMPLES * FRAME_SAMPLES
                k = read_ahead_frames(whole, p_system(model, stereo(x[:n], lead_audio, tid, floor_from=x))[skip:])
                rows.append((tid, frac, k))
                worst = max(worst, k)
        print(json.dumps({**info, "cuts": len(rows), "worst_read_ahead_frames": worst,
                          "worst_read_ahead_ms": worst * 20,
                          "cuts_with_any_read_ahead": sum(k > 0 for *_, k in rows)}, indent=1))
        return
    out = {**info, "p_system": {}, "n_frames": {}}
    for tid, x in _turns(a.dir, meta):
        p = p_system(model, stereo(x, lead_audio, tid))[skip:]
        out["p_system"][tid] = [round(v, 5) for v in p]
        out["n_frames"][tid] = len(p)
    path = a.dir / "vap.json"
    path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{len(out['p_system'])} turns -> {path} (VAP {commit}, state dict {info['state_dict_sha256'][:12]}, "
          f"prompt {'none' if not skip else info['prompt_sha256'][:12] + f', {skip} frames dropped'})")


if __name__ == "__main__":
    main()
