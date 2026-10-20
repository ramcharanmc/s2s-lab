"""
An inaudible watermark on the project's synthetic speech (M43; v1 §9: "Synthesized 'patient' audio watermarked and
labelled synthetic in filenames and metadata"). The labels were already there (severity.write_synthetic, the
demo's WAV tags); this is the mark that survives when they are stripped.

AudioSeal (Meta; code and weights MIT; pinned in requirements/watermark.txt): a small network adds a signal shaped
to the speech's own envelope, and a matching detector says, short frame by short frame, whether it is there.
Since AudioSeal 0.2 nothing is resampled inside it. Marks are made at MARK_RATE, the rate Kokoro speaks at, and
read at MARK_RATE: a clip at any other rate is resampled to it before it is read. On the smoke test (M43, log row
3) a 24 kHz mark read at 24 kHz survived a trip through 16 kHz and through Opus; read at 16 kHz it was not found.

    wm = Watermarker(); wm.load()
    marked = wm.embed(audio, 24_000)            # float32, the same length, within [-1, 1]
    wm.detect(marked, 24_000)                   # {"share": 1.0, "bits": 16, "marked": True}

"share" is what AudioSeal's detect_watermark calls a probability: the share of frames whose detector score is
over one half. A clip counts as marked when the share is at least THRESHOLD. "bits" is how many of the 16 bits
carried with the mark read back as the project's MESSAGE: the mark says it is s2s-lab's, not only that it is a mark.

The first mark after loading takes seconds (AudioSeal compiles its layers); later ones a fraction of real time.
Recorded audio of a person is never marked or changed: only what this project synthesizes.
"""

from __future__ import annotations

import math

import numpy as np

GENERATOR, DETECTOR = "audioseal_wm_16bits", "audioseal_detector_16bits"
MARK_RATE = 24_000
MESSAGE = tuple(int(b) for b in format(0x5332, "016b"))     # "S2": s2s-lab's 16 bits
THRESHOLD = 0.5
TAG = "AudioSeal 0.2.0 watermark (audioseal_wm_16bits, message 0x5332)"


def to_rate(audio, sr: int, rate: int = MARK_RATE) -> np.ndarray:
    """float32 mono at `rate`, by polyphase resampling."""
    from scipy.signal import resample_poly
    x = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    if sr == rate or not x.size:
        return x
    g = math.gcd(sr, rate)
    return resample_poly(x, rate // g, sr // g).astype(np.float32)


class Watermarker:
    """AudioSeal's generator and detector, loaded once. `generator` and `detector` stand in for them in tests:
    get_watermark(x, message=...) -> the mark, and detect_watermark(x) -> (share, bits), on (1, 1, n) tensors."""

    def __init__(self, device: str = "cpu", generator=None, detector=None):
        self.device, self._gen, self._det = device, generator, detector

    def load(self) -> "Watermarker":
        if self._gen is None or self._det is None:
            from audioseal import AudioSeal
            self._gen = AudioSeal.load_generator(GENERATOR).to(self.device).eval()
            self._det = AudioSeal.load_detector(DETECTOR).to(self.device).eval()
        return self

    def embed(self, audio, sr: int) -> np.ndarray:
        """The audio with the mark added: the same length, clipped to [-1, 1]. Only at MARK_RATE."""
        import torch
        if sr != MARK_RATE:
            raise ValueError(f"marks are made at {MARK_RATE} Hz, the rate Kokoro speaks at, not {sr}")
        x = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        if not x.size:
            return x
        self.load()
        t = torch.from_numpy(x.copy()).to(self.device).view(1, 1, -1)
        msg = torch.tensor([MESSAGE], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            mark = self._gen.get_watermark(t, message=msg)
        return (t + mark).clamp(-1.0, 1.0).view(-1).cpu().numpy().astype(np.float32)

    def detect(self, audio, sr: int) -> dict:
        """{"share": of frames flagged, "bits": of MESSAGE read back, "marked": share >= THRESHOLD}, read at
        MARK_RATE whatever the clip's own rate."""
        import torch
        x = to_rate(audio, sr)
        if not x.size:
            return {"share": 0.0, "bits": 0, "marked": False}
        self.load()
        t = torch.from_numpy(x.copy()).to(self.device).view(1, 1, -1)
        with torch.inference_mode():
            share, bits = self._det.detect_watermark(t)
        share = float(np.asarray(share.cpu() if hasattr(share, "cpu") else share).reshape(-1)[0])
        got = np.asarray(bits.cpu() if hasattr(bits, "cpu") else bits).reshape(-1)[: len(MESSAGE)]
        return {"share": round(share, 4), "bits": int(sum(int(a) == b for a, b in zip(got, MESSAGE))),
                "marked": share >= THRESHOLD}


def level_db(clean, marked) -> float:
    """How far below the speech the mark sits: the RMS of (marked - clean) against the RMS of clean, in dB."""
    c, m = np.asarray(clean, dtype=np.float64), np.asarray(marked, dtype=np.float64)
    noise = float(np.sqrt(np.mean((m - c) ** 2))) if c.size else 0.0
    signal = float(np.sqrt(np.mean(c ** 2))) if c.size else 0.0
    return round(20 * math.log10(noise / signal), 1) if noise > 0 and signal > 0 else float("-inf")
