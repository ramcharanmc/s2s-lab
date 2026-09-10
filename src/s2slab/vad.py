"""
Voice activity detection. Pure CPU, no model.

This is an ENERGY VAD with an adaptive threshold, and it is a deliberate
placeholder. Silero VAD (ONNX, CPU, ~1 ms/frame) is the real thing and drops in
behind the same signature. The interface is what matters:

    frame_speech_prob(audio, sr) -> list[float], one probability per frame

Everything downstream consumes that list and nothing else, so swapping the
implementation cannot ripple. Keep it that way.

What an energy VAD gets WRONG, and why it still earns its place here:

  * Unvoiced fricatives (/s/, /f/, /th/) are low-energy and read as silence.
    A speaker ending on "...to access" gets endpointed mid-word.
  * Steady background noise raises the floor until nothing is silence.
  * Breath and lip noise read as speech.

Those failures are not incidental -- they are precisely what makes fixed
thresholds cut off disfluent and elderly speakers, which is the fairness result
this project is built to measure. Starting with the weak VAD means the
improvement from a real one is something you can quantify rather than assume.
"""

from __future__ import annotations

import numpy as np

FRAME_MS = 20.0     # 20 ms is the endpointing convention; 10 ms doubles the
                    # cost for no decision-quality gain at these thresholds.

def frame_signal(audio: np.ndarray, sr: int,
                 frame_ms: float = FRAME_MS) -> np.ndarray:
    """Split into non-overlapping frames. Returns shape (n_frames, hop).

    A trailing partial frame is DISCARDED rather than zero-padded. Padding
    would create an artificially quiet final frame that reads as silence, and
    the last frame is exactly where an endpointer is making its decision.
    """
    hop = int(sr * frame_ms / 1000)
    if hop <= 0:
        raise ValueError(f"frame_ms={frame_ms} too small for sr={sr}")
    n = len(audio) // hop
    if n == 0:
        return np.empty((0, hop), dtype="float32")
    return audio[: n * hop].reshape(n, hop)


def frame_rms(audio: np.ndarray, sr: int,
              frame_ms: float = FRAME_MS) -> np.ndarray:
    """Per-frame RMS energy. The +1e-12 guards sqrt(0) on digital silence."""
    frames = frame_signal(audio, sr, frame_ms)
    if frames.size == 0:
        return np.empty(0, dtype="float64")
    return np.sqrt((frames.astype("float64") ** 2).mean(axis=1) + 1e-12)

def frame_speech_prob(audio: np.ndarray, sr: int,
                      frame_ms: float = FRAME_MS,
                      noise_percentile: float = 10.0,
                      snr_db_full: float = 20.0) -> list[float]:
    """Per-frame probability that the frame contains speech, in [0, 1].

    The noise floor is estimated as a low percentile of frame energy rather
    than from a fixed dBFS value or a leading-silence assumption -- neither
    survives contact with real recordings at different levels.

    Probability ramps linearly in dB above that floor: 0.0 at the floor, 1.0 at
    `snr_db_full` above it. A ramp rather than a hard 0/1 matters because the
    decision-theoretic policy needs a graded posterior, not a bit.

    Returns [] for audio too short to frame -- callers must handle an empty
    list rather than assuming at least one frame exists.
    """
    rms = frame_rms(audio, sr, frame_ms)
    if rms.size == 0:
        return []

    db = 20.0 * np.log10(rms + 1e-12)
    floor = np.percentile(db, noise_percentile)

    # An all-silence clip has no dynamic range; without this guard the ramp
    # divides by ~0 and reports confident speech for pure digital silence.
    if db.max() - floor < 3.0:
        return [0.0] * len(db)

    prob = np.clip((db - floor) / snr_db_full, 0.0, 1.0)
    return [float(p) for p in prob]