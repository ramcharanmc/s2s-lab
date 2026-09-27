"""
FT-3: page 09's severity targets as FastPitch's own inputs -- no edit of a waveform.

FastPitch predicts, per text token, a duration in mel frames and a pitch value in the model's
normalized units ((Hz - pitch_mean) / pitch_std, 0 for unvoiced). Its forward pass accepts both
as inputs in place of the predictions (nemo/collections/tts/modules/fastpitch.py: `durs` and
`pitch` on `forward`, used when no spectrogram is given). So the four targets of
s2slab.severity.Targets become arithmetic on two arrays, done here, and the model renders the
result. Everything in this module is a pure function on numpy arrays, tested without NeMo.

    f0_offset_hz    pitch + offset / pitch_std, on speech tokens
    f0_dev_scale    a * pitch, on speech tokens: the centre is 0 = the speaker's mean, page 09's
                    own centre (the Praat render used the measured median; see severity.py)
    rate            speech tokens' durations / rate
    pause_scale     durations * pause_scale, on the tokens of each PAUSE: a run of pause tokens
                    (punctuation, the space beside it) predicted to last MIN_PAUSE_S or more --
                    the same 150 ms rule as the Praat render's prosody.pause_segments. Shorter
                    runs (the space between two words) are speech, and follow the rate.

The pauses' positions in the output come from the token durations themselves (frames x hop),
so s2slab.severity.achieved measures a FastPitch render exactly as it measured the Praat one:
same median offset, same middle-half spread, same speech time, same silent time in the pauses.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from s2slab.severity import Render, Targets

MIN_PAUSE_S = 0.15          # prosody.pause_segments' floor: below this a gap is not a pause
MIN_SPEECH_FRAMES = 1       # a speech token keeps at least one frame however fast the rate


def pause_token_ids(tokens: list[str], punct: list[str], space: str = " ",
                    pad: int | None = None, blank: int | None = None) -> set[int]:
    """The ids of tokens that are silence in the text: the space, every punctuation mark, and
    the pad and blank tokens if the tokenizer has them. `tokens` is the tokenizer's id -> string
    list (tokenizer.tokens); `punct` its PUNCT_LIST."""
    ids = {i for i, t in enumerate(tokens) if t == space or t in punct}
    ids |= {i for i in (pad, blank) if i is not None}
    return ids


def pause_runs(durs: np.ndarray, is_pause: np.ndarray, hop_s: float, min_s: float = MIN_PAUSE_S) -> list[tuple[int, int]]:
    """[(first, last + 1)] token-index ranges of the pauses: maximal runs of pause tokens whose
    total predicted duration is at least `min_s`."""
    durs, is_pause = np.asarray(durs), np.asarray(is_pause, dtype=bool)
    runs, i, n = [], 0, len(durs)
    while i < n:
        if not is_pause[i]:
            i += 1
            continue
        j = i
        while j < n and is_pause[j]:
            j += 1
        if durs[i:j].sum() * hop_s >= min_s:
            runs.append((i, j))
        i = j
    return runs


def in_runs(n: int, runs: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for a, b in runs:
        mask[a:b] = True
    return mask


def control_durations(durs: np.ndarray, pause_mask: np.ndarray, t: Targets) -> np.ndarray:
    """The token durations, in frames, after the targets: speech (everything outside a pause)
    divided by the rate, pauses multiplied by pause_scale; integers, speech never below
    MIN_SPEECH_FRAMES."""
    durs = np.asarray(durs, dtype=float)
    out = np.where(pause_mask, durs * t.pause_scale, durs / t.rate)
    out = np.rint(out).astype(int)
    out[~pause_mask] = np.maximum(out[~pause_mask], MIN_SPEECH_FRAMES)
    out[pause_mask] = np.maximum(out[pause_mask], 0)
    return out


def control_pitch(pitch: np.ndarray, pause_mask: np.ndarray, t: Targets, pitch_std: float) -> np.ndarray:
    """The per-token pitch, in the model's normalized units, after the targets: speech tokens
    scaled about 0 (the speaker's mean) by f0_dev_scale and shifted by offset / pitch_std; pause
    tokens as predicted."""
    pitch = np.asarray(pitch, dtype=np.float32)
    shifted = t.f0_dev_scale * pitch + t.f0_offset_hz / pitch_std
    return np.where(pause_mask, pitch, shifted).astype(np.float32)


def intervals(durs: np.ndarray, runs: list[tuple[int, int]], hop_s: float) -> list[tuple[float, float]]:
    """Each pause run as (start, end) seconds in the audio these durations produce."""
    starts = np.concatenate([[0], np.cumsum(np.asarray(durs, dtype=float))]) * hop_s
    return [(float(starts[a]), float(starts[b])) for a, b in runs]


class Plan(NamedTuple):
    """What one render will ask of the model, and where its pauses will be."""
    durs: np.ndarray            # frames per token, int
    pitch: np.ndarray           # normalized units per token, float32
    runs: list                  # the pauses as token ranges
    pauses_in: list             # (start, end) s in the NEUTRAL render (the predicted durations)
    pauses_out: list            # the same pauses in THIS render


def plan(durs_pred: np.ndarray, pitch_pred: np.ndarray, is_pause: np.ndarray, hop_s: float,
         pitch_std: float, t: Targets | None) -> Plan:
    """The controlled inputs for one target set; with t=None, the model's own prediction (the
    neutral render), so both go through the same forward pass."""
    durs_pred, pitch_pred = np.asarray(durs_pred), np.asarray(pitch_pred, dtype=np.float32)
    runs = pause_runs(durs_pred, is_pause, hop_s)
    mask = in_runs(len(durs_pred), runs)
    if t is None:
        return Plan(durs_pred.astype(int), pitch_pred, runs, intervals(durs_pred, runs, hop_s), intervals(durs_pred, runs, hop_s))
    durs = control_durations(durs_pred, mask, t)
    return Plan(durs, control_pitch(pitch_pred, mask, t, pitch_std), runs,
                intervals(durs_pred, runs, hop_s), intervals(durs, runs, hop_s))


def as_render(audio: np.ndarray, p: Plan) -> Render:
    """severity.Render for severity.achieved: the audio and the pauses' positions, in and out."""
    return Render(np.asarray(audio, dtype=np.float32), p.pauses_in, p.pauses_out)
