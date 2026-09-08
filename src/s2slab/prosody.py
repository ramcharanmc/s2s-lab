"""
Acoustic measurements over a waveform. Pure CPU, no GPU, no Modal.

This module is a VERSIONED INTERFACE. Every number the project reports is
computed here, so changing a definition invalidates every previously reported
result. Change a definition only alongside a dated note saying what moved.

The one rule that matters:

    Never derive a syllable count from phonation time.

Articulation rate is syllables divided by phonation time. If the syllable count
is itself proportional to phonation time, the two cancel and the rate becomes a
constant that no longer depends on the audio at all. The reference repo shipped
exactly that bug -- articulation rate sat at ~8.3 syll/s for every input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class ProsodyFeatures:
    """Everything measurable from one utterance's waveform."""

    n_syllables: int          # the count both rates are built from
    speech_rate: float        # syll/s INCLUDING pauses
    articulation_rate: float  # syll/s EXCLUDING pauses
    pause_mean_s: float
    pause_p90_s: float
    n_pauses: int
    f0_mean: float
    f0_std: float

def pause_segments(audio: np.ndarray, sr: int, min_pause_s: float = 0.15,
                   frame_ms: float = 20) -> list[tuple[float, float]]:
    """Find silences. Returns (start_s, duration_s) per pause.

    Threshold is adaptive -- the 20th percentile of frame energy, floored at 5%
    of peak. A fixed dBFS threshold breaks the moment recording level changes,
    and we compare across cohorts recorded differently.

    The 150 ms minimum is what separates a PAUSE from the ordinary gap between
    words. Below that you are measuring articulation, not pausing.
    """
    hop = int(sr * frame_ms / 1000)
    n = len(audio) // hop
    if n == 0:
        return []

    frames = audio[: n * hop].reshape(n, hop)
    energy = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    thresh = max(np.percentile(energy, 20), energy.max() * 0.05)
    silent = energy < thresh

    out: list[tuple[float, float]] = []
    run_start = None
    for i, is_silent in enumerate(silent):
        if is_silent and run_start is None:
            run_start = i
        elif not is_silent and run_start is not None:
            dur = (i - run_start) * frame_ms / 1000
            if dur >= min_pause_s:
                out.append((run_start * frame_ms / 1000, dur))
            run_start = None
    return out

# Tested:cd /mnt/c/Users/ramch/Desktop/Ram/Resources/Audio/Code/s2s-lab && source .venv/bin/activate && python -c "
# import sys; sys.path.insert(0,'src')
# import numpy as np
# from s2slab.prosody import pause_segments
# sr=16000
# tone=lambda s: 0.3*np.sin(2*np.pi*200*np.arange(int(s*sr))/sr)
# sig=np.concatenate([tone(0.5), np.zeros(int(0.4*sr)), tone(0.5), np.zeros(int(0.1*sr)), tone(0.3)])
# print('pauses:', [(round(a,2), round(b,2)) for a,b in pause_segments(sig.astype('float32'), sr)])
# "
# pauses: [(0.5, 0.4)]

def count_syllable_nuclei(audio: np.ndarray, sr: int, silence_db: float = -25.0,
                          min_dip_db: float = 2.0) -> int:
    """Count syllable nuclei as voiced intensity peaks separated by a dip.

    De Jong & Wempe (2009). Returns 0 when nothing is found -- callers must
    treat that as "rate undefined" rather than dividing by it.

    NEVER replace this with anything derived from phonation time. See the
    module docstring for what that does to articulation rate.
    """
        # Praat cannot run an intensity analysis on audio shorter than
    # 6.4 / minimum_pitch seconds. That is not an error condition -- a clip too
    # short to analyse simply has no measurable nuclei -- so return 0 rather
    # than letting Praat raise from three frames down.
    if len(audio) / sr < 6.4 / 50.0:
        return 0

    import parselmouth

    snd = parselmouth.Sound(audio.astype("float64"), sampling_frequency=sr)

    # 1. Loudness over time. minimum_pitch=50 sets the analysis window; lower
    #    means smoother, and too smooth merges adjacent syllables together.
    intensity = snd.to_intensity(minimum_pitch=50.0, time_step=0.01)
    v = np.asarray(intensity.values[0], dtype="float64")
    t = np.asarray(intensity.xs(), dtype="float64")

    finite = np.isfinite(v)
    if finite.sum() < 3:
        return 0
    v = np.where(finite, v, -300.0)

    # 2. Floor relative to the loudest peak, not an absolute dBFS value --
    #    same reason pause_segments uses an adaptive threshold.
    floor = v[finite].max() + silence_db

    peaks = [
        i for i in range(1, len(v) - 1)
        if v[i] > v[i - 1] and v[i] >= v[i + 1] and v[i] > floor
    ]

    # 3. The dip test. Two peaks with no real valley between them are ONE
    #    nucleus, and the louder of the two represents it.
    kept: list[int] = []
    for i in peaks:
        if not kept:
            kept.append(i)
            continue
        j = kept[-1]
        dip = v[j:i + 1].min()
        if min(v[j], v[i]) - dip >= min_dip_db:
            kept.append(i)
        elif v[i] > v[j]:
            kept[-1] = i

    # 4. The voicing test. A syllable nucleus is a vowel; vowels are voiced.
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    f0 = pitch.selected_array["frequency"]
    pt = np.asarray(pitch.xs(), dtype="float64")
    if pt.size == 0:
        return 0

    voiced_peaks = 0
    for i in kept:
        k = int(np.argmin(np.abs(pt - t[i])))
        if f0[k] > 0:
            voiced_peaks += 1
    return voiced_peaks

# cd /mnt/c/Users/ramch/Desktop/Ram/Resources/Audio/Code/s2s-lab && source .venv/bin/activate && python -c "
# import sys; sys.path.insert(0,'src')
# import numpy as np
# from s2slab.prosody import count_syllable_nuclei
# sr=16000
# def utt(n_syll, syll_s=0.22, pause_s=0.18, f0=150.0):
#     out=[]; ph=0.0
#     for i in range(n_syll):
#         n=int(syll_s*sr); seg=np.empty(n,dtype='float32')
#         for k in range(n):
#             ph += 2*np.pi*f0/sr
#             seg[k]=0.3*(np.sin(np.pi*k/n)**0.6)*np.sin(ph)
#         out.append(seg)
#         if i%3==2: out.append(np.zeros(int(pause_s*sr),dtype='float32'))
#     return np.concatenate(out)
# for n in (5,7,11,13):
#     print(f'  built {n:>3} syllables  ->  detected {count_syllable_nuclei(utt(n), sr):>3}')
# "
#   built   5 syllables  ->  detected   5
#   built   7 syllables  ->  detected   7
#   built  11 syllables  ->  detected  11
#   built  13 syllables  ->  detected  13

def extract_prosody(audio: np.ndarray, sr: int,
                    n_syllables: int | None = None) -> ProsodyFeatures:
    """Full prosodic feature set for one utterance.

    Pass `n_syllables` when the text is known -- it is exact, and detection is
    only needed when you have audio and no transcript. Both paths feed the same
    rate arithmetic, so results stay comparable either way.
    """
    import parselmouth

    snd = parselmouth.Sound(audio.astype("float64"), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]          # unvoiced frames are 0, not missing -- drop them

    pauses = pause_segments(audio, sr)
    total_s = len(audio) / sr
    pause_s = sum(d for _, d in pauses)
    phonation_s = max(1e-6, total_s - pause_s)

    syll = n_syllables if n_syllables is not None else count_syllable_nuclei(audio, sr)

    return ProsodyFeatures(
        n_syllables=syll,
        # NaN, not 0. "Zero syllables per second" is a measurement claim;
        # "I could not measure this" is the truth. Do not conflate them.
        speech_rate=(syll / total_s) if syll else float("nan"),
        articulation_rate=(syll / phonation_s) if syll else float("nan"),
        pause_mean_s=float(np.mean([d for _, d in pauses])) if pauses else 0.0,
        pause_p90_s=float(np.percentile([d for _, d in pauses], 90)) if pauses else 0.0,
        n_pauses=len(pauses),
        f0_mean=float(voiced.mean()) if voiced.size else float("nan"),
        f0_std=float(voiced.std()) if voiced.size else float("nan"),
    )

# cd /mnt/c/Users/ramch/Desktop/Ram/Resources/Audio/Code/s2s-lab && source .venv/bin/activate && python -c "
# import sys; sys.path.insert(0,'src')
# import numpy as np, parselmouth
# from s2slab.prosody import extract_prosody
# sr=16000
# def utt(n_syll, rate=1.0, pscale=1.0, f0=150.0):
#     syll_s=0.22/rate; pause_s=0.18*pscale; out=[]; ph=0.0
#     for i in range(n_syll):
#         n=int(syll_s*sr); seg=np.empty(n,dtype='float32')
#         for k in range(n):
#             ph += 2*np.pi*f0/sr
#             seg[k]=0.3*(np.sin(np.pi*k/n)**0.6)*np.sin(ph)
#         out.append(seg)
#         if i%3==2: out.append(np.zeros(int(pause_s*sr),dtype='float32'))
#     return np.concatenate(out)
# print(f\"{'sev':>5} {'syll':>5} {'artic':>7} {'speech':>7} {'pause_p90':>10} {'OLD artic':>10}\")
# for sev in (0.0,0.25,0.5,0.75,1.0):
#     a=utt(13, rate=1.0-0.5*sev, pscale=1.0+3.0*sev)
#     f=extract_prosody(a,sr)
#     p=parselmouth.Sound(a.astype('float64'),sampling_frequency=sr).to_pitch(time_step=0.01,pitch_floor=60,pitch_ceiling=400)
#     vf=p.selected_array['frequency']; vf=vf[vf>0]
#     old_syll=max(1,int(vf.size/12))
#     tot=len(a)/sr; from s2slab.prosody import pause_segments
#     ph_s=max(1e-6, tot-sum(d for _,d in pause_segments(a,sr)))
#     print(f'{sev:>5} {f.n_syllables:>5} {f.articulation_rate:>7.3f} {f.speech_rate:>7.3f} {f.pause_p90_s:>10.2f} {old_syll/ph_s:>10.3f}')
# "
#   sev  syll   artic  speech  pause_p90  OLD artic
#   0.0    13   4.545   3.631       0.18      8.392
#  0.25    13   3.954   2.871       0.32      8.212
#   0.5    13   3.374   2.316       0.44      8.306
#  0.75    13   2.829   1.880       0.58      8.268
#   1.0    13   2.273   1.512       0.72      8.217
