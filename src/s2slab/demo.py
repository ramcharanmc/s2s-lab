"""
The demo: the ladder shown on demand. Two parts, in this order.

  MEASURED  what the committed run files say, turn by turn: the caller's clip, what was heard, what
            was replied, and where the time went. No GPU, no model: it reads runs/.
  LIVE      a few seconds of the visitor's own speech through the rungs that are loaded, each reply
            with its own timings. Behind an access code and a daily spend cap.

Neither plan has a live demo, and v1 argues against one as the centrepiece (its "demo trap"), so
the measured part comes first and the live part says what it is: one turn, on whatever machine
is awake, not a benchmark.

    PYTHONPATH=src DEMO_ACCESS_CODE=... python -m uvicorn s2slab.demoweb:laptop --factory --port 8000
        the whole demo on the laptop's CPU (rungs r0, r1, r2), for development (requirements/demo.txt)
    modal serve modal_app/demo.py
        the page on a CPU container, the live turn on a T4 that sleeps (modal_app/demo.py)

This module is the demo without the web: decoding, the access code, the spend cap, the measured
data, a live turn as a record. demoweb.py puts a web app around it.

What the live part does with a visitor's audio: decodes it in memory, hands it to the rungs, returns
the replies, and keeps nothing. No audio and no transcript is written to a disk, a volume or a log.

The spend cap counts the time the GPU is up, not the time a request takes: a container that woke
for one turn stays up for its scale-down window. charge() models that (Budget, below).
"""

from __future__ import annotations

import base64
import datetime
import hmac
import io
import json
import pathlib
import sys
import time

import numpy as np

from . import rungs as ladder

SR = 16_000                         # what the rungs are handed
MIN_CLIP_S, MAX_CLIP_S = 0.3, 15.0
MAX_UPLOAD_BYTES = 2_000_000        # 15 s of browser audio is well under this
SILENT_PEAK = 1e-3
CODE_ENV = "DEMO_ACCESS_CODE"
MIN_CODE_LEN = 10                   # a shorter code is treated as none: it could be found by trying
WRONG_CODES, LOCK_S = 30, 300.0     # this many wrong codes inside LOCK_S close the live part for LOCK_S (M32a)
NOT_LOADED = "This rung could not be loaded on this machine."
NOT_ANSWERED = "This rung failed on this recording."
NO_MACHINE = "The machine this rung runs on did not answer."
WARM_UP_S = 4.0                     # a full-duplex model's throwaway call at load: a second of silence, and this much after it
PAGE = pathlib.Path(__file__).with_name("demo_page.html")

# What the Measured part shows: one committed run per rung, and the same four clips for all of them.
MEASURED = (
    ("R0", "naive cascade: recognizer, then LLM, then TTS, each waiting for the last", "runs/r0-gpu/evalset-modal-t4-pr002.json"),
    ("R1", "streaming cascade: the reply handed out clause by clause", "runs/r1-gpu/evalset-modal-t4-pr002.json"),
    ("R2", "two-tier: a slow recognizer (large-v3-turbo) drives the reply", "runs/r2-gpu-turbo/evalset-modal-t4-pr004.json"),
    ("R3", "speech-LLM: Qwen2-Audio hears the turn, no transcript in the loop", "runs/r3-gpu/evalset-modal-a10.json"),
    ("R4", "speech-to-speech, half duplex: Qwen2.5-Omni speaks the reply itself", "runs/r4-gpu/evalset-modal-l40s.json"),
    ("R4 streamed", "the same model, its reply decoded a block at a time and handed out, its first half second held; "
                    "in its own run, beside it, R4 took 6.8 s to first sound (PR-015)", "runs/r4-gpu/pr015.json"),
    ("R5 Moshi", "full duplex: no turns, the model decides when to talk", "runs/r5-moshi/evalset-modal-a10.json"),
    ("R5 PersonaPlex", "full duplex, with a role and a voice prompt before every call", "runs/r5-personaplex/evalset-modal-a10.json"),
)
CLIPS = ("clean-07", "clean-11", "clean-23", "clean-37")      # LibriSpeech test-clean, CC BY 4.0; set by pick_clips()
FROM_RUNG = {"runs/r4-gpu/pr015.json": "r4-gpu-stream"}        # a run file of several rungs: the one its row shows
# The verbatim panel: PR-011's run, small.en beside small.en + FT-2's LoRA, on two clips of the disfluent cohort.
VERBATIM_RUN = "runs/ft2/pr011.json"
VERBATIM_CLIPS = ("disfluent-29", "disfluent-42")             # DisfluencySpeech test split, Apache-2.0; set by pick_verbatim()
VERBATIM_NOTE = ("These two clips are the ones where the difference is largest, chosen for that; the cohort's numbers are "
                 "beside them. One speaker, performing disfluency, trained on and tested on: same-speaker adaptation.")
# The prosody panel: PR-006's run, R3 asked one question of two deliveries of the same sentence.
PROSODY_RUN = "runs/prosody/pr006.json"
PROSODY_PAIRS = ("ang-t10", "fea-t04", "sad-t05")             # CREMA-D test pairs; set by pick_pairs(). No audio is served
EMOTION_WORDS = {"SAD": "sad", "FEA": "fearful", "ANG": "angry"}
PROSODY_NOTE = ("Acted emotion, and a benchmark the model may have met in training. One pair for each emotion, the one "
                "with the middle difference of its thirteen: a usual case, not the best. The clips are CREMA-D's and are "
                "not played here: its audio is not redistributed by this project.")
PROSODY_SOURCE = ("CREMA-D: Cao, Cooper, Keutmann, Gur, Nenkova, Verma, IEEE Transactions on Affective Computing 5(4), 2014; "
                  "Open Database License 1.0")
VOICE_NOTE = ("Measured from the recording, in the page's server, by Praat. Numbers, not a reading of how anyone feels: "
              "nothing here can tell that. Loudness, and a speaking rate taken from the sound, are left out: both follow the "
              "microphone and the level more than the speaker.")
READ_WITH_CARE = ("The rows come from different GPUs and different days, and the same rung measured 17% slower three days "
                  "apart: only rungs run in one container are comparable. R5's number is a response gap in stream time, "
                  "not a turn's latency.")


# --------------------------------------------------------------------------- #
# the visitor's audio
# --------------------------------------------------------------------------- #
def decode(data: bytes) -> np.ndarray:
    """Whatever the browser recorded (WebM or Ogg Opus, MP4 AAC, WAV) as mono float32 at 16 kHz, in
    memory. Raises ValueError, with a sentence for the visitor, for what cannot be a turn."""
    import av
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"the recording is {len(data) / 1e6:.1f} MB; the most this takes is {MAX_UPLOAD_BYTES / 1e6:.0f} MB")
    try:
        parts = []
        with av.open(io.BytesIO(data)) as container:
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
            for frame in container.decode(audio=0):
                parts += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
            parts += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    except Exception as e:                                      # PyAV raises many types; none is the visitor's problem
        raise ValueError("that is not audio this can read") from e
    audio = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
    seconds = len(audio) / SR
    if seconds < MIN_CLIP_S:
        raise ValueError(f"the recording is {seconds:.1f} s long; say at least a few words")
    if seconds > MAX_CLIP_S:
        raise ValueError(f"the recording is {seconds:.0f} s long; the most this takes is {MAX_CLIP_S:.0f} s")
    if not np.isfinite(audio).all() or np.abs(audio).max() < SILENT_PEAK:
        raise ValueError("the recording is silent")
    return audio


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# who may use the live part, and how much it may cost
# --------------------------------------------------------------------------- #
def code_usable(expected: str | None) -> bool:
    return bool(expected) and len(expected) >= MIN_CODE_LEN


def code_ok(given: str | None, expected: str | None) -> bool:
    """The access code, compared in constant time. With no code configured, or one shorter than
    MIN_CODE_LEN, nobody gets in."""
    if not code_usable(expected) or not given:
        return False
    return hmac.compare_digest(given.encode(), expected.encode())


class Door:
    """Wrong codes, counted. WRONG_CODES of them inside LOCK_S close the live part to everyone for
    LOCK_S, the right code included: thirty tries in five minutes find no code of ten characters, and
    a few strangers guessing do not close it on the people who were given the code (M32a; it was
    five in ten minutes, for ten, before the demo was shown widely).

    The count is one for all visitors, not one each: behind a proxy a visitor's address is what a
    header says it is. So someone sending wrong codes can keep the live part closed; they cannot
    open it, and a closed live part costs nothing. `store` as in Budget."""

    def __init__(self, store, tries: int = WRONG_CODES, lock_s: float = LOCK_S):
        self.store, self.tries, self.lock_s = store, tries, lock_s

    def locked_s(self, t: float) -> float:
        """Seconds until the live part opens again; 0 if it is open."""
        return max(0.0, float(self.store.get("locked_until", 0.0)) - t)

    def wrong(self, t: float) -> None:
        recent = [x for x in self.store.get("wrong_codes", []) if t - x < self.lock_s] + [t]
        if len(recent) >= self.tries:
            self.store["locked_until"], recent = t + self.lock_s, []
        self.store["wrong_codes"] = recent


class Budget:
    """The daily spend cap, in GPU seconds that are UP, not seconds that are busy.

    A container woken by a request is billed until its scale-down window after the LAST request
    has passed. So the first request of a burst is charged its own time, the window, and the cold
    start; a request that arrives while the container is still up is charged only the time by
    which it extends the bill. `store` is any mapping with get() and item assignment (a dict here,
    a modal.Dict in the deployment); the day is the UTC date.

    One cap, however many machines: `machines` names the ones beside the first, each with its own
    rate and window, {"a10": (1.10, 120.0)}. Each is billed until its own window has passed."""

    def __init__(self, store, cap_dollars: float, dollars_per_hour: float, window_s: float = 60.0, cold_start_s: float = 45.0,
                 machines: dict | None = None):
        self.store, self.cap, self.rate = store, cap_dollars, dollars_per_hour
        self.window_s, self.cold_start_s, self.machines = window_s, cold_start_s, dict(machines or {})

    @staticmethod
    def day(t: float) -> str:
        return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d")

    def spent(self, t: float) -> float:
        return float(self.store.get(f"dollars:{self.day(t)}", 0.0))

    def left(self, t: float) -> float:
        return max(0.0, self.cap - self.spent(t))

    def allows(self, t: float) -> bool:
        return self.spent(t) < self.cap

    def charge(self, start: float, end: float, machine: str | None = None) -> float:
        """Record one request to one machine (None: the first), from `start` to `end` (epoch
        seconds). Returns the dollars it added."""
        rate, window_s = self.machines[machine] if machine else (self.rate, self.window_s)
        key = f"billed_until:{machine}" if machine else "billed_until"
        billed_until = float(self.store.get(key, 0.0))
        if start > billed_until:                                  # the container had gone to sleep
            seconds = self.cold_start_s + (end - start) + window_s
        else:
            seconds = max(0.0, end + window_s - billed_until)
        dollars = seconds / 3600 * rate
        self.store[key] = end + window_s
        self.store[f"dollars:{self.day(end)}"] = self.spent(end) + dollars
        return dollars


# --------------------------------------------------------------------------- #
# the measured part
# --------------------------------------------------------------------------- #
def _p50(values) -> float | None:
    v = [x for x in values if x is not None and x == x]
    return round(float(np.median(v)), 1) if v else None


def pick_clips(manifest: dict, n: int = 4) -> tuple[str, ...]:
    """The n shortest clips of the clean cohort: how CLIPS was chosen, kept so it can be checked."""
    clean = [i for i in manifest["items"] if i["cohort"] == "clean"]
    return tuple(sorted(i["id"] for i in sorted(clean, key=lambda i: (i["duration_s"], i["id"]))[:n]))


def pick_verbatim(run: dict, items: dict, n: int = 2) -> tuple[str, ...]:
    """The n disfluent clips where the LoRA removed the most raw errors: how VERBATIM_CLIPS was
    chosen, kept so it can be checked. The page says they were chosen this way."""
    from . import wer
    base, lora = run["models"]["small.en"]["turns"], run["models"]["ft2"]["turns"]
    gain = {c: wer.wer(items[c]["text"], base[c]["text"], wer.light).errors - wer.wer(items[c]["text"], t["text"], wer.light).errors
            for c, t in lora.items() if t["cohort"] == "disfluent"}
    return tuple(sorted(sorted(gain, key=lambda c: (-gain[c], c))[:n]))


def verbatim(root, items: dict, clips=VERBATIM_CLIPS, rel: str = VERBATIM_RUN) -> dict | None:
    """The verbatim panel's data from PR-011's run file; None if the file is not there."""
    path = pathlib.Path(root) / rel
    if not path.exists():
        return None
    run = json.loads(path.read_text(encoding="utf-8"))
    c, base, lora = run["criteria"], run["models"]["small.en"]["turns"], run["models"]["ft2"]["turns"]
    return {"file": rel, "registration": run["registration"], "machine": run["machine"], "commit": run["commit"], "note": VERBATIM_NOTE,
            "cohort": {"clips": c["n_target"], "base_raw_wer": round(100 * c["s1"]["base_raw_wer"], 1),
                       "lora_raw_wer": round(100 * c["s1"]["lora_raw_wer"], 1), "fillers": c["s2"]["fillers_in_refs"],
                       "base_kept": c["s2"]["base_kept"], "lora_kept": c["s2"]["lora_kept"]},
            "clips": [{"id": k, "seconds": items[k]["duration_s"], "reference": items[k]["text"], "base": base[k]["text"],
                       "lora": lora[k]["text"], "base_ms": base[k]["asr_ms"], "lora_ms": lora[k]["asr_ms"],
                       "source": "DisfluencySpeech, test split, Apache-2.0"} for k in clips]}


def pick_pairs(run: dict) -> tuple[str, ...]:
    """For each emotion, the pair whose difference p(Yes) loaded - p(Yes) neutral is the middle one
    of its pairs: how PROSODY_PAIRS was chosen, kept so it can be checked."""
    wording = next(iter(run["wordings"]))
    p = {c: a["p_yes"][wording] for c, a in run["answers"].items()}
    out = []
    for emotion in EMOTION_WORDS:
        mine = sorted((p[pr["loaded"]] - p[pr["neutral"]], pr["id"]) for pr in run["pairs"] if pr["emotion"] == emotion)
        if mine:
            out.append(mine[len(mine) // 2][1])
    return tuple(sorted(out))


def prosody(root, pairs=PROSODY_PAIRS, rel: str = PROSODY_RUN) -> dict | None:
    """The prosody panel's data from PR-006's run file; None if the file is not there."""
    from .prosodycheck import SENTENCES
    path = pathlib.Path(root) / rel
    if not path.exists():
        return None
    run = json.loads(path.read_text(encoding="utf-8"))
    wording = next(iter(run["wordings"]))
    p = {c: a["p_yes"][wording] for c, a in run["answers"].items()}
    by_id = {pr["id"]: pr for pr in run["pairs"]}
    return {"file": rel, "registration": "PR-006", "run": run["run"], "question": run["wordings"][wording], "note": PROSODY_NOTE,
            "source": PROSODY_SOURCE,
            "scores": [{"emotion": EMOTION_WORDS[e], "answered": a, "ranked": r, "pairs": n} for e, (a, r, n) in run["scores"][wording].items()],
            "pairs": [{"id": k, "emotion": EMOTION_WORDS[by_id[k]["emotion"]], "sentence": SENTENCES[by_id[k]["sentence"]],
                       "neutral": round(p[by_id[k]["neutral"]], 2), "loaded": round(p[by_id[k]["loaded"]], 2)} for k in pairs]}


def voice(audio: np.ndarray) -> dict | None:
    """What a recording's voice carried, as three numbers: median pitch, pitch range (10th to 90th
    percentile, in semitones), and the seconds that had pitch. prosodycheck's definitions, so a
    visitor's numbers and the probe set's mean the same. None where Praat is not installed.

    Takes mono float32 at 16 kHz. Pitch is None when less than a tenth of a second has any."""
    import math
    try:
        from .prosodycheck import FRAME_S, MIN_VOICED_FRAMES, PITCH_CEILING_HZ, PITCH_FLOOR_HZ, pitch_hz
        voiced = pitch_hz(audio)
    except ImportError:
        return None
    out = {"voiced_s": round(voiced.size * FRAME_S, 1), "pitch_hz": None, "range_st": None,
           "floor_hz": PITCH_FLOOR_HZ, "ceiling_hz": PITCH_CEILING_HZ, "note": VOICE_NOTE}
    if voiced.size >= MIN_VOICED_FRAMES:
        lo, hi = np.percentile(voiced, [10, 90])
        out.update(pitch_hz=round(float(np.median(voiced)), 1), range_st=round(12 * math.log2(hi / lo), 1))
    return out


def measured(root=".", clips=CLIPS, rows=MEASURED, verbatim_clips=VERBATIM_CLIPS, verbatim_run=VERBATIM_RUN,
             prosody_pairs=PROSODY_PAIRS, prosody_run=PROSODY_RUN, findings_files=None) -> dict:
    """The Measured part's data, from the committed run files under `root`. A rung whose file is
    missing is listed as missing, not left out. "audio" names every clip the page may play."""
    root = pathlib.Path(root)
    manifest = json.loads((root / "evalset" / "manifest.json").read_text(encoding="utf-8"))
    items = {i["id"]: i for i in manifest["items"]}
    panel = verbatim(root, items, verbatim_clips, verbatim_run)
    out = {"note": READ_WITH_CARE, "rungs": [], "verbatim": panel, "prosody": prosody(root, prosody_pairs, prosody_run),
           "findings": findings(root, FINDINGS if findings_files is None else findings_files),
           "audio": list(clips) + [c["id"] for c in (panel or {"clips": []})["clips"]],
           "clips": [{"id": c, "seconds": items[c]["duration_s"], "reference": items[c]["text"].capitalize() + ".",
                      "source": f"LibriSpeech test-clean, utterance {items[c]['source_id']}, CC BY 4.0"} for c in clips]}
    for label, what, rel in rows:
        path = root / rel
        if not path.exists():
            out["rungs"].append({"label": label, "what": what, "file": rel, "missing": True})
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        turns = run["rungs"][FROM_RUNG[rel]]["turns"] if rel in FROM_RUNG else run["turns"]
        row = {"label": label, "what": what, "file": rel, "machine": run.get("machine"), "commit": run.get("commit"),
               "turns_in_run": len(turns), "turns": []}
        if "gap_ms" in next(iter(turns.values())):                 # full duplex: a conversation, not a turn
            gaps = [t["gap_ms"] for t in turns.values()]
            row.update(kind="duplex", answered=sum(g is not None for g in gaps), gap_p50_ms=_p50(gaps))
            for c in clips:
                t = turns[c]
                row["turns"].append({"clip": c, "gap_ms": t["gap_ms"], "reply_text": t["reply_text"], "setup_s": t["setup_s"],
                                     "overlap": t["overlap"]})
        else:
            row.update(kind="turn", first_audio_p50_ms=_p50(t["first_audio_ms"] for t in turns.values()),
                       total_p50_ms=_p50(t["total_ms"] for t in turns.values()))
            for c in clips:
                t = turns[c]
                row["turns"].append({"clip": c, "transcript": t.get("transcript"), "reply_text": t["reply_text"],
                                     "stages_ms": t.get("stages_ms", {}), "first_audio_ms": t["first_audio_ms"],
                                     "total_ms": t["total_ms"], "reply_s": t["reply_s"]})
        out["rungs"].append(row)
    return out


# --------------------------------------------------------------------------- #
# the measured part: what else was registered and run
# --------------------------------------------------------------------------- #
# One panel a registration, from its committed run file. No audio: the barge-in set is the
# author's own voice, and the renders are synthetic speech kept out of git.
FINDINGS = ("runs/bargein/pr003-70ms.json", "runs/ft1/pr010.json", "runs/severity/pr009.json", "runs/ft3/pr012.json",
            "runs/m26/pr013.json", "runs/turn-taking/pr014-dev-vap.json", "runs/r4-gpu/pr015.json")
BARGEIN_POLICIES = (("B0_duration_200ms", "stop after 200 ms of caller sound"), ("B0_duration_400ms", "after 400 ms"),
                    ("B0_duration_800ms", "after 800 ms"), ("B0_duration_1500ms", "after 1500 ms"),
                    ("B1_words", "stop when the recognizer has written words"), ("B2_speechlike", "stop when the words are not a backchannel"))


def _share(c: dict) -> str:
    return f"{c['met']} of {c['of']} ({100 * c['met'] / c['of']:.1f}%)" + ("" if c["pass"] else ": failed")


def _severity(run: dict, rel: str, registration: str, title: str, question: str, note: str) -> dict:
    from .severity import S2_MIN_RANGE_IN_ORDER, S4_OFFSET_SHARE, S5_RANGE_SHARE      # the bars, from where the criteria are
    c = run["criteria"]
    named = (("S1", "pitch offset, rate and pauses in order, every sweep"),
             ("S2", f"pitch range in order, at least {S2_MIN_RANGE_IN_ORDER} of {c['S2']['of']}"),
             ("S3", "rate and pauses within 10%, every check"),
             ("S4", f"pitch offset within 3 Hz or 10%, at least {100 * S4_OFFSET_SHARE:.0f}%"),
             ("S5", f"pitch range within 10%, at least {100 * S5_RANGE_SHARE:.0f}%"))
    return {"registration": registration, "title": title, "question": question, "supported": bool(c["supported"]),
            "rows": [[what, _share(c[k])] for k, what in named], "note": note, "commit": run.get("commit"), "file": rel}


def _bargein(run: dict, rel: str) -> dict:
    wrong = {k: sum(not o["correct"] for o in v) for k, v in run["outcomes"].items()}
    n = len(next(iter(run["outcomes"].values())))
    best = min(wrong[k] for k in wrong if k.startswith("B0"))
    return {"registration": "PR-003", "title": "When the caller breaks in",
            "question": "Does a barge-in policy that reads the caller's words make fewer wrong stop or keep-going decisions than the "
                        "best duration threshold?",
            "supported": wrong["B2_speechlike"] < best,
            "rows": [[what, f"wrong on {wrong[k]} of {n}"] for k, what in BARGEIN_POLICIES if k in wrong],
            "note": "Twenty recordings of the author's own voice, breaking in on a reply or only saying 'mm-hm'. They are not played "
                    "here. Supported, narrowly: one set, one speaker, written by the person who built the policy.",
            "commit": None, "file": rel}


def _ft1(run: dict, rel: str) -> dict:
    c = run["criteria"]
    return {"registration": "PR-010", "title": "A LoRA on a full-duplex model", "supported": bool(c["supported"]),
            "question": "Does a LoRA trained on two-channel dialogues move Moshi's turn gaps toward the dialogues' own?",
            "rows": [["distance from the human gaps, base Moshi", f"{c['distance_base_ms']:.0f} ms"],
                     ["distance, Moshi with the LoRA", f"{c['distance_lora_ms']:.0f} ms"],
                     ["the change", f"{-c['gain_ms']:+.0f} ms, 95% interval {-c['ci95_ms'][1]:+.0f} to {-c['ci95_ms'][0]:+.0f}: further away"]],
            "note": f"{c['n_dialogues']} test dialogues the LoRA never saw. It moved the model, away from the target: a fine-tune "
                    "that ran and did not do what it was trained for, reported as such.",
            "commit": run.get("commit"), "file": rel}


def _quant(run: dict, rel: str) -> dict:
    c, s, size = run["criteria"], run["summary"], {k: v["size_mb"] for k, v in run["rungs"].items()}
    return {"registration": "PR-013", "title": "A synthesizer made smaller", "supported": bool(c["claim_a"]),
            "question": "Quantized to 8 bits, does Kokoro's pitch drift while its words stay?",
            "rows": [["the 8-bit copy's pitch, against the full one", f"{c['A2']['mean']:+.2f} Hz on average, 95% interval {c['A2']['lo']:+.2f} to "
                                                                      f"{c['A2']['hi']:+.2f}; within 3 Hz on {c['A1']['within']} of {c['A1']['of']} renders"],
                     ["what the same model does from one run to the next", f"{c['A2']['floor_p95']:.2f} Hz at the 95th percentile"],
                     ["its words, by a recognizer", f"{c['A3']['quantized']['errors']} errors in {c['A3']['quantized']['words']} words, "
                                                    f"against {c['A3']['reference']['errors']}"],
                     ["size", f"{size['onnx-q8']:.0f} MB against {size['onnx-fp32']:.0f} MB"],
                     ["speed on a laptop's CPU", f"{s['onnx-q8']['rtf_median']:.2f} of real time against {s['onnx-fp32']['rtf_median']:.2f}: slower"],
                     ["the 16-bit copy", f"spoke on {c['B1']['audio']} of {c['B1']['of']} renders; no audio on the rest"]],
            "note": "Claim A, the 8-bit copy, supported; Claim B, that the 16-bit copy is usable as it is, not. Whether a listener "
                    "hears a 4 Hz shift was not tested.",
            "commit": run.get("commit"), "file": rel}


def _pr014(run: dict, rel: str) -> dict:
    rows = [[f"setting {r['theta']:g}", f"{r['cut']} of {r['n']} turns cut off, a median wait of {r['p50']:,.0f} ms"] for r in run["rows"]]
    return {"registration": "PR-014", "title": "Voice activity projection, on real speech", "supported": False, "status": "not run",
            "question": "Does an endpointer that ends the turn when a voice-activity-projection model expects the system to speak "
                        "cut off fewer disfluent speakers than the best word-based one, without making anyone wait longer?",
            "rows": rows + [["the rule: the fewest cut-offs with a median wait of at most 1,000 ms", "no setting met it"]],
            "note": "On the author's 30 recordings, with the system's prompt played first, the model rated the speaker's own pauses "
                    "above the end of the turn, so most turns were ended by the 1,500 ms fallback, not by the model. With no "
                    "setting, the test set was never run: reported as not run.",
            "commit": None, "file": rel}


def _pr015(run: dict, rel: str) -> dict:
    from .pr015 import STREAM, _wer, criteria
    c = criteria(run)
    d, R = c["describe"], run["rungs"]
    fs, total = d["first_sound"], d["total"]
    turns = lambda name: list(R[name]["turns"].values())                       # noqa: E731
    rows = [["first sound, the median: streamed against R4",
             f"{fs['stream'][0]:,.0f} ms against {fs['r4'][0]:,.0f} ({fs['stream'][0] / fs['r4'][0]:.2f} of R4's)"],
            ["first sound, p90", f"{fs['stream'][1]:,.0f} ms against {fs['r4'][1]:,.0f} ({fs['stream'][1] / fs['r4'][1]:.2f})"],
            ["turns where playback ran dry", f"{len(d['stalls'])} of {d['turns']}"],
            ["the words, as Whisper hears the reply", f"{_wer(turns(STREAM)):.1f}% wrong against {_wer(turns('r4-gpu')):.1f}%"],
            ["the whole reply, the median", f"{total['stream'][0]:,.0f} ms against {total['r4'][0]:,.0f} "
                                            f"({total['stream'][0] / total['r4'][0]:.2f} times)"],
            ["R4 with its Talker compiled, not streamed", f"{fs['fast'][0]:,.0f} ms to first sound: no sooner"]]
    return {"registration": "PR-015", "title": "R4, streamed", "supported": c["verdict"].startswith("SUPPORTED"),
            "question": "Does streaming R4's reply let the caller hear it sooner, without stalling, a much longer turn or "
                        "garbled words?",
            "rows": rows,
            "note": "The same model, the same words and the same voice: the reply decoded a block at a time and handed out, "
                    "playback held for its first half second. Times on the server, the three in one container, 200 clips.",
            "commit": run.get("commit"), "file": rel}


def findings(root, files=FINDINGS) -> list[dict]:
    """The panels of what else was registered and run, each from its run file; a file that is
    not there is left out."""
    make = {"runs/bargein/pr003-70ms.json": _bargein, "runs/ft1/pr010.json": _ft1, "runs/m26/pr013.json": _quant,
            "runs/turn-taking/pr014-dev-vap.json": _pr014, "runs/r4-gpu/pr015.json": _pr015,
            "runs/severity/pr009.json": lambda run, rel: _severity(
                run, rel, "PR-009", "A severity score, put onto speech",
                "Does a severity score render onto neutral speech as the mapping says: lower, slower, flatter, with longer pauses?",
                "Synthetic speech edited after the fact; no clinical validity is claimed or implied. Four criteria of five met; "
                "pitch offset missed its bar."),
            "runs/ft3/pr012.json": lambda run, rel: _severity(
                run, rel, "PR-012", "The same targets, as a synthesizer's own inputs",
                "Given the same targets as its own durations and pitch, does FastPitch meet the criteria the edited speech missed?",
                "Synthetic speech; no clinical validity is claimed or implied. Rate was exact and pitch on target in little more "
                "than half the checks: all five criteria failed, as predicted.")}
    out = []
    for rel in files:
        path = pathlib.Path(root) / rel
        if path.exists():
            out.append(make[rel](json.loads(path.read_text(encoding="utf-8")), rel))
    return out


# --------------------------------------------------------------------------- #
# the severity render, to be heard
# --------------------------------------------------------------------------- #
# PR-009's render, live: one of the registration's sentences in one of its voices, neutral, and
# edited to one of its scores. Nothing else can be asked of it: no other text, no other score.
RENDER_NOTE = ("Synthetic speech, edited to a specification. No clinical validity is claimed or implied, and this is not a "
               "diagnosis of anyone: the numbers are one hand-written mapping from a rating scale to pitch, range, rate and "
               "pauses, put onto a synthesizer's neutral voice.")
RENDER_WARNING = "At this score the mapping asks for more than about 30% of rate or 40 Hz of pitch: past that, edited speech sounds processed."
RENDER_FIELDS = (("f0_offset_hz", "pitch, Hz"), ("f0_dev_scale", "pitch range, times"), ("rate", "rate, times"),
                 ("pause_scale", "pauses, times"))


def kokoro_speaker(device: str = "cpu"):
    """speak(text, voice) -> float32 at 24 kHz, through R0's Kokoro on `device` (severity.kokoro_speaker, with a device)."""
    from .r0 import KokoroTTS
    tts = KokoroTTS(device=device)
    tts.load()

    def speak(text: str, voice: str) -> np.ndarray:
        parts = [np.asarray(r.audio, dtype=np.float32) for r in tts.pipeline(text, voice=voice) if r.audio is not None]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return speak


def synthetic_wav(audio: np.ndarray, sr: int, title: str, comment: str) -> bytes:
    """A WAV that says in its own metadata that it is synthetic, as every file severity.py writes does."""
    import soundfile as sf
    buf = io.BytesIO()
    with sf.SoundFile(buf, "w", samplerate=sr, channels=1, subtype="PCM_16", format="WAV") as f:
        f.title, f.comment, f.software = title, comment, "s2s-lab severity.py"
        f.write(np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0))
    return buf.getvalue()


def render_choices() -> dict:
    """What the page may ask for: the registration's voices, sentences and scores, and no other."""
    from . import severity
    return {"voices": list(severity.TEST_VOICES), "sentences": dict(severity.TEST_SENTENCES),
            "scores": {k: list(v) for k, v in severity.SCORES.items()},
            "scales": {"depression": severity.PHQ9.name, "mania": severity.YMRS.name}, "note": RENDER_NOTE}


def render_severity(speak, voice: str, sentence: str, condition: str, score: float) -> dict:
    """One sentence, neutral and rendered, with the targets and what was measured again from the
    render. Raises ValueError for anything outside render_choices()."""
    from . import severity
    ok = render_choices()
    if voice not in ok["voices"] or sentence not in ok["sentences"] or condition not in ok["scores"] or score not in ok["scores"][condition]:
        raise ValueError("only the registered voices, sentences and scores can be rendered")
    t0 = time.perf_counter()
    text, sr = ok["sentences"][sentence], severity.KOKORO_SR
    neutral = severity.trim(speak(text, voice), sr)
    t = severity.CONDITIONS[condition](score)
    r = severity.render(neutral, sr, t)
    got = severity.achieved(neutral, r, sr)
    clean = lambda x: None if x != x else round(float(x), 3)
    about = f"{severity.SYNTHETIC_NOTE} Voice {voice}. Text: {text}"
    return {"voice": voice, "sentence": sentence, "text": text, "condition": condition, "scale": ok["scales"][condition], "score": score,
            "target": {f: clean(getattr(t, f)) for f, _ in RENDER_FIELDS}, "achieved": {f: clean(getattr(got, f)) for f, _ in RENDER_FIELDS},
            "misses": severity.misses(got, t), "neutral_s": round(len(neutral) / sr, 2), "rendered_s": round(len(r.audio) / sr, 2),
            "warning": RENDER_WARNING if abs(t.rate - 1) > 0.3 or abs(t.f0_offset_hz) > 40 else None,
            "neutral_wav": base64.b64encode(synthetic_wav(neutral, sr, "synthetic neutral", about)).decode("ascii"),
            "rendered_wav": base64.b64encode(synthetic_wav(r.audio, sr, f"synthetic {condition} {score:g}", about)).decode("ascii"),
            "total_ms": round(1000 * (time.perf_counter() - t0), 1), "note": RENDER_NOTE}


def registered_step(root, voice: str, sentence: str, condition: str, score: float, rel: str = "runs/severity/pr009.json") -> dict | None:
    """What PR-009's run measured for the same voice, sentence, condition and score; None if the file is not there."""
    path = pathlib.Path(root) / rel
    if not path.exists():
        return None
    run = json.loads(path.read_text(encoding="utf-8"))
    clean = lambda x: None if x is None or x != x else round(float(x), 3)
    for row in run["rows"]:
        if (row["voice"], row["sentence"], row["condition"]) == (voice, sentence, condition):
            for step in row["steps"]:
                if step["score"] == score:
                    c = run["criteria"]
                    return {"achieved": {f: clean(step["achieved"][f]) for f, _ in RENDER_FIELDS}, "misses": step["misses"],
                            "registration": run["registration"], "commit": run["commit"], "file": rel, "supported": bool(c["supported"]),
                            "result": f"{sum(v['pass'] for k, v in c.items() if k != 'supported')} of its five criteria met; pitch within "
                                      f"3 Hz or 10% in {c['S4']['met']} of {c['S4']['of']} checks, where it needed nine in ten"}
    return None


# --------------------------------------------------------------------------- #
# the live part
# --------------------------------------------------------------------------- #
class Call:
    """A full-duplex model as the live part holds it. It is not a rung: it has no finished turn to
    answer. The visitor's recording is the caller's side of one call, streamed as duplex.converse
    streams an eval clip: silence while the model opens the call, the recording, silence in which
    to answer. What comes back is duplex.measure's numbers and the call to listen to, the caller
    on the left and the model on the right.

    NOT A CONVERSATION. The visitor cannot hear the model while speaking, so nothing the model says
    can change what the visitor says. It is the eval run's protocol, on one more clip."""

    def __init__(self, model):
        self.model, self.rung_id = model, model.rung_id

    def load(self) -> float:
        """Load the model, then one throwaway call of silence, as the rungs take a throwaway turn:
        the first call after loading runs at about 1.7 of real time and the ones after it at 0.7
        (the eval run's first clip, and the demo's first call), and that belongs to start-up."""
        from . import duplex
        t0 = time.perf_counter()
        self.model.load()
        duplex.converse(self.model, np.zeros(duplex.IN_SR, np.float32), duplex.IN_SR, lead_s=0.0, tail_s=WARM_UP_S)
        return (time.perf_counter() - t0) * 1000

    def record(self, audio: np.ndarray, sr: int) -> dict:
        from . import duplex
        conv = duplex.converse(self.model, audio, sr)
        m = duplex.measure(audio, conv, sr)
        return {"kind": "duplex", "transcript": None, "reply_text": m["reply_text"], "gap_ms": m["gap_ms"], "setup_s": m["setup_s"],
                "overlap": m["overlap"], "opened": m["opened"], "backchannels": m["backchannels"], "interruptions": m["interruptions"],
                "rtf": m["rtf"], "stream_s": m["stream_s"], "lead_s": conv.lead_s, "total_ms": round(1000 * (conv.wall_s + conv.begin_s), 1),
                "reply_wav": base64.b64encode(wav_bytes(duplex.stereo(audio, conv, sr), duplex.SR)).decode("ascii")}


def any_rung(rung_id: str):
    """A registered rung, or a full-duplex model behind a Call."""
    from . import r5
    return Call(r5.get(rung_id)) if rung_id in r5.MODELS else ladder.get(rung_id)


class Live:
    """The rungs the live part offers, each loaded once: on first use, or all at once by
    load_all(). `offered` maps the label a visitor sees to a registered rung id: {"R0": "r0-gpu"}.

    A rung that does not load, or fails on a recording, is answered for with a sentence, and the
    other rungs still answer. What went wrong goes to the server's log, never to the visitor."""

    def __init__(self, offered: dict, get=any_rung, speaker=None):
        self.offered, self._get, self._loaded, self.load_ms = dict(offered), get, {}, {}
        self.unavailable, self._answered = {}, False
        self._speaker, self._speak = speaker, None                # the severity render's voice: made on first use

    def render(self, voice: str, sentence: str, condition: str, score: float) -> dict:
        """The severity render (render_severity), on this machine."""
        if self._speak is None:
            self._speak = (self._speaker or kokoro_speaker)()
        return render_severity(self._speak, voice, sentence, condition, score)

    def rung(self, label: str):
        if label not in self.offered:
            raise KeyError(f"no rung {label!r} here; there are {', '.join(self.offered)}")
        if label not in self._loaded:
            r = self._get(self.offered[label])
            self.load_ms[label] = round(r.load(), 1)
            self._loaded[label] = r
        return self._loaded[label]

    def _failed(self, label: str, doing: str, e: Exception, sentence: str) -> dict:
        print(f"rung {label} ({self.offered[label]}) failed while {doing}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return {"label": label, "rung": self.offered[label], "error": sentence}

    def load_all(self) -> float:
        """Load every rung offered, now. Returns the seconds it took."""
        t0 = time.perf_counter()
        for label in self.offered:
            try:
                self.rung(label)
            except Exception as e:                                  # a rung's models: any error is "not here"
                self.unavailable[label] = self._failed(label, "loading", e, NOT_LOADED)["error"]
        return time.perf_counter() - t0

    def turns(self, labels, audio: np.ndarray) -> dict:
        """One recording through each rung named. "cold": this is the first recording answered
        since the process began, so whoever waited for it also waited for the machine."""
        cold, self._answered = not self._answered, True
        return {"turns": [self.turn(label, audio) for label in labels], "cold": cold}

    def turn(self, label: str, audio: np.ndarray) -> dict:
        """One turn through one rung, as a record the page can show. The reply travels as a WAV."""
        if label not in self.offered:
            raise KeyError(f"no rung {label!r} here; there are {', '.join(self.offered)}")
        if label in self.unavailable:
            return {"label": label, "rung": self.offered[label], "error": self.unavailable[label]}
        try:
            rung = self.rung(label)
        except Exception as e:
            self.unavailable[label] = NOT_LOADED
            return self._failed(label, "loading", e, NOT_LOADED)
        try:
            if hasattr(rung, "record"):                           # a Call: one conversation, not a turn
                return {"label": label, "rung": self.offered[label], **rung.record(audio, SR)}
            t = rung.respond(audio, SR)
        except Exception as e:
            return self._failed(label, "answering", e, NOT_ANSWERED)
        clean = lambda x: None if x != x else round(float(x), 1)
        return {"label": label, "rung": self.offered[label], "transcript": t.transcript, "reply_text": t.reply_text,
                "stages_ms": {k: round(float(v), 1) for k, v in t.stages_ms.items()}, "total_ms": round(float(t.total_ms), 1),
                "first_audio_ms": clean(t.first_audio_ms), "first_sound_ms": clean(t.first_sound_ms),
                "stall_ms": round(float(t.stall_ms), 1), "reply_s": round(t.reply_s, 2),
                "reply_wav": base64.b64encode(wav_bytes(t.audio, t.sr)).decode("ascii") if len(t.audio) else None}


class Remote:
    """The live part where the rungs are on another machine: what the web app holds in place of a
    Live. `call(labels, samples)` reaches the other end, where answer() hands them to a Live. The
    recording travels as float32 samples at 16 kHz."""

    def __init__(self, offered: dict, call, render=None):
        self.offered, self._call, self._render = dict(offered), call, render
        if render is not None:                                    # only a machine that renders says it can
            self.render = lambda voice, sentence, condition, score: self._render(voice, sentence, condition, score)

    def turns(self, labels, audio: np.ndarray) -> dict:
        return self._call(list(labels), np.ascontiguousarray(audio, dtype=np.float32).tobytes())


def answer(live: Live, labels, samples: bytes) -> dict:
    """The other end of a Remote."""
    return live.turns(labels, np.frombuffer(samples, dtype=np.float32))


class Several:
    """The live part over more than one machine. `first` answers every recording; each of the
    `others` answers only when a visitor asks for one of its rungs, because it wakes a GPU of its
    own. The machines are asked at the same time, so the wait is the longest of theirs and not
    the sum. A machine that does not answer is a sentence on its rungs' cards; the rest answer.

    `others` maps a machine's name (as in Budget.machines) to a Live or a Remote; `says` maps it
    to what a visitor is told: {"a10": "an A10"}."""

    def __init__(self, first, others: dict, says: dict | None = None, clock=time.time):
        self.first, self.others, self._clock = first, dict(others), clock
        self.says = {None: None, **{name: name for name in self.others}, **(says or {})}     # the first is named by whoever serves it
        self.default = list(first.offered)
        self.offered = dict(first.offered)
        self.optional = {}
        if hasattr(first, "render"):                              # the render is the first machine's: Kokoro is there
            self.render = first.render
        for name, machine in self.others.items():
            self.offered.update(machine.offered)
            self.optional.update({label: name for label in machine.offered})

    def _ask(self, name, machine, labels, audio) -> tuple[dict, dict]:
        start = self._clock()
        try:
            out = machine.turns(labels, audio)
        except Exception as e:
            print(f"{self.says[name] or 'the first machine'} did not answer: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            out = {"turns": [{"label": label, "rung": machine.offered[label], "error": NO_MACHINE} for label in labels], "cold": False}
        used = {"machine": name, "says": out.get("machine") or self.says[name], "start": start, "end": self._clock(),
                "cold": bool(out.get("cold")), "load_s": out.get("load_s"),
                "busy_s": sum(t.get("total_ms", 0.0) for t in out["turns"]) / 1000}
        return out, used

    def turns(self, labels, audio: np.ndarray) -> dict:
        from concurrent.futures import ThreadPoolExecutor
        asked = [(name, machine, [label for label in labels if label in machine.offered])
                 for name, machine in [(None, self.first), *self.others.items()]]
        asked = [a for a in asked if a[2]]
        with ThreadPoolExecutor(max_workers=max(1, len(asked))) as pool:
            answers = list(pool.map(lambda a: self._ask(a[0], a[1], a[2], audio), asked))
        by_label = {t["label"]: dict(t, machine=used["says"]) for out, used in answers for t in out["turns"]}
        first = answers[0] if answers else ({}, {})
        return {"turns": [by_label[label] for label in labels], "used": [used for _, used in answers],
                "cold": any(used["cold"] for _, used in answers), "machine": first[1].get("says"), "load_s": first[0].get("load_s")}
