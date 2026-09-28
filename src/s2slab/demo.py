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
WRONG_CODES, LOCK_S = 5, 600.0      # this many wrong codes inside LOCK_S close the live part for LOCK_S
NOT_LOADED = "This rung could not be loaded on this machine."
NOT_ANSWERED = "This rung failed on this recording."
PAGE = pathlib.Path(__file__).with_name("demo_page.html")

# What the Measured part shows: one committed run per rung, and the same four clips for all of them.
MEASURED = (
    ("R0", "naive cascade: recognizer, then LLM, then TTS, each waiting for the last", "runs/r0-gpu/evalset-modal-t4-pr002.json"),
    ("R1", "streaming cascade: the reply handed out clause by clause", "runs/r1-gpu/evalset-modal-t4-pr002.json"),
    ("R2", "two-tier: a slow recognizer (large-v3-turbo) drives the reply", "runs/r2-gpu-turbo/evalset-modal-t4-pr004.json"),
    ("R3", "speech-LLM: Qwen2-Audio hears the turn, no transcript in the loop", "runs/r3-gpu/evalset-modal-a10.json"),
    ("R4", "speech-to-speech, half duplex: Qwen2.5-Omni speaks the reply itself", "runs/r4-gpu/evalset-modal-l40s.json"),
    ("R5 Moshi", "full duplex: no turns, the model decides when to talk", "runs/r5-moshi/evalset-modal-a10.json"),
    ("R5 PersonaPlex", "full duplex, with a role and a voice prompt before every call", "runs/r5-personaplex/evalset-modal-a10.json"),
)
CLIPS = ("clean-07", "clean-11", "clean-23", "clean-37")      # LibriSpeech test-clean, CC BY 4.0; set by pick_clips()
# The verbatim panel: PR-011's run, small.en beside small.en + FT-2's LoRA, on two clips of the disfluent cohort.
VERBATIM_RUN = "runs/ft2/pr011.json"
VERBATIM_CLIPS = ("disfluent-29", "disfluent-42")             # DisfluencySpeech test split, Apache-2.0; set by pick_verbatim()
VERBATIM_NOTE = ("These two clips are the ones where the difference is largest, chosen for that; the cohort's numbers are "
                 "beside them. One speaker, performing disfluency, trained on and tested on: same-speaker adaptation.")
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
    LOCK_S, the right code included: five tries in ten minutes find no code of ten characters.

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
    a modal.Dict in the deployment); the day is the UTC date."""

    def __init__(self, store, cap_dollars: float, dollars_per_hour: float, window_s: float = 60.0, cold_start_s: float = 45.0):
        self.store, self.cap, self.rate = store, cap_dollars, dollars_per_hour
        self.window_s, self.cold_start_s = window_s, cold_start_s

    @staticmethod
    def day(t: float) -> str:
        return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d")

    def spent(self, t: float) -> float:
        return float(self.store.get(f"dollars:{self.day(t)}", 0.0))

    def left(self, t: float) -> float:
        return max(0.0, self.cap - self.spent(t))

    def allows(self, t: float) -> bool:
        return self.spent(t) < self.cap

    def charge(self, start: float, end: float) -> float:
        """Record one request, from `start` to `end` (epoch seconds). Returns the dollars it added."""
        billed_until = float(self.store.get("billed_until", 0.0))
        if start > billed_until:                                  # the container had gone to sleep
            seconds = self.cold_start_s + (end - start) + self.window_s
        else:
            seconds = max(0.0, end + self.window_s - billed_until)
        dollars = seconds / 3600 * self.rate
        self.store["billed_until"] = end + self.window_s
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


def measured(root=".", clips=CLIPS, rows=MEASURED, verbatim_clips=VERBATIM_CLIPS, verbatim_run=VERBATIM_RUN) -> dict:
    """The Measured part's data, from the committed run files under `root`. A rung whose file is
    missing is listed as missing, not left out. "audio" names every clip the page may play."""
    root = pathlib.Path(root)
    manifest = json.loads((root / "evalset" / "manifest.json").read_text(encoding="utf-8"))
    items = {i["id"]: i for i in manifest["items"]}
    panel = verbatim(root, items, verbatim_clips, verbatim_run)
    out = {"note": READ_WITH_CARE, "rungs": [], "verbatim": panel,
           "audio": list(clips) + [c["id"] for c in (panel or {"clips": []})["clips"]],
           "clips": [{"id": c, "seconds": items[c]["duration_s"], "reference": items[c]["text"].capitalize() + ".",
                      "source": f"LibriSpeech test-clean, utterance {items[c]['source_id']}, CC BY 4.0"} for c in clips]}
    for label, what, rel in rows:
        path = root / rel
        if not path.exists():
            out["rungs"].append({"label": label, "what": what, "file": rel, "missing": True})
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        turns = run["turns"]
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
                row["turns"].append({"clip": c, "transcript": t["transcript"], "reply_text": t["reply_text"],
                                     "stages_ms": t["stages_ms"], "first_audio_ms": t["first_audio_ms"], "total_ms": t["total_ms"],
                                     "reply_s": t["reply_s"]})
        out["rungs"].append(row)
    return out


# --------------------------------------------------------------------------- #
# the live part
# --------------------------------------------------------------------------- #
class Live:
    """The rungs the live part offers, each loaded once: on first use, or all at once by
    load_all(). `offered` maps the label a visitor sees to a registered rung id: {"R0": "r0-gpu"}.

    A rung that does not load, or fails on a recording, is answered for with a sentence, and the
    other rungs still answer. What went wrong goes to the server's log, never to the visitor."""

    def __init__(self, offered: dict, get=ladder.get):
        self.offered, self._get, self._loaded, self.load_ms = dict(offered), get, {}, {}
        self.unavailable, self._answered = {}, False

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

    def __init__(self, offered: dict, call):
        self.offered, self._call = dict(offered), call

    def turns(self, labels, audio: np.ndarray) -> dict:
        return self._call(list(labels), np.ascontiguousarray(audio, dtype=np.float32).tobytes())


def answer(live: Live, labels, samples: bytes) -> dict:
    """The other end of a Remote."""
    return live.turns(labels, np.frombuffer(samples, dtype=np.float32))
