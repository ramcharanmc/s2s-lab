"""
M43's development check, on the laptop, free: does the watermark stay findable, stay out of what was never marked,
and leave the severity render as it was? Against a rule written before the check (docs/m43-log.md).

    PYTHONPATH=src python -m s2slab.watermarkdev           step 1a: the full mark on the dev set (recorded: STOP)
    PYTHONPATH=src python -m s2slab.watermarkdev alpha     step 1b: weaker marks on the dev set; the rule picks one
    PYTHONPATH=src python -m s2slab.watermarkdev confirm   step 1c: that mark, once, on PR-009's held-out set

THE MATERIAL. severity's renders: on DEV its two dev voices and four dev sentences, on the CONFIRMATION PR-009's two
held-out voices and eight held-out sentences; every one of the twelve scores (both scales), at 24 kHz. As controls
never marked: the same clips unmarked, and the eval set's 200 real clips (16 kHz).

FOR EACH RENDER, marked as the demo will mark it (the neutral clip and the render, each on its own):
  * severity's four measures (severity.achieved) on the marked pair against the unmarked pair;
  * the mark read back as written, after a 16-bit WAV, after a trip through 16 kHz, and after the browser's
    WebM/Opus at 48 kHz (reported only);
  * how far below the speech the mark sits (watermark.level_db).
FOR EACH NEUTRAL CLIP as Kokoro spoke it, untrimmed (a cascade's reply): the leading silence the rungs measure
(rungs.silence_before_speech_ms), marked against unmarked.

THE GATE (unchanged since step 1a): on every clip,
  * every marked render and neutral clip reads as marked after the 16-bit WAV and after the trip through 16 kHz;
  * nothing unmarked reads as marked: the renders, the neutral clips, the eval set's 200;
  * the pitch offset moves under 0.3 Hz, and the pitch range, the rate and the pauses under 1%: a tenth of
    PR-009's own tolerances (3 Hz; 10%);
  * a reply's leading silence moves under 20 ms, one of the measure's frames.

STEP 1a: the full mark (alpha 1) on DEV. Recorded: STOP, the pitch offset and range over their bars.
STEP 1b, the rule written before it: the gate for each weaker mark in ALPHAS on DEV; the mark CHOSEN is the
  strongest that passes in full (a stronger mark is the harder to strip). None passes: STOP, nothing wired in.
STEP 1c, the rule written before it: the chosen mark, once, on PR-009's held-out voices and sentences, the same gate.
  GO: the mark is wired in at that strength. Otherwise STOP, nothing wired in.
"""

from __future__ import annotations

import io
import json
import math
import pathlib
import subprocess

import numpy as np

RUN = pathlib.Path("runs/watermark/dev.json")
ALPHA_RUN = pathlib.Path("runs/watermark/alpha-dev.json")
CONFIRM_RUN = pathlib.Path("runs/watermark/confirm.json")
OFFSET_HZ_MAX, REL_MAX, SILENCE_MS_MAX = 0.3, 0.01, 20.0
RELATIVE = ("f0_dev_scale", "rate", "pause_scale")
ALPHAS = (0.5, 0.3, 0.2, 0.1)           # step 1b, strongest first


def quantize16(x) -> np.ndarray:
    """A 16-bit WAV's samples, read back as float32."""
    return (np.round(np.clip(np.asarray(x, dtype=np.float32), -1, 1) * 32767) / 32767).astype(np.float32)


def opus_roundtrip(x, sr: int) -> tuple[np.ndarray, int]:
    """Through WebM/Opus at 48 kHz, as a browser records and plays it; (audio, 48000)."""
    import av

    from .watermark import to_rate
    x48 = to_rate(x, sr, 48_000)
    buf = io.BytesIO()
    out = av.open(buf, "w", format="webm")
    stream = out.add_stream("libopus", rate=48_000)
    stream.layout = "mono"
    n = stream.codec_context.frame_size or 960
    for i in range(0, len(x48), n):
        chunk = x48[i:i + n]
        if len(chunk) < n:
            chunk = np.pad(chunk, (0, n - len(chunk)))
        frame = av.AudioFrame.from_ndarray(chunk[None, :], format="flt", layout="mono")
        frame.sample_rate, frame.pts = 48_000, i
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()
    buf.seek(0)
    with av.open(buf) as cin:
        parts = [f.to_ndarray().reshape(-1) for f in cin.decode(audio=0)]
    return np.concatenate(parts).astype(np.float32), 48_000


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def _readings(wm, x, sr: int, opus: bool) -> dict:
    from .watermark import to_rate
    out = {"as_written": wm.detect(x, sr), "wav16": wm.detect(quantize16(x), sr),
           "via16k": wm.detect(to_rate(x, sr, 16_000), 16_000)}
    if opus:
        out["opus"] = wm.detect(*opus_roundtrip(x, sr))
    return out


def marked_sets(speak, wm, alphas, voices=None, sentences=None, opus: bool = True) -> dict:
    """{alpha: {"neutrals": [...], "renders": [...]}}: every clip spoken and rendered once, then marked at each alpha."""
    from . import severity
    from .rungs import silence_before_speech_ms
    from .watermark import MARK_RATE, level_db
    sr = severity.KOKORO_SR
    assert sr == MARK_RATE
    voices = severity.DEV_VOICES if voices is None else voices
    sentences = severity.DEV_SENTENCES if sentences is None else sentences
    out = {a: {"neutrals": [], "renders": []} for a in alphas}
    for voice in voices:
        for sid, text in sentences.items():
            raw = speak(text, voice)
            neutral = severity.trim(raw, sr)
            nid = f"{voice}-{sid}"
            plain_silence, neutral_unmarked = _num(silence_before_speech_ms(raw, sr)), wm.detect(neutral, sr)
            renders = []
            for condition, scores in severity.SCORES.items():
                for score in scores:
                    t = severity.CONDITIONS[condition](score)
                    r = severity.render(neutral, sr, t)
                    renders.append((f"{nid}-{condition}-{score:g}", r, severity.achieved(neutral, r, sr),
                                    wm.detect(r.audio, sr)))
            for a in alphas:
                neutral_marked = wm.embed(neutral, sr, alpha=a)
                out[a]["neutrals"].append({"id": nid, "silence_ms": [plain_silence,
                                                                     _num(silence_before_speech_ms(wm.embed(raw, sr, alpha=a), sr))],
                                           "unmarked": neutral_unmarked, **_readings(wm, neutral_marked, sr, False)})
                for rid, r, plain, unmarked in renders:
                    marked = severity.Render(wm.embed(r.audio, sr, alpha=a), r.pauses_in, r.pauses_out)
                    got = severity.achieved(neutral_marked, marked, sr)
                    out[a]["renders"].append({"id": rid, "plain": {f: _num(getattr(plain, f)) for f in severity.FIELDS},
                                              "marked_measures": {f: _num(getattr(got, f)) for f in severity.FIELDS},
                                              "level_db": level_db(r.audio, marked.audio), "unmarked": unmarked,
                                              **_readings(wm, marked.audio, sr, opus)})
    return out


def run(speak, wm, controls, opus: bool = True) -> dict:
    """Step 1a: the full mark on DEV. controls: [(id, audio, sr)], clips never marked."""
    one = marked_sets(speak, wm, (1.0,), opus=opus)[1.0]
    return {**one, "controls": [{"id": cid, **wm.detect(audio, csr)} for cid, audio, csr in controls]}


def moved(row: dict) -> dict:
    """How far the mark moved each measure: Hz for the pitch offset, a share of the plain value for the rest."""
    out = {}
    for f, a in row["plain"].items():
        b = row["marked_measures"][f]
        if a is None or b is None:
            out[f] = None
        elif f == "f0_offset_hz":
            out[f] = abs(b - a)
        else:
            out[f] = abs(b - a) / abs(a) if a else None
    return out


def gate(blob: dict) -> dict:
    """The gate (module docstring), on {"neutrals", "renders", "controls"}."""
    marked = blob["neutrals"] + blob["renders"]
    lost = [r["id"] for r in marked if not (r["wav16"]["marked"] and r["via16k"]["marked"])]
    alarms = ([r["id"] for r in marked if r["unmarked"]["marked"]] + [c["id"] for c in blob["controls"] if c["marked"]])
    worst = {f: max((moved(r)[f] for r in blob["renders"] if moved(r)[f] is not None), default=0.0)
             for f in ("f0_offset_hz",) + RELATIVE}
    over = [f for f, v in worst.items() if v >= (OFFSET_HZ_MAX if f == "f0_offset_hz" else REL_MAX)]
    shifts = [abs(n["silence_ms"][1] - n["silence_ms"][0]) for n in blob["neutrals"]
              if None not in n["silence_ms"]]
    silence = max(shifts, default=0.0)
    checks = {"found": (not lost, f"{len(marked) - len(lost)} of {len(marked)} marked clips found after a 16-bit WAV "
                                  f"and a trip through 16 kHz"),
              "no false alarm": (not alarms, f"{len(alarms)} of {len(marked) + len(blob['controls'])} unmarked clips "
                                             f"read as marked"),
              "render unchanged": (not over, "the most the mark moved: pitch offset "
                                             f"{worst['f0_offset_hz']:.3f} Hz (under {OFFSET_HZ_MAX}), "
                                             + ", ".join(f"{f} {100 * worst[f]:.2f}%" for f in RELATIVE)
                                             + f" (under {100 * REL_MAX:.0f}%)"),
              "silence unchanged": (silence < SILENCE_MS_MAX, f"a reply's leading silence moved {silence:.0f} ms at most "
                                                              f"(under {SILENCE_MS_MAX:.0f})")}
    go = all(ok for ok, _ in checks.values())
    return {"go": go, "checks": checks, "lost": lost, "alarms": alarms, "worst": worst, "silence_ms": silence}


def at(blob: dict, alpha) -> dict:
    """One strength of a step-1b or step-1c blob, as gate() takes it."""
    return {**blob["alphas"][f"{float(alpha):g}"], "controls": blob["controls"]}


def choose(blob: dict) -> float | None:
    """Step 1b's rule: the strongest mark in ALPHAS that passes the gate in full; None if none does."""
    passing = [a for a in ALPHAS if f"{a:g}" in blob["alphas"] and gate(at(blob, a))["go"]]
    return max(passing) if passing else None


def _describe(blob: dict) -> list[str]:
    R = blob["renders"]
    p = lambda v, q: float(np.percentile(v, q)) if v else float("nan")             # noqa: E731
    lines = [f"  the mark below the speech: {p([r['level_db'] for r in R], 50):.1f} dB at the median "
             f"({min(r['level_db'] for r in R):.1f} to {max(r['level_db'] for r in R):.1f})"]
    for k, what in (("as_written", "as written"), ("wav16", "after a 16-bit WAV"), ("via16k", "after 16 kHz"),
                    ("opus", "after WebM/Opus (reported only)")):
        rows = [r[k] for r in R if k in r]
        if rows:
            lines.append(f"  {what}: found on {sum(x['marked'] for x in rows)} of {len(rows)}, share p50 "
                         f"{p([x['share'] for x in rows], 50):.3f}, all 16 bits on {sum(x['bits'] == 16 for x in rows)}")
    ctrl = [c["share"] for c in blob["controls"]] + [r["unmarked"]["share"] for r in R]
    lines.append(f"  unmarked clips: the share at most {max(ctrl, default=0.0):.3f}")
    return lines


def format_dev(blob: dict) -> str:
    g = gate(blob)
    lines = [f"M43, the watermark's development check: {len(blob['neutrals'])} neutral clips and {len(blob['renders'])} "
             f"renders (severity's dev set, 24 kHz), {len(blob['controls'])} eval-set clips never marked", ""]
    lines += [f"{name:<18} {'PASS' if ok else 'FAIL'}  {detail}" for name, (ok, detail) in g["checks"].items()]
    lines += ["", f"The rule (written before the check): {'GO: wire the mark in' if g['go'] else 'STOP: nothing wired in'}",
              "", "Descriptive, not the rule:"] + _describe(blob)
    return "\n".join(lines)


def format_alphas(blob: dict) -> str:
    chosen = choose(blob)
    lines = [f"M43, step 1b: weaker marks on the dev set, {len(blob['controls'])} eval-set clips never marked", ""]
    for a in ALPHAS:
        if f"{a:g}" not in blob["alphas"]:
            continue
        one = at(blob, a)
        g = gate(one)
        lines += [f"alpha {a:g}: {'PASS' if g['go'] else 'FAIL'}"]
        lines += [f"  {name:<18} {'PASS' if ok else 'FAIL'}  {detail}" for name, (ok, detail) in g["checks"].items()]
        lines += _describe(one) + [""]
    lines.append("The rule (written before the run): " + (f"CHOSEN alpha {chosen:g}, the strongest that passes; "
                                                          f"confirm it once on PR-009's held-out set" if chosen is not None
                                                          else "STOP: no weaker mark passes; nothing wired in"))
    return "\n".join(lines)


def format_confirm(blob: dict) -> str:
    one = at(blob, blob["alpha"])
    g = gate(one)
    lines = [f"M43, step 1c: alpha {blob['alpha']:g} once on PR-009's held-out set: {len(one['neutrals'])} neutral clips "
             f"and {len(one['renders'])} renders, {len(blob['controls'])} eval-set clips never marked", ""]
    lines += [f"{name:<18} {'PASS' if ok else 'FAIL'}  {detail}" for name, (ok, detail) in g["checks"].items()]
    lines += ["", f"The rule (written before the run): {'GO: wire the mark in at alpha ' + format(blob['alpha'], 'g') if g['go'] else 'STOP: nothing wired in'}",
              "", "Descriptive, not the rule:"] + _describe(one)
    return "\n".join(lines)


def _committed(path: pathlib.Path) -> bool:
    return subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], capture_output=True).returncode == 0


def main(argv=None) -> None:
    import argparse
    import warnings

    from . import severity
    from .evalset import load, verify
    from .sweep import _git_commit
    from .watermark import TAG, Watermarker
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", nargs="?", default="full", choices=["full", "alpha", "confirm"])
    a = ap.parse_args(argv)
    path = {"full": RUN, "alpha": ALPHA_RUN, "confirm": CONFIRM_RUN}[a.step]
    if path.exists():
        raise SystemExit(f"{path} exists: each step is recorded once")
    chosen = None
    if a.step == "confirm":
        if not ALPHA_RUN.exists() or not _committed(ALPHA_RUN):
            raise SystemExit(f"{ALPHA_RUN} must be run and committed first: the strength is fixed before the confirmation")
        chosen = choose(json.loads(ALPHA_RUN.read_text(encoding="utf-8")))
        if chosen is None:
            raise SystemExit("step 1b chose no strength: nothing to confirm")
    problems = verify()
    if problems:
        raise SystemExit("the eval set is not the frozen set:\n  " + "\n  ".join(problems[:10]))
    wm = Watermarker().load()
    wm.embed(np.zeros(24_000, np.float32), 24_000)                   # the first mark compiles AudioSeal's layers
    controls = [(it.id, it.audio, 16_000) for it in load()]
    head = {"commit": _git_commit(), "watermark": TAG}
    speak = severity.kokoro_speaker()
    if a.step == "full":
        blob = {"run": "M43, the watermark's development check", **head, **run(speak, wm, controls)}
        text = format_dev
    else:
        alphas = ALPHAS if a.step == "alpha" else (chosen,)
        voices, sentences = ((None, None) if a.step == "alpha" else (severity.TEST_VOICES, severity.TEST_SENTENCES))
        sets = marked_sets(speak, wm, alphas, voices, sentences)
        blob = {"run": f"M43, step {'1b: weaker marks on the dev set' if a.step == 'alpha' else '1c: the confirmation'}",
                **head, "alphas": {f"{k:g}": v for k, v in sets.items()},
                "controls": [{"id": cid, **wm.detect(audio, csr)} for cid, audio, csr in controls]}
        if a.step == "confirm":
            blob["alpha"] = chosen
        text = format_alphas if a.step == "alpha" else format_confirm
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    print(text(blob))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
