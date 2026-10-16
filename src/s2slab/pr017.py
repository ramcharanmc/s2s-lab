"""
PR-017's run and analysis: R4 streamed with its start timed from the turn, against the start buffer frozen
for PR-015 and PR-016, on the frozen eval set, once.

Every definition here is fixed by docs/preregistration.md (PR-017), committed with this file and before any
eval-set turn of r4-gpu-stream-timed existed. The run answers the 200 eval clips with r4-gpu-stream-timed (its
margin frozen at 50 ms by M37's development rule) in one container, Whisper small.en hearing every reply. Each
turn keeps when each piece was made, before any hold; what the 0.5 s buffer would have done to the very same
turn is computed from those times (r4streamdev.buffered, which reproduces StartBuffer), so the two rules are
compared on the same codes, the same pieces and the same host. This module then says which of V1, V2 and S1
to S3 hold.

    modal run modal_app/r4_bench.py::pr017 --confirm PR-017      once; saves runs/r4-gpu/pr017.json

    V1  on all 200 turns the audio handed out is the audio made (the same samples)   else: not a valid test
    V2  at most 2 of 200 replies cut at their cache's end                            else: not a valid test
    S1  first-sound p50, timed over the buffer, <= 0.90     upper end of the 95% interval
    S2  first-sound p90, timed over the buffer, <= 0.92     upper end of the 95% interval
    S3  at most 2 of 200 timed turns stall at all, and none for more than 100 ms

THE BOOTSTRAP IS PAIRED, as PR-015's and PR-016's (compare.py): 10,000 resamples of the 200 clip ids, seed 0,
both rules' percentiles recomputed on the same resampled clips.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from .compare import N_BOOT, SEED, percentile_ratio, resamples
from .r4 import OUT_SR
from .r4sized import ROOM_CODES, SIZES
from .r4stream import BLOCK_CODES, FIRST_CODES, START_BUFFER_S
from .r4timed import MARGIN_S

REGISTRATION = "PR-017"
RUN = pathlib.Path("runs/r4-gpu/pr017.json")
TIMED = "r4-gpu-stream-timed"
CUT_MAX = 2                                    # V2
P50_MAX, P90_MAX = 0.90, 0.92                  # S1, S2
STALL_TURNS_MAX, STALL_LONGEST_MS = 2, 100.0   # S3
FIXED_WAIT_MS = 1100                           # development's smallest stall-free fixed wait: descriptive only


def eval_clips(confirm: str = "", root=None) -> tuple[dict, dict]:
    """The 200 eval clips as {id: audio} and {id: cohort}. PR-017's confirmation's: refused without it."""
    if confirm != REGISTRATION:
        raise PermissionError(f"the eval clips are {REGISTRATION}'s confirmation's: pass confirm=\"{REGISTRATION}\"")
    from .evalset import EVAL_ROOT, load, verify
    problems = verify(root or EVAL_ROOT)
    if problems:
        raise RuntimeError("the eval set is not the frozen set:\n  " + "\n  ".join(problems[:10]))
    items = load(root or EVAL_ROOT)
    return {it.id: it.audio for it in items}, {it.id: it.cohort for it in items}


def run(clips: dict, sr: int, cohorts: dict, asr=None, base=None, make_fast=None) -> dict:
    """r4-gpu-stream-timed as frozen, every clip, its times kept and the buffer's turn computed beside it; no WAVs."""
    from .r0 import WhisperASR
    from .r4 import OmniS2S
    from .r4stream import FastOmni
    from .r4streamdev import buffered, to_16k, turn_record
    from .r4timed import TimedStreamingSpeechToSpeech
    asr = asr or WhisperASR("small.en", device="cuda", compute_type="float16")
    asr.load()
    base = base or OmniS2S()
    rung = TimedStreamingSpeechToSpeech(make_fast(base) if make_fast else FastOmni(base))
    load_ms = rung.load()
    left, right = rung._context()
    turns = {}
    for clip, audio in clips.items():
        turn = rung.respond(audio, sr)
        t = rung.last_timing
        rec = turn_record(turn, asr.transcribe(to_16k(turn.audio)), rung.last_codes)
        turns[clip] = dict(rec, **{k: rung.last_turn.get(k) for k in ("places", "prompt", "cut")}, cohort=cohorts[clip],
                           made=[[round(ms, 1), n] for ms, n in t["made"]], code_ms=[round(x, 1) for x in t["code_ms"]],
                           decode_ms=t["decode_ms"],
                           planned_ms=None if t["planned_ms"] is None else round(t["planned_ms"], 1),
                           buffer=buffered(t["made"], OUT_SR, START_BUFFER_S, rec["leading_silence_ms"]))
    return {"clips": list(clips), "rung": TIMED, "load_ms": round(load_ms, 1), "load_phases": rung.load_phases,
            "first": rung.first, "block": rung.block, "left": left, "right": right,
            "window_ms": {str(n): ms for n, ms in sorted(rung.window_ms.items())}, "turns": turns,
            "setting": {"first_codes": FIRST_CODES, "block_codes": BLOCK_CODES, "margin_s": MARGIN_S,
                        "buffer_s": START_BUFFER_S, "sizes": list(SIZES), "room_codes": ROOM_CODES}}


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def criteria(blob: dict, n_boot: int = N_BOOT) -> dict:
    """Everything PR-017 asks of the run: {"valid", "v", "checks": {name: (passed, detail)}, "verdict", "describe"}.
    Raises ValueError if a turn has no first sound under either rule (PR-017 drops no clip)."""
    from .r4timed import needed_start, release, second_window, simulate
    ids = blob["clips"]
    T = [blob["turns"][i] for i in ids]
    silent = [i for i, t in zip(ids, T) if _missing(t["first_sound_ms"]) or _missing(t["buffer"]["first_sound_ms"])]
    if silent:
        raise ValueError(f"no first sound for {silent[:5]}: PR-017 drops no clip -- report this as a deviation")
    idx = resamples(len(ids), n_boot)
    same = sum(sum(n for _, n in t["chunks"]) == sum(n for _, n in t["made"]) for t in T)
    cut = sum(bool(t.get("cut")) for t in T)
    timed = [t["first_sound_ms"] for t in T]
    buf = [t["buffer"]["first_sound_ms"] for t in T]
    s1, s2 = percentile_ratio(timed, buf, 50, idx), percentile_ratio(timed, buf, 90, idx)
    stalls = [t["stall_ms"] for t in T]
    n_stall, longest = sum(v > 0 for v in stalls), max(stalls)
    v = {"V1 the same audio": (same == len(ids), f"the audio handed out is the audio made on {same} of {len(ids)}; needs all"),
         "V2 no reply cut": (cut <= CUT_MAX, f"{cut} replies cut at their cache's end; needs <= {CUT_MAX}")}
    checks = {
        "S1 median": (s1.hi <= P50_MAX, f"first-sound p50 ratio {s1}; needs upper end <= {P50_MAX}"),
        "S2 tail": (s2.hi <= P90_MAX, f"first-sound p90 ratio {s2}; needs upper end <= {P90_MAX}"),
        "S3 stalls": (n_stall <= STALL_TURNS_MAX and longest <= STALL_LONGEST_MS,
                      f"{n_stall} turns stall (max {STALL_TURNS_MAX}), the longest {longest:.0f} ms (max {STALL_LONGEST_MS:.0f})"),
    }
    valid = all(ok for ok, _ in v.values())
    failed = [name for name, (ok, _) in checks.items() if not ok]
    verdict = ("NOT A VALID TEST: " + ", ".join(k for k, (ok, _) in v.items() if not ok) if not valid
               else "SUPPORTED: all three criteria hold" if not failed else "NOT SUPPORTED: failed " + ", ".join(failed))
    p = lambda values, q: float(np.percentile(values, q)) if values else math.nan     # noqa: E731
    window = blob["window_ms"][str(second_window(blob["first"], blob["block"], blob["left"], blob["right"]))]
    margin = blob["setting"]["margin_s"]
    fixed = [release(t["made"], t["made"][0][0] + FIXED_WAIT_MS, t["leading_silence_ms"]) for t in T]
    sim = [simulate(t["made"], t["code_ms"], t["leading_silence_ms"], margin, window, blob["block"]) for t in T]
    late = [t["first_audio_ms"] - t["planned_ms"] for t in T if t["planned_ms"] is not None]
    slack = [t["first_audio_ms"] - needed_start(t["made"]) for t in T if len(t["made"]) > 1]
    cohorts: dict = {}
    for t in T:
        cohorts.setdefault(t["cohort"], []).append((t["first_sound_ms"], t["buffer"]["first_sound_ms"]))
    places: dict = {}
    for t in T:
        places[str(t.get("places"))] = places.get(str(t.get("places")), 0) + 1
    words = sum(t["words"] for t in T)
    describe = {
        "turns": len(ids), "window_ms_at_load": window,
        "first_sound": {"timed": (p(timed, 50), p(timed, 90)), "buffer": (p(buf, 50), p(buf, 90)),
                        "fixed": (p([x["first_sound_ms"] for x in fixed], 50), p([x["first_sound_ms"] for x in fixed], 90))},
        "fixed_stalled": sum(x["stall_ms"] > 0 for x in fixed),
        "buffer_stalled": sum(t["buffer"]["stall_ms"] > 0 for t in T),
        "total_p50": p([t["total_ms"] for t in T], 50),
        "timer_late_ms": (p(late, 50), max(late) if late else math.nan),
        "slack_ms": (p(slack, 50), min(slack) if slack else math.nan),
        "simulated_off_ms": (p([abs(s["first_sound_ms"] - t["first_sound_ms"]) for s, t in zip(sim, T)], 90),
                             max(abs(s["first_sound_ms"] - t["first_sound_ms"]) for s, t in zip(sim, T))),
        "second_decode_ms": (p([t["decode_ms"][1] for t in T if len(t["decode_ms"]) > 1], 50),
                             max((t["decode_ms"][1] for t in T if len(t["decode_ms"]) > 1), default=math.nan)),
        "cohort_first_sound_p50": {c: (p([x[0] for x in xs], 50), p([x[1] for x in xs], 50)) for c, xs in cohorts.items()},
        "places": places, "stalls": sorted((x for x in stalls if x > 0), reverse=True),
        "wer": 100 * sum(t["word_errors"] for t in T) / words if words else math.nan,
    }
    return {"valid": valid, "v": v, "checks": checks, "verdict": verdict, "describe": describe}


def format_pr017(blob: dict, result: dict) -> str:
    """The text recorded under PR-017's Results, verbatim."""
    d = result["describe"]
    mark = lambda ok: "PASS" if ok else "FAIL"                                     # noqa: E731
    st = blob["setting"]
    lines = [f"PR-017: {TIMED} (start planned from the turn, margin {st['margin_s'] * 1000:.0f} ms) against the "
             f"{st['buffer_s']:g} s start buffer on the same turns; first block {st['first_codes']} codes, then "
             f"{st['block_codes']}; caches of {', '.join(map(str, st['sizes']))} places",
             f"  commit {blob['commit']} · {blob['machine']} · {blob['versions']}",
             f"  paired bootstrap: {N_BOOT} resamples of {d['turns']} clips, seed {SEED}", ""]
    lines += [f"{name:<20} {mark(ok)}  {detail}" for name, (ok, detail) in result["v"].items()]
    lines += [f"{name:<20} {mark(ok)}  {detail}" for name, (ok, detail) in result["checks"].items()]
    fs = d["first_sound"]
    lines += ["", result["verdict"], "", "Descriptive, not criteria:",
              f"  first sound, ms, p50 / p90: timed {fs['timed'][0]:.0f} / {fs['timed'][1]:.0f}; the {st['buffer_s']:g} s "
              f"buffer {fs['buffer'][0]:.0f} / {fs['buffer'][1]:.0f}; a fixed {FIXED_WAIT_MS:,} ms wait {fs['fixed'][0]:.0f} / "
              f"{fs['fixed'][1]:.0f} ({d['fixed_stalled']} turns stalled)",
              f"  the buffer's turns stalled: {d['buffer_stalled']}; the turn's total p50 {d['total_p50']:.0f} ms (the same "
              f"turn under both rules)",
              f"  the second window decoded in {d['window_ms_at_load']:.1f} ms at load; in the turns p50 "
              f"{d['second_decode_ms'][0]:.0f}, max {d['second_decode_ms'][1]:.0f} ms",
              f"  the timer handed on after its plan by p50 {d['timer_late_ms'][0]:.1f} ms, max {d['timer_late_ms'][1]:.1f}; "
              f"playback began before a stall would by p50 {d['slack_ms'][0]:.0f} ms, min {d['slack_ms'][1]:.0f}",
              f"  the simulation from the pieces against the live first sound: off by p90 {d['simulated_off_ms'][0]:.1f} ms, "
              f"max {d['simulated_off_ms'][1]:.1f}",
              "  first-sound p50 by cohort, timed vs buffer: " + ", ".join(
                  f"{c} {a:.0f} vs {r:.0f}" for c, (a, r) in d["cohort_first_sound_p50"].items()),
              "  the caches: " + ", ".join(f"{n} places on {k} turns" for n, k in sorted(d["places"].items())),
              f"  stalls, timed: {len(d['stalls'])} turns" + (f", longest {d['stalls'][0]:.0f} ms" if d["stalls"] else ""),
              f"  Whisper's word error rate on the replies: {d['wer']:.1f}% (the same audio under both rules)"]
    return "\n".join(lines)


def save(blob: dict, path: pathlib.Path = RUN) -> pathlib.Path:
    from .sidechannel import refuse_to_overwrite
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    return path
