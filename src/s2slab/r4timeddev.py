"""
M37, step 2: the timed start on development clips, before anything is registered.

One container, the 30 probe clips (development; the eval set is not touched): r4-gpu-stream-sized as deployed
(M34, PR-016), run with no hold, every turn's times kept: each code's, each piece's as decoded, each window's
decode; and at load, each DiT window's decode as replayed. What a start rule would have done is computed from
those times afterwards (r4timed.simulate) for every margin in MARGINS_MS, and today's 0.5 s buffer from the
same pieces (r4streamdev.buffered): every rule scored on the very same turns. Whisper small.en hears each
reply against the Thinker's text, as in M34.

THE RULES, written before the run (docs/m37-log.md):
    the margin      the smallest of MARGINS_MS with no stalled turn (any stall over 0 ms, as in M34).
    the stop        no margin stall-free, or the chosen one's first-sound p50 less than STOP_GAIN_MS sooner
                    than today's 0.5 s buffer on the same turns: M37 stops there, nothing registered.
    alongside       a fixed wait after the first block, the smallest of WAITS_MS with no stalled turn: what
                    timing from the turn adds. Reported; it decides nothing.

    modal run modal_app/r4_bench.py::timeddev       one L40S; saves runs/r4-gpu/timed-dev-l40s.json
"""

from __future__ import annotations

from .r4 import OUT_SR

MARGINS_MS = tuple(range(0, 501, 50))
WAITS_MS = tuple(range(0, 2001, 50))
STOP_GAIN_MS = 150
TODAY_S = 0.5                                    # r4stream.START_BUFFER_S, frozen for PR-015 and PR-016
LISTEN = 3                                       # replies kept as WAVs
RUNG = "r4-gpu-stream-sized"


def run(clips: dict, sr: int, asr=None, base=None, make_fast=None, listen: int = LISTEN) -> dict:
    """r4-gpu-stream-sized with no hold on every clip, its times kept. asr and make_fast as in r4streamdev.run."""
    from .r0 import WhisperASR
    from .r4 import OmniS2S
    from .r4sized import SizedStreamingSpeechToSpeech
    from .r4stream import FastOmni
    from .r4streamdev import to_16k, turn_record, wav_bytes
    asr = asr or WhisperASR("small.en", device="cuda", compute_type="float16")
    asr.load()
    base = base or OmniS2S()
    rung = SizedStreamingSpeechToSpeech(make_fast(base) if make_fast else FastOmni(base), start_buffer_s=0.0)
    load_ms = rung.load()
    left, right = rung._context()
    turns, listened = {}, {}
    for i, (clip, audio) in enumerate(clips.items()):
        turn = rung.respond(audio, sr)
        t = rung.last_timing
        turns[clip] = dict(turn_record(turn, asr.transcribe(to_16k(turn.audio)), rung.last_codes),
                           **{k: rung.last_turn.get(k) for k in ("places", "prompt", "cut")},
                           code_ms=[round(x, 1) for x in t["code_ms"]],
                           made=[[round(ms, 1), n] for ms, n in t["made"]], decode_ms=t["decode_ms"])
        if i < listen:
            listened[f"{RUNG}/{clip}.wav"] = wav_bytes(turn.audio)
    return {"clips": list(clips), "rung": RUNG, "load_ms": round(load_ms, 1), "load_phases": rung.load_phases,
            "first": rung.first, "block": rung.block, "left": left, "right": right,
            "window_ms": {str(n): ms for n, ms in sorted(rung.window_ms.items())}, "turns": turns, "listen": listened}


def _stats(per: dict) -> dict:
    from .r4streamdev import _pct
    v = list(per.values())
    return {"first_sound_p50": _pct([x["first_sound_ms"] for x in v], 50),
            "first_sound_p90": _pct([x["first_sound_ms"] for x in v], 90),
            "stalled": sum(x["stall_ms"] > 0 for x in v), "stall_p90": _pct([x["stall_ms"] for x in v], 90)}


def score(blob: dict) -> dict:
    """Every rule on the same turns: today's buffer, the timed start at each margin, a fixed wait at each."""
    from .r4streamdev import buffered
    from .r4timed import release, second_window, simulate
    window = blob["window_ms"][str(second_window(blob["first"], blob["block"], blob["left"], blob["right"]))]
    turns = blob["turns"]

    def rule(fn) -> dict:
        per = {clip: fn(t) for clip, t in turns.items()}
        return {**_stats(per), "turns": per}

    return {"window_ms": window,
            "today": rule(lambda t: buffered(t["made"], OUT_SR, TODAY_S, t["leading_silence_ms"])),
            "margins": {m: rule(lambda t, m=m: simulate(t["made"], t["code_ms"], t["leading_silence_ms"], m / 1000,
                                                        window, blob["block"])) for m in MARGINS_MS},
            "waits": {w: rule(lambda t, w=w: release(t["made"], t["made"][0][0] + w, t["leading_silence_ms"]))
                      for w in WAITS_MS}}


def choose(blob: dict) -> dict:
    """The rules above, applied: the margin, whether M37 goes on to PR-017, and the fixed wait alongside."""
    s = score(blob)
    strip = lambda r: {k: v for k, v in r.items() if k != "turns"}          # noqa: E731
    fixed = next((w for w in WAITS_MS if s["waits"][w]["stalled"] == 0), None)
    out = {"today": strip(s["today"]), "fixed_wait_ms": fixed, "fixed": None if fixed is None else strip(s["waits"][fixed])}
    margin = next((m for m in MARGINS_MS if s["margins"][m]["stalled"] == 0), None)
    if margin is None:
        return {**out, "margin_ms": None, "timed": None, "gain_ms": None, "go": False,
                "why": f"every margin from {MARGINS_MS[0]} to {MARGINS_MS[-1]} ms stalls a turn"}
    timed = strip(s["margins"][margin])
    gain = round(out["today"]["first_sound_p50"] - timed["first_sound_p50"], 1)
    go = gain >= STOP_GAIN_MS
    return {**out, "margin_ms": margin, "timed": timed, "gain_ms": gain, "go": go,
            "why": f"first sound p50 {gain:,.0f} ms sooner than today's {TODAY_S:g} s buffer"
                   + ("" if go else f", under the {STOP_GAIN_MS} ms the stop rule asks")}


def diagnose(blob: dict) -> dict:
    """What the plan rests on, per turn: the ms a code when the first piece exists, the second window's decode
    in the turn against the load's, and the wait playback needed after the first piece against the plan's
    (margin 0)."""
    from .r4streamdev import _pct
    from .r4timed import needed_start, second_window
    from .r4stream import planned_start
    window = blob["window_ms"][str(second_window(blob["first"], blob["block"], blob["left"], blob["right"]))]
    per_code, decode, need, short = [], [], [], []
    for t in blob["turns"].values():
        made, codes = t["made"], t["code_ms"]
        if len(made) < 2:
            continue
        t1, n1 = made[0]
        known = [c for c in codes if c <= t1]
        per_code.append((known[-1] - known[0]) / (len(known) - 1) if len(known) > 1 else 0.0)
        decode.append(t["decode_ms"][1])
        need.append(needed_start(made) - t1)
        short.append(need[-1] - (planned_start(t1, known, blob["block"], window, n1 / OUT_SR, 0.0) - t1))
    q = lambda v: {"p50": _pct(v, 50), "p90": _pct(v, 90), "max": round(max(v), 1) if v else None}   # noqa: E731
    return {"turns": len(need), "window_ms_at_load": window, "ms_a_code": q(per_code), "second_decode_ms": q(decode),
            "needed_wait_ms": q(need), "plan_short_by_ms": q(short)}


def format_timed_dev(blob: dict) -> str:
    from .r4timed import second_window
    s, c, d = score(blob), choose(blob), diagnose(blob)
    f = lambda x: "-" if x is None else f"{x:,.0f}"                          # noqa: E731
    places: dict = {}
    for t in blob["turns"].values():
        places[t["places"]] = places.get(t["places"], 0) + 1
    w = second_window(blob["first"], blob["block"], blob["left"], blob["right"])
    lines = [f"R4 streamed ({blob['rung']}), its start timed from the turn, on {len(blob['clips'])} probe clips (development)", "",
             "The caches: " + ", ".join(f"{n} places on {k} turns" for n, k in sorted(places.items()))
             + f"; replies cut: {sum(bool(t['cut']) for t in blob['turns'].values())}.",
             f"The second window ({w} codes) decoded in {d['window_ms_at_load']:,.1f} ms at load; in the turns "
             f"p50 {f(d['second_decode_ms']['p50'])}, max {f(d['second_decode_ms']['max'])} ms.",
             f"The turn's ms a code at its first block: p50 {d['ms_a_code']['p50']}, max {d['ms_a_code']['max']}.",
             f"The wait playback needed after the first block: p50 {f(d['needed_wait_ms']['p50'])}, p90 "
             f"{f(d['needed_wait_ms']['p90'])}, max {f(d['needed_wait_ms']['max'])} ms; the plan (margin 0) short by "
             f"p50 {f(d['plan_short_by_ms']['p50'])}, max {f(d['plan_short_by_ms']['max'])} ms.", "",
             "| start rule | first sound p50 ms | p90 | stalled turns | stall p90 ms |", "|---|---:|---:|---:|---:|"]
    row = lambda name, r: (f"| {name} | {f(r['first_sound_p50'])} | {f(r['first_sound_p90'])} | {r['stalled']} | "  # noqa: E731
                           f"{f(r['stall_p90'])} |")
    lines.append(row(f"today: the {TODAY_S:g} s buffer", s["today"]))
    lines += [row(f"timed, margin {m} ms", s["margins"][m]) for m in MARGINS_MS]
    lines.append(row("fixed wait after the first block, " + ("none stall-free" if c["fixed_wait_ms"] is None
                                                              else f"{c['fixed_wait_ms']} ms (the smallest stall-free)"),
                     c["fixed"] or {"first_sound_p50": None, "first_sound_p90": None, "stalled": "-", "stall_p90": None}))
    lines += ["", "The rules (written before the run):",
              "  the margin: " + ("none" if c["margin_ms"] is None else f"{c['margin_ms']} ms, the smallest with no stalled turn"),
              f"  {'GO: register PR-017' if c['go'] else 'STOP: nothing registered'}: {c['why']}"]
    return "\n".join(lines)
