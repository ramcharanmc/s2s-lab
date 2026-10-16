"""
M37: R4 streamed, its start timed from the turn (s2slab.r4timed), and its development run (s2slab.r4timeddev):
the plan; the timer that hands the held pieces on while the Talker's thread is busy; the simulation that
scores every rule on the same turns, and agrees with the timer; the rules. The rung and the run themselves
on the tiny Talker of tests/test_talkerloop.py (skipped without transformers).
"""

import time

import numpy as np
import pytest

from s2slab import r4timed, r4timeddev
from s2slab.r4 import OUT_SR
from s2slab.r4stream import SAMPLES_PER_CODE, BlockDecoder, CodeStreamer, FastOmni, TimedStart, planned_start
from s2slab.r4streamdev import buffered, heard

FIRST = 23 * SAMPLES_PER_CODE          # the first piece: 24 codes, less the 20 ms held back for the crossfade
BLOCK = 96 * SAMPLES_PER_CODE


def wall():
    t0 = time.perf_counter()
    return lambda: (time.perf_counter() - t0) * 1000


def streamed(per=6.5, decode=520.0, blocks=3, lead=200.0):
    """A turn as M34's ran, with no hold: 36 codes, the first window decoded (50 ms), then 96 codes and a
    decode for each later block; the Talker's thread writes no code while it decodes."""
    codes, made, t = [], [], 1000.0
    for _ in range(36):
        codes.append(t)
        t += per
    t = codes[-1] + 50.0
    made.append([t, FIRST])
    for _ in range(blocks):
        for _ in range(96):
            t += per
            codes.append(t)
        t += decode
        made.append([t, BLOCK])
    return {"code_ms": codes, "made": made, "leading_silence_ms": lead, "places": 2048, "cut": False,
            "decode_ms": [50.0] + [decode] * blocks}


def blob_of(turns, window=520.0):
    return {"clips": list(turns), "rung": "r4-gpu-stream-sized", "first": 24, "block": 96, "left": 24, "right": 12,
            "window_ms": {"36": 50.0, "132": window}, "turns": turns}


# ------------------------------------------------------------------ the plan and the timer


def test_the_plan_is_the_block_at_the_turns_own_rate_plus_the_window_less_the_first_piece():
    codes = [100.0 + 10 * i for i in range(36)]                          # 10 ms a code
    assert planned_start(1000, codes, 96, 500, 0.46, 0) == pytest.approx(1000 + 960 + 500 - 460)
    assert planned_start(1000, codes, 96, 500, 0.46, 0.1) == pytest.approx(2000 + 100)
    assert planned_start(1000, codes, 0, 0, 0.46, 0) == 1000             # never before the first piece
    assert planned_start(1000, [5.0], 96, 500, 0.46, 0) == pytest.approx(1040)   # one code: no rate yet


def test_the_timer_hands_on_while_the_talkers_thread_is_busy_and_agrees_with_the_simulation():
    clock, got, made = wall(), [], []
    start = TimedStart(lambda p: got.append((clock(), len(p))), lambda now, n: now + 100, clock)
    made.append((clock(), FIRST))
    start.add([np.zeros(FIRST)])
    assert got == []
    time.sleep(0.3)                                                   # the second window's decode: nobody ticks
    made.append((clock(), BLOCK))
    start.add([np.zeros(BLOCK)])
    start.flush()
    assert [n for _, n in got] == [FIRST, BLOCK] and got[0][0] < made[1][0]      # handed on during the decode
    sim, live = r4timed.release(made, made[0][0] + 100, 0.0), heard(got, OUT_SR, 0.0)
    assert abs(sim["start_ms"] - got[0][0]) < 40 and abs(sim["first_sound_ms"] - live["first_sound_ms"]) < 40


def test_a_plan_already_due_hands_on_at_once_and_the_end_hands_on_whatever_the_plan():
    got = []
    now = TimedStart(lambda p: got.append(len(p)), lambda now, n: now, wall())
    now.add([np.zeros(3), np.zeros(4)])
    assert got == [3, 4] and now.timer is None
    got = []
    late = TimedStart(lambda p: got.append(len(p)), lambda now, n: now + 60_000, wall())
    late.add([np.zeros(3)])
    late.add([np.zeros(4)])
    assert got == []
    late.flush()
    assert got == [7]
    late.timer.join(1)
    assert not late.timer.is_alive()                                   # its timer stopped with the turn


def test_a_failed_turn_cancels_the_timer_and_hands_nothing_on():
    got = []
    start = TimedStart(lambda p: got.append(len(p)), lambda now, n: now + 50, wall())
    start.add([np.zeros(3)])
    start.cancel()
    time.sleep(0.15)
    start.add([np.zeros(4)])
    assert got == []


def test_the_streamer_and_the_decoder_keep_their_times():
    ticks, got = iter([1.0, 2.0]), []
    s = CodeStreamer(got.append, clock=lambda: next(ticks))
    s.put(np.zeros(9))
    for c in (5, 6):
        s.put(np.array([c]))
    assert got == [5, 6] and s.times == [1.0, 2.0]
    dec = BlockDecoder(lambda w: np.zeros(len(w) * SAMPLES_PER_CODE), first=24, block=96, left=24, right=12)
    for c in range(300):
        dec.push(c)
    dec.finish()
    assert len(dec.decode_ms) == len(dec.windows) == 4 and min(dec.decode_ms) >= 0


# ------------------------------------------------------------------ the simulation and the rules


def test_the_timed_start_meets_the_second_block_and_today_waits_for_it():
    t = streamed(per=6.5, decode=520.0)
    t1, t2 = t["made"][0][0], t["made"][1][0]
    timed = r4timed.simulate(t["made"], t["code_ms"], 200.0, 0.01, 520.0, 96)       # a 10 ms margin
    assert timed["stall_ms"] == 0 and timed["start_ms"] == pytest.approx(t1 + 96 * 6.5 + 520 - 460 + 10)
    assert r4timed.needed_start(t["made"]) == pytest.approx(t2 - 460)
    today = buffered(t["made"], OUT_SR, 0.5, 200.0)                   # 0.46 s is under the buffer: it waits
    assert today["first_audio_ms"] == t2 and today["stall_ms"] == 0
    assert today["first_sound_ms"] - timed["first_sound_ms"] == pytest.approx(450)
    short = r4timed.simulate(t["made"], t["code_ms"], 200.0, 0.0, 440.0, 96)          # the load's decode too short
    assert short["stall_ms"] == pytest.approx(80)


def test_a_reply_that_ends_before_the_plan_plays_at_its_end():
    made = [[1000.0, FIRST], [1100.0, 4 * SAMPLES_PER_CODE]]
    out = r4timed.release(made, 5000.0, 0.0)
    assert out["start_ms"] == 1100.0 and out["stall_ms"] == 0
    assert r4timed.release(made, 0.0, 0.0)["start_ms"] == 1000.0          # never before the first piece


def test_the_rules_take_the_smallest_stall_free_margin_and_go_on_when_it_gains_enough():
    slow_decode = streamed(per=6.5, decode=600.0)                     # its decode 80 ms over the load's
    big_cache = streamed(per=10.5, decode=520.0)                      # 6,144 places: slower codes, planned for
    c = r4timeddev.choose(blob_of({"a": slow_decode, "b": big_cache}))
    assert c["margin_ms"] == 100 and c["timed"]["stalled"] == 0 and c["go"]
    assert c["gain_ms"] >= r4timeddev.STOP_GAIN_MS and c["today"]["stalled"] == 0
    assert c["fixed_wait_ms"] == 1100                                 # 6,144 places need 1,068 ms after the first
    text = r4timeddev.format_timed_dev(blob_of({"a": slow_decode, "b": big_cache}))
    assert "GO: register PR-017" in text and "margin 100 ms" in text and "fixed wait" in text


def test_the_rules_stop_when_the_margin_eats_the_gain_or_no_margin_is_stall_free():
    worst = streamed(decode=1000.0)                                   # 480 ms over the load's: 500 needed
    c = r4timeddev.choose(blob_of({"a": worst, "b": streamed(), "c": streamed()}))
    assert c["margin_ms"] == 500 and not c["go"] and "under the 150 ms" in c["why"]
    assert "STOP" in r4timeddev.format_timed_dev(blob_of({"a": worst, "b": streamed(), "c": streamed()}))
    c = r4timeddev.choose(blob_of({"a": streamed(decode=1200.0)}))
    assert c["margin_ms"] is None and not c["go"] and "every margin" in c["why"]


def test_what_the_plan_rests_on_is_reported():
    d = r4timeddev.diagnose(blob_of({"a": streamed(per=6.5, decode=600.0)}))
    assert d["ms_a_code"]["p50"] == 6.5 and d["second_decode_ms"]["p50"] == 600.0
    assert d["needed_wait_ms"]["p50"] == pytest.approx(96 * 6.5 + 600 - 460, abs=0.1)
    assert d["plan_short_by_ms"]["p50"] == pytest.approx(80, abs=0.1)


# ------------------------------------------------------------------ the rung and the run, on the tiny Talker


def test_the_timed_rung_writes_the_sized_streams_audio_and_holds_by_its_plan():
    pytest.importorskip("transformers")
    from test_r4sized import TinyBase, same

    from s2slab.r4sized import SizedStreamingSpeechToSpeech
    fast = FastOmni(TinyBase(), compile_fn=same)
    sized = SizedStreamingSpeechToSpeech(fast, start_buffer_s=0.0)
    due = r4timed.TimedStreamingSpeechToSpeech(fast, margin_s=-1e6)     # its plan always due: nothing held
    never = r4timed.TimedStreamingSpeechToSpeech(fast, margin_s=1e6)    # never due: all of it at the end
    for rung in (sized, due, never):
        rung.load()
    assert set(due.window_ms) == {36, 132}                             # each window's decode timed at load
    clip = np.full(16_000, 2.0, dtype=np.float32)
    a, b, c = sized.respond(clip, 16_000), due.respond(clip, 16_000), never.respond(clip, 16_000)
    assert sized.last_codes == due.last_codes == never.last_codes
    assert np.array_equal(a.audio, b.audio) and np.array_equal(a.audio, c.audio)
    assert len(b.chunks) == len(a.chunks) > 1 and len(c.chunks) == 1
    t = sized.last_timing
    assert [n for _, n in t["made"]] == [n for _, n in a.chunks]        # with no hold, made is what went out
    assert len(t["code_ms"]) == len(sized.last_codes) and t["code_ms"] == sorted(t["code_ms"])
    assert len(t["decode_ms"]) == len(sized.last_windows)


def test_the_development_run_keeps_every_turns_times():
    pytest.importorskip("transformers")
    from test_r4sized import TinyBase, same
    from test_r4stream import FakeASR

    clips = {f"p{i}": np.full(16_000, float(i), dtype=np.float32) for i in range(3)}
    blob = r4timeddev.run(clips, 16_000, asr=FakeASR(), base=TinyBase(), make_fast=lambda b: FastOmni(b, compile_fn=same),
                          listen=1)
    assert blob["rung"] == "r4-gpu-stream-sized" and set(blob["window_ms"]) == {"36", "132"}
    for t in blob["turns"].values():
        assert [n for _, n in t["made"]] == [n for _, n in t["chunks"]] and len(t["code_ms"]) == t["n_codes"]
        assert t["places"] == 2048 and not t["cut"]
    assert len(blob["listen"]) == 1 and "The rules (written before the run)" in r4timeddev.format_timed_dev(blob)
