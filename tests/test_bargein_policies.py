"""
Tests for s2slab.bargein_policies and s2slab.bargein_eval -- no models, no recordings.

The caller is a quiet room with tone bursts at known times, so every test
knows when the sound started, how long it lasted and what the right answer is.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2slab.bargein import BARGEIN_ROOT, CELLS, BargeIn
from s2slab.bargein_eval import (WINDOW_AFTER_S, WINDOW_BEFORE_S, Outcome, context_for,
                                 format_report, score, score_set, summarize)
from s2slab.bargein_policies import (HANG_MS, POLICIES, BargeInContext, BargeInPolicy,
                                     DurationThreshold, causal_speech_prob, get, sound_run_ms)

SR = 16_000


def room(total_s: float, *bursts: tuple[float, float], seed: int = 0) -> np.ndarray:
    """A quiet room (-80 dB noise) with a 200 Hz tone from `start` for `lasts` seconds, per burst."""
    rng = np.random.default_rng(seed)
    x = 1e-4 * rng.standard_normal(int(total_s * SR)).astype(np.float32)
    for start, lasts in bursts:
        n, a = int(lasts * SR), int(start * SR)
        x[a:a + n] += (0.3 * np.sin(2 * np.pi * 200 * np.arange(n) / SR)).astype(np.float32)
    return x


def item(item_id: str, cell: str, onset_s: float, lasts_s: float, *extra, lag_s: float = 0.0,
         kind: str = "speech", total_s: float = 12.0) -> BargeIn:
    """A loaded item whose caller sound is a tone at onset_s (agent clock). The
    recording carries lag_s of overdub lag, as a real one does."""
    audio = room(total_s, (onset_s + lag_s, lasts_s), *[(a + lag_s, b) for a, b in extra])
    return BargeIn(item_id, cell, kind, cell.startswith("stop"), "A1", audio, onset_s, lasts_s,
                   0.2, 10.0, [], 3.0, 4.0, 0.3)


def ctx(audio, start_s=0.0, end_s=None, **kw) -> BargeInContext:
    return BargeInContext(audio, SR, start_s, end_s if end_s is not None else len(audio) / SR, **kw)


# --------------------------------------------------------------------------- #
# Signals that cannot see the future
# --------------------------------------------------------------------------- #


def test_the_vad_at_time_t_does_not_depend_on_audio_after_t():
    """vad.frame_speech_prob takes its noise floor from the whole clip. This one must not."""
    quiet_later = causal_speech_prob(room(8.0, (2.0, 0.5)), SR)
    loud_later = causal_speech_prob(room(8.0, (2.0, 0.5), (4.0, 3.0)), SR)
    k = int(3.5 * 50)                                   # 50 frames a second
    assert quiet_later[:k] == loud_later[:k]


def test_a_tone_in_a_quiet_room_reads_as_sound_and_the_room_does_not():
    prob = causal_speech_prob(room(4.0, (2.0, 0.5)), SR)
    assert all(p >= 0.9 for p in prob[int(2.1 * 50):int(2.4 * 50)])
    assert max(prob[int(1.0 * 50):int(1.9 * 50)]) < 0.5
    assert causal_speech_prob(np.zeros(0, np.float32), SR) == []


def test_a_sound_survives_a_short_gap_and_ends_after_a_long_one():
    """ "wait, wait" is one sound. Two sounds HANG_MS apart are two."""
    sound, quiet = [1.0], [0.0]
    run = sound_run_ms(sound * 10 + quiet * 5 + sound * 10 + quiet * 20 + sound * 3)
    assert run[9] == 200 and run[14] == 200 and run[24] == 400      # 100 ms gap: still counting
    assert run[24 + int(HANG_MS / 20)] == 0                         # 300 ms of quiet: over
    assert run[-1] == 60                                            # the next sound starts from zero


def test_the_context_answers_by_time():
    c = ctx(room(6.0, (2.0, 1.0)), partials=[(2.6, "no"), (3.1, "no wait")],
            f0_track=[0.0] * 200 + [180.0] * 100 + [0.0] * 300)
    assert c.sound_ms_at(1.9) == 0 and 380 <= c.sound_ms_at(2.4) <= 420
    assert c.partial_at(2.5) == "" and c.partial_at(2.6) == "no" and c.partial_at(5.0) == "no wait"
    assert c.f0_before(2.1, ms=200) == [0.0] * 10 + [180.0] * 10
    assert c.f0_before(0.05) == [0.0] * 5                # never reaches before zero or past t
    assert c.latest_partial(2.5) is None and c.latest_partial(3.5).text == "no wait"
    assert c.event_start_at(1.9) is None and c.event_start_at(2.5) == pytest.approx(2.0, abs=0.04)
    assert c.event_start_at(3.2) == pytest.approx(2.0, abs=0.04)          # 200 ms after the sound: still the event
    assert c.event_start_at(3.6) is None                                  # HANG_MS of quiet: over


# --------------------------------------------------------------------------- #
# B0
# --------------------------------------------------------------------------- #


def test_b0_stops_once_the_caller_has_made_enough_sound():
    decision = DurationThreshold(400).decide(ctx(room(6.0, (2.0, 1.0))))
    assert decision.fired and decision.stop_s == pytest.approx(2.4, abs=0.05)
    assert "400 ms" in decision.reason


def test_b0_keeps_going_for_a_sound_shorter_than_its_threshold():
    decision = DurationThreshold(800).decide(ctx(room(6.0, (2.0, 0.5))))
    assert not decision.fired and "keep going" in str(decision)


def test_a_decision_does_not_change_when_the_future_does():
    audio = room(8.0, (2.0, 1.0))
    a = DurationThreshold(400).decide(ctx(audio))
    cut = audio[: int((a.stop_s + 0.001) * SR)]
    b = DurationThreshold(400).decide(ctx(cut))
    assert a.stop_s == b.stop_s


def test_a_policy_only_decides_inside_its_window():
    """A loud noise at 1 s is history, not the item: the window opens at 3 s."""
    audio = room(9.0, (1.0, 1.0), (5.0, 1.0))
    assert DurationThreshold(400).decide(ctx(audio, start_s=3.0)).stop_s == pytest.approx(5.4, abs=0.05)
    assert not DurationThreshold(400).decide(ctx(audio, start_s=3.0, end_s=4.5)).fired


def test_the_registry_holds_four_baselines_and_the_word_readers():
    assert sorted(n for n, p in POLICIES.items() if p.baseline) == [
        "B0_duration_1500ms", "B0_duration_200ms", "B0_duration_400ms", "B0_duration_800ms"]
    assert get("B0_duration_400ms").baseline and not get("B0_duration_400ms").needs_partials
    assert get("B1_words").needs_partials and not get("B1_words").baseline
    assert "B1_words_min1" in POLICIES
    with pytest.raises(KeyError, match="B0_duration_200ms"):
        get("B9")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_the_recording_is_moved_onto_the_agents_clock():
    """The label says 5.0 s on the agent's clock; the recording, 200 ms late, has the
    sound at 5.2 s. In the context it must be at 5.0 s again."""
    c = context_for(item("ss01", "stop-short", 5.0, 0.6, lag_s=0.2), lag_s=0.2)
    assert c.sound_ms_at(4.98) == 0 and c.sound_ms_at(5.12) >= 80
    assert (c.start_s, c.end_s) == (5.0 - WINDOW_BEFORE_S, pytest.approx(5.6 + WINDOW_AFTER_S))


def test_a_stop_item_is_a_hit_with_a_latency_or_a_missed_interruption():
    short = item("ss01", "stop-short", 5.0, 0.6)
    (hit,) = score(DurationThreshold(400), [short], 0.0)
    assert hit.correct and hit.error == "" and hit.latency_ms == pytest.approx(400, abs=50)
    (miss,) = score(DurationThreshold(800), [short], 0.0)
    assert not miss.correct and miss.error == "missed interruption" and miss.latency_ms is None


def test_a_continue_item_is_correct_or_a_false_barge_in():
    ack = item("cl01", "continue-long", 5.0, 2.0)
    assert score(DurationThreshold(400), [ack], 0.0)[0].error == "false barge-in"
    assert score(DurationThreshold(1500), [item("cs01", "continue-short", 5.0, 0.5)], 0.0)[0].correct


def test_a_stray_noise_outside_the_window_is_not_the_item():
    """A chair creaks for a second at 1.0 s; "mm-hm" comes at 6.0 s. B0 at 800 ms
    would stop for the chair, but the chair is not what the item is about."""
    (o,) = score(DurationThreshold(800), [item("cs01", "continue-short", 6.0, 0.5, (1.0, 1.0))], 0.0)
    assert o.correct


def make_set() -> list[BargeIn]:
    lengths = {"stop-long": 3.0, "stop-short": 0.5, "continue-short": 0.5, "continue-long": 2.0}
    items = [item(f"{''.join(w[0] for w in cell.split('-'))}{k:02d}", cell, 4.0 + 0.1 * k, lengths[cell])
             for cell in CELLS for k in range(1, 6)]
    items[10].kind = items[11].kind = "noise"            # two of the continue-short items
    return items


def test_no_duration_threshold_is_right_about_the_two_by_two():
    """The design, through the real machinery: VAD, windows, scoring."""
    items = make_set()
    for ms, wrong_cells in ((200, {"continue-short", "continue-long"}),
                            (800, {"stop-short", "continue-long"})):
        s = summarize(score(DurationThreshold(ms), items, 0.0))
        assert {c for c, n in s["wrong"].items() if n} == wrong_cells and s["wrong_total"] == 10
    s = summarize(score(DurationThreshold(200), items, 0.0))
    assert (s["missed"], s["false_bargein"], s["wrong_noise"], s["n_noise"]) == (0, 10, 2, 2)
    assert len(s["hit_latencies_ms"]) == 10 and all(150 <= v <= 300 for v in s["hit_latencies_ms"])


def test_the_report_lists_latencies_and_never_a_bare_median():
    items = make_set()
    text = format_report({p: score(get(p), items, 0.0) for p in ("B0_duration_200ms", "B0_duration_800ms")})
    assert "| B0_duration_200ms | 0 | 0 | 5 | 5 | **10 of 20** | 0 | 10 | 2 of 2 |" in text
    assert "fastest / middle / slowest of 10 hits" in text and "p50" not in text
    assert "B0_duration_800ms: ss01 (missed interruption)" in text
    assert "cl01 (false barge-in at +" in text


def test_a_policy_that_is_always_right_says_so():
    class Oracle(BargeInPolicy):
        name = "oracle"

        def decide(self, ctx):
            raise AssertionError("not used")

    outcomes = [Outcome("ss01", "stop-short", "speech", True, 5.4, "heard 'no'", 400.0),
                Outcome("cs01", "continue-short", "speech", False, None, "no rule fired", None)]
    assert "oracle: every decision right" in format_report({"oracle": outcomes})


# --------------------------------------------------------------------------- #
# The frozen set is a test set
# --------------------------------------------------------------------------- #


def test_only_baselines_may_be_scored_on_the_frozen_set_without_a_registration(monkeypatch):
    class Clever(DurationThreshold):
        baseline = False

    monkeypatch.setitem(POLICIES, "B1_clever", Clever(300))
    POLICIES["B1_clever"].name = "B1_clever"
    with pytest.raises(PermissionError, match="--confirm PR-003"):
        score_set(["B1_clever"], BARGEIN_ROOT)


@pytest.mark.skipif(not (BARGEIN_ROOT / "audio" / "sl03.wav").exists(),
                    reason="the barge-in recordings are not on this machine")
def test_the_baselines_run_on_the_real_recordings():
    text = score_set(["B0_duration_400ms"])
    assert "20 items" in text and "of 20**" in text