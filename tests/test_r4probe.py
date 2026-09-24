"""
Tests for s2slab.r4probe -- the reply-prosody measure and the container loop,
with synthetic voices and a fake rung. No model.

A "voice" here is a harmonic tone whose pitch glides between two frequencies,
so its F0 range is known in advance: 150 -> 200 Hz is about 5 semitones,
150 -> 300 Hz about 12.
"""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib

import numpy as np
import pytest
import soundfile as sf

from s2slab.prosodyset import Pair
from s2slab.r4probe import (TIE_ST, analyse, answer_clips, compare, direction, format_criteria, format_report,
                            measure, pair_deltas, save, to_16k, wav_bytes, write_replies)
from s2slab.rungs import Reply, Rung

pytest.importorskip("parselmouth")
ROOT = pathlib.Path(__file__).resolve().parents[1]
SR = 24_000


def glide(f0_from: float, f0_to: float, seconds: float = 1.5, sr: int = SR, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    f0 = np.linspace(f0_from, f0_to, len(t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    return (amp * sum(np.sin(k * phase) / k for k in range(1, 6))).astype(np.float32)


def pair(emotion: str, k: int) -> Pair:
    a = f"10{k:02d}"
    return Pair(f"{emotion.lower()}-d{k:02d}", "dev", emotion, a, "DFA", f"{a}_DFA_NEU_XX", f"{a}_DFA_{emotion}_XX",
                0.9, 0.9, 10, 10, 60.0, 0.9)


def test_a_wider_glide_measures_as_a_wider_range_and_24k_is_resampled():
    narrow, wide = measure(wav_bytes(glide(150, 200), SR)), measure(wav_bytes(glide(150, 300), SR))
    assert narrow["seconds"] == 1.5 and 3.0 < narrow["range_st"] < 6.0 and 9.0 < wide["range_st"] < 13.0
    assert 160 < narrow["pitch_hz"] < 190                                    # not 2/3 of it: the rate was honoured
    assert len(to_16k(np.zeros(SR, np.float32), SR)) == 16_000


def test_silence_has_no_pitch_and_the_pair_does_not_move():
    silent = measure(wav_bytes(np.zeros(SR, np.float32), SR))
    assert silent["range_st"] is None and silent["pitch_hz"] is None
    p = pair("ANG", 1)
    rows = pair_deltas([p], {p.neutral: silent, p.loaded: measure(wav_bytes(glide(150, 300), SR))},
                       {p.neutral: "", p.loaded: "Oh no!"})
    assert rows[0]["d_range_st"] is None and rows[0]["direction"] is None and rows[0]["words_upset"] == 2


def test_deltas_are_upset_reply_minus_calm_reply():
    p = pair("SAD", 2)
    m = {p.neutral: {"pitch_hz": 200.0, "range_st": 4.0, "level_db": -20.0, "seconds": 2.0},
         p.loaded: {"pitch_hz": 400.0, "range_st": 7.5, "level_db": -17.0, "seconds": 3.0}}
    row = pair_deltas([p], m, {p.neutral: "Sure.", p.loaded: "I'm so sorry to hear that."})[0]
    assert (row["d_range_st"], row["d_pitch_st"], row["d_level_db"], row["direction"]) == (3.5, 12.0, 3.0, "wider")
    assert (row["words_calm"], row["words_upset"]) == (1, 6)


class GlideRung(Rung):
    """Answers a loud clip with a wide glide and a quiet one with a narrow glide."""

    rung_id, description = "glide", "a test double"

    def __init__(self):
        self.loads = 0

    def _load(self):
        self.loads += 1

    def _respond(self, audio, sr, sink):
        loud = float(np.abs(audio).max()) > 0.1
        return Reply(glide(150, 300 if loud else 200), SR, None, "Oh dear!" if loud else "Okay.")


def clip(amp: float) -> bytes:
    return wav_bytes(np.full(16_000, amp, np.float32), 16_000)


def test_the_container_loop_answers_every_clip_and_checks_a_repeat():
    p = pair("FEA", 3)
    clips = {p.neutral: clip(0.01), p.loaded: clip(0.5)}
    got = answer_clips(GlideRung(), clips, repeat=2)
    assert set(got["replies"]) == set(clips) and got["repeat"] == {p.neutral: 0.0, p.loaded: 0.0}
    assert got["replies"][p.loaded]["text"] == "Oh dear!"
    back, sr = sf.read(io.BytesIO(got["replies"][p.loaded]["wav"]), dtype="float32")
    assert sr == SR and len(back) == int(1.5 * SR)
    measures, deltas = analyse([p], {"glide": got})
    assert deltas["glide"][0]["direction"] == "wider" and deltas["glide"][0]["d_range_st"] > 4


def test_the_report_counts_moving_pairs_per_rung_and_emotion(tmp_path):
    pairs = [pair("SAD", 1), pair("ANG", 2)]
    clips = {c: clip(a) for p in pairs for c, a in ((p.neutral, 0.01), (p.loaded, 0.5))}
    answers = {"r4-gpu": answer_clips(GlideRung(), clips, repeat=1)}
    answers["r3-gpu"] = {"replies": {c: {**r, "wav": answers["r4-gpu"]["replies"][pairs[0].neutral]["wav"]}
                                     for c, r in answers["r4-gpu"]["replies"].items()}, "repeat": {}}
    measures, deltas = analyse(pairs, answers)
    text = format_report(pairs, deltas, {r: a["repeat"] for r, a in answers.items()})
    assert "| R4 (Qwen2.5-Omni speaks) | all | 2 | 2 | 0 | 0 | 0 |" in text
    assert "| R3 (Qwen2-Audio -> Kokoro) | all | 2 | 0 | 2 | 0 | 0 |" in text      # every reply the same narrow glide: same
    assert "the same clip answered twice: 1001_DFA_NEU_XX identical" in text
    assert "| sad-d01 | +5.3 + | +0.0 |" in text and "(+ wider, - narrower: beyond +-0.5 st)" in text   # R4 first, as answered
    write_replies(answers, tmp_path / "replies")
    assert len(list((tmp_path / "replies" / "r4-gpu").glob("*.wav"))) == 4
    save(tmp_path / "run.json", "2 dev pairs", pairs, answers, measures, deltas)
    kept = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert "wav" not in kept["replies"]["r4-gpu"][pairs[0].neutral] and kept["deltas"]["r4-gpu"][1]["direction"] == "wider"
    with pytest.raises(FileExistsError):
        save(tmp_path / "run.json", "again", pairs, answers, measures, deltas)


def test_the_modal_module_has_the_pairs_entrypoint_and_its_function():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module.answer, "remote") and hasattr(module.pairs, "info")


# --------------------------------------------------------------------------- #
# The band, and PR-008's criteria
# --------------------------------------------------------------------------- #


def test_the_band_calls_jitter_same():
    assert TIE_ST == 0.5
    assert [direction(d) for d in (None, 0.0, 0.05, -0.5, 0.5, 0.51, -0.51, 3.0)] == \
        [None, "same", "same", "same", "same", "wider", "narrower", "wider"]


def rows(dirs: str) -> list:
    """'w' wider, 'n' narrower, 's' same, 'x' unmeasured -- one row per pair."""
    d = {"w": 1.0, "n": -1.0, "s": 0.0, "x": None}
    return [{"id": f"p{k:02d}", "emotion": "ANG", "d_range_st": d[c], "direction": direction(d[c])}
            for k, c in enumerate(dirs)]


def test_s1_and_s2_hold_when_r4_widens_and_r3_does_not():
    r4, r3 = rows("w" * 14 + "n" * 2 + "s" * 3), rows("s" * 17 + "w" * 2)                     # 14 fixed, 2 broken
    r = compare({"r4-gpu": r4, "r3-gpu": r3})
    assert r["r4"] == {"wider": 14, "same": 3, "narrower": 2, None: 0} and r["p_one"] < 0.01
    assert len(r["fixed"]) == 14 and r["broken"] == ["p17", "p18"] and r["S1"] and r["S2"]   # R3 wider where R4 is same
    text = format_criteria(r, "PR-008")
    assert "PR-008: SUPPORTED" in text and "PR-008 secondary: R4's delivery moves consistently -- wider" in text
    assert "PR-008" not in format_criteria(r)


def test_a_flattening_r4_fails_s1_and_the_secondary_names_the_direction():
    r4, r3 = rows("w" * 4 + "n" * 16 + "s" * 5), rows("w" * 5 + "s" * 20)
    r = compare({"r4-gpu": r4, "r3-gpu": r3})
    assert r["S1"] is False and r["S2"] is False and r["p_two"] < 0.05
    text = format_criteria(r, "PR-008")
    assert "| S1 R4's reply wider for the upset caller | 4 wider, 16 narrower, 5 same |" in text
    assert "NOT SUPPORTED: failed S1, S2" in text
    assert "PR-008 secondary: R4's delivery moves consistently -- narrower for the upset caller" in text


def test_an_even_split_has_no_direction():
    r = compare({"r4-gpu": rows("w" * 8 + "n" * 9 + "x"), "r3-gpu": rows("s" * 18)})
    assert r["r4"][None] == 1 and r["p_two"] > 0.5
    assert "secondary: no consistent direction" in format_criteria(r, "PR-008")
