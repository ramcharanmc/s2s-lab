"""
Tests for s2slab.ownvoice: the window rule on made-up takes, the freeze / verify /
load of a tiny set, what the Modal side is sent, and the tagger table with a fake
extractor. No microphone, no model.
"""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib

import numpy as np
import pytest
import soundfile as sf

from s2slab.affect import Tagger
from s2slab.ownvoice import PAD_S, RAW, clips, freeze, load, split_answers, tag, wav_bytes, window, windowed
from s2slab.prosodyset import Pair, verify, write_pairs

ROOT = pathlib.Path(__file__).resolve().parents[1]
SR = 16_000


def take(click_at=0.1, speech=(2.0, 3.0), length=6.0, amp=0.3, gap=None, seed=0) -> np.ndarray:
    """A made-up take: silence, an optional record click (loud, 0.15 s), noise as
    speech between the given seconds, an optional silent gap inside it."""
    rng = np.random.default_rng(seed)
    a = np.zeros(int(length * SR), np.float32)
    if click_at is not None:
        n = int(0.15 * SR)
        a[int(click_at * SR): int(click_at * SR) + n] = 0.5 * rng.standard_normal(n).astype(np.float32)
    s, e = int(speech[0] * SR), int(speech[1] * SR)
    a[s:e] = amp * rng.standard_normal(e - s).astype(np.float32)
    if gap:
        a[int(gap[0] * SR): int(gap[1] * SR)] = 0.0
    return a


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #


def test_the_window_is_the_longest_sound_padded_and_the_record_click_is_left_out():
    assert window(take()) == pytest.approx((2.0 - PAD_S, 3.0 + PAD_S))
    assert window(take(click_at=None)) == pytest.approx((2.0 - PAD_S, 3.0 + PAD_S))


def test_a_short_pause_inside_the_phrase_is_kept_and_a_long_one_splits_it():
    assert window(take(gap=(2.4, 2.7))) == pytest.approx((1.7, 3.3))                      # 0.3 s: one phrase
    a = take(click_at=None, speech=(1.0, 1.5)) + take(click_at=None, speech=(2.3, 3.5), seed=1)
    assert window(a) == pytest.approx((2.0, 3.8))                                          # 0.8 s apart: the longer wins


def test_the_window_is_clamped_to_the_take_and_silence_is_refused():
    assert window(take(click_at=None, speech=(0.1, 0.5), length=0.7)) == pytest.approx((0.0, 0.7))
    with pytest.raises(ValueError, match="no sound"):
        window(np.zeros(SR, np.float32))


def test_windowed_audio_round_trips_through_wav_bytes_at_16_khz_pcm16():
    w = windowed(take())
    assert len(w) == int((1.0 + 2 * PAD_S) * SR)
    back, sr = sf.read(io.BytesIO(wav_bytes(w)), dtype="float32")
    assert sr == SR and len(back) == len(w) and np.abs(back - np.clip(w, -1, 1)).max() < 1e-4   # PCM clips at 1
    assert sf.info(io.BytesIO(wav_bytes(w))).subtype == "PCM_16"


# --------------------------------------------------------------------------- #
# A tiny set: freeze, verify, load, and what is sent
# --------------------------------------------------------------------------- #


def own_pair(k: int, emotion: str) -> Pair:
    return Pair(f"{emotion.lower()}-o{k:02d}", "dev", emotion, "own", f"p{k}", f"own_p{k}_CAL", f"own_p{k}_{emotion}",
                0.0, 0.0, 0, 0, -1.0, 0.0)


def tiny_set(root: pathlib.Path, sr: int = SR, click=0.1) -> list[Pair]:
    pairs = [own_pair(1, "SAD"), own_pair(2, "ANG")]
    write_pairs(pairs, root / "pairs.tsv")
    (root / "audio").mkdir()
    for k, p in enumerate(pairs):
        sf.write(str(root / "audio" / f"{p.neutral}.wav"), take(click, amp=0.05, seed=10 + k), sr, subtype="PCM_16")
        sf.write(str(root / "audio" / f"{p.loaded}.wav"), take(click, amp=0.3, seed=20 + k), sr, subtype="PCM_16")
    return pairs


def test_the_set_loads_only_once_frozen_and_verifies_until_a_take_changes(tmp_path):
    pairs = tiny_set(tmp_path)
    with pytest.raises(RuntimeError, match="not frozen"):
        load(tmp_path)
    manifest = json.loads(freeze(tmp_path).read_text(encoding="utf-8"))
    assert manifest["set"] == "prosody-own" and manifest["pairs"] == 2 and len(manifest["clips"]) == 4
    assert manifest["window"] == {"floor_db": -30.0, "gap_s": 0.5, "pad_s": 0.3}
    assert verify(tmp_path) == []
    got = load(tmp_path)
    assert [p for p, _, _ in got] == pairs and all(len(c) == 6 * SR and len(u) == 6 * SR for _, c, u in got)
    with pytest.raises(FileExistsError, match="never re-frozen"):
        freeze(tmp_path)
    clip = tmp_path / "audio" / f"{pairs[0].loaded}.wav"
    sf.write(str(clip), take(amp=0.31), SR, subtype="PCM_16")
    assert verify(tmp_path) == [f"audio/{pairs[0].loaded}.wav: changed"]
    with pytest.raises(RuntimeError, match="changed"):
        load(tmp_path)


def test_a_missing_take_or_the_wrong_rate_cannot_be_frozen(tmp_path):
    pairs = tiny_set(tmp_path / "a", sr=44_100)
    with pytest.raises(ValueError, match="44100 Hz"):
        freeze(tmp_path / "a")
    tiny_set(tmp_path / "b")
    (tmp_path / "b" / "audio" / f"{pairs[1].loaded}.wav").unlink()
    with pytest.raises(FileNotFoundError, match="not recorded yet: own_p2_ANG"):
        freeze(tmp_path / "b")


def test_every_take_is_sent_twice_windowed_under_its_name_and_as_recorded_under_raw(tmp_path):
    pairs = tiny_set(tmp_path)
    freeze(tmp_path)
    got, data = clips(tmp_path)
    assert got == pairs and len(data) == 8
    for p in pairs:
        for name in (p.neutral, p.loaded):
            assert data[name + RAW] == (tmp_path / "audio" / f"{name}.wav").read_bytes()
            w, sr = sf.read(io.BytesIO(data[name]), dtype="float32")
            assert sr == SR and len(w) == int((1.0 + 2 * PAD_S) * SR)


def test_the_answers_come_apart_by_the_raw_suffix():
    answers = {"own_p1_CAL": {"p_yes": {"upset": 0.2}}, "own_p1_CAL" + RAW: {"p_yes": {"upset": 0.4}},
               "own_p1_SAD": {"p_yes": {"upset": 0.9}}, "own_p1_SAD" + RAW: {"p_yes": {"upset": 0.8}}}
    windowed_, raw = split_answers(answers)
    assert windowed_ == {"own_p1_CAL": {"p_yes": {"upset": 0.2}}, "own_p1_SAD": {"p_yes": {"upset": 0.9}}}
    assert raw == {"own_p1_CAL": {"p_yes": {"upset": 0.4}}, "own_p1_SAD": {"p_yes": {"upset": 0.8}}}


# --------------------------------------------------------------------------- #
# The tagger on the set, with a fake extractor
# --------------------------------------------------------------------------- #


def test_the_tagger_table_is_given_windowed_and_as_recorded(tmp_path):
    tiny_set(tmp_path, click=None)
    freeze(tmp_path)
    # one feature: mean |sample| - 0.1; positive is tagged sad, negative neutral. Windowed, the
    # upset takes (noise of amplitude 0.3, mean |sample| 0.24) clear it and the calm ones (0.04) do not;
    # as recorded, 5 s of silence dilute the upset takes to 0.04 -- the reason for the window.
    tagger = Tagger(("NEU", "SAD", "FEA", "ANG"), ("f",), np.zeros(1), np.ones(1),
                    np.array([[-50.0], [50.0], [0.0], [0.0]]), np.zeros(4))
    calls = []

    def extract(audio):
        calls.append(len(audio))
        return np.array([float(np.abs(audio).mean()) - 0.1])

    text = tag(tagger, tmp_path, tmp_path / "cache", extract=extract)
    head, rest = text.split("\n\nas recorded:\n")
    assert head.startswith("windowed:\n")
    assert "| SAD | 1 | 1 | 1 | 1 | 1 |" in head and "| ANG | 1 | 0 | 1 | 1 | 0 |" in head
    assert head.endswith("pairs not told apart:\n  none")
    assert "| SAD | 1 | 0 | 0 | 1 | 0 |" in rest and "| ANG | 1 | 0 | 0 | 1 | 0 |" in rest
    assert "sad-o01: neutral clip tagged neutral" in rest and "loaded clip tagged neutral" in rest
    assert sorted(calls) == [int(1.6 * SR)] * 4 + [6 * SR] * 4                      # windowed takes are 1.6 s, raw 6 s
    assert (tmp_path / "cache" / "own-windowed.npz").exists() and (tmp_path / "cache" / "own-raw.npz").exists()
    assert tag(tagger, tmp_path, tmp_path / "cache", extract=extract) == text and len(calls) == 8   # cached


def test_look_lists_every_take_and_every_pair(tmp_path):
    pytest.importorskip("parselmouth")
    from s2slab.ownvoice import look
    tiny_set(tmp_path)
    freeze(tmp_path)
    text = look(tmp_path)
    assert text.count("| own_p") == 4 and "| sad-o01 | I'm fine | SAD |" in text and "| ang-o02 | That's great | ANG |" in text


def test_the_modal_module_has_the_own_voice_entrypoint():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("r3_probe", ROOT / "modal_app" / "r3_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(getattr(module, "own", None))
