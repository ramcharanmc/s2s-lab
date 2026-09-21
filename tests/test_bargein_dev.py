"""
Tests for s2slab.bargein_dev -- the synthetic development set, built with fake voices.

The fake agent speaks one word every 0.3 s; the fake caller is 0.25 s of
silence (TTS padding), a tone lasting 0.15 s per word, and 0.25 s more
silence. So every onset, length and label is known.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import pytest
import soundfile as sf

from s2slab.bargein import CELLS, Word, load_set, phrase_span_s, read_agent, read_agent_words, read_script
from s2slab.bargein_dev import (DEV_ROOT, ENTRY_S, LAUGH_TEXT, NOISE_AT_S, VOICES, build, cough,
                                noise_items, place)
from s2slab.bargein_eval import score, summarize
from s2slab.bargein_policies import DurationThreshold
from s2slab.probes import read_labels

SR = 16_000
REPO_DEV = pathlib.Path(__file__).resolve().parents[1] / DEV_ROOT
PAD_S, WORD_S = 0.25, 0.15


def fake_agent(text: str):
    words = [Word(w.strip(".,?"), k * 0.3, (k + 1) * 0.3) for k, w in enumerate(text.split())]
    rng = np.random.default_rng(len(text))
    n = int(len(words) * 0.3 * 24_000)
    return np.concatenate([0.1 * rng.standard_normal(n), np.zeros(48_000)]).astype(np.float32), 24_000, words


class FakeCaller:
    def __init__(self):
        self.calls = []

    def __call__(self, text: str, voice: str) -> np.ndarray:
        self.calls.append((text, voice))
        n = int(len(text.split()) * WORD_S * SR)
        tone = 0.5 * np.sin(2 * np.pi * 200 * np.arange(n) / SR)
        pad = np.zeros(int(PAD_S * SR))
        return np.concatenate([pad, tone, pad]).astype(np.float32)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("dev")
    for name in ("agent.tsv", "script.tsv"):
        shutil.copy(REPO_DEV / name, root / name)
    caller = FakeCaller()
    lines = build(root, agent_synth=fake_agent, caller_synth=caller)
    return root, caller, lines


# --------------------------------------------------------------------------- #
# The committed script
# --------------------------------------------------------------------------- #


def test_the_dev_script_is_forty_items_on_ten_new_agent_lines():
    script = read_script(REPO_DEV)
    assert len(script) == 40 and len(read_agent(REPO_DEV)) == 10
    for cell in CELLS:
        assert sum(1 for it in script.values() if it["cell"] == cell) == 10
    noise = [it["text"] for it in script.values() if it["kind"] == "noise"]
    assert sorted(noise) == ["[cough]", "[cough]", "[short laugh]", "[short laugh]"]


def test_the_dev_set_shares_no_caller_line_and_no_agent_line_with_the_test_set():
    test_root = REPO_DEV.parent / "bargein"
    spoken = lambda root: {it["text"].lower() for it in read_script(root).values() if it["kind"] == "speech"}  # noqa: E731
    assert not spoken(REPO_DEV) & spoken(test_root)
    assert not set(read_agent(REPO_DEV).values()) & set(read_agent(test_root).values())


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def test_every_item_is_built_labelled_and_loads_like_a_real_one(built):
    root, _, lines = built
    items = load_set(root)
    assert len(items) == 40 == len(lines)
    assert all(it.problems() == [] for it in items.values())          # no leak, no clipping
    assert {it.agent_id for it in items.values()} == {f"D{k:02d}" for k in range(1, 11)}


def test_the_label_is_where_the_voice_is_not_where_the_padding_is(built):
    """The fake caller pads 0.25 s of silence in front, as a TTS does. The label
    must start at the tone, and the tone must sit where the script says."""
    root, _, _ = built
    script, words = read_script(root), read_agent_words(root)
    for item_id in ("sl03", "ss07", "cl10"):
        (region,) = read_labels(root / "labels" / f"{item_id}.txt")
        item = script[item_id]
        after_end = phrase_span_s(words[item["agent"]], item["after"])[1]
        assert ENTRY_S[0] - 0.03 <= region.start_s - after_end <= ENTRY_S[1] + 0.03
        assert region.end_s - region.start_s == pytest.approx(len(item["text"].split()) * WORD_S, abs=0.05)
        audio, _ = sf.read(str(root / "audio" / f"{item_id}.wav"), dtype="float32")
        k = int(region.start_s * SR)
        assert np.abs(audio[k - 800: k - 80]).max() < 0.01 < np.abs(audio[k + 80: k + 800]).max()


def test_callers_never_use_the_agents_voice_and_rotate_through_the_others(built):
    _, caller, _ = built
    voices = [v for _, v in caller.calls]
    assert "af_heart" not in voices and set(voices) == set(VOICES)


def test_a_cough_is_noise_and_a_laugh_is_a_voice_saying_ha(built):
    root, caller, _ = built
    assert [t for t, _ in caller.calls].count(LAUGH_TEXT) == 2
    assert "[cough]" not in [t for t, _ in caller.calls]
    items = load_set(root)
    assert 0.3 <= items["cs02"].duration_s <= 0.8 and items["cs02"].kind == "noise"


def test_a_build_is_never_repeated_over_an_existing_one(built):
    root, _, _ = built
    with pytest.raises(FileExistsError, match="already built"):
        build(root, agent_synth=fake_agent, caller_synth=FakeCaller())


def test_the_same_seed_places_every_caller_at_the_same_onset(built, tmp_path):
    root, _, lines = built
    for name in ("agent.tsv", "script.tsv"):
        shutil.copy(root / name, tmp_path / name)
    assert build(tmp_path, agent_synth=fake_agent, caller_synth=FakeCaller()) == lines


def test_place_refuses_a_sound_that_does_not_fit():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="does not fit"):
        place(FakeCaller()("one two three", "am_adam"), 9.9, 10.0, rng)
    with pytest.raises(ValueError, match="silent"):
        place(np.zeros(SR, np.float32), 1.0, 10.0, rng)


def test_a_cough_has_two_bursts_and_no_pitch():
    from s2slab.bargein import loud_spans
    from s2slab.probes import _pitch_track, _voiced_s
    c = cough(np.random.default_rng(1))
    x = np.concatenate([np.zeros(SR // 2, np.float32), 0.2 * c / np.abs(c).max(), np.zeros(SR // 2, np.float32)])
    assert 0.3 <= len(c) / SR <= 0.7
    assert _voiced_s(_pitch_track(x), 0.0, len(x) / SR) < 0.05
    assert len(loud_spans(x)) >= 1
# --------------------------------------------------------------------------- #
# Real noise clips
# --------------------------------------------------------------------------- #


def test_a_noise_clip_becomes_a_keep_going_item_in_a_real_room(tmp_path):
    """Like an ESC-50 clip: 44.1 kHz, the sound in the middle, DIGITAL silence around it."""
    rng = np.random.default_rng(2)
    clip = np.zeros(5 * 44_100, np.float32)
    clip[44_100: 2 * 44_100] = 0.3 * rng.standard_normal(44_100) * np.hanning(44_100)
    sf.write(str(tmp_path / "cough-a.wav"), clip, 44_100, subtype="PCM_16")
    sf.write(str(tmp_path / "cough-b.wav"), np.concatenate([clip, clip]), 44_100, subtype="PCM_16")
    short, long_ = noise_items(tmp_path)
    assert (short.id, short.kind, short.should_stop, short.cell) == ("cough-a", "noise", False, "continue-short")
    assert long_.cell == "continue-long"                                     # two coughs four seconds apart
    assert short.onset_s == NOISE_AT_S and 0.4 < short.duration_s < 1.1
    assert (short.audio == 0).mean() < 0.01                                  # a room, not digital silence
    outcome, = score(DurationThreshold(200), [short], 0.0)
    assert outcome.error == "false barge-in" and outcome.overrun_words == 0  # no agent line: nothing overrun



# --------------------------------------------------------------------------- #
# The design survives synthesis
# --------------------------------------------------------------------------- #


def test_b0_is_as_lost_on_the_dev_set_as_on_the_real_one(built):
    """With the fake caller every short item is under 0.5 s and every long one
    over 0.9 s, so a duration threshold is wrong about two cells, whatever it is."""
    root, _, _ = built
    items = list(load_set(root).values())
    for ms in (200, 600):
        s = summarize(score(DurationThreshold(ms), items, 0.0))
        assert s["wrong_total"] >= 15, (ms, s["wrong"])