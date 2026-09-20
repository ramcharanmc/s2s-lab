"""
Tests for s2slab.bargein -- on a made-up set, with no Kokoro and no microphone.

The fake agent "speaks" one word every 0.3 s as a quiet tone. The fake caller
is silence with one burst at a known time: a 200 Hz tone for speech (it has a
pitch), white noise for a cough (it has none). So every test knows the onset,
the length and the right answer.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import numpy as np
import pytest
import soundfile as sf

from s2slab.bargein import (BARGEIN_ROOT, CALIBRATION, CELLS, MANIFEST, Word, agent_leak_db,
                            check, calibrate, duration_policy_errors, freeze, leak, load_item,
                            load_set, loud_spans, measure_lag, phrase_span_s, propose_label,
                            read_agent, read_agent_words, read_script, render, verify)
from s2slab.probes import Chunk, load_wav, read_labels, write_labels

SR = 16_000
AGENT_SR = 24_000
WORD_S = 0.3
REPO_SET = pathlib.Path(__file__).resolve().parents[1] / BARGEIN_ROOT


def fake_synth(text: str):
    """One word every 0.3 s, as noise that rises and falls like syllables -- like
    speech, it lines up with itself at one shift only -- then two silent seconds."""
    words = [Word(w.strip(".,?"), k * WORD_S, (k + 1) * WORD_S) for k, w in enumerate(text.split())]
    rng = np.random.default_rng(len(text))
    n = int(len(words) * WORD_S * AGENT_SR)
    voice = 0.1 * rng.standard_normal(n) * (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / (WORD_S * AGENT_SR)))
    return np.concatenate([voice, np.zeros(2 * AGENT_SR)]).astype(np.float32), AGENT_SR, words


def burst(onset_s: float, lasts_s: float, total_s: float, noise: bool = False, click_at=None,
          leak=None, gain=1.0):
    """leak: audio mixed in from time zero -- the agent reaching the microphone."""
    rng = np.random.default_rng(0)
    x = 1e-4 * rng.standard_normal(int(total_s * SR)).astype(np.float32)
    if leak is not None:
        m = min(len(x), len(leak))
        x[:m] += leak[:m]
    n, a = int(lasts_s * SR), int(onset_s * SR)
    x[a:a + n] += gain * (0.08 * rng.standard_normal(n) if noise
                          else 0.3 * np.sin(2 * np.pi * 200 * np.arange(n) / SR)).astype(np.float32)
    if click_at is not None:
        k = int(click_at * SR)
        x[k:k + int(0.2 * SR)] += 0.3 * rng.standard_normal(int(0.2 * SR)).astype(np.float32)
    return x


@pytest.fixture
def root(tmp_path):
    """The real script and agent lines, a fake render, and a zero-lag calibration."""
    for name in ("agent.tsv", "script.tsv"):
        shutil.copy(REPO_SET / name, tmp_path / name)
    render(tmp_path, synth=fake_synth)
    (tmp_path / CALIBRATION).write_text(json.dumps({"agent": "A3", "lag_ms": 0.0}), encoding="utf-8")
    return tmp_path


def record(root, item_id, lasts_s, delay_s=0.2, lag_s=0.0, label=True, leak_level=0.0, **kwargs):
    """Write a fake recording of one item: the burst starts delay_s after the
    agent finishes the item's `after` phrase. Returns the onset.
    leak_level: how much of the agent's own audio reaches the microphone; 0.1
    puts it about 30 dB under the caller, as the pilot's first headphones did."""
    item = read_script(root)[item_id]
    if leak_level:
        kwargs["leak"] = leak_level * load_wav(root / "agent" / f"{item['agent']}.wav")
    words = read_agent_words(root)[item["agent"]]
    onset = phrase_span_s(words, item["after"])[1] + delay_s
    total = words[-1].end_s + 3.0          # the caller stays silent past the end of the reply
    (root / "audio").mkdir(exist_ok=True)
    sf.write(str(root / "audio" / f"{item_id}.wav"),
             burst(onset + lag_s, lasts_s, total, noise=item["kind"] == "noise", **kwargs),
             SR, subtype="PCM_16")
    if label:
        write_labels([Chunk(onset + lag_s, onset + lag_s + lasts_s, item["text"])],
                     root / "labels" / f"{item_id}.txt")
    return onset




def record_all(root, short_s=0.4, long_s=2.0):
    for item_id, item in read_script(root).items():
        record(root, item_id, short_s if item["cell"].endswith("short") else long_s)



# --------------------------------------------------------------------------- #
# The committed script
# --------------------------------------------------------------------------- #


def test_the_script_is_twenty_items_five_per_cell():
    script = read_script(REPO_SET)
    assert len(script) == 20 and len(read_agent(REPO_SET)) == 5
    for cell in CELLS:
        items = [it for it in script.values() if it["cell"] == cell]
        assert len(items) == 5 and len({it["agent"] for it in items}) == 5
        assert all(it["should_stop"] == cell.startswith("stop") for it in items)
    assert [i for i, it in script.items() if it["kind"] == "noise"] == ["cs02", "cs04"]


def test_an_ambiguous_after_phrase_is_refused(root):
    """A3 says "Your appointment" and "your current medication": which one?"""
    path = root / "script.tsv"
    path.write_text(path.read_text(encoding="utf-8").replace("\tthe fourteenth\t", "\tyour\t"),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        read_script(root)


def test_an_unbalanced_script_is_refused(root):
    path = root / "script.tsv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(ln for ln in lines if not ln.startswith("cl05")), encoding="utf-8")
    with pytest.raises(ValueError, match="continue-long"):
        read_script(root)


# --------------------------------------------------------------------------- #
# The agent's side
# --------------------------------------------------------------------------- #


def test_render_writes_every_reply_and_when_each_word_is_spoken(root):
    assert sorted(p.name for p in (root / "agent").iterdir()) == [f"A{k}.wav" for k in range(1, 6)]
    words = read_agent_words(root)
    assert words["A3"][0] == Word("Your", 0.0, 0.3)
    assert len(words["A3"]) == 33
    assert sf.info(str(root / "agent" / "A3.wav")).samplerate == AGENT_SR


def test_render_never_replaces_audio_that_may_have_been_recorded_against(root):
    with pytest.raises(FileExistsError, match="already rendered"):
        render(root, synth=fake_synth)


def test_a_phrase_is_found_whatever_its_case_or_punctuation():
    words = [Word(w, k * 1.0, k + 1.0) for k, w in enumerate(["At", "half", "past", "two.", "Please"])]
    assert phrase_span_s(words, "half past two") == (1.0, 4.0)
    with pytest.raises(ValueError, match="found 0 times"):
        phrase_span_s(words, "quarter past two")


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def _speechlike(seconds=6.0):
    """Noise with a slow loudness envelope: like speech, it lines up with itself at one shift only."""
    rng = np.random.default_rng(1)
    t = np.arange(int(seconds * SR)) / SR
    return (rng.standard_normal(len(t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 0.2).astype(np.float32)


def test_the_lag_is_found_through_a_bad_acoustic_path():
    """Delayed 40 ms, quieter, low-passed by the headphone-to-microphone gap, in a noisy room."""
    played = _speechlike()
    rng = np.random.default_rng(2)
    heard = np.convolve(played, np.ones(8) / 8, mode="same") * 0.3
    recorded = np.concatenate([np.zeros(int(0.040 * SR), np.float32), heard])[:len(played)]
    recorded = recorded + 0.01 * rng.standard_normal(len(recorded)).astype(np.float32)
    lag_ms, sharpness = measure_lag(recorded, played)
    assert lag_ms == pytest.approx(40, abs=1) and sharpness > 8


def test_a_recording_of_the_room_has_no_clear_lag():
    rng = np.random.default_rng(3)
    _, sharpness = measure_lag(0.05 * rng.standard_normal(6 * SR).astype(np.float32), _speechlike())
    assert sharpness < 8


def test_calibrate_writes_the_lag_or_refuses(root):
    (root / CALIBRATION).unlink()
    (root / "audio").mkdir()
    played = _speechlike()
    sf.write(str(root / "agent" / "A3.wav"), played, SR, subtype="PCM_16")     # something with structure
    sf.write(str(root / "audio" / "calibration.wav"),
             np.concatenate([np.zeros(int(0.025 * SR), np.float32), played * 0.5]), SR, subtype="PCM_16")
    assert "+25.0 ms" in calibrate(root)
    assert json.loads((root / CALIBRATION).read_text(encoding="utf-8"))["lag_ms"] == 25.0
    rng = np.random.default_rng(4)
    sf.write(str(root / "audio" / "calibration.wav"),
             0.05 * rng.standard_normal(6 * SR).astype(np.float32), SR, subtype="PCM_16")
    with pytest.raises(ValueError, match="no clear match"):
        calibrate(root)



# --------------------------------------------------------------------------- #
# The caller's side
# --------------------------------------------------------------------------- #


def test_a_draft_label_covers_the_sound_and_ignores_the_record_click(root):
    onset = record(root, "ss03", 0.5, label=False, click_at=0.1)
    (region,) = read_labels(propose_label("ss03", root))
    assert region.start_s == pytest.approx(onset, abs=0.06)
    assert region.end_s == pytest.approx(onset + 0.5, abs=0.06)
    assert region.text == "no, wait"


def test_a_cough_is_labelled_though_it_has_no_pitch(root):
    onset = record(root, "cs02", 0.4, label=False)
    (region,) = read_labels(propose_label("cs02", root))
    assert region.start_s == pytest.approx(onset, abs=0.06) and region.text == "[cough]"

def test_a_draft_finds_the_caller_through_a_leak_of_the_agent(root):
    """The pilot, first round: the agent leaked into the microphone 30 dB under
    the caller. Measured against the noise floor that is "speech" all the way
    through; measured against the caller it is nothing."""
    onset = record(root, "ss03", 0.5, label=False, leak_level=0.1)
    (region,) = read_labels(propose_label("ss03", root))
    assert region.start_s == pytest.approx(onset, abs=0.06)
    assert region.end_s == pytest.approx(onset + 0.5, abs=0.06)


def test_loud_spans_joins_a_short_gap_and_drops_a_blip():
    x = np.zeros(4 * SR, np.float32)
    for a, b in ((1.0, 1.4), (1.5, 2.0), (3.0, 3.04)):          # a 0.1 s gap; then a 40 ms blip
        x[int(a * SR):int(b * SR)] = 0.3 * np.sin(2 * np.pi * 200 * np.arange(int(b * SR) - int(a * SR)) / SR)
    ((start, end),) = loud_spans(x)
    assert start == pytest.approx(1.0, abs=0.03) and end == pytest.approx(2.0, abs=0.03)


def test_a_draft_never_replaces_corrected_labels(root):
    record(root, "ss03", 0.5)
    with pytest.raises(FileExistsError, match="--force"):
        propose_label("ss03", root)
    propose_label("ss03", root, overwrite=True)


def test_an_item_lands_on_the_agents_timeline(root):
    """The overdub lag is taken off: a recording that is 50 ms late everywhere
    gives the same onset as one that is not."""
    (root / CALIBRATION).write_text(json.dumps({"agent": "A3", "lag_ms": 50.0}), encoding="utf-8")
    onset = record(root, "sl03", 2.0, delay_s=0.25, lag_s=0.050)
    item = load_item("sl03", root)
    assert item.onset_s == pytest.approx(onset) and item.duration_s == pytest.approx(2.0)
    assert item.entry_delay_s == pytest.approx(0.25)
    assert (item.cell, item.kind, item.should_stop, item.agent_id) == ("stop-long", "speech", True, "A3")
    assert len(item.audio) == int((33 * WORD_S + 3.0) * SR)


def test_how_much_of_the_reply_the_caller_had_heard(root):
    """sl03 comes in after "half past two": the agent's 12th word. 0.25 s later
    the 13th word is still being spoken, so 12 have been heard."""
    record(root, "sl03", 2.0, delay_s=0.25)
    item = load_item("sl03", root)
    assert item.words_finished_by(item.onset_s) == 12
    assert item.words_finished_by(item.onset_s + 0.1) == 13


@pytest.mark.parametrize("breakage, message", [
    (lambda r: write_labels([Chunk(4.0, 4.2, "no"), Chunk(4.5, 4.9, "wait")], r / "labels" / "ss03.txt"),
     "exactly one region"),
    (lambda r: write_labels([Chunk(0.5, 0.9, "no, wait")], r / "labels" / "ss03.txt"), "before the agent even"),
    (lambda r: sf.write(str(r / "audio" / "ss03.wav"), np.zeros(5 * SR, np.float32), SR), "ends before the agent"),
    (lambda r: (r / CALIBRATION).unlink(), "calibrate"),
])
def test_an_item_that_would_give_a_wrong_answer_is_refused(root, breakage, message):
    record(root, "ss03", 0.5)
    breakage(root)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        load_item("ss03", root)

# --------------------------------------------------------------------------- #
# Leak and clipping: a take that would fool every policy
# --------------------------------------------------------------------------- #


def test_a_clean_take_has_next_to_no_leak(root):
    record(root, "ss03", 0.5)
    item = load_item("ss03", root)
    assert item.leak_db < 4 and item.problems() == []


def test_a_leaking_take_is_measured_and_called_unfit(root):
    record(root, "ss03", 0.5, leak_level=0.1)
    item = load_item("ss03", root)
    assert item.leak_db > 15 and item.leak_match > 10
    assert "agent leak" in item.problems()[0]


def test_breaths_are_loud_but_are_not_the_agent(root):
    """sl03 in the pilot: +7 dB while the agent spoke, from two breaths. They do
    not line up with the agent's audio, so the take is not called a leak."""
    record(root, "ss03", 0.5)
    rng = np.random.default_rng(5)
    path = root / "audio" / "ss03.wav"
    x, _ = sf.read(str(path), dtype="float32")
    for at in (0.8, 1.6, 5.5, 7.0, 8.2):
        k = int(at * SR)
        x[k:k + int(0.4 * SR)] += 0.003 * rng.standard_normal(int(0.4 * SR)).astype(np.float32)
    sf.write(str(path), x, SR, subtype="PCM_16")
    item = load_item("ss03", root)
    assert item.leak_db > 6 and item.leak_match < 10 and item.problems() == []


def test_the_callers_own_voice_is_not_counted_as_leak(root):
    """A long interruption covers much of the reply; it must not read as leak."""
    record(root, "sl03", 4.0)
    assert load_item("sl03", root).leak_db < 4


def test_a_take_at_full_scale_is_called_unfit(root):
    record(root, "ss03", 0.5, gain=3.4)              # 0.3 x 3.4 > 1.0: soundfile clips it
    assert "full scale" in load_item("ss03", root).problems()[0]


def test_the_leak_command_reads_a_silent_take(root):
    total = read_agent_words(root)["A3"][-1].end_s + 3.0
    (root / "audio").mkdir()
    sf.write(str(root / "audio" / "leaktest.wav"), burst(1.0, 0.0, total), SR, subtype="PCM_16")
    assert "clean" in leak(root)
    sf.write(str(root / "audio" / "leaktest.wav"),
             burst(1.0, 0.0, total, leak=0.1 * load_wav(root / "agent" / "A3.wav")), SR, subtype="PCM_16")
    assert "too much" in leak(root)
    with pytest.raises(FileNotFoundError, match="stay silent"):
        leak(root, name="nothing-here")


def test_leak_needs_a_quiet_tail_to_measure_the_room():
    assert np.isnan(agent_leak_db(np.zeros(5 * SR, np.float32), agent_s=10.0, lag_s=0.0))


def test_load_set_takes_only_what_has_been_recorded_and_labelled(root):
    record(root, "sl03", 2.0)
    record(root, "cs03", 0.4, label=False)
    assert list(load_set(root)) == ["sl03"]


# --------------------------------------------------------------------------- #
# The design does what it says
# --------------------------------------------------------------------------- #


def test_no_duration_threshold_beats_chance_on_this_set(root):
    """The point of the 2 x 2: whatever N is, two cells come out wrong."""
    record_all(root)
    items = list(load_set(root).values())
    assert len(items) == 20
    for ms, wrong_cells in ((200, {"continue-short", "continue-long"}),
                            (800, {"stop-short", "continue-long"}),
                            (3000, {"stop-long", "stop-short"})):
        wrong = duration_policy_errors(items, ms)
        assert {c for c, n in wrong.items() if n} == wrong_cells
        assert sum(wrong.values()) == 10


def test_check_lists_every_item_and_previews_the_naive_policy(root):
    record_all(root)
    text = check(root)
    assert "20 of 20 items load." in text
    assert "| sl03 | stop-long | speech |" in text and "12 of 33" in text and "matches agent" in text
    assert "| 800 | 0 | 5 | 0 | 5 | 10 |" in text


def test_check_names_files_that_are_not_item_ids(root):
    """The pilot: sl03 saved as s103.wav -- the letter l as the digit 1 -- and silently skipped."""
    record(root, "sl03", 2.0)
    shutil.copy(root / "audio" / "sl03.wav", root / "audio" / "s103.wav")
    shutil.copy(root / "audio" / "sl03.wav", root / "audio" / "leaktest.wav")
    text = check(root)
    assert "s103.wav" in text and "leaktest.wav" not in text


def test_check_tells_you_to_re_record_a_leaking_take(root):
    record(root, "cl03", 2.0, leak_level=0.1)
    assert "RE-RECORD: agent leak" in check(root)


def test_check_flags_a_short_item_that_ran_long_and_a_broken_one(root):
    record(root, "ss03", 1.5)
    record(root, "cs03", 0.4)
    write_labels([Chunk(0.5, 0.9, "right")], root / "labels" / "cs03.txt")
    text = check(root)
    assert "longer than a short item should be" in text
    assert "before the agent even" in text and "1 of 20 items load." in text


# --------------------------------------------------------------------------- #
# Freezing
# --------------------------------------------------------------------------- #


def test_freeze_needs_every_item_and_then_verifies(root):
    record(root, "sl03", 2.0)
    with pytest.raises(FileNotFoundError):
        freeze(root)
    record_all(root)
    manifest = json.loads(freeze(root).read_text(encoding="utf-8"))
    assert manifest["set"] == "bargein" and len(manifest["items"]) == 20 and len(manifest["agent"]) == 5
    assert set(manifest["files"]) == {"agent.tsv", "script.tsv", "agent_words.tsv", CALIBRATION}
    assert verify(root) == []
    with pytest.raises(FileExistsError, match="never re-frozen"):
        freeze(root)

def test_a_set_with_an_unfit_take_cannot_be_frozen(root):
    record_all(root)
    record(root, "cl01", 2.0, leak_level=0.1)
    with pytest.raises(ValueError, match="cl01: agent leak"):
        freeze(root)


def test_verify_notices_a_change_anywhere(root):
    record_all(root)
    freeze(root)
    write_labels([Chunk(4.0, 4.6, "no, wait")], root / "labels" / "ss03.txt")
    audio, sr = sf.read(str(root / "agent" / "A1.wav"), dtype="float32")
    sf.write(str(root / "agent" / "A1.wav"), audio * 0.5, sr, subtype="PCM_16")
    (root / CALIBRATION).write_text(json.dumps({"agent": "A3", "lag_ms": 9.0}), encoding="utf-8")
    assert verify(root) == [f"{CALIBRATION}: changed", "agent/A1.wav: changed", "ss03: labels changed"]


def test_text_files_verify_with_either_line_ending(root):
    record_all(root)
    freeze(root)
    path = root / "script.tsv"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert verify(root) == []


def test_verify_says_so_when_the_set_is_not_frozen(root):
    assert verify(root) == [f"no {MANIFEST}: the set is not frozen yet"]


@pytest.mark.skipif(not (REPO_SET / MANIFEST).exists() or not (REPO_SET / "audio" / "sl03.wav").exists(),
                    reason="the barge-in set is not frozen, or its audio is not on this machine")
def test_the_real_recordings_still_match_their_manifest():
    assert verify(REPO_SET) == []
    assert verify(REPO_SET) == []