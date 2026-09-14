"""
Tests for s2slab.probes.

Recordings are synthesized into a temporary probe folder -- a voiced tone for
each chunk of speech, a faint noise floor everywhere else -- so each test knows
exactly where the speech, the pauses and the true end are. The real recordings
are not in git, so the one test that reads them skips itself when they are
absent.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from s2slab.policies import (INCOMPLETE_TAILS, POLICIES, DecisionTheoretic,
                             FixedThreshold)
from s2slab.probes import (LEAD_KEEP_S, MIN_TRAIL_S, PROBE_ROOT, TARGET_SR, Chunk,
                           check, detect_chunks, load_item, load_set, load_wav,
                           propose_labels, read_labels, read_script, write_labels)

SR = TARGET_SR

# 0.3 s quiet, "i would like to change the", 0.7 s pause, "delivery address", 1.5 s quiet
TWO_CHUNKS = [("gap", 0.3), ("speech", 1.0), ("gap", 0.7), ("speech", 1.0), ("gap", 1.5)]
FUNCTION_TEXT = "i would like to change the | delivery address"


def _noise(seconds: float, rng, level: float = 3e-4) -> np.ndarray:
    """A real room is never digitally silent; the VAD's noise floor needs one."""
    return (level * rng.standard_normal(int(seconds * SR))).astype("float32")


def _tone(seconds: float, f0: float = 150.0, amp: float = 0.3) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)      # 10 ms fades: no clicks
    return (amp * env * np.sin(2 * np.pi * f0 * t)).astype("float32")


def make_recording(layout, seed: int = 0):
    """layout: [("speech" | "gap" | "noise", seconds), ...] -> (audio, speech spans).

    "noise" is loud but has no pitch -- a stand-in for the record click, a
    breath or a stop click. It is not counted as a speech span.
    """
    rng = np.random.default_rng(seed)
    parts, spans, t = [], [], 0.0
    for kind, secs in layout:
        seg = _noise(secs, rng)
        if kind == "speech":
            seg = seg + _tone(secs)[: len(seg)]
            spans.append((t, t + secs))
        elif kind == "noise":
            seg = seg + (0.15 * rng.standard_normal(len(seg))).astype("float32")
        parts.append(seg)
        t += secs
    return np.concatenate(parts), spans


def make_probe_root(tmp_path, items: dict):
    """items: {id: (type, script text, layout, chunk texts or None)}.

    Writes script.tsv and audio/<id>.wav, plus labels/<id>.txt placed exactly on
    the synthesized speech when chunk texts are given.
    """
    root = tmp_path / "probe"
    (root / "audio").mkdir(parents=True)
    rows = ["id\ttype\ttext"]
    for item_id, (kind, text, layout, chunk_texts) in items.items():
        rows.append(f"{item_id}\t{kind}\t{text}")
        audio, spans = make_recording(layout)
        sf.write(str(root / "audio" / f"{item_id}.wav"), audio, SR, subtype="PCM_16")
        if chunk_texts is not None:
            write_labels([Chunk(a, b, w) for (a, b), w in zip(spans, chunk_texts)],
                         root / "labels" / f"{item_id}.txt")
    (root / "script.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def labelled_item(tmp_path, layout=TWO_CHUNKS,
                  texts=("i would like to change the", "delivery address")):
    return make_probe_root(tmp_path, {"t01": ("function", FUNCTION_TEXT, layout, list(texts))})


def _row(table: str, item_id: str) -> list[str]:
    line = next(ln for ln in table.splitlines() if ln.startswith(f"| {item_id} "))
    return [c.strip() for c in line.strip().strip("|").split("|")]


# --------------------------------------------------------------------------- #
# The real script
# --------------------------------------------------------------------------- #


def test_real_script_has_thirty_items_in_three_groups():
    kinds = [v["type"] for v in read_script().values()]
    assert len(kinds) == 30
    assert (kinds.count("function"), kinds.count("content"), kinds.count("natural")) == (12, 12, 6)


def test_script_types_match_the_word_before_each_pause():
    """The script's own claim, checked against T1's list: a function item
    pauses after a word T1 flags, a content item after one it cannot. If a line
    is edited carelessly, the item silently moves to the wrong group."""
    for item_id, item in read_script().items():
        for chunk in item["chunks"][:-1]:
            last = chunk.split()[-1]
            if item["type"] == "function":
                assert last in INCOMPLETE_TAILS, (item_id, last)
            else:
                assert last not in INCOMPLETE_TAILS, (item_id, last)


def test_natural_items_have_a_prompt_and_no_chunks():
    for item in read_script().values():
        if item["type"] == "natural":
            assert item["chunks"] == []
            assert item["text"].startswith("[")


def test_scripted_text_splits_at_the_pause_marks(tmp_path):
    root = make_probe_root(tmp_path, {"t01": ("content", "a b | c d | e", TWO_CHUNKS, None)})
    assert read_script(root / "script.tsv")["t01"]["chunks"] == ["a b", "c d", "e"]


# --------------------------------------------------------------------------- #
# load_wav
# --------------------------------------------------------------------------- #


def test_other_rates_and_stereo_become_16k_mono(tmp_path):
    """The first n01 take was saved at 44.1 kHz; read as 16 kHz it would have
    been stretched to 2.76x its length."""
    t = np.arange(44100) / 44100
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 150 * t)] * 2, axis=1).astype("float32")
    path = tmp_path / "x.wav"
    sf.write(str(path), stereo, 44100, subtype="PCM_16")
    y = load_wav(path)
    assert y.ndim == 1 and y.dtype == np.float32
    assert abs(len(y) - SR) <= 2


def test_16k_mono_passes_through_unchanged(tmp_path):
    x = _tone(0.5)
    path = tmp_path / "x.wav"
    sf.write(str(path), x, SR, subtype="PCM_16")
    assert np.allclose(load_wav(path), x, atol=1e-4)


# --------------------------------------------------------------------------- #
# Label files
# --------------------------------------------------------------------------- #


def test_labels_round_trip(tmp_path):
    chunks = [Chunk(0.08, 3.32, "i would like to change the"),
              Chunk(4.66, 7.84, "delivery address for my order")]
    assert read_labels(write_labels(chunks, tmp_path / "l.txt")) == chunks


def test_spectral_and_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("0.1\t0.5\thello\n\\\t100\t2000\n\n1.0\t2.0\tworld\n", encoding="utf-8")
    assert [c.text for c in read_labels(p)] == ["hello", "world"]


def test_labels_come_back_in_time_order(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("2.0\t3.0\tsecond\n0.1\t1.0\tfirst\n", encoding="utf-8")
    assert [c.text for c in read_labels(p)] == ["first", "second"]


def test_a_label_without_text_reads_as_empty(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("0.1\t0.5\n", encoding="utf-8")
    assert read_labels(p)[0].text == ""


# --------------------------------------------------------------------------- #
# detect_chunks
# --------------------------------------------------------------------------- #


def test_finds_each_stretch_of_speech():
    audio, spans = make_recording(TWO_CHUNKS)
    got = detect_chunks(audio)
    assert len(got) == 2
    for (a, b), (x, y) in zip(got, spans):
        assert abs(a - x) < 0.06 and abs(b - y) < 0.06


def test_gaps_shorter_than_a_pause_are_merged():
    audio, _ = make_recording([("gap", 0.3), ("speech", 0.8), ("gap", 0.2),
                               ("speech", 0.8), ("gap", 1.5)])
    assert len(detect_chunks(audio)) == 1


def test_a_click_before_speech_is_dropped():
    """The c01 pilot problem: a 60 ms click, then silence, then the first word."""
    audio, _ = make_recording([("gap", 0.3), ("speech", 0.06), ("gap", 0.8),
                               ("speech", 1.0), ("gap", 1.5)])
    got = detect_chunks(audio)
    assert len(got) == 1
    assert got[0][0] > 1.0


# --------------------------------------------------------------------------- #
# propose_labels
# --------------------------------------------------------------------------- #


def test_a_scripted_item_keeps_its_longest_gaps_as_the_pauses(tmp_path):
    """Two gaps are detected but the script has one pause: the 1.2 s gap is the
    pause, and the 0.35 s gap inside the first sentence is merged."""
    layout = [("gap", 0.3), ("speech", 0.9), ("gap", 0.35), ("speech", 0.8),
              ("gap", 1.2), ("speech", 1.0), ("gap", 1.5)]
    root = make_probe_root(tmp_path, {"t01": ("function", FUNCTION_TEXT, layout, None)})
    chunks = read_labels(propose_labels("t01", root))
    assert [c.text for c in chunks] == ["i would like to change the", "delivery address"]
    assert chunks[0].end_s == pytest.approx(2.35, abs=0.06)       # 0.3+0.9+0.35+0.8
    assert chunks[1].start_s == pytest.approx(3.55, abs=0.06)     # + the 1.2 s pause


def test_a_natural_item_gets_question_marks_to_fill_in(tmp_path):
    root = make_probe_root(tmp_path, {"n99": ("natural", "[say something]", TWO_CHUNKS, None)})
    assert [c.text for c in read_labels(propose_labels("n99", root))] == ["?", "?"]


def test_propose_never_overwrites_corrected_labels(tmp_path):
    """REGRESSION: the pilot lost n01's typed-in words when propose ran again."""
    root = labelled_item(tmp_path, texts=("corrected one", "corrected two"))
    with pytest.raises(FileExistsError):
        propose_labels("t01", root)
    assert [c.text for c in read_labels(root / "labels" / "t01.txt")] == \
           ["corrected one", "corrected two"]


def test_propose_overwrites_only_when_asked(tmp_path):
    root = labelled_item(tmp_path, texts=("corrected one", "corrected two"))
    propose_labels("t01", root, overwrite=True)
    assert [c.text for c in read_labels(root / "labels" / "t01.txt")] == \
           ["i would like to change the", "delivery address"]


# --------------------------------------------------------------------------- #
# load_item: validation
# --------------------------------------------------------------------------- #


def _with_labels(tmp_path, chunks, layout=TWO_CHUNKS):
    root = make_probe_root(tmp_path, {"t01": ("function", FUNCTION_TEXT, layout, None)})
    write_labels(chunks, root / "labels" / "t01.txt")
    return root


def test_refuses_a_chunk_without_words(tmp_path):
    root = _with_labels(tmp_path, [Chunk(0.3, 1.3, "?"), Chunk(2.0, 3.0, "delivery address")])
    with pytest.raises(ValueError, match="no words"):
        load_item("t01", root)


def test_refuses_overlapping_regions(tmp_path):
    root = _with_labels(tmp_path, [Chunk(0.3, 2.2, "a the"), Chunk(2.0, 3.0, "b c")])
    with pytest.raises(ValueError, match="overlap"):
        load_item("t01", root)


def test_refuses_a_point_label(tmp_path):
    root = _with_labels(tmp_path, [Chunk(0.3, 0.3, "a the"), Chunk(2.0, 3.0, "b c")])
    with pytest.raises(ValueError, match="ends before it starts"):
        load_item("t01", root)


def test_refuses_a_recording_that_stops_too_soon(tmp_path):
    """Too little silence after the last word and the slowest policy never
    decides -- the first n01 take, which had 0.74 s."""
    short = [("gap", 0.3), ("speech", 1.0), ("gap", 0.7), ("speech", 1.0), ("gap", 0.6)]
    root = make_probe_root(tmp_path, {"t01": ("function", FUNCTION_TEXT, short,
                                              ["a the", "b c"])})
    with pytest.raises(ValueError, match="re-record"):
        load_item("t01", root)


def test_required_trailing_silence_covers_the_slowest_possible_decision():
    """The slowest decision on the ladder is T2 after a function word with
    rising pitch: 500 base + 400 hold + 150 = 1050 ms. MIN_TRAIL_S has to
    exceed it, or that turn can end with no decision at all."""
    def worst_ms(p):
        if isinstance(p, DecisionTheoretic):
            return p.min_silence_frames * 20.0
        return (getattr(p, "silence_ms", 0) + getattr(p, "extension_ms", 0)
                + getattr(p, "yield_bonus_ms", 0))
    assert MIN_TRAIL_S * 1000 > max(worst_ms(p) for p in POLICIES.values())


def test_the_kept_lead_in_is_shorter_than_any_policy_waits():
    """The crop keeps LEAD_KEEP_S before the first word. If that were as long
    as the shortest threshold, a policy could end the turn before it began."""
    shortest = min(p.silence_ms for p in POLICIES.values() if isinstance(p, FixedThreshold))
    assert LEAD_KEEP_S * 1000 < shortest


# --------------------------------------------------------------------------- #
# load_item: what it builds
# --------------------------------------------------------------------------- #


def test_audio_is_cropped_to_just_before_the_first_word(tmp_path):
    u = load_item("t01", labelled_item(tmp_path))
    t0 = 0.3 - LEAD_KEEP_S
    assert u.true_end_s == pytest.approx(3.0 - t0)
    assert len(u.audio) == pytest.approx((4.5 - t0) * SR, abs=2)


def test_the_last_word_of_each_chunk_ends_at_the_chunk_end(tmp_path):
    """The guarantee behind spreading word times evenly: at every pause, the
    word T1 reads is exactly right."""
    u = load_item("t01", labelled_item(tmp_path))
    t0 = 0.3 - LEAD_KEEP_S
    ends = {w: t for w, t in u.words}
    assert ends["the"] == pytest.approx(1.3 - t0)
    assert ends["address"] == pytest.approx(3.0 - t0)


def test_the_partial_during_a_pause_ends_on_the_word_before_it(tmp_path):
    u = load_item("t01", labelled_item(tmp_path))
    in_pause = 1.65 - (0.3 - LEAD_KEEP_S)
    assert u.context().partial_at(in_pause).split()[-1] == "the"


def test_words_are_lowercased(tmp_path):
    u = load_item("t01", labelled_item(tmp_path, texts=("I Would Like To Change The",
                                                        "Delivery Address")))
    assert all(w == w.lower() for w, _ in u.words)


def test_pauses_are_the_gaps_between_chunks(tmp_path):
    assert load_item("t01", labelled_item(tmp_path)).mid_turn_pauses_s == \
           [pytest.approx(0.7)]


def test_the_pitch_track_covers_the_audio_and_hears_the_voice(tmp_path):
    u = load_item("t01", labelled_item(tmp_path))
    assert len(u.f0_track) == pytest.approx(len(u.audio) / SR / 0.010, abs=2)
    voiced = [f for f in u.f0_track if f > 0]
    assert np.median(voiced) == pytest.approx(150.0, rel=0.1)
    in_pause = int((1.65 - (0.3 - LEAD_KEEP_S)) / 0.010)
    assert u.f0_track[in_pause] == 0.0


def test_the_cohort_names_the_item_type(tmp_path):
    assert load_item("t01", labelled_item(tmp_path)).cohort == "probe_function"


# --------------------------------------------------------------------------- #
# load_set and check
# --------------------------------------------------------------------------- #


def test_load_set_skips_unfinished_items_and_keeps_script_order(tmp_path):
    root = make_probe_root(tmp_path, {
        "t01": ("function", FUNCTION_TEXT, TWO_CHUNKS, ["a the", "b c"]),
        "t02": ("function", FUNCTION_TEXT, TWO_CHUNKS, None),          # not labelled
        "t03": ("function", FUNCTION_TEXT, TWO_CHUNKS, ["a the", "b c"]),
    })
    with open(root / "script.tsv", "a", encoding="utf-8") as fh:
        fh.write(f"t04\tfunction\t{FUNCTION_TEXT}\n")                  # not recorded
    assert list(load_set(root)) == ["t01", "t03"]


def test_t1_holds_a_medium_pause_after_a_function_word_that_t0_cuts(tmp_path):
    """The band the revised protocol targets. A 0.7 s pause after "the": T0 at
    500 ms interrupts; T1 holds, because it needs 900 ms after a function word."""
    cells = _row(check(labelled_item(tmp_path)), "t01")
    t0_500, t1_500 = cells[3], cells[6]
    assert t0_500 == "CUT after 'the'"
    assert t1_500.startswith("+")


def test_every_policy_cuts_a_pause_longer_than_any_threshold(tmp_path):
    """The pilot result in miniature: a 1.3 s pause beats every wait on the
    ladder, so the policies become indistinguishable."""
    long_pause = [("gap", 0.3), ("speech", 1.0), ("gap", 1.3), ("speech", 1.0), ("gap", 1.5)]
    cells = _row(check(labelled_item(tmp_path, layout=long_pause)), "t01")
    assert all(c == "CUT after 'the'" for c in cells[3:])


# --------------------------------------------------------------------------- #
# The real pilot recordings (skipped when the audio is not on this machine)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not all((PROBE_ROOT / d / f"{i}.{ext}").exists()
                            for i in ("f01", "c01", "n01")
                            for d, ext in (("audio", "wav"), ("labels", "txt"))),
                    reason="pilot recordings are not in git")
def test_the_pilot_recordings_load_and_validate():
    items = load_set()
    assert {"f01", "c01", "n01"} <= set(items)
    for u in items.values():
        assert len(u.audio) / u.sr - u.true_end_s >= MIN_TRAIL_S


# --------------------------------------------------------------------------- #
# Voice-aware drafting: clicks, breaths and noise have no pitch
# --------------------------------------------------------------------------- #


def test_voiced_seconds_hears_a_tone_and_not_noise():
    from s2slab.probes import _pitch_track, _voiced_s
    audio, _ = make_recording([("gap", 0.2), ("speech", 1.0), ("noise", 1.0), ("gap", 0.2)])
    track = _pitch_track(audio)
    assert _voiced_s(track, 0.2, 1.2) > 0.8        # the tone: nearly all voiced
    assert _voiced_s(track, 1.2, 2.2) < 0.10       # loud noise: no voice


def test_first_voiced_run_ignores_a_stray_voiced_frame():
    """One frame of 'pitch' found in noise must not count as the voice starting."""
    from s2slab.probes import _first_voiced_run_s
    track = [0.0, 0.0, 150.0, 0.0, 0.0, 0.0] + [150.0] * 10
    assert _first_voiced_run_s(track, 0.0, 1.0) == pytest.approx(0.06)
    assert _first_voiced_run_s([0.0, 150.0, 150.0, 0.0] * 5, 0.0, 1.0) is None


def test_noise_before_speech_is_left_out_of_the_draft(tmp_path):
    """The record click and room noise at the start of most f recordings."""
    layout = [("noise", 0.8), ("gap", 0.6), ("speech", 1.0), ("gap", 0.7),
              ("speech", 1.0), ("gap", 1.5)]
    root = make_probe_root(tmp_path, {"t01": ("function", FUNCTION_TEXT, layout, None)})
    chunks = read_labels(propose_labels("t01", root))
    assert [c.text for c in chunks] == ["i would like to change the", "delivery address"]
    assert chunks[0].start_s == pytest.approx(1.4, abs=0.06)


def test_a_stray_sound_after_the_end_is_not_taken_as_the_pause(tmp_path):
    """REGRESSION: in f02, f07, f09, f10 and f12 a stop click seconds after the
    last word was its own region, the gap before it was the longest, and the
    draft chose THAT gap as the pause -- merging away the real one."""
    layout = [("gap", 0.3), ("speech", 1.0), ("gap", 0.7), ("speech", 1.0),
              ("gap", 2.5), ("noise", 0.3), ("gap", 1.5)]
    root = make_probe_root(tmp_path, {"t01": ("function", FUNCTION_TEXT, layout, None)})
    chunks = read_labels(propose_labels("t01", root))
    assert len(chunks) == 2
    assert chunks[0].end_s == pytest.approx(1.3, abs=0.06)       # the real pause...
    assert chunks[1].start_s == pytest.approx(2.0, abs=0.06)     # ...is the one kept
    assert chunks[1].end_s == pytest.approx(3.0, abs=0.06)       # not the stop click


def test_noise_running_into_the_first_word_is_trimmed_off(tmp_path):
    """Noise less than a pause away from speech is merged into the same span
    by detect_chunks, so dropping voice-less spans is not enough: the first
    region is also started at the voice."""
    layout = [("gap", 0.3), ("noise", 0.8), ("gap", 0.1), ("speech", 1.0),
              ("gap", 0.7), ("speech", 1.0), ("gap", 1.5)]
    root = make_probe_root(tmp_path, {"t01": ("function", FUNCTION_TEXT, layout, None)})
    start = read_labels(propose_labels("t01", root))[0].start_s
    voice = 0.3 + 0.8 + 0.1
    assert voice - 0.10 <= start <= voice


def test_draft_lead_in_plus_crop_stays_under_the_shortest_threshold(tmp_path):
    """The draft starts a little before the voice and the loader keeps
    LEAD_KEEP_S more. Together they must stay under the shortest policy wait,
    or a policy could end the turn before the first word."""
    layout = [("noise", 0.8), ("gap", 0.6), ("speech", 1.0), ("gap", 0.7),
              ("speech", 1.0), ("gap", 1.5)]
    root = make_probe_root(tmp_path, {"t01": ("function", FUNCTION_TEXT, layout, None)})
    propose_labels("t01", root)
    u = load_item("t01", root)
    first_speech = next(i for i, p in enumerate(u.context().speech_prob) if p >= 0.5)
    shortest = min(p.silence_ms for p in POLICIES.values() if isinstance(p, FixedThreshold))
    assert first_speech * 20.0 < shortest
