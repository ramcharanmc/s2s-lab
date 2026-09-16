"""
Tests for s2slab.evalset -- all offline.

The real datasets are replaced by FakeSource, which serves rows and tones
exactly like ParquetSource serves rows and recordings, so the rules, the
download discipline, the writing and the checksums are tested without a
download. One test reads the real built set, and skips when it isn't there.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from s2slab.evalset import (EVAL_ROOT, MANIFEST, TARGET_SR, build, candidates, choose,
                            cover_groups, eval_clips, load, round_robin, seeded_order,
                            slow_down, summary, to_16k_mono, verify)


class FakeSource:
    """Rows plus a 150 Hz tone per clip, of a length each test chooses."""

    def __init__(self, rows, seconds=lambda row: 5.0, sr=22_050):
        self._rows, self.seconds, self.sr = rows, seconds, sr
        self.fetched_mb, self.fetched = 0.0, []

    def rows(self):
        return [dict(r) for r in self._rows]

    def group_mb(self, file, group):
        return 10.0

    def audio(self, row):
        self.fetched.append(row["index"])
        t = np.arange(int(self.seconds(row) * self.sr)) / self.sr
        return (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), self.sr


def clean_rows(n=12, speakers=4):
    return [{"id": f"s{i % speakers}-{i}", "speaker_id": i % speakers,
             "text": "one two three four five six seven", "file": "f", "group": 0, "index": i}
            for i in range(n)]


def accented_rows():
    rows, i = [], 0
    for group, accent in [(0, "Kenyan English"), (1, "Spanish"), (2, "Kenyan English"),
                          (3, "Mainstream US English"), (4, "Don't know")]:
        for speaker in ("A", "B"):
            for _ in range(3):
                rows.append({"speaker": f"{accent}-{group}-{speaker}", "accent": accent,
                             "l1": accent, "gender": "female",
                             "text": "we went to the market on sunday", "file": "f",
                             "group": group, "index": i})
                i += 1
    return rows


def disfluent_rows(n=12):
    return [{"transcript_a": "well I I mean we should go now please",
             "transcript_c": "we should go now please" if i % 3 else "well I I mean we should go now please",
             "file": "f", "group": 0, "index": i} for i in range(n)]


def fake_sources(seconds=lambda row: 5.0):
    return {"clean": FakeSource(clean_rows(), seconds), "accented": FakeSource(accented_rows(), seconds),
            "disfluent": FakeSource(disfluent_rows(), seconds)}


# --------------------------------------------------------------------------- #
# Choosing clips
# --------------------------------------------------------------------------- #


def test_seeded_order_repeats_for_a_seed_and_differs_across_seeds():
    rows = list(range(20))
    assert seeded_order(rows, 1) == seeded_order(rows, 1)
    assert seeded_order(rows, 1) != seeded_order(rows, 2)
    assert sorted(seeded_order(rows, 1)) == rows


def test_round_robin_takes_one_from_each_group_in_turn():
    rows = ["a1", "a2", "a3", "b1", "c1", "c2"]
    assert round_robin(rows, key=lambda r: r[0]) == ["a1", "b1", "c1", "a2", "c2", "a3"]


def test_cover_groups_covers_every_label_in_the_fewest_groups():
    groups = {"g1": ({"x"}, 5.0), "g2": ({"x", "y"}, 50.0), "g3": ({"z"}, 1.0), "g4": ({"y"}, 1.0)}
    chosen = cover_groups(groups)
    assert set().union(*(groups[g][0] for g in chosen)) == {"x", "y", "z"}
    assert chosen[0] == "g2"                  # the only group adding two labels
    assert len(chosen) == 2


def test_cover_groups_prefers_the_cheaper_of_two_equal_groups():
    assert cover_groups({"big": ({"x"}, 90.0), "small": ({"x"}, 9.0)}) == ["small"]


def test_choose_keeps_n_with_at_most_two_per_speaker():
    rows = [{"who": w} for w in "AAAABBBBCC"]
    kept = choose(rows, 5, accept=lambda r: "clip", speaker=lambda r: r["who"])
    assert [r["who"] for r, _ in kept] == ["A", "A", "B", "B", "C"]


def test_choose_skips_rejected_clips_and_never_fetches_a_capped_speaker():
    fetched = []

    def accept(row):
        fetched.append(row["i"])
        return None if row["i"] == 1 else "clip"

    rows = [{"i": i, "who": "A"} for i in range(6)]
    kept = choose(rows, 3, accept, speaker=lambda r: r["who"])
    assert [r["i"] for r, _ in kept] == [0, 2]      # 1 rejected; A capped after two
    assert fetched == [0, 1, 2]                      # 3, 4, 5 never downloaded


def test_clean_candidates_drop_texts_under_six_words():
    rows = clean_rows(4) + [{"id": "x", "speaker_id": 9, "text": "too short", "file": "f",
                             "group": 0, "index": 99}]
    assert "x" not in {r["id"] for r in candidates("clean", rows)}


def test_accented_candidates_follow_their_rules():
    rows = accented_rows() + [{"speaker": "T", "accent": "Spanish", "l1": "Spanish", "gender": "male",
                               "text": "we went to <OVERLAP> the market on sunday", "file": "f",
                               "group": 1, "index": 999}]
    got = candidates("accented", rows)
    accents = {r["accent"] for r in got}
    assert accents == {"Kenyan English", "Spanish"}                  # not mainstream US, not "don't know"
    assert all("<" not in r["text"] for r in got)
    assert {r["group"] for r in got} in ({0, 1}, {1, 2})             # one Kenyan group is enough
    assert {got[0]["accent"], got[1]["accent"]} == accents          # interleaved by accent


def test_disfluent_candidates_are_only_clips_with_disfluencies():
    got = candidates("disfluent", disfluent_rows())
    assert got and all(r["transcript_a"] != r["transcript_c"] for r in got)
    assert len(got) == 8                                             # 4 of 12 are fluent


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def test_any_rate_and_channels_become_16k_mono():
    stereo = np.zeros((22_050, 2), dtype=np.float32)
    out = to_16k_mono(stereo, 22_050)
    assert out.ndim == 1 and out.dtype == np.float32
    assert abs(len(out) - TARGET_SR) <= 1


def test_slow_down_lengthens_by_one_over_the_rate_and_keeps_the_pitch():
    import parselmouth
    t = np.arange(2 * TARGET_SR) / TARGET_SR
    tone = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    slow = slow_down(tone, TARGET_SR, 0.75)
    assert len(slow) / TARGET_SR == pytest.approx(2 / 0.75, abs=0.05)
    f0 = parselmouth.Sound(slow.astype(np.float64), TARGET_SR).to_pitch().selected_array["frequency"]
    assert np.median(f0[f0 > 0]) == pytest.approx(150, abs=3)


# --------------------------------------------------------------------------- #
# Build, verify, load
# --------------------------------------------------------------------------- #


def test_build_writes_four_cohorts_and_a_manifest(tmp_path):
    manifest = json.loads(build(tmp_path, fake_sources(), per_cohort=3).read_text(encoding="utf-8"))
    ids = [i["id"] for i in manifest["items"]]
    assert ids == [f"{c}-{k:02d}" for c in ("clean", "accented", "disfluent", "slow") for k in range(3)]
    assert all((tmp_path / "audio" / f"{i}.wav").exists() for i in ids)
    assert manifest["sources"]["clean"]["revision"] and manifest["seed"]


def test_the_slow_cohort_is_the_clean_cohort_slowed(tmp_path):
    items = json.loads(build(tmp_path, fake_sources(), per_cohort=3).read_text(encoding="utf-8"))["items"]
    by_id = {i["id"]: i for i in items}
    for k in range(3):
        slow, clean = by_id[f"slow-{k:02d}"], by_id[f"clean-{k:02d}"]
        assert slow["derived_from"] == clean["id"] and slow["text"] == clean["text"]
        assert slow["duration_s"] == pytest.approx(clean["duration_s"] / 0.75, abs=0.05)


def test_build_skips_clips_outside_three_to_fifteen_seconds(tmp_path):
    srcs = fake_sources(seconds=lambda row: 2.0 if row["index"] % 2 else 5.0)
    items = json.loads(build(tmp_path, srcs, per_cohort=2).read_text(encoding="utf-8"))["items"]
    assert all(3.0 <= i["duration_s"] <= 15.0 for i in items if i["cohort"] != "slow")


def test_build_refuses_a_second_time_and_when_a_source_runs_short(tmp_path):
    build(tmp_path / "a", fake_sources(), per_cohort=2)
    with pytest.raises(FileExistsError):
        build(tmp_path / "a", fake_sources(), per_cohort=2)
    with pytest.raises(RuntimeError, match="clean"):
        build(tmp_path / "b", fake_sources(seconds=lambda row: 20.0), per_cohort=2)


def test_verify_is_clean_then_names_changed_and_missing_recordings(tmp_path):
    build(tmp_path, fake_sources(), per_cohort=2)
    assert verify(tmp_path) == []
    wav = tmp_path / "audio" / "accented-01.wav"
    data = bytearray(wav.read_bytes())
    data[-1] ^= 1
    wav.write_bytes(bytes(data))
    (tmp_path / "audio" / "slow-00.wav").unlink()
    assert verify(tmp_path) == ["accented-01: recording changed", "slow-00: recording missing"]


def test_load_returns_audio_and_the_reference_text(tmp_path):
    build(tmp_path, fake_sources(), per_cohort=2)
    items = load(tmp_path, cohorts={"disfluent"})
    assert [i.id for i in items] == ["disfluent-00", "disfluent-01"]
    assert items[0].text.startswith("well")                     # transcript_a: every word spoken
    assert items[0].meta["text_fluent"] == "we should go now please"
    assert items[0].audio.dtype == np.float32
    assert list(eval_clips(tmp_path, {"clean"})) == ["clean-00", "clean-01"]


def test_summary_has_a_row_per_cohort(tmp_path):
    build(tmp_path, fake_sources(), per_cohort=2)
    text = summary(tmp_path)
    assert all(f"| {c} |" in text for c in ("clean", "accented", "disfluent", "slow"))


def test_importing_the_eval_set_needs_no_download_library():
    """Only building needs pyarrow and huggingface_hub; loading needs neither."""
    root = pathlib.Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    code = ("import sys, s2slab.evalset; "
            "print([m for m in ('pyarrow', 'huggingface_hub') if m in sys.modules])")
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.stdout.strip() == "[]", proc.stderr


@pytest.mark.skipif(not (EVAL_ROOT / MANIFEST).exists(), reason="the eval set is not built on this machine")
def test_the_built_eval_set_still_matches_its_manifest():
    assert verify() == []