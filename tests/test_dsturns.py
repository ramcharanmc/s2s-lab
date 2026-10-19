"""
Tests for s2slab.dsturns, P1b on real disfluent speech (M40): the eval set's rules on a split, a build that
writes each clip once with its checksum and marks the eval set's own, the test turns refused without PR-019,
and the rule written before the DEV run. A stand-in dataset with synthetic audio replaces the download.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from s2slab import dsturns

SR = 16_000


def _clip(pause_s: float, seed: int = 0) -> np.ndarray:
    """Room noise, a voiced 1.4 s burst, a pause, a voiced 1.6 s burst."""
    rng = np.random.default_rng(seed)
    t = lambda s: np.arange(int(s * SR)) / SR                                  # noqa: E731
    noise = lambda s: rng.normal(0, 0.002, int(s * SR)).astype("float32")     # noqa: E731
    tone = lambda s, f: (0.3 * np.sin(2 * np.pi * f * t(s)) * np.hanning(int(s * SR))).astype("float32")  # noqa: E731
    return np.concatenate([noise(0.4), tone(1.4, 160) + noise(1.4), noise(pause_s), tone(1.6, 150) + noise(1.6)])


class Source:
    """DisfluencySpeech's shape: rows of text, audio by row."""

    fetched_mb = 1.5

    def __init__(self, seconds):
        self.seconds = seconds
        texts = [("uh, I, I think we should go now please", "I think we should go now please"),
                 ("we should go now please friend", "we should go now please friend"),          # fluent: not eligible
                 ("um so the the the thing is", "so the thing is"),
                 ("so, uh, my account number is four seven", "so my account number is four seven"),
                 ("well, uh, I don't, I don't do it much", "well I don't do it much")]
        self._rows = [{"file": "data/test-00000-of-00001.parquet", "group": 0, "index": i, "transcript_a": a,
                       "transcript_c": c} for i, (a, c) in enumerate(texts)]

    def rows(self):
        return [dict(r) for r in self._rows]

    def audio(self, row):
        s = self.seconds[row["index"]]
        return (_clip(0.6, row["index"])[: int(s * SR)] if s < 4.5 else np.resize(_clip(0.6), int(s * SR))), SR


def test_a_split_keeps_the_eval_sets_rules_in_file_order_and_each_clip_once(tmp_path):
    src = Source({0: 4.4, 1: 4.4, 2: 4.4, 3: 2.0, 4: 16.0})
    path = dsturns.build("test", tmp_path, src, in_evalset={"data/test-00000-of-00001.parquet#0/0"})
    blob = json.loads(path.read_text())
    assert [i["source_id"] for i in blob["items"]] == ["data/test-00000-of-00001.parquet#0/0", "data/test-00000-of-00001.parquet#0/2"]
    assert [i["id"] for i in blob["items"]] == ["ds-test-000", "ds-test-001"]
    assert [i["in_evalset"] for i in blob["items"]] == [True, False]
    assert blob["skipped"] == {"under_3_s": 1, "over_15_s": 1} and blob["fetched_mb"] == 1.5
    assert blob["source"]["revision"] == "b7da294fe3a70dd96df6640893f2a5dfc2c87638" and blob["dataset_split"] == "test"
    assert dsturns.verify("test", tmp_path) == []
    with pytest.raises(FileExistsError, match="never changes"):
        dsturns.build("test", tmp_path, src, in_evalset=set())
    (tmp_path / "audio" / "test" / "ds-test-001.wav").write_bytes(b"changed")
    assert dsturns.verify("test", tmp_path) == ["ds-test-001: changed"]


def test_the_splits_are_the_eval_sets_pin_and_its_50_are_in_the_test_split():
    assert dsturns.spec("dev").pattern == "^data/validation-" and dsturns.spec("test").pattern == "^data/test-"
    assert dsturns.spec("dev").revision == dsturns.SOURCES["disfluent"].revision
    ids = dsturns.evalset_ids()
    assert len(ids) == 50 and all(i.startswith("data/test-") for i in ids)


def test_the_test_turns_are_refused_without_the_registration(tmp_path):
    with pytest.raises(PermissionError, match="PR-019"):
        dsturns.turns("test", root=tmp_path)


def test_dev_turns_go_through_the_real_turn_harness_and_the_rule_reads_t0_at_300ms(tmp_path):
    src = Source({0: 4.4, 1: 4.4, 2: 4.4, 3: 4.4, 4: 4.4})
    dsturns.build("dev", tmp_path, src, in_evalset=set())
    items = dsturns.turns("dev", root=tmp_path, words_fn=lambda x: [("think", 1.6), ("now", 3.6)])
    assert [i for i, _ in items] == ["ds-dev-000", "ds-dev-001", "ds-dev-002", "ds-dev-003"] and items[0][1].cohort == "disfluent"
    assert items[0][1].mid_turn_pauses_s and 0.4 < items[0][1].mid_turn_pauses_s[0] < 0.8
    pols = [p for p in dsturns.dev_policies() if p.name in ("T0_fixed_300ms", "T0_fixed_900ms")]
    blob = dsturns.report(items, pols)
    rows = {r["policy"]: r for r in blob["rows"]}
    assert rows["T0_fixed_300ms"]["cut"] == 4 and rows["T0_fixed_900ms"]["cut"] == 0      # a 0.6 s pause in each
    assert rows["T0_fixed_300ms"]["cut_ids"] == ["ds-dev-000", "ds-dev-001", "ds-dev-002", "ds-dev-003"]
    assert blob["pauses"]["turns_with_one_over_300ms"] == 4
    g = dsturns.gate(blob)
    assert g["go"] and g["share"] == 1.0 and "4 of 4" in g["why"]
    assert not dsturns.gate({"rows": [{"policy": "T0_fixed_300ms", "cut": 1, "n": 10}]})["go"]      # 10%: STOP
    assert dsturns.gate({"rows": [{"policy": "T0_fixed_300ms", "cut": 2, "n": 10}]})["go"]          # 20%: GO
    text = dsturns.format_dev(blob)
    assert "GO: register PR-019" in text and "| T0_fixed_300ms | 4 of 4 |" in text


def test_the_dev_sweep_names_every_setting_once():
    names = [p.name for p in dsturns.dev_policies()]
    assert len(names) == len(set(names)) == 8 + 4 + 12 + 5
    assert "T2_prosodic_500ms_s40" in names and "T0_fixed_1500ms" in names and "T3_decision_r32" in names
