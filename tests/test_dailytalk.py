"""
Tests for s2slab.dailytalk (FT-1's data): the folds and the test-fold lock, loading at Moshi's rate,
the human reference and the report -- on small synthetic stereo files, no download needed.
"""

import numpy as np
import pytest

from s2slab import dailytalk
from s2slab.turngaps import summary

SR_IN = 44_100


def buzz(spans, length, sr):
    x = np.zeros(int(length * sr), np.float32)
    for s, e in spans:
        i, j = int(s * sr), int(e * sr)
        x[i:j] = 0.2 * np.sign(np.sin(2 * np.pi * 180 * np.arange(j - i) / sr))
    return x


@pytest.fixture
def root(tmp_path):
    """Dialogues 10 (test fold), 11 and 21 (dev), 12 (train): the left answers the right 300 ms after each turn."""
    import soundfile as sf
    d = tmp_path / "data_stereo"
    d.mkdir()
    for n in (10, 11, 12, 21):
        left = buzz([(0.2, 1.0), (2.1, 3.0)], 4.0, SR_IN)        # opens, then answers at 1.8 + 0.3
        right = buzz([(1.2, 1.8)], 4.0, SR_IN)
        sf.write(str(d / f"{n}.wav"), np.stack([left, right], axis=1), SR_IN, subtype="PCM_16")
    return tmp_path


def test_the_folds_are_set_by_the_dialogue_number():
    assert [dailytalk.fold(n) for n in (0, 10, 2540, 1, 491, 2, 99)] == ["test", "test", "test", "dev", "dev", "train", "train"]
    assert dailytalk.REVISION.startswith("33e1b50") and dailytalk.SR == 24_000


def test_only_the_files_on_this_machine_are_listed_by_fold(root):
    assert dailytalk.dialogue_ids("dev", root) == [11, 21] and dailytalk.dialogue_ids("test", root) == [10]
    assert dailytalk.dialogue_ids("train", root) == [12]


def test_a_test_dialogue_is_locked_and_a_dev_one_loads_at_24k(root):
    with pytest.raises(PermissionError, match="PR-010"):
        dailytalk.load(10, root)
    left, _ = dailytalk.load(10, root, confirm="PR-010")
    left, right = dailytalk.load(11, root)
    assert left.dtype == np.float32 and len(left) == len(right) == pytest.approx(4.0 * 24_000, abs=2)


def test_the_reference_is_the_left_answering_the_right(root):
    ref = dailytalk.reference(*dailytalk.load(11, root))
    assert ref["n_transitions"] == 1 and ref["fto_ms"] == pytest.approx([300.0], abs=25)
    assert ref["overlapped"] == 0 and ref["overlap_ratio"] == 0.0


def test_the_report_sets_a_model_beside_the_reference_over_the_dialogues_both_cover():
    from s2slab.turngaps import IPU
    user = [IPU(1.0, 2.0), IPU(4.0, 5.0)]
    ref = {1: summary(user, [IPU(2.4, 3.0), IPU(5.4, 6.0)]), 2: summary(user, [IPU(2.4, 3.0)])}
    model = {1: summary(user, [IPU(1.8, 3.0), IPU(5.1, 6.0)]), 3: summary(user, [IPU(2.1, 2.5)])}   # 3: no reference
    text = dailytalk.report(model, ref, "moshi")
    assert text.startswith("1 dialogues") and "| turn changes | 2 | 2 |" in text
    assert "| overlapping turn changes | 1 | 0 |" in text
    # model -200, +100 against reference 400, 400: they move 600 and 300 ms, (600 + 300) / 2 = 450
    assert "distance between the two offset distributions (Wasserstein-1, pooled): 450 ms" in text
