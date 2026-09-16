"""
Tests for s2slab.wer.

Every count is hand-worked: each expected number is written out in the test, so
a wrong alignment cannot quietly agree with itself. The normalizer tests use
real mismatches from this project -- "FOURTEENTH" against "the 14th", "Mr."
against "mister", and a dropped "uh".
"""

from __future__ import annotations

import json
import math

import pytest

from s2slab.wer import (Errors, align, corpus_wer, format_table, light, mistakes,
                        references, score, standard, wer)

REF = "the cat sat on the mat"


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #


def test_one_wrong_word_in_six():
    got = wer(REF, "the cat sat on a mat")
    assert (got.sub, got.delete, got.insert, got.n_words) == (1, 0, 0, 6)
    assert got.rate == pytest.approx(1 / 6)


def test_a_missed_word_and_an_invented_word_are_counted_apart():
    missed = wer(REF, "the cat sat on mat")
    invented = wer(REF, "the big cat sat on the mat")
    assert (missed.delete, missed.insert) == (1, 0)
    assert (invented.delete, invented.insert) == (0, 1)
    assert missed.rate == invented.rate == pytest.approx(1 / 6)


def test_identical_is_zero_and_a_silent_recognizer_is_one():
    assert wer(REF, REF).rate == 0.0
    assert wer(REF, "").rate == 1.0            # six deletions over six words


def test_wer_can_pass_one_hundred_percent():
    got = wer("yes", "yes i think so absolutely")
    assert got.insert == 4 and got.rate == pytest.approx(4.0)


def test_an_empty_reference_has_no_rate_but_still_counts_inventions():
    got = wer("", "hello there")
    assert got.n_words == 0 and got.insert == 2
    assert math.isnan(got.rate)


def test_corpus_wer_weighs_by_words_not_by_turns():
    """A one-word turn must not count as much as a nine-word one. Mean of the
    two rates would be 50%; the corpus rate is one error in ten words."""
    pairs = [("yes", "no"),                                             # 1 of 1 wrong
             ("the cat sat on the mat every single day",
              "the cat sat on the mat every single day")]               # 0 of 9 wrong
    total = corpus_wer(pairs)
    assert (total.n_utterances, total.n_words, total.errors) == (2, 10, 1)
    assert total.rate == pytest.approx(0.1)


def test_errors_add_up_across_utterances():
    assert (wer(REF, "the cat sat on a mat") + wer(REF, REF)).n_utterances == 2


def test_mistakes_lists_only_what_went_wrong():
    assert mistakes(REF, "the cat sat on a mat") == [("sub", "the", "a")]
    assert mistakes(REF, REF) == []


def test_alignment_is_readable():
    ops = align("a b c".split(), "a x c".split())
    assert ops == [("equal", "a", "a"), ("sub", "b", "x"), ("equal", "c", "c")]


# --------------------------------------------------------------------------- #
# The two normalizers
# --------------------------------------------------------------------------- #


def test_raw_normalization_keeps_the_words_as_spoken():
    assert light("Well, uh, I don't -- I don't know!") == "well uh i don't i don't know"
    assert light("THE 14TH") == "the 14th"


def test_raw_wer_sees_a_number_written_two_ways_as_an_error():
    assert wer("the fourteenth", "the 14th").rate == pytest.approx(0.5)


def test_normalized_wer_matches_number_words_to_digits():
    pytest.importorskip("transformers")
    assert standard("THE FOURTEENTH") == standard("the 14th")
    assert wer("the fourteenth", "the 14th", standard).rate == 0.0


def test_normalized_wer_expands_abbreviations():
    pytest.importorskip("transformers")
    assert wer("MISTER QUILTER IS HERE", "Mr. Quilter is here", standard).rate == 0.0


def test_normalized_wer_hides_a_dropped_filler_and_raw_wer_does_not():
    """Why both numbers are reported. The recognizer missed the "uh" the
    speaker said; the standard normalizer deletes fillers from both sides, so
    only the raw score can see it -- and the disfluent cohort exists to."""
    pytest.importorskip("transformers")
    ref, hyp = "well uh i think so", "well i think so"
    assert wer(ref, hyp).errors == 1                 # raw: one word missed
    assert wer(ref, hyp, standard).errors == 0       # normalized: cannot see it


# --------------------------------------------------------------------------- #
# Scoring a run against the eval set
# --------------------------------------------------------------------------- #


def fake_evalset(tmp_path):
    items = [{"id": "clean-00", "cohort": "clean", "text": "the cat sat on the mat"},
             {"id": "clean-01", "cohort": "clean", "text": "a dog barked"},
             {"id": "disfluent-00", "cohort": "disfluent", "text": "well uh i went home",
              "text_fluent": "i went home"}]
    (tmp_path / "manifest.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    return tmp_path


def test_references_come_from_the_manifest(tmp_path):
    root = fake_evalset(tmp_path)
    assert references(root)["clean-01"] == ("clean", "a dog barked")
    fluent = references(root, key="text_fluent")
    assert set(fluent) == {"disfluent-00"}           # only that cohort has one
    assert fluent["disfluent-00"][1] == "i went home"


def test_score_groups_by_cohort_and_reports_both_numbers(tmp_path):
    root = fake_evalset(tmp_path)
    scored = score({"clean-00": "the cat sat on a mat", "clean-01": "a dog barked",
                    "disfluent-00": "well i went home"}, root)
    assert set(scored) == {"clean", "disfluent"}
    assert scored["clean"]["raw"].n_utterances == 2
    assert scored["clean"]["raw"].n_words == 9       # 6 + 3
    assert scored["clean"]["raw"].rate == pytest.approx(1 / 9)


def test_score_refuses_an_id_the_eval_set_does_not_know(tmp_path):
    with pytest.raises(KeyError, match="mystery-01"):
        score({"mystery-01": "hello"}, fake_evalset(tmp_path))


def test_the_table_has_a_row_per_cohort_and_a_total(tmp_path):
    root = fake_evalset(tmp_path)
    text = format_table(score({"clean-00": "the cat sat on the mat",
                               "disfluent-00": "well i went home"}, root), title="R0")
    assert "### R0" in text
    assert "| clean |" in text and "| disfluent |" in text and "| **all** |" in text


def test_a_run_can_be_scored_against_the_tidied_reference(tmp_path):
    """The disfluent cohort keeps a second reference with the fillers removed,
    so a recognizer that drops them can be scored either way."""
    root = fake_evalset(tmp_path)
    hyp = {"disfluent-00": "i went home"}
    assert score(hyp, root)["disfluent"]["raw"].errors == 2            # "well" and "uh" missed
    assert score(hyp, root, key="text_fluent")["disfluent"]["raw"].errors == 0


@pytest.mark.skipif(not (pytest.importorskip("s2slab.evalset").EVAL_ROOT / "manifest.json").exists(),
                    reason="the eval set is not built on this machine")
def test_the_real_eval_set_offers_a_reference_for_every_clip():
    refs = references()
    assert len(refs) == 200
    assert {c for c, _ in refs.values()} == {"clean", "accented", "disfluent", "slow"}
    assert all(text.strip() for _, text in refs.values())