"""
Tests for s2slab.diagnose.

Two kinds of test here. The helpers get small hand-built inputs with answers you
can check on paper. The analysis itself gets INVARIANTS -- statements that must
hold for every interrupted turn because of how the policies are built. The
invariants are the valuable part: if one fails, either the analysis is reading
the wrong moment or a policy has changed underneath it.
"""

from __future__ import annotations

import pytest

from s2slab.diagnose import (Interruption, breakdown, classify_tail,
                             format_breakdown, interruptions, silent_run_ms)
from s2slab.policies import INCOMPLETE_TAILS, SemanticThreshold, get
from s2slab.simulate import make_all
from s2slab.sweep import evaluate

FRAME_MS = 20.0


@pytest.fixture(scope="module")
def disfluent():
    """The same turns the sweep scores for seed 0. Built once for the module --
    each utterance caches its VAD pass, so every policy reuses it."""
    return make_all(80, seed=0)["disfluent"]


# --------------------------------------------------------------------------- #
# classify_tail
# --------------------------------------------------------------------------- #


def test_function_word_tail():
    assert classify_tail("my account number is") == ("is", "function")


def test_content_word_tail():
    assert classify_tail("my account number is four seven") == ("seven", "content")


def test_nothing_said_yet():
    assert classify_tail("") == ("", "none")
    assert classify_tail("   ") == ("", "none")
    assert classify_tail(None) == ("", "none")


def test_punctuation_and_case_are_ignored():
    assert classify_tail("My number IS,") == ("is", "function")


def test_verdict_matches_what_t1_actually_checks():
    """The analysis is only meaningful if it classifies a partial exactly the way
    T1 does. Checked against T1's own method across many partials."""
    t1 = SemanticThreshold(500)
    partials = ["my account number is", "four seven", "the thing is i",
                "wanted to", "and the", "tuesday morning", "", "It's,"]
    for p in partials:
        _, tail = classify_tail(p)
        assert (tail == "function") == t1._should_hold(p), p


def test_every_listed_word_is_a_function_word():
    for w in INCOMPLETE_TAILS:
        assert classify_tail(f"something {w}")[1] == "function", w


# --------------------------------------------------------------------------- #
# silent_run_ms
# --------------------------------------------------------------------------- #


def test_run_in_the_middle():
    probs = [0.9, 0.9, 0.1, 0.1, 0.1, 0.9]
    for frame in (2, 3, 4):              # any frame inside the run gives its full length
        assert silent_run_ms(probs, frame, FRAME_MS) == 60.0


def test_speech_frame_has_no_silence():
    assert silent_run_ms([0.9, 0.1, 0.1], 0, FRAME_MS) == 0.0


def test_run_touching_the_start_and_the_end():
    assert silent_run_ms([0.1, 0.1, 0.9], 0, FRAME_MS) == 40.0
    assert silent_run_ms([0.9, 0.1, 0.1, 0.1], 3, FRAME_MS) == 60.0


def test_frame_outside_the_buffer_is_not_an_error():
    assert silent_run_ms([0.1, 0.1], 5, FRAME_MS) == 0.0
    assert silent_run_ms([0.1, 0.1], -1, FRAME_MS) == 0.0
    assert silent_run_ms([], 0, FRAME_MS) == 0.0


# --------------------------------------------------------------------------- #
# interruptions -- agreement with the sweep
# --------------------------------------------------------------------------- #


def test_counts_agree_with_the_sweep(disfluent):
    """Same definition of "interrupted" as sweep.evaluate, or the two modules
    would report different numbers for the same policy on the same turns."""
    for name in ("T0_fixed_500ms", "T1_semantic_500ms", "T2_prosodic_500ms"):
        assert len(interruptions(get(name), disfluent)) == \
               evaluate(get(name), disfluent).n_cutoff, name


def test_every_record_is_a_real_interruption(disfluent):
    for r in interruptions(get("T0_fixed_500ms"), disfluent):
        assert isinstance(r, Interruption)
        assert r.cut_s < r.true_end_s
        assert r.early_ms > 0


def test_records_follow_cohort_order(disfluent):
    turns = [r.turn for r in interruptions(get("T0_fixed_500ms"), disfluent)]
    assert turns == sorted(turns)
    assert all(0 <= t < len(disfluent) for t in turns)


def test_every_cut_lands_inside_a_silence(disfluent):
    """A policy only fires during silence, so the cut frame must be silent. If
    this fails, the frame arithmetic in interruptions() is off by one."""
    for r in interruptions(get("T0_fixed_500ms"), disfluent):
        assert r.pause_ms > 0


# --------------------------------------------------------------------------- #
# interruptions -- invariants that follow from how the policies work
# --------------------------------------------------------------------------- #


def test_t0_only_cuts_pauses_of_at_least_its_threshold(disfluent):
    """T0 at 500 ms fires after 25 silent frames, so every pause it cut inside
    was at least 500 ms long."""
    for r in interruptions(get("T0_fixed_500ms"), disfluent):
        assert r.pause_ms >= 500.0 - 1e-9


def test_a_t1_cut_after_a_function_word_needs_a_pause_longer_than_the_hold(disfluent):
    """THE BUILT-IN CHECK FROM THE REPORT.

    After a function word T1 needs 500 + 400 = 900 ms of silence, so every
    function-word interruption must sit in a pause of at least 900 ms. A shorter
    one means the analysis read the transcript at the wrong moment.
    """
    recs = interruptions(get("T1_semantic_500ms"), disfluent)
    function_cuts = [r for r in recs if r.tail == "function"]
    assert function_cuts, "fixture too small to contain any"
    for r in function_cuts:
        assert r.pause_ms >= 900.0 - 1e-9, r


def test_a_t1_cut_after_a_content_word_needs_at_least_the_base_threshold(disfluent):
    for r in interruptions(get("T1_semantic_500ms"), disfluent):
        if r.tail == "content":
            assert r.pause_ms >= 500.0 - 1e-9, r


def test_t1_cannot_change_a_content_word_cut(disfluent):
    """Every turn T0 cut after a content word, T1 cuts at the SAME moment.

    Before that pause both policies need the same 500 ms, so they reach it
    together; at it, the transcript ends on a content word, so T1 has nothing to
    hold on. This is why the fix is a better completeness check, not a longer
    hold: the hold never gets a chance on these turns.
    """
    t0 = {(r.turn, r.cut_s) for r in interruptions(get("T0_fixed_500ms"), disfluent)
          if r.tail == "content"}
    t1 = {(r.turn, r.cut_s) for r in interruptions(get("T1_semantic_500ms"), disfluent)
          if r.tail == "content"}
    assert t0 <= t1


def test_t1_rescues_pauses_not_turns(disfluent):
    """Some turns T1 held through at a function word get cut LATER, at a
    content-word pause. So T1 has at least as many content-word cuts as T0 --
    the extra ones are turns that moved, not new failures."""
    t0 = interruptions(get("T0_fixed_500ms"), disfluent)
    t1 = interruptions(get("T1_semantic_500ms"), disfluent)
    t0_turns = {r.turn for r in t0}
    moved = [r for r in t1 if r.tail == "content"
             and r.turn in t0_turns
             and r.cut_s > next(x.cut_s for x in t0 if x.turn == r.turn)]
    assert moved, "no turn moved -- the test would pass vacuously; grow the fixture"
    assert all(t0_r.tail == "function"
               for t0_r in t0 if t0_r.turn in {m.turn for m in moved})


# --------------------------------------------------------------------------- #
# breakdown / format_breakdown
# --------------------------------------------------------------------------- #


def _rec(tail: str, word: str, pause: float, turn: int = 0) -> Interruption:
    return Interruption(turn=turn, cut_s=1.0, true_end_s=2.0,
                        last_word=word, tail=tail, pause_ms=pause)


def test_breakdown_counts_and_shares():
    recs = [_rec("content", "seven", 600), _rec("content", "seven", 700),
            _rec("content", "thing", 800), _rec("function", "is", 1000)]
    b = breakdown(recs)
    assert b["content"]["n"] == 3 and b["function"]["n"] == 1
    assert b["content"]["share"] == pytest.approx(0.75)
    assert b["content"]["pause_p50_ms"] == pytest.approx(700.0)
    assert b["content"]["top_words"][0] == ("seven", 2)


def test_breakdown_leaves_out_kinds_that_never_occur():
    b = breakdown([_rec("content", "seven", 600)])
    assert set(b) == {"content"}


def test_breakdown_of_nothing_is_empty():
    assert breakdown([]) == {}


def test_shares_add_up_to_one(disfluent):
    b = breakdown(interruptions(get("T0_fixed_500ms"), disfluent))
    assert sum(s["share"] for s in b.values()) == pytest.approx(1.0)


def test_top_words_are_capped_at_five():
    recs = [_rec("content", w, 600) for w in "a b c d e f g".split()]
    assert len(breakdown(recs)["content"]["top_words"]) == 5


def test_report_has_a_header_and_one_row_per_kind():
    recs = [_rec("content", "seven", 600), _rec("function", "is", 1000)]
    table = format_breakdown(recs, "T1_semantic_500ms", "disfluent", 250)
    assert "2 of 250 turns interrupted" in table
    rows = [ln for ln in table.splitlines()
            if ln.startswith("| ") and "last word" not in ln]
    assert len(rows) == 2
