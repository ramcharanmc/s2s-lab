"""
Tests for s2slab.truncation -- cutting the agent's turn where the caller stopped hearing it.
"""

from __future__ import annotations

import pathlib
from typing import NamedTuple

import pytest

from s2slab.truncation import truncate_reply


class W(NamedTuple):
    text: str
    start_s: float
    end_s: float


REPLY = "Your appointment is booked for Tuesday the fourteenth at half past two. Please arrive ten minutes early."


def timed(text: str, word_s: float = 0.5) -> list[W]:
    """One word every word_s, as a synthesizer would report them: no punctuation."""
    return [W(w.strip(".,?!"), k * word_s, (k + 1) * word_s) for k, w in enumerate(text.split())]


def test_the_turn_is_cut_after_the_last_word_that_finished_playing():
    t = truncate_reply(REPLY, timed(REPLY), stop_s=5.2)           # "half" ended at 5.0, "past" ends at 5.5
    assert t.heard == "Your appointment is booked for Tuesday the fourteenth at half"
    assert t.unheard == "past two. Please arrive ten minutes early."
    assert (t.n_heard, t.n_words) == (10, 17)


def test_a_word_cut_in_half_was_not_heard():
    assert truncate_reply(REPLY, timed(REPLY), stop_s=5.49).heard.endswith("at half")
    assert truncate_reply(REPLY, timed(REPLY), stop_s=5.50).heard.endswith("half past")


def test_a_full_stop_stays_with_the_word_it_ends():
    t = truncate_reply(REPLY, timed(REPLY), stop_s=6.0)
    assert t.heard.endswith("half past two.") and t.unheard.startswith("Please")


def test_stopped_before_the_first_word_nothing_was_heard():
    t = truncate_reply(REPLY, timed(REPLY), stop_s=0.3)
    assert t.heard == "" and t.unheard == REPLY and t.n_heard == 0


def test_not_stopped_at_all_everything_was_heard():
    t = truncate_reply(REPLY, timed(REPLY), stop_s=99.0)
    assert t.heard == REPLY and t.unheard == "" and t.n_heard == t.n_words == 17


def test_spelling_and_punctuation_come_from_the_reply_not_from_the_word_list():
    reply = "Sure -- I don't mind; it's 12.50, isn't it?"
    words = [W(w, k, k + 1.0) for k, w in enumerate(["sure", "I", "don't", "mind", "it's", "12.50", "isn't", "it"])]
    assert truncate_reply(reply, words, stop_s=4.0).heard == "Sure -- I don't mind;"
    assert truncate_reply(reply, words, stop_s=6.0).heard == "Sure -- I don't mind; it's 12.50,"


def test_word_times_that_belong_to_another_text_are_refused():
    with pytest.raises(ValueError, match="'Thursday'"):
        truncate_reply(REPLY, [W("Your", 0, 0.5), W("Thursday", 0.5, 1.0)], stop_s=2.0)


@pytest.mark.skipif(not (pathlib.Path(__file__).resolve().parents[1] / "probes/bargein/agent_words.tsv").exists(),
                    reason="the barge-in set's word times are not on this machine")
def test_every_real_agent_line_can_be_cut_at_every_word():
    """Kokoro's own word list against the five committed agent lines: every
    word must be found, and cutting after word k must leave k words heard."""
    from s2slab.bargein import BARGEIN_ROOT, read_agent, read_agent_words
    root = pathlib.Path(__file__).resolve().parents[1] / BARGEIN_ROOT
    for agent_id, text in read_agent(root).items():
        words = read_agent_words(root)[agent_id]
        for k, word in enumerate(words):
            t = truncate_reply(text, words, stop_s=word.end_s)
            assert t.n_heard == k + 1 and len(t.heard.split()) == k + 1, (agent_id, word.text)
        assert truncate_reply(text, words, 99.0).heard == text

