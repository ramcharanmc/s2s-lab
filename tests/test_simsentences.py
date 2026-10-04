"""
Tests for s2slab.simsentences: the rule that drew the whole-sentence sets from DailyTalk's dev fold.
Fake alignments only; the real transcripts are not in the repo.
"""

from __future__ import annotations

import pytest

from s2slab.simsentences import pool, sentences


def _aligned(text: str) -> list:
    return [[w, [0.0, 0.0], "SPEAKER_MAIN"] for w in text.split()]


def test_a_sentence_ends_at_end_punctuation_is_normalised_and_kept_only_at_8_to_14_words():
    a = _aligned("Hi. I would like to book a table for two tonight, please. "
                 "Can I pay with my $20 card at the door now? Yes, that's it.")
    assert sentences(a) == ["i would like to book a table for two tonight please"]


def test_an_apostrophe_inside_a_word_stays():
    assert sentences(_aligned("I'm sure you'll love it and we'll see.")) == ["i'm sure you'll love it and we'll see"]


def test_a_sentence_of_fifteen_words_is_dropped():
    assert sentences(_aligned("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen.")) == []


def test_dev_and_eval_take_disjoint_dialogues_of_the_dev_fold_only():
    assert [pool(n) for n in (1, 11, 21, 31, 491)] == ["dev", "eval", "dev", "eval", "eval"]
    for n in (0, 10, 2):                         # the test fold is PR-010's; train is not on this machine
        with pytest.raises(ValueError):
            pool(n)
