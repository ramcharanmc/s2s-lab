"""
Tests for s2slab.chunking -- no models, no audio, just text.

The most important property is the last group: a reply cut into tokens any
way at all gives the same clauses as the whole reply. R1 streams, the tests
here mostly do not, and that property is what makes the difference safe.
"""

from __future__ import annotations

import pytest

from s2slab.chunking import MIN_WORDS, ClauseChunker, split_clauses

# Replies in the shape the 0.5B model writes: one or two short sentences.
REPLIES = [
    "Sure, I can help with that. What is the new address?",
    "Your appointment is on Tuesday, and I've noted the change.",
    "I'm sorry to hear that; let me check your account for you.",
    "Dr. Smith can see you at 2:30 p.m. tomorrow, is that okay?",
    'He said "yes." Then the order was cancelled.',
    "The total is 12.50 dollars.",
    "Thank you!",
]


# --------------------------------------------------------------------------- #
# Where a reply is cut
# --------------------------------------------------------------------------- #


def test_a_sentence_end_is_always_a_boundary():
    assert split_clauses("Sure. What is the new address?") == ["Sure.", "What is the new address?"]
    assert split_clauses("Great! Anything else? No.") == ["Great!", "Anything else?", "No."]


def test_a_short_comma_clause_is_not_spoken_alone():
    """ "Sure," alone would sound clipped. """
    assert split_clauses("Sure, I can help with that.") == ["Sure, I can help with that."]


def test_a_comma_splits_once_the_clause_is_long_enough():
    assert split_clauses("Your appointment is on Tuesday, and I've noted the change.") == [
        "Your appointment is on Tuesday,", "and I've noted the change."]


def test_the_word_threshold_is_exactly_min_words():
    words = " ".join(["word"] * MIN_WORDS)
    shorter = " ".join(["word"] * (MIN_WORDS - 1))
    assert split_clauses(f"{words}, then more.") == [f"{words},", "then more."]
    assert split_clauses(f"{shorter}, then more.") == [f"{shorter}, then more."]


def test_semicolons_and_colons_follow_the_comma_rule():
    assert split_clauses("I'm sorry to hear that; let me check.") == [
        "I'm sorry to hear that;", "let me check."]
    assert split_clauses("Note: it is late.") == ["Note: it is late."]


def test_numbers_and_times_are_not_boundaries():
    """No whitespace after the mark, so no boundary."""
    assert split_clauses("The total is 12.50 dollars at 2:30 today, 1,000 times over.") == [
        "The total is 12.50 dollars at 2:30 today,", "1,000 times over."]


def test_abbreviations_do_not_end_a_sentence():
    assert split_clauses("Dr. Smith will call you. Mr. Jones will not.") == [
        "Dr. Smith will call you.", "Mr. Jones will not."]
    assert split_clauses("Bring an ID, e.g. a passport.") == ["Bring an ID, e.g. a passport."]


def test_closing_quotes_and_brackets_stay_with_their_clause():
    assert split_clauses('He said "yes." Then he left.') == ['He said "yes."', "Then he left."]
    assert split_clauses("It is closed (for now.) Try later.") == [
        "It is closed (for now.)", "Try later."]


def test_a_newline_is_always_a_boundary():
    assert split_clauses("First item\n\nSecond item") == ["First item", "Second item"]


def test_nothing_is_lost_or_reordered():
    for reply in REPLIES:
        assert " ".join(split_clauses(reply)).split() == reply.split()


def test_empty_and_blank_replies_give_no_clauses():
    assert split_clauses("") == []
    assert split_clauses("   \n ") == []


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def test_a_boundary_waits_for_the_character_after_it():
    """ "Sure." might be "Sure.5" -- only the next character says. """
    chunker = ClauseChunker()
    assert chunker.feed("Sure.") == []
    assert chunker.feed(" What") == ["Sure."]
    assert chunker.feed(" is it?") == []
    assert chunker.flush() == ["What is it?"]


def test_flush_starts_the_next_reply_clean():
    chunker = ClauseChunker()
    chunker.feed("left over")
    assert chunker.flush() == ["left over"]
    assert chunker.flush() == []
    assert chunker.feed("New. ") == ["New."]


@pytest.mark.parametrize("reply", REPLIES)
def test_any_two_way_cut_gives_the_same_clauses(reply):
    whole = split_clauses(reply)
    for i in range(len(reply) + 1):
        chunker = ClauseChunker()
        assert chunker.feed(reply[:i]) + chunker.feed(reply[i:]) + chunker.flush() == whole


@pytest.mark.parametrize("reply", REPLIES)
def test_character_by_character_gives_the_same_clauses(reply):
    chunker = ClauseChunker()
    streamed = [c for ch in reply for c in chunker.feed(ch)] + chunker.flush()
    assert streamed == split_clauses(reply)