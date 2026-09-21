"""
State truncation: after a barge-in, the agent's turn ends where the caller stopped hearing it.

The v2 plan, R1: "If the agent generated three sentences and the user
interrupted after one, appending all three to history means the agent believes
it said things the user never heard. Every later turn is grounded on a false
premise." The fix is to cut the turn at the playback position.

    reply     "Your appointment is booked for Tuesday the fourteenth at half past two. Please arrive ..."
    stop at   3.30 s -- "half" ended at 3.10 s, "past" ends at 3.49 s
    heard     "Your appointment is booked for Tuesday the fourteenth at half"
    unheard   "past two. Please arrive ten minutes early, ..."

A WORD CUT IN HALF WAS NOT HEARD. The cut goes after the last word that had
finished playing: the history may then claim a little less than the caller
caught, never more.

The word times come from the synthesizer (bargein.kokoro_synth keeps Kokoro's
own), so this needs no recognizer and no alignment model: the system that spoke
knows when it said what.
"""

from __future__ import annotations

from typing import NamedTuple


class Truncated(NamedTuple):
    heard: str          # what goes into the dialogue history
    unheard: str        # what the agent had prepared or spoken and the caller did not hear
    n_heard: int        # words heard
    n_words: int        # words in the whole reply


def _letters(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def _find(text: str, pos: int, target: str) -> tuple[int, int] | None:
    """Where the next word spelling `target` (letters and digits only) sits in
    text at or after pos: (start, end), or None. "don't" matches "dont"."""
    i = pos
    while i < len(text):
        if not text[i].isalnum():
            i += 1
            continue
        got, j = "", i
        while j < len(text) and len(got) < len(target):
            if text[j].isalnum():
                got += text[j].lower()
            j += 1
        if got == target:
            return i, j
        while i < len(text) and text[i].isalnum():      # not this word: move past it
            i += 1
    return None


def truncate_reply(text: str, words, stop_s: float) -> Truncated:
    """Cut a reply at the last word that had finished playing by stop_s.

    Takes:   the reply as written; its words with times -- anything with
             .text, .start_s and .end_s, in order; and when playback stopped,
             on the same clock as the word times.
    Calls:   _find() once per word, walking the text left to right, so the
             heard part keeps the original spelling, capitals and punctuation.
    Returns: Truncated(heard, unheard, n_heard, n_words). Punctuation directly
             after the last heard word stays with it: "...half past two."
    Raises:  ValueError if a timed word is not in the text, in order: the times
             then belong to some other text.
    """
    cut, pos, n_heard, n_words = 0, 0, 0, 0
    for word in words:
        target = _letters(word.text)
        if not target:
            continue
        found = _find(text, pos, target)
        if found is None:
            raise ValueError(f"{word.text!r} is not in the reply after position {pos}: wrong word times?")
        pos, n_words = found[1], n_words + 1
        if word.end_s <= stop_s:
            cut, n_heard = pos, n_heard + 1
    while 0 < cut < len(text) and not text[cut].isalnum() and not text[cut].isspace():
        cut += 1                                        # keep the full stop with the word it ends
    if n_heard == n_words:
        return Truncated(text.strip(), "", n_heard, n_words)
    return Truncated(text[:cut].strip(), text[cut:].strip(), n_heard, n_words)
