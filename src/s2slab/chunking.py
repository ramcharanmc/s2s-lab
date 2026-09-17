"""
Clause chunking: cut a reply into pieces a TTS can speak while the LLM is still writing.

R1's whole gain is here. R0 waits for the last token, then speaks. R1 speaks
the first clause as soon as it is complete -- so the question is when a piece
of text is complete enough to say out loud without sounding cut off.

    tokens arrive:  "Your appoint" "ment is on Tues" "day, and I" "'ve noted it."
    feed() returns:  []            []                 ["Your appointment is on Tuesday,"]
    flush() returns: ["and I've noted it."]

THE RULES, in order of confidence:
  newline           always a boundary (a list the prompt asked the model not to write)
  . ! ?             a boundary -- unless the word before is an abbreviation
                    ("Dr. Smith"), because splitting there makes TTS pause mid-name
  , ; :             a boundary only once the clause has MIN_WORDS words. "Sure,"
                    spoken alone sounds clipped; "Your appointment is on Tuesday,"
                    does not
Every boundary also needs the WHITESPACE after it. "2.50" and "2:30" are not
boundaries, and the only way to know is to see the next character -- so a
chunk is released one character late, never early.

STREAM-SAFE BY CONSTRUCTION. Whether a boundary splits depends only on the text
before it, never on how that text was cut into tokens. So feeding a reply
whole, token by token or character by character gives the same clauses --
which the tests check for every possible cut.

TUNED ON PROBE CLIPS ONLY. MIN_WORDS and the abbreviation list may be adjusted
on the 30 probe-clip replies (dev data). They are frozen in PR-002 before R1
touches the eval set, whose replies are test data and are not looked at here.
"""

from __future__ import annotations

import re

MIN_WORDS = 4     # a , ; or : splits only after this many words in the clause

# Words that end in a full stop without ending a sentence. Lower case, no final dot.
ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "e.g", "i.e", "a.m", "p.m",
})

# Punctuation, any closing quotes or brackets that belong to it, then the whitespace
# that proves it is finished. Or a newline on its own.
_BOUNDARY = re.compile(r"""([.!?;:,]+)(["'”’)\]]*)\s|\n""")
_SENTENCE_END = frozenset(".!?")



class ClauseChunker:
    """Collects streamed text and releases complete clauses.

    One per reply: feed() every piece of text as it arrives, then flush() once
    the LLM is done.
    """

    def __init__(self, min_words: int = MIN_WORDS):
        self.min_words = min_words
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        """Takes: the next piece of the reply, any length, "" included.
        Returns: every clause completed by it, in order -- often none.
        """
        self.buffer += text
        out = []
        while (cut := self._first_split()) is not None:
            clause_end, rest_start = cut
            clause = self.buffer[:clause_end].strip()
            self.buffer = self.buffer[rest_start:]
            if clause:
                out.append(clause)
        return out

    def flush(self) -> list[str]:
        """Takes: nothing -- call once, after the last token.
        Returns: whatever is left as one final clause, or [] if only whitespace.
        """
        rest, self.buffer = self.buffer.strip(), ""
        return [rest] if rest else []

    def _first_split(self) -> tuple[int, int] | None:
        """(where the clause ends, where the rest starts) for the first boundary
        in the buffer that qualifies, or None if there is none yet."""
        for m in _BOUNDARY.finditer(self.buffer):
            if m.group(0) == "\n":
                return m.start(), m.end()
            clause_end = m.end(2)                  # keep the punctuation and its closers
            if self._splits(self.buffer[:clause_end], m.group(1)):
                return clause_end, m.end()
        return None

    def _splits(self, clause: str, punct: str) -> bool:
        """Does this boundary end a clause worth speaking on its own?"""
        if _SENTENCE_END.intersection(punct):
            last = clause.split()[-1].lower().rstrip(".\"'”’)]") if clause.split() else ""
            return not (punct == "." and last in ABBREVIATIONS)
        return len(clause.split()) >= self.min_words



def split_clauses(text: str, min_words: int = MIN_WORDS) -> list[str]:
    """The clauses of a finished reply: feed it whole, then flush.

    Takes: a reply. Returns: its clauses, in order. Same result as streaming it.
    """
    chunker = ClauseChunker(min_words)
    return chunker.feed(text) + chunker.flush()


if __name__ == "__main__":
    import sys

    for clause in split_clauses(" ".join(sys.argv[1:]) or
                                "Your appointment is on Tuesday, and I've noted it. Anything else?"):
        print(f"[{len(clause.split()):2d} words] {clause}")    