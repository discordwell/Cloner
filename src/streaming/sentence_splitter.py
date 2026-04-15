"""
Sentence splitter for streaming LLM output.

Accumulates tokens and emits complete sentences as they form.
Designed for real-time use: feed tokens one at a time, get callbacks
when sentence boundaries are detected.
"""

import re
from typing import Callable, Optional

# Common abbreviations that end with a period but aren't sentence endings
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "ave", "blvd",
    "vs", "etc", "approx", "dept", "est", "govt", "inc", "corp", "ltd",
    "gen", "sgt", "cpl", "pvt", "capt", "lt", "col", "maj",
    "e.g", "i.e", "u.s", "u.k",
}

# Sentence-ending punctuation
_SENTENCE_ENDINGS = ".!?;"


class SentenceSplitter:
    """
    Accumulates LLM tokens and emits complete sentences.

    Usage:
        sentences = []
        splitter = SentenceSplitter(on_sentence=sentences.append)
        for token in llm_stream:
            splitter.feed(token)
        splitter.flush()
    """

    def __init__(
        self,
        on_sentence: Callable[[str], None],
        min_length: int = 10,
        max_buffer: int = 500,
    ):
        """
        Args:
            on_sentence: Called with each complete sentence.
            min_length: Minimum chars before emitting (avoids tiny fragments).
            max_buffer: Force-flush if buffer exceeds this size without a boundary.
        """
        self._on_sentence = on_sentence
        self._min_length = min_length
        self._max_buffer = max_buffer
        self._buffer = ""

    def feed(self, token: str):
        """Feed a token from the LLM stream."""
        self._buffer += token
        self._try_emit()

    def flush(self):
        """Emit whatever remains in the buffer."""
        text = self._buffer.strip()
        if text:
            self._on_sentence(text)
        self._buffer = ""

    def reset(self):
        """Clear the buffer without emitting."""
        self._buffer = ""

    def _try_emit(self):
        """Check for sentence boundaries and emit if found."""
        # Force-flush on max buffer
        if len(self._buffer) >= self._max_buffer:
            # Find the last space to avoid splitting mid-word
            last_space = self._buffer.rfind(" ", 0, self._max_buffer)
            if last_space > self._min_length:
                self._emit_up_to(last_space + 1)
            return

        # Scan for sentence boundaries
        i = 0
        while i < len(self._buffer):
            ch = self._buffer[i]

            if ch in _SENTENCE_ENDINGS:
                # Check if this is a real sentence boundary
                if self._is_sentence_boundary(i):
                    candidate = self._buffer[:i + 1].strip()
                    if len(candidate) >= self._min_length:
                        self._emit_up_to(i + 1)
                        # After emitting, reset scan position
                        i = 0
                        continue

            i += 1

    def _is_sentence_boundary(self, pos: int) -> bool:
        """
        Determine if the punctuation at pos is a real sentence boundary.

        Returns False for abbreviations (Dr., U.S.), decimal numbers (3.14),
        and mid-sentence periods without a following uppercase letter or end.
        """
        ch = self._buffer[pos]
        text_before = self._buffer[:pos]

        # For . specifically, apply heuristics
        if ch == ".":
            # Check for abbreviation: word before the period
            word_match = re.search(r'(\w+(?:\.\w+)*)$', text_before)
            if word_match:
                word = word_match.group(1).lower()
                if word in _ABBREVIATIONS:
                    return False

            # Check for decimal number: digit before period
            if text_before and text_before[-1].isdigit():
                after = self._buffer[pos + 1:]
                # If nothing follows yet, defer — next token might be a digit
                if not after:
                    return False
                # If next char is a digit, it's a decimal (3.14)
                if after[0].isdigit():
                    return False

        # Look at what follows the punctuation
        after = self._buffer[pos + 1:]

        if not after:
            # End of buffer — can't tell yet. Don't emit unless we have
            # a good amount of text (the next token will clarify).
            return len(self._buffer[:pos + 1].strip()) >= self._min_length * 2

        # If followed by a space then uppercase letter (or end), it's a boundary
        if after[0] == " ":
            rest = after[1:]
            if not rest:
                # Space at end of buffer — likely a boundary but wait for next token
                return len(self._buffer[:pos + 1].strip()) >= self._min_length * 2
            if rest[0].isupper() or rest[0] == '"' or rest[0] == "'":
                return True
            # Lowercase after period+space: probably not a sentence end (e.g. "vs. the")
            if ch == ".":
                return False
            # For ! ? ; followed by space+lowercase, still treat as boundary
            return True

        # If followed by newline, it's a boundary
        if after[0] == "\n":
            return True

        # If followed by a quote or closing paren, it's a boundary
        if after[0] in ('"', "'", ")", "\u201d"):
            return True

        return False

    def _emit_up_to(self, pos: int):
        """Emit text up to pos and trim buffer."""
        text = self._buffer[:pos].strip()
        self._buffer = self._buffer[pos:].lstrip()
        if text:
            self._on_sentence(text)
