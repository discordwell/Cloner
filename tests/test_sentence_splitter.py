"""Tests for the streaming sentence splitter."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.streaming.sentence_splitter import SentenceSplitter


def _split(text: str, **kwargs) -> list:
    """Helper: feed full text character-by-character and return sentences."""
    results = []
    splitter = SentenceSplitter(on_sentence=results.append, **kwargs)
    for ch in text:
        splitter.feed(ch)
    splitter.flush()
    return results


def _split_tokens(tokens: list, **kwargs) -> list:
    """Helper: feed tokens (multi-char strings) and return sentences."""
    results = []
    splitter = SentenceSplitter(on_sentence=results.append, **kwargs)
    for token in tokens:
        splitter.feed(token)
    splitter.flush()
    return results


class TestBasicSplitting:
    def test_single_sentence(self):
        assert _split("Hello world.", min_length=1) == ["Hello world."]

    def test_two_sentences(self):
        result = _split("Hello there. How are you?", min_length=1)
        assert result == ["Hello there.", "How are you?"]

    def test_three_sentences(self):
        result = _split("One. Two. Three.", min_length=1)
        assert result == ["One.", "Two.", "Three."]

    def test_exclamation_mark(self):
        result = _split("Wow! That is great.", min_length=1)
        assert result == ["Wow!", "That is great."]

    def test_question_mark(self):
        result = _split("Is this working? Yes it is.", min_length=1)
        assert result == ["Is this working?", "Yes it is."]

    def test_semicolon(self):
        result = _split("First part; second part.", min_length=1)
        assert result == ["First part;", "second part."]

    def test_newline_boundary(self):
        result = _split("First line.\nSecond line.", min_length=1)
        assert result == ["First line.", "Second line."]


class TestMinLength:
    def test_filters_short_fragments(self):
        result = _split("Hi. How are you today?", min_length=10)
        # "Hi." is only 3 chars, below min_length of 10
        # Should combine with next sentence or emit at flush
        assert len(result) >= 1
        assert "How are you today?" in result[-1]

    def test_respects_min_length(self):
        result = _split(
            "This is a long sentence. And another one here.",
            min_length=10,
        )
        assert len(result) == 2
        assert result[0] == "This is a long sentence."
        assert result[1] == "And another one here."


class TestAbbreviations:
    def test_dr_no_split(self):
        result = _split("Dr. Smith is here today.", min_length=1)
        assert result == ["Dr. Smith is here today."]

    def test_mr_no_split(self):
        result = _split("Mr. Jones said hello.", min_length=1)
        assert result == ["Mr. Jones said hello."]

    def test_mrs_no_split(self):
        result = _split("Mrs. Smith is here.", min_length=1)
        assert result == ["Mrs. Smith is here."]

    def test_etc_no_split(self):
        result = _split("Cats, dogs, etc. are animals.", min_length=1)
        assert result == ["Cats, dogs, etc. are animals."]

    def test_vs_no_split(self):
        result = _split("It was us vs. them today.", min_length=1)
        assert result == ["It was us vs. them today."]


class TestDecimalNumbers:
    def test_decimal_no_split(self):
        result = _split("The value is 3.14 approximately.", min_length=1)
        assert result == ["The value is 3.14 approximately."]

    def test_price_no_split(self):
        result = _split("It costs $9.99 per month.", min_length=1)
        assert result == ["It costs $9.99 per month."]


class TestTokenStreaming:
    """Test with multi-character tokens like real LLM output."""

    def test_token_by_token(self):
        tokens = ["Hello", " there", ".", " How", " are", " you", "?"]
        result = _split_tokens(tokens, min_length=1)
        assert result == ["Hello there.", "How are you?"]

    def test_sentence_split_across_tokens(self):
        tokens = ["This is great", ". And", " this too", "."]
        result = _split_tokens(tokens, min_length=1)
        assert result == ["This is great.", "And this too."]

    def test_large_tokens(self):
        tokens = [
            "Sure, let me explain. ",
            "The system works by streaming tokens. ",
            "Each sentence is processed independently.",
        ]
        result = _split_tokens(tokens, min_length=10)
        assert len(result) == 3


class TestMaxBuffer:
    def test_force_flush_on_max_buffer(self):
        # A long string with no sentence boundaries
        long_text = "word " * 120  # 600 chars
        result = _split(long_text, min_length=1, max_buffer=200)
        assert len(result) >= 2
        # All text should be emitted
        combined = " ".join(result)
        assert len(combined) > 0


class TestEdgeCases:
    def test_empty_input(self):
        result = _split("", min_length=1)
        assert result == []

    def test_whitespace_only(self):
        result = _split("   ", min_length=1)
        assert result == []

    def test_no_punctuation(self):
        result = _split("Hello world", min_length=1)
        # Should emit on flush
        assert result == ["Hello world"]

    def test_multiple_periods(self):
        result = _split("Wait... What?", min_length=1)
        # Ellipsis should not split into fragments
        assert "What?" in result[-1]

    def test_quoted_sentence(self):
        result = _split('He said "Hello." Then left.', min_length=1)
        assert len(result) >= 1

    def test_flush_emits_remainder(self):
        results = []
        splitter = SentenceSplitter(on_sentence=results.append, min_length=1)
        splitter.feed("Partial sentence without ending")
        assert results == []
        splitter.flush()
        assert results == ["Partial sentence without ending"]

    def test_reset_clears_buffer(self):
        results = []
        splitter = SentenceSplitter(on_sentence=results.append, min_length=1)
        splitter.feed("Some text")
        splitter.reset()
        splitter.flush()
        assert results == []


class TestLLMStyleOutput:
    """Test with realistic LLM streaming patterns."""

    def test_typical_response(self):
        tokens = [
            "Sure", ",", " I", "'d", " be", " happy", " to", " help", ".",
            " The", " main", " thing", " to", " know", " is", " that",
            " streaming", " reduces", " latency", ".",
            " Let", " me", " explain", " how", ".",
        ]
        result = _split_tokens(tokens, min_length=10)
        assert len(result) == 3
        assert result[0] == "Sure, I'd be happy to help."
        assert "streaming reduces latency" in result[1]

    def test_short_acknowledgment_then_explanation(self):
        tokens = [
            "Great", " question", "!",
            " The", " answer", " involves", " several", " key",
            " concepts", " that", " work", " together", ".",
        ]
        result = _split_tokens(tokens, min_length=10)
        assert result[0] == "Great question!"
        assert "answer involves" in result[1]
