"""Tests for autopitch.scripts._llm (OpenAI wrapper edge cases)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts._llm import complete


class TestComplete:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            complete("hello", model="gpt-4o")
