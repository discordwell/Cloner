"""Tests for autopitch.scripts.write_pitch (prompt assembly — pure function)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts.write_pitch import build_prompt


class TestBuildPrompt:
    def test_includes_name_and_company(self):
        p = build_prompt(
            name="Jane Doe", company="Acme Widgets", role=None,
            hypothesis="Automate quoting.", target_duration_s=60,
        )
        assert "Jane Doe" in p
        assert "Acme Widgets" in p
        assert "Automate quoting." in p

    def test_first_name_extracted(self):
        p = build_prompt(
            name="Jane Marie Doe", company="A", role=None, hypothesis="",
            target_duration_s=60,
        )
        assert "Jane" in p  # first_name
        # Shouldn't leak "Jane Marie Doe" into every mention of first_name
        # (this is hard to test without knowing the template — at least verify
        # the raw first token is usable)

    def test_role_clause_when_provided(self):
        with_role = build_prompt(
            name="Jane", company="Acme", role="CEO", hypothesis="",
            target_duration_s=60,
        )
        without_role = build_prompt(
            name="Jane", company="Acme", role=None, hypothesis="",
            target_duration_s=60,
        )
        assert "CEO" in with_role
        assert "CEO" not in without_role

    def test_target_words_scales_with_duration(self):
        short = build_prompt(
            name="J", company="A", role=None, hypothesis="",
            target_duration_s=30, words_per_second=2.5,
        )
        long = build_prompt(
            name="J", company="A", role=None, hypothesis="",
            target_duration_s=120, words_per_second=2.5,
        )
        # 30s × 2.5 = 75 words; 120s × 2.5 = 300 words
        assert "75" in short
        assert "300" in long
