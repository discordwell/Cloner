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


class TestBuildPromptRobustness:
    def test_hypothesis_with_braces_does_not_crash(self):
        """The hypothesis is LLM-generated markdown and routinely carries literal
        braces (the analyze_site skeleton uses {short title}, and pages embed JSON
        / CSS). A str.format-based builder is one template edit away from KeyError;
        the placeholder-substitution builder passes braces through untouched."""
        messy = '## Opportunity 1: {short title}\nUse config {"x": 1} for RAG.'
        p = build_prompt(
            name="Jane Doe", company="Acme", role=None,
            hypothesis=messy, target_duration_s=60,
        )
        assert messy in p

    def test_brace_in_value_is_not_re_substituted(self):
        """Single pass: a {company} that arrives via the hypothesis stays literal
        rather than being recursively expanded."""
        p = build_prompt(
            name="Jane", company="Acme", role=None,
            hypothesis="they wrote {company} on their homepage",
            target_duration_s=60,
        )
        assert "{company} on their homepage" in p

    def test_empty_name_does_not_crash(self):
        """name.split()[0] used to IndexError on an empty/whitespace-only name."""
        p = build_prompt(
            name="", company="Acme", role=None,
            hypothesis="x", target_duration_s=60,
        )
        assert "Acme" in p
