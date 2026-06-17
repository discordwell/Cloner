"""Tests for autopitch.scripts.analyze_site (prompt assembly — pure function)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts.analyze_site import build_prompt


class TestBuildPrompt:
    def test_substitutes_known_placeholders(self):
        p = build_prompt(
            company="Acme Widgets",
            url="https://acme.example.com",
            site_text="We build custom cabinets.",
        )
        assert "Acme Widgets" in p
        assert "https://acme.example.com" in p
        assert "We build custom cabinets." in p
        # The {company}/{url}/{site_text} placeholders are gone once filled.
        assert "{company}" not in p
        assert "{url}" not in p
        assert "{site_text}" not in p

    def test_preserves_literal_example_braces(self):
        """Regression: the template's example skeleton ({short title},
        {2-3 sentences ...}) must survive — str.format used to raise
        KeyError: 'short title' here and broke the whole stage."""
        p = build_prompt(company="Acme", url="https://acme.com", site_text="x")
        assert "{short title}" in p

    def test_site_text_with_braces_does_not_crash(self):
        """Scraped pages routinely contain { } (JSON, CSS, code)."""
        messy = 'config = {"theme": "dark"} and a CSS rule { color: red; }'
        p = build_prompt(company="Acme", url="https://acme.com", site_text=messy)
        assert messy in p

    def test_substitution_is_single_pass(self):
        """A placeholder appearing inside an inserted value is left literal —
        it must not be recursively re-substituted."""
        p = build_prompt(
            company="Acme",
            url="https://acme.com",
            site_text="they literally wrote {company} on their homepage",
        )
        # The {company} that came in via site_text stays as-is.
        assert "{company} on their homepage" in p
