"""Tests for autopitch.scripts.scrape_site (pure HTML parsing helpers)."""

import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts.scrape_site import _clean_text, _collect_logo_candidates


FIXTURE_HTML = """
<!doctype html>
<html>
<head>
    <title>Acme Widgets</title>
    <meta property="og:image" content="/og.png">
    <meta name="twitter:image" content="https://cdn.acme.com/twitter.png">
    <link rel="apple-touch-icon" href="/apple.png">
    <link rel="icon" href="favicon.ico">
</head>
<body>
    <nav>Home About</nav>
    <h1>We make widgets</h1>
    <p>Custom widget manufacturing since 1987.</p>
    <script>tracking()</script>
    <footer>© 2026</footer>
</body>
</html>
"""


class TestCleanText:
    def test_extracts_headings_and_paragraphs(self):
        soup = BeautifulSoup(FIXTURE_HTML, "html.parser")
        text = _clean_text(soup)
        assert "We make widgets" in text
        assert "Custom widget manufacturing since 1987." in text

    def test_strips_scripts(self):
        soup = BeautifulSoup(FIXTURE_HTML, "html.parser")
        text = _clean_text(soup)
        assert "tracking" not in text


class TestLogoCandidates:
    def test_og_image_first(self):
        soup = BeautifulSoup(FIXTURE_HTML, "html.parser")
        cands = _collect_logo_candidates(soup, "https://acme.com/")
        sources = [c[0] for c in cands]
        assert sources[0] == "og:image"
        # og:image was relative — should be resolved to absolute URL
        assert cands[0][1] == "https://acme.com/og.png"

    def test_includes_all_tiers(self):
        soup = BeautifulSoup(FIXTURE_HTML, "html.parser")
        cands = _collect_logo_candidates(soup, "https://acme.com/")
        sources = [c[0] for c in cands]
        assert "twitter:image" in sources
        assert "apple-touch-icon" in sources
        assert "favicon" in sources
        # fallback to /favicon.ico always appended
        assert "favicon.ico" in sources

    def test_falls_back_when_no_og(self):
        html = "<html><head><link rel='icon' href='/fav.png'></head></html>"
        soup = BeautifulSoup(html, "html.parser")
        cands = _collect_logo_candidates(soup, "https://x.com/")
        sources = [c[0] for c in cands]
        assert "og:image" not in sources
        assert "favicon" in sources
