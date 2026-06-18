"""Tests for autopitch.scripts.scrape_site (pure HTML parsing + logo download)."""

import sys
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts import scrape_site
from autopitch.scripts.scrape_site import _clean_text, _collect_logo_candidates


def _png_bytes() -> bytes:
    """A small but >200-byte PNG (noise compresses past the tiny-file reject)."""
    buf = BytesIO()
    Image.effect_noise((64, 64), 80).convert("RGB").save(buf, "PNG")
    return buf.getvalue()


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


class TestDownloadImage:
    """_download_image fetches via the capped helper, normalizes to PNG, and
    fails closed (returns False, writes nothing) on any error."""

    def test_saves_valid_image_as_rgba_png(self, monkeypatch, tmp_path):
        monkeypatch.setattr(scrape_site, "fetch_bytes", lambda url, **kw: _png_bytes())
        out = tmp_path / "logo.png"
        assert scrape_site._download_image("http://x/logo.png", out, session=None) is True
        assert Image.open(out).mode == "RGBA"

    def test_passes_byte_cap_and_session(self, monkeypatch, tmp_path):
        seen = {}

        def capture(url, **kw):
            seen.update(kw)
            seen["url"] = url
            return _png_bytes()

        monkeypatch.setattr(scrape_site, "fetch_bytes", capture)
        sentinel_session = object()
        scrape_site._download_image("http://x/l.png", tmp_path / "l.png", sentinel_session)
        assert seen["url"] == "http://x/l.png"
        assert seen["max_bytes"] == scrape_site.MAX_IMAGE_BYTES
        assert seen["session"] is sentinel_session

    def test_returns_false_and_writes_nothing_on_cap_overflow(self, monkeypatch, tmp_path):
        def over_cap(url, **kw):
            raise ValueError("download exceeds cap")

        monkeypatch.setattr(scrape_site, "fetch_bytes", over_cap)
        out = tmp_path / "logo.png"
        assert scrape_site._download_image("http://x/huge", out, session=None) is False
        assert not out.exists()

    def test_rejects_tiny_response(self, monkeypatch, tmp_path):
        monkeypatch.setattr(scrape_site, "fetch_bytes", lambda url, **kw: b"\x89PNG")
        out = tmp_path / "logo.png"
        assert scrape_site._download_image("http://x/stub", out, session=None) is False
        assert not out.exists()


class TestScrape:
    """scrape() fetches the homepage HTML through the capped text helper (not
    resp.text), then persists html/text and the best logo."""

    def test_routes_html_through_capped_fetch_text(self, monkeypatch, tmp_path):
        seen = {}

        def fake_fetch_text(url, **kw):
            seen.update(kw)
            seen["url"] = url
            return FIXTURE_HTML

        monkeypatch.setattr(scrape_site, "fetch_text", fake_fetch_text)
        # First logo candidate (og:image) "succeeds".
        monkeypatch.setattr(
            scrape_site, "_download_image",
            lambda url, out, session: (out.write_bytes(_png_bytes()) or True))

        result = scrape_site.scrape("https://acme.com/", tmp_path)

        # The HTML went through fetch_text with the byte cap and the UA session.
        assert seen["url"] == "https://acme.com/"
        assert seen["max_bytes"] == scrape_site.MAX_HTML_BYTES
        assert seen["session"] is not None
        # Artifacts persisted, title + first logo tier resolved.
        assert (tmp_path / "site.html").read_text() == FIXTURE_HTML
        assert "We make widgets" in (tmp_path / "site.txt").read_text()
        assert result.title == "Acme Widgets"
        assert result.logo_source == "og:image"

    def test_continues_without_logo_when_all_candidates_fail(self, monkeypatch, tmp_path):
        monkeypatch.setattr(scrape_site, "fetch_text", lambda url, **kw: FIXTURE_HTML)
        monkeypatch.setattr(scrape_site, "_download_image",
                            lambda url, out, session: False)
        result = scrape_site.scrape("https://acme.com/", tmp_path)
        assert result.logo_path is None
        assert result.logo_source is None
        # text extraction still happened
        assert (tmp_path / "site.txt").exists()
