"""Scrape a company homepage for text + logo.

Writes:
  {run_dir}/site.html
  {run_dir}/site.txt
  {run_dir}/logo.png          (best guess: og:image -> apple-touch-icon -> favicon)

Usage:
  python -m autopitch.scripts.scrape_site --url https://acme.example.com --run runs/jane-doe-acme
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

from autopitch.scripts.download import fetch_bytes, fetch_text

logger = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)
DEFAULT_TIMEOUT = 15
MAX_TEXT_CHARS = 8000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
# A homepage is still a third-party URL; cap the HTML so a hostile or
# misconfigured server can't stream an unbounded body into memory. Generous
# enough for any real homepage's source (inline CSS/JS included; images are
# external).
MAX_HTML_BYTES = 10 * 1024 * 1024


@dataclass
class ScrapeResult:
    url: str
    title: Optional[str]
    text_path: Path
    html_path: Path
    logo_path: Optional[Path]
    logo_source: Optional[str]       # "og:image" | "apple-touch-icon" | "favicon" | None


def _clean_text(soup: BeautifulSoup) -> str:
    """Extract readable text — strip scripts, styles, navs, footers."""
    for tag in soup(["script", "style", "noscript", "svg", "form"]):
        tag.decompose()
    chunks = []
    for el in soup.select("h1, h2, h3, h4, p, li"):
        t = el.get_text(separator=" ", strip=True)
        if t and len(t) > 2:
            chunks.append(t)
    return "\n".join(chunks)[:MAX_TEXT_CHARS]


def _collect_logo_candidates(soup: BeautifulSoup, base_url: str) -> List[tuple[str, str]]:
    """Return [(source, absolute_url)] for logo candidates, in preference order."""
    out: List[tuple[str, str]] = []

    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        out.append(("og:image", urljoin(base_url, og["content"])))

    twitter = soup.find("meta", attrs={"name": "twitter:image"})
    if twitter and twitter.get("content"):
        out.append(("twitter:image", urljoin(base_url, twitter["content"])))

    for rel in ("apple-touch-icon", "apple-touch-icon-precomposed"):
        link = soup.find("link", rel=rel)
        if link and link.get("href"):
            out.append((rel, urljoin(base_url, link["href"])))

    icon = soup.find("link", rel=lambda v: v and "icon" in v.lower())
    if icon and icon.get("href"):
        out.append(("favicon", urljoin(base_url, icon["href"])))

    # Fallback: /favicon.ico at root
    parsed = urlparse(base_url)
    out.append(("favicon.ico", f"{parsed.scheme}://{parsed.netloc}/favicon.ico"))
    return out


def _download_image(url: str, out_path: Path, session: requests.Session) -> bool:
    """Download (capped) and normalize to PNG. Returns True on success."""
    try:
        data = fetch_bytes(url, timeout=DEFAULT_TIMEOUT,
                           max_bytes=MAX_IMAGE_BYTES, session=session)
        if len(data) < 200:
            return False
        img = Image.open(BytesIO(data)).convert("RGBA")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        return True
    except Exception as e:
        logger.debug("logo candidate %s failed: %s", url, e)
        return False


def scrape(url: str, run_dir: Path, timeout: int = DEFAULT_TIMEOUT,
            user_agent: str = DEFAULT_UA) -> ScrapeResult:
    """Fetch url, persist html/text/logo into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = user_agent

    logger.info("scraping %s", url)
    html = fetch_text(url, timeout=timeout, max_bytes=MAX_HTML_BYTES, session=session)

    html_path = run_dir / "site.html"
    html_path.write_text(html, encoding="utf-8")

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None

    text = _clean_text(soup)
    text_path = run_dir / "site.txt"
    text_path.write_text(text, encoding="utf-8")

    logo_path: Optional[Path] = None
    logo_source: Optional[str] = None
    for source, candidate_url in _collect_logo_candidates(soup, url):
        target = run_dir / "logo.png"
        if _download_image(candidate_url, target, session):
            logo_path = target
            logo_source = source
            logger.info("logo: %s (%s)", candidate_url, source)
            break

    if logo_path is None:
        logger.warning("no logo candidate succeeded; continuing without logo")

    return ScrapeResult(
        url=url, title=title, text_path=text_path, html_path=html_path,
        logo_path=logo_path, logo_source=logo_source,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--run", required=True, help="Run directory (e.g. autopitch/runs/jane-doe-acme)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = scrape(args.url, Path(args.run))
    print(f"title:       {result.title}")
    print(f"text:        {result.text_path} ({result.text_path.stat().st_size}B)")
    print(f"logo:        {result.logo_path} ({result.logo_source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
