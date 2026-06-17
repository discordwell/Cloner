"""Download a URL — to a local file, or into memory — with a hard byte cap that
is enforced *as the body streams*.

The cap matters because every consumer here fetches third-party URLs: ChatGPT
image output (the agent), a prospect's logo (scrape_site), and arbitrary image
search hits (find_photo). Trusting the ``Content-Length`` header alone is not
enough — a server can omit it or under-report it, and then ``resp.content``
reads an unbounded body straight into memory. So we count the bytes we actually
receive and abort the moment they cross the limit.

Usage:
  python -m autopitch.scripts.download --url "https://..." --out path/to/file.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_BYTES = 20 * 1024 * 1024


def _stream_capped(resp: requests.Response, max_bytes: int) -> Iterator[bytes]:
    """Yield body chunks from a streaming response, enforcing ``max_bytes``.

    Pre-checks the ``Content-Length`` header when it is present and numeric, so an
    honestly-oversized response is rejected before any bytes transfer. Then counts
    the bytes actually streamed, so a missing or under-reported header cannot slip
    past the cap. Raises ``ValueError`` the instant the limit is crossed.
    """
    declared = resp.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise ValueError(f"declared size {declared}B exceeds {max_bytes}B cap")
    total = 0
    for chunk in resp.iter_content(chunk_size=8192):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"download exceeds {max_bytes}B cap")
        yield chunk


def fetch_bytes(url: str, *, timeout: int = DEFAULT_TIMEOUT,
                 max_bytes: int = MAX_BYTES,
                 session: Optional[requests.Session] = None,
                 headers: Optional[Dict[str, str]] = None) -> bytes:
    """Fetch ``url`` into memory, capped at ``max_bytes``.

    Raises ``ValueError`` if the body exceeds the cap, or a ``requests`` exception
    on network/HTTP errors. Pass ``session`` to reuse an existing session (and its
    headers); ``headers`` are applied per-request.
    """
    getter = session.get if session is not None else requests.get
    with getter(url, timeout=timeout, stream=True, headers=headers) as resp:
        resp.raise_for_status()
        return b"".join(_stream_capped(resp, max_bytes))


def download(url: str, out_path: Path, timeout: int = DEFAULT_TIMEOUT,
              max_bytes: int = MAX_BYTES) -> int:
    """Stream-download ``url`` to ``out_path``, capped at ``max_bytes``.

    Returns bytes written. Raises ``ValueError`` if the download exceeds the cap.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size = 0
    with requests.get(url, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in _stream_capped(resp, max_bytes):
                size += len(chunk)
                f.write(chunk)
    return size


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        size = download(args.url, Path(args.out), args.timeout, args.max_bytes)
    except Exception as e:
        print(f"download failed: {e}", file=sys.stderr)
        return 1
    print(f"saved: {args.out} ({size}B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
