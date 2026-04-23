"""Download a URL to a local file. Used by the autopitch agent to save ChatGPT
image output without shell-escaping concerns.

Usage:
  python -m autopitch.scripts.download --url "https://..." --out path/to/file.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_BYTES = 20 * 1024 * 1024


def download(url: str, out_path: Path, timeout: int = DEFAULT_TIMEOUT,
              max_bytes: int = MAX_BYTES) -> int:
    """Stream-download url to out_path. Returns bytes written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        size = 0
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"download exceeds {max_bytes}B cap")
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
