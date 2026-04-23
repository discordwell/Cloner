"""Call GPT to produce an AI-opportunity hypothesis from scraped site text.

Usage:
  python -m autopitch.scripts.analyze_site --run autopitch/runs/jane-doe-acme \
      --company "Acme Widgets" --url https://acme.example.com
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from autopitch.scripts._llm import complete

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def analyze(run_dir: Path, company: str, url: str,
             model: str = "gpt-5.4") -> Path:
    """Read run_dir/site.txt, call LLM, write hypothesis.md."""
    site_txt = (run_dir / "site.txt").read_text(encoding="utf-8")
    template = (PROMPTS_DIR / "analyze_site.txt").read_text(encoding="utf-8")
    prompt = template.format(company=company, url=url, site_text=site_txt)

    logger.info("analyzing site for %s (%d chars of text)", company, len(site_txt))
    output = complete(prompt, model=model, max_tokens=900, temperature=0.7)

    out_path = run_dir / "hypothesis.md"
    out_path.write_text(output, encoding="utf-8")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--company", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out = analyze(Path(args.run), args.company, args.url, model=args.model)
    print(f"saved: {out} ({out.stat().st_size}B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
