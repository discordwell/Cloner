"""Generate a ~60s first-person-self-addressing pitch script.

Usage:
  python -m autopitch.scripts.write_pitch --run autopitch/runs/jane-doe-acme \
      --name "Jane Doe" --company "Acme Widgets" [--role CEO] [--duration 60]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional

from autopitch.scripts._llm import complete

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
DEFAULT_WPS = 2.4

# Like analyze_site, fill only the known placeholders in a single pass. The
# hypothesis is LLM-generated markdown that can carry literal braces (the
# analyze_site skeleton uses {short title}), and the template may grow example
# braces of its own — str.format() would raise KeyError on either. Substituting
# just these names leaves every other brace untouched and never re-scans an
# inserted value (a {company} arriving via the hypothesis stays literal).
_PLACEHOLDERS = re.compile(
    r"\{(first_name|name|company|role_clause|hypothesis"
    r"|target_duration_s|words_per_second|target_words)\}"
)


def build_prompt(name: str, company: str, role: Optional[str],
                  hypothesis: str, target_duration_s: float,
                  words_per_second: float = DEFAULT_WPS) -> str:
    """Fill the pitch template, preserving any literal braces in the inputs."""
    template = (PROMPTS_DIR / "pitch_script.txt").read_text(encoding="utf-8")
    parts = name.split()
    first_name = parts[0] if parts else name
    role_clause = f"who runs {company} as {role}, " if role else ""
    target_words = int(round(target_duration_s * words_per_second))
    values = {
        "name": name,
        "first_name": first_name,
        "company": company,
        "role_clause": role_clause,
        "hypothesis": hypothesis,
        "target_duration_s": str(int(target_duration_s)),
        "words_per_second": str(words_per_second),
        "target_words": str(target_words),
    }
    return _PLACEHOLDERS.sub(lambda m: values[m.group(1)], template)


def write_pitch(run_dir: Path, name: str, company: str,
                 role: Optional[str] = None,
                 target_duration_s: float = 60,
                 words_per_second: float = DEFAULT_WPS,
                 model: str = "gpt-5.4") -> Path:
    hyp_path = run_dir / "hypothesis.md"
    hypothesis = hyp_path.read_text(encoding="utf-8") if hyp_path.exists() else ""

    prompt = build_prompt(name, company, role, hypothesis, target_duration_s, words_per_second)
    logger.info("writing %ds pitch for %s", int(target_duration_s), name)
    pitch = complete(prompt, model=model, max_tokens=500, temperature=0.85)

    out_path = run_dir / "pitch.txt"
    out_path.write_text(pitch, encoding="utf-8")
    word_count = len(pitch.split())
    logger.info("pitch: %d words (target ~%d)",
                word_count, int(target_duration_s * words_per_second))
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--company", required=True)
    p.add_argument("--role", default=None)
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--wps", type=float, default=DEFAULT_WPS)
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out = write_pitch(
        Path(args.run), args.name, args.company,
        role=args.role, target_duration_s=args.duration,
        words_per_second=args.wps, model=args.model,
    )
    print(f"saved: {out}")
    print("---")
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
