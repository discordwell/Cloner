"""CLI entry for autopitch — creates a blueprint for a prospect, ready for the agent.

Usage:
  python -m autopitch.scripts.run --name "Jane Doe" --url https://acme.example.com

  # With extras
  python -m autopitch.scripts.run \
      --name "Jane Doe" \
      --url https://acme.example.com \
      --company "Acme Widgets" \
      --linkedin https://linkedin.com/in/janedoe \
      --role CEO \
      --region midwest_us \
      --duration 45

After the blueprint is written, invoke the agent to actually produce the video:
  claude --agent autopitch "Run autopitch for autopitch/runs/{slug}"

or paste that instruction into an interactive Claude Code session.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)

RUNS_DIR = Path("autopitch/runs")

_STAGES = [
    "scrape",
    "find_photo",
    "find_voice",
    "clone_voice",
    "cartoonify_portrait",
    "cartoonify_scene",
    "analyze",
    "write_pitch",
    "tts",
    "lipsync",
]

_ASSET_KEYS = [
    "photo_raw", "photo_cartoon", "scene_cartoon", "voice_sample",
    "logo", "site_text", "hypothesis", "pitch_text", "pitch_audio", "final_mp4",
]


def slugify(text: str) -> str:
    """Lowercase, alphanumeric + hyphen, collapse repeats."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unknown"


def derive_company_from_url(url: str) -> str:
    host = urlparse(url).netloc or url
    host = host.replace("www.", "")
    return host.split(".")[0].capitalize()


def _build_frontmatter(name: str, url: str, company: str,
                        linkedin: Optional[str],
                        role: Optional[str], region: Optional[str],
                        gender: Optional[str], age: Optional[str],
                        target_duration_s: float, slug: str) -> Dict[str, Any]:
    extras: Dict[str, Any] = {}
    if role:
        extras["role"] = role
    if region:
        extras["region_hint"] = region
    if gender:
        extras["gender_hint"] = gender
    if age:
        extras["age_hint"] = age

    stages: Dict[str, Any] = {s: {"status": "pending"} for s in _STAGES}
    stages["find_voice"].update({"source": None, "fallback_reason": None})
    stages["clone_voice"].update({"voice_id": None})

    return {
        "name": name,
        "company": company,
        "url": url,
        "linkedin": linkedin,
        "extras": extras,
        "target_duration_s": int(target_duration_s),
        "slug": slug,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages": stages,
        "assets": {k: None for k in _ASSET_KEYS},
    }


def build_blueprint(name: str, url: str, company: str,
                     linkedin: Optional[str] = None,
                     role: Optional[str] = None,
                     region: Optional[str] = None,
                     gender: Optional[str] = None,
                     age: Optional[str] = None,
                     target_duration_s: float = 60,
                     slug: Optional[str] = None) -> tuple[str, str]:
    """Return (slug, blueprint_markdown). YAML is safely escaped."""
    slug = slug or f"{slugify(name)}-{slugify(company)}"
    fm = _build_frontmatter(
        name=name, url=url, company=company, linkedin=linkedin,
        role=role, region=region, gender=gender, age=age,
        target_duration_s=target_duration_s, slug=slug,
    )
    frontmatter = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True,
                                  default_flow_style=False).rstrip()
    body = (
        "---\n"
        f"{frontmatter}\n"
        "---\n"
        "\n"
        "# Notes\n"
        "\n"
        "(Free-form notes. The agent appends here as it works — why it picked a fallback,\n"
        "why it rejected a photo candidate, any quirks observed.)\n"
    )
    return slug, body


def create_run(name: str, url: str, company: Optional[str] = None, **kwargs) -> Path:
    """Create a run directory with blueprint.md. Returns the run dir path."""
    company = company or derive_company_from_url(url)
    slug, body = build_blueprint(name=name, url=url, company=company, **kwargs)
    run_dir = RUNS_DIR / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "blueprint.md").write_text(body, encoding="utf-8")
    return run_dir


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--company", default=None, help="Override company name (else derived from URL)")
    p.add_argument("--linkedin", default=None)
    p.add_argument("--role", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--gender", default=None)
    p.add_argument("--age", default=None)
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--slug", default=None, help="Override auto-generated slug")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_dir = create_run(
        name=args.name, url=args.url, company=args.company,
        linkedin=args.linkedin, role=args.role, region=args.region,
        gender=args.gender, age=args.age,
        target_duration_s=args.duration, slug=args.slug,
    )

    print(f"blueprint: {run_dir / 'blueprint.md'}")
    print()
    print("Next: invoke the autopitch agent to run the pipeline. From this repo:")
    print()
    print(f'  claude --agent autopitch "Run autopitch for {run_dir}"')
    print()
    print("Or, in an interactive Claude Code session, just paste that instruction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
