"""Tests for autopitch.scripts.run (slug/blueprint creation)."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts.run import (
    build_blueprint,
    create_run,
    derive_company_from_url,
    slugify,
)


def _parse_frontmatter(md: str) -> dict:
    assert md.startswith("---\n")
    rest = md[4:]
    end = rest.index("\n---\n")
    return yaml.safe_load(rest[:end])


class TestSlugify:
    def test_simple(self):
        assert slugify("Jane Doe") == "jane-doe"

    def test_mixed_punctuation(self):
        assert slugify("Jane D'Oe & Co.") == "jane-d-oe-co"

    def test_collapses_repeats(self):
        assert slugify("Jane   Doe!!!") == "jane-doe"

    def test_empty_fallback(self):
        assert slugify("!!!") == "unknown"


class TestDeriveCompany:
    def test_www_prefix(self):
        assert derive_company_from_url("https://www.acme.com") == "Acme"

    def test_no_www(self):
        assert derive_company_from_url("https://acme.example.com") == "Acme"

    def test_bare_host(self):
        assert derive_company_from_url("acme.com") == "Acme"


class TestBuildBlueprint:
    def test_frontmatter_has_core_fields(self):
        slug, md = build_blueprint(
            name="Jane Doe", url="https://acme.com", company="Acme Widgets",
        )
        assert slug == "jane-doe-acme-widgets"
        fm = _parse_frontmatter(md)
        assert fm["name"] == "Jane Doe"
        assert fm["url"] == "https://acme.com"
        assert fm["company"] == "Acme Widgets"
        assert fm["linkedin"] is None
        assert fm["target_duration_s"] == 60

    def test_all_stages_pending(self):
        _, md = build_blueprint(name="Jane", url="https://a.co", company="A")
        fm = _parse_frontmatter(md)
        expected_stages = {
            "scrape", "find_photo", "find_voice", "clone_voice",
            "cartoonify_portrait", "cartoonify_scene",
            "analyze", "write_pitch", "tts", "lipsync",
        }
        assert set(fm["stages"].keys()) == expected_stages
        for name, stage in fm["stages"].items():
            assert stage["status"] == "pending", f"{name} not pending"

    def test_extras_populate(self):
        _, md = build_blueprint(
            name="Jane", url="https://a.co", company="A",
            role="CEO", region="us", gender="female",
        )
        fm = _parse_frontmatter(md)
        assert fm["extras"] == {"role": "CEO", "region_hint": "us", "gender_hint": "female"}

    def test_custom_duration(self):
        _, md = build_blueprint(
            name="Jane", url="https://a.co", company="A",
            target_duration_s=45,
        )
        fm = _parse_frontmatter(md)
        assert fm["target_duration_s"] == 45

    def test_hostile_names_are_safely_escaped(self):
        """Names with YAML-special chars must still produce parseable frontmatter."""
        tricky = 'Jane "Q" O\'Doe: the Third'
        _, md = build_blueprint(
            name=tricky, url="https://a.co", company="A & B, Inc.",
        )
        fm = _parse_frontmatter(md)
        assert fm["name"] == tricky
        assert fm["company"] == "A & B, Inc."


class TestCreateRun:
    def test_writes_blueprint_to_runs_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = create_run(name="Jane Doe", url="https://acme.example.com")
        assert run_dir.exists()
        assert (run_dir / "blueprint.md").exists()
        text = (run_dir / "blueprint.md").read_text()
        assert "name: Jane Doe" in text
        assert "company: Acme" in text  # derived from URL
