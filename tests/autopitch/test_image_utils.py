"""Tests for autopitch.scripts.image_utils (QC + IO)."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts.image_utils import (
    crop_to_content,
    load_and_validate,
    validate_image_quality,
)


def _solid_color(size, color=(128, 128, 128)):
    return Image.new("RGB", size, color)


def _noise(size, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestValidateImageQuality:
    def test_rejects_tiny_image(self):
        img = _noise((100, 100))
        ok, reason = validate_image_quality(img)
        assert not ok
        assert "too small" in reason

    def test_rejects_uniform_color(self):
        img = _solid_color((1024, 1024))
        ok, reason = validate_image_quality(img)
        assert not ok
        assert "uniform" in reason or "variance" in reason

    def test_accepts_noise(self):
        img = _noise((1024, 1024))
        ok, reason = validate_image_quality(img)
        assert ok, reason


class TestLoadAndValidate:
    def test_rejects_tiny_file(self, tmp_path):
        tiny = tmp_path / "tiny.png"
        tiny.write_bytes(b"\x89PNG\r\n\x1a\n")
        ok, reason = load_and_validate(str(tiny))
        assert not ok
        assert "too small" in reason

    def test_accepts_valid_png(self, tmp_path):
        src = tmp_path / "valid.png"
        _noise((1024, 1024)).save(str(src))
        out = tmp_path / "out.png"
        ok, reason = load_and_validate(str(src), str(out))
        assert ok, reason
        assert out.exists()

    def test_missing_file(self, tmp_path):
        ok, reason = load_and_validate(str(tmp_path / "nonexistent.png"))
        assert not ok
        assert "not found" in reason


class TestCropToContent:
    def test_noop_on_fully_opaque(self):
        img = _noise((200, 200)).convert("RGBA")
        result = crop_to_content(img)
        assert result.size == (200, 200)

    def test_crops_to_nonzero_alpha_region(self):
        """Build an RGBA with a 50x50 opaque box in the center."""
        img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        box = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        img.paste(box, (75, 75))
        result = crop_to_content(img, padding=0)
        # Should crop roughly to 50x50
        assert 45 <= result.width <= 55
        assert 45 <= result.height <= 55
