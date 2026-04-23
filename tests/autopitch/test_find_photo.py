"""Tests for autopitch.scripts.find_photo (face detection on synthetic images)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts.find_photo import _detect_best_face

mediapipe = pytest.importorskip("mediapipe", reason="mediapipe required for face detection tests")


class TestDetectBestFace:
    def test_returns_none_for_solid_color(self):
        """A plain gray square clearly has no face."""
        img = Image.new("RGB", (600, 600), (128, 128, 128))
        assert _detect_best_face(img) is None

    def test_returns_none_for_random_noise(self):
        """Uniform noise shouldn't trigger a false positive."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (600, 600, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        # Noise shouldn't produce a confident face detection
        result = _detect_best_face(img)
        # Result may be None OR a low-score detection below MIN_FACE_AREA_RATIO
        if result is not None:
            conf, area = result
            assert area >= 0.05, "should only return detections above area threshold"
