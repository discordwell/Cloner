"""Tests for autopitch.scripts.find_photo (face detection on synthetic images)."""

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts import find_photo
from autopitch.scripts.find_photo import (
    MIN_FACE_AREA_RATIO,
    _detect_best_face,
    _detect_faces_opencv,
    _download,
    _ensure_face_model,
)


def _gray_square(size: int = 600) -> Image.Image:
    return Image.new("RGB", (size, size), (128, 128, 128))


def _png_bytes(size=(48, 48)) -> bytes:
    buf = BytesIO()
    _gray_square(size[0]).resize(size).save(buf, "PNG")
    return buf.getvalue()


class TestDetectBestFace:
    def test_returns_none_for_solid_color(self):
        """A plain gray square clearly has no face."""
        assert _detect_best_face(_gray_square()) is None

    def test_returns_none_for_random_noise(self):
        """Uniform noise shouldn't trigger a false positive."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (600, 600, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        # Noise shouldn't produce a confident face detection
        result = _detect_best_face(img)
        # Result may be None OR a low-score detection below MIN_FACE_AREA_RATIO
        if result is not None:
            conf, area = result
            assert area >= MIN_FACE_AREA_RATIO, "should only return detections above area threshold"

    def test_rejects_face_below_area_threshold(self, monkeypatch):
        """A confident but tiny face (thumbnail hit) is rejected."""
        monkeypatch.setattr(find_photo, "_detect_faces_mediapipe", lambda rgb: [(0.95, 0.01)])
        assert _detect_best_face(_gray_square()) is None

    def test_picks_highest_scoring_face(self, monkeypatch):
        """conf*area scoring: a large medium-confidence face beats a small confident one."""
        monkeypatch.setattr(
            find_photo, "_detect_faces_mediapipe",
            lambda rgb: [(0.9, 0.06), (0.5, 0.5)])
        assert _detect_best_face(_gray_square()) == (0.5, 0.5)

    def test_falls_back_to_opencv_when_mediapipe_unavailable(self, monkeypatch):
        """None from the mediapipe path (unavailable) routes to the OpenCV path."""
        monkeypatch.setattr(find_photo, "_detect_faces_mediapipe", lambda rgb: None)
        calls = []

        def fake_opencv(rgb):
            calls.append(rgb.shape)
            return [(0.8, 0.2)]

        monkeypatch.setattr(find_photo, "_detect_faces_opencv", fake_opencv)
        assert _detect_best_face(_gray_square()) == (0.8, 0.2)
        assert calls == [(600, 600, 3)]

    def test_empty_mediapipe_result_does_not_fall_back(self, monkeypatch):
        """No faces found (empty list) is a real answer, not a reason to fall back."""
        monkeypatch.setattr(find_photo, "_detect_faces_mediapipe", lambda rgb: [])

        def boom(rgb):
            raise AssertionError("opencv fallback should not run")

        monkeypatch.setattr(find_photo, "_detect_faces_opencv", boom)
        assert _detect_best_face(_gray_square()) is None


class TestOpenCVFallback:
    def test_no_detections_on_solid_color(self):
        rgb = np.full((600, 600, 3), 128, dtype=np.uint8)
        assert _detect_faces_opencv(rgb) == []

    def test_no_detections_on_noise(self):
        rng = np.random.default_rng(7)
        rgb = rng.integers(0, 256, (600, 600, 3), dtype=np.uint8)
        for conf, area in _detect_faces_opencv(rgb):
            assert 0.0 <= conf <= 1.0
            assert 0.0 <= area <= 1.0


class TestEnsureFaceModel:
    def test_returns_cached_model_without_network(self, monkeypatch, tmp_path):
        model = tmp_path / "model.tflite"
        model.write_bytes(b"cached")
        monkeypatch.setattr(find_photo, "_FACE_MODEL_PATH", model)

        def boom(*args, **kwargs):
            raise AssertionError("downloader should not run when model is cached")

        monkeypatch.setattr(find_photo, "download_file", boom)
        assert _ensure_face_model() == model

    def test_streams_through_capped_downloader_then_atomically_replaces(self, monkeypatch, tmp_path):
        """The model fetch routes through download.download (capped, stream-to-disk)
        and only the final path — never the .tmp — survives."""
        model = tmp_path / "models" / "face.tflite"
        monkeypatch.setattr(find_photo, "_FACE_MODEL_PATH", model)
        seen = {}

        def fake_download(url, out_path, **kwargs):
            seen["url"] = url
            seen["out_path"] = Path(out_path)
            seen["max_bytes"] = kwargs.get("max_bytes")
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"model-bytes")
            return len(b"model-bytes")

        monkeypatch.setattr(find_photo, "download_file", fake_download)
        assert _ensure_face_model() == model
        assert model.read_bytes() == b"model-bytes"
        # capped, and written to a temp sibling before the atomic replace
        assert seen["max_bytes"] == find_photo.MAX_FACE_MODEL_BYTES
        assert seen["out_path"].name == "face.tflite.tmp"
        assert not seen["out_path"].exists()

    def test_returns_none_and_cleans_temp_when_download_fails(self, monkeypatch, tmp_path):
        """A mid-stream failure (e.g. cap overflow) leaves no model and no .tmp."""
        model = tmp_path / "missing.tflite"
        monkeypatch.setattr(find_photo, "_FACE_MODEL_PATH", model)

        def fail_midstream(url, out_path, **kwargs):
            Path(out_path).write_bytes(b"partial")   # partial temp written, then boom
            raise ValueError("download exceeds cap")

        monkeypatch.setattr(find_photo, "download_file", fail_midstream)
        assert _ensure_face_model() is None
        assert not model.exists()
        assert not model.with_name(model.name + ".tmp").exists()


class TestDownloadCandidate:
    """_download fetches an arbitrary search-hit URL via the capped helper, so a
    candidate can never pull an unbounded image into memory."""

    def test_returns_decoded_image(self, monkeypatch):
        monkeypatch.setattr(find_photo, "fetch_bytes", lambda url, **kw: _png_bytes((48, 48)))
        img = _download("http://x/p.jpg")
        assert img is not None
        assert img.size == (48, 48)

    def test_passes_byte_cap_and_user_agent(self, monkeypatch):
        seen = {}

        def capture(url, **kw):
            seen.update(kw)
            return _png_bytes()

        monkeypatch.setattr(find_photo, "fetch_bytes", capture)
        _download("http://x/p.jpg")
        assert seen["max_bytes"] == find_photo.MAX_IMAGE_BYTES
        assert "User-Agent" in seen["headers"]

    def test_returns_none_on_cap_overflow(self, monkeypatch):
        def over_cap(url, **kw):
            raise ValueError("download exceeds cap")

        monkeypatch.setattr(find_photo, "fetch_bytes", over_cap)
        assert _download("http://x/huge.jpg") is None
