"""Find a frontal photo of a person via Bing Image Search + face detection.

Behavior:
  1. Query `"{name}" "{company}"` via Bing Image Search API (or SerpAPI).
  2. Download the top N candidates.
  3. Run face detection (mediapipe BlazeFace, falling back to OpenCV Haar);
     pick the highest-confidence frontal face (face must occupy >=5% of the
     image area to reject thumbnail hits).
  4. Save as {run_dir}/photo_raw.jpg.

Usage:
  python -m autopitch.scripts.find_photo --name "Jane Doe" --company "Acme" \
      --run autopitch/runs/jane-doe-acme
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
from PIL import Image

from autopitch.scripts.download import download as download_file
from autopitch.scripts.download import fetch_bytes

logger = logging.getLogger(__name__)

BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
DEFAULT_TIMEOUT = 15
MIN_FACE_AREA_RATIO = 0.05  # face bbox must be >= 5% of image area
MAX_IMAGE_BYTES = 16 * 1024 * 1024  # cap per candidate fetch (search hits are arbitrary URLs)
MAX_FACE_MODEL_BYTES = 64 * 1024 * 1024  # BlazeFace is ~230KB; this is a safe ceiling

# mediapipe >= 0.10.21 removed the legacy `mp.solutions` API, and the Tasks
# API that replaced it does not bundle a model file — so the BlazeFace model
# is cached in data/models/ (gitignored) and downloaded on first use.
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
_FACE_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "models" / "blaze_face_short_range.tflite"
)


@dataclass
class Candidate:
    url: str
    width: int
    height: int
    source: str


@dataclass
class PhotoResult:
    saved_path: Path
    source_url: str
    confidence: float
    face_area_ratio: float


def _query_bing(name: str, company: str, api_key: str, count: int) -> List[Candidate]:
    q = f'"{name}" "{company}"'
    params = {
        "q": q,
        "count": count,
        "imageType": "Photo",
        "safeSearch": "Moderate",
        "size": "Medium",
    }
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    resp = requests.get(BING_ENDPOINT, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    out = []
    for item in data.get("value", []):
        out.append(Candidate(
            url=item["contentUrl"],
            width=item.get("width", 0),
            height=item.get("height", 0),
            source="bing",
        ))
    return out


def _query_serpapi(name: str, company: str, api_key: str, count: int) -> List[Candidate]:
    q = f'"{name}" "{company}"'
    params = {
        "engine": "google_images",
        "q": q,
        "api_key": api_key,
        "num": count,
    }
    resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    out = []
    for item in data.get("images_results", [])[:count]:
        out.append(Candidate(
            url=item.get("original") or item.get("thumbnail"),
            width=item.get("original_width", 0),
            height=item.get("original_height", 0),
            source="serpapi",
        ))
    return out


def _ensure_face_model() -> Optional[Path]:
    """Return the cached BlazeFace model path, downloading it on first use.

    Streams to a temp file through the capped downloader (then atomically
    replaces), rather than reading the whole body into memory — the file lives
    on disk anyway, and the cap keeps a misbehaving CDN from blowing up memory.
    """
    if _FACE_MODEL_PATH.exists():
        return _FACE_MODEL_PATH
    tmp = _FACE_MODEL_PATH.with_name(_FACE_MODEL_PATH.name + ".tmp")
    try:
        download_file(_FACE_MODEL_URL, tmp, timeout=DEFAULT_TIMEOUT,
                      max_bytes=MAX_FACE_MODEL_BYTES)
        tmp.replace(_FACE_MODEL_PATH)
        return _FACE_MODEL_PATH
    except Exception as e:
        logger.warning("could not download face detection model: %s", e)
        tmp.unlink(missing_ok=True)
        return None


def _detect_faces_mediapipe(rgb: np.ndarray) -> Optional[List[tuple[float, float]]]:
    """Detect faces with the mediapipe Tasks API (BlazeFace).

    Returns a list of (confidence, face_area_ratio) — empty when no faces —
    or None when mediapipe or its model is unavailable, so the caller can
    fall back to OpenCV.
    """
    # Import lazily — mediapipe is slow to load
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as e:
        logger.debug("mediapipe unavailable: %s", e)
        return None

    model_path = _ensure_face_model()
    if model_path is None:
        return None

    h, w = rgb.shape[:2]
    total = float(h * w)
    try:
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=0.4,
        )
        with mp_vision.FaceDetector.create_from_options(options) as detector:
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    except Exception as e:
        logger.warning("mediapipe face detection failed: %s", e)
        return None

    out = []
    for det in result.detections:
        conf = float(det.categories[0].score) if det.categories else 0.0
        bbox = det.bounding_box  # pixel coordinates, unlike the legacy API
        out.append((conf, max(0.0, bbox.width * bbox.height) / total))
    return out


def _detect_faces_opencv(rgb: np.ndarray) -> List[tuple[float, float]]:
    """Haar-cascade fallback (model ships with opencv-python; works offline).

    Less accurate than BlazeFace — misses tilted faces — but keeps the
    pipeline usable when mediapipe or its model download is unavailable.
    """
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    total = float(h * w)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        logger.warning("could not load OpenCV haarcascade")
        return []
    min_side = max(24, int(0.1 * min(h, w)))
    rects, _, weights = cascade.detectMultiScale3(
        gray, scaleFactor=1.1, minNeighbors=6,
        minSize=(min_side, min_side), outputRejectLevels=True)
    out = []
    for (x, y, fw, fh), weight in zip(rects, weights):
        # levelWeights are unbounded stage scores (roughly 0-10); squash to
        # 0-1 so scores stay comparable with the mediapipe path.
        conf = min(1.0, float(np.ravel(weight)[0]) / 10.0)
        out.append((conf, (fw * fh) / total))
    return out


def _detect_best_face(img: Image.Image) -> Optional[tuple[float, float]]:
    """Detect faces and return (confidence, face_area_ratio) of the best one.

    Returns None when no detected face passes MIN_FACE_AREA_RATIO.
    """
    rgb = np.ascontiguousarray(np.array(img.convert("RGB")))
    detections = _detect_faces_mediapipe(rgb)
    if detections is None:
        detections = _detect_faces_opencv(rgb)
    if not detections:
        return None
    # Score combines confidence and relative area so a large confident face
    # wins over a tiny confident one.
    best_conf, best_area = max(detections, key=lambda d: d[0] * d[1])
    if best_area < MIN_FACE_AREA_RATIO:
        return None
    return best_conf, best_area


def _download(url: str) -> Optional[Image.Image]:
    try:
        data = fetch_bytes(
            url, timeout=DEFAULT_TIMEOUT, max_bytes=MAX_IMAGE_BYTES,
            headers={"User-Agent": "Mozilla/5.0 (compatible; autopitch/1.0)"},
        )
        return Image.open(BytesIO(data))
    except Exception as e:
        logger.debug("download %s failed: %s", url, e)
        return None


def find_photo(name: str, company: str, run_dir: Path,
                max_candidates: int = 8,
                bing_key: Optional[str] = None,
                serpapi_key: Optional[str] = None) -> Optional[PhotoResult]:
    """Query image search, pick best frontal, save to {run_dir}/photo_raw.jpg."""
    bing_key = bing_key or os.getenv("BING_IMAGE_SEARCH_KEY")
    serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY")

    if bing_key:
        candidates = _query_bing(name, company, bing_key, max_candidates)
        logger.info("bing returned %d candidates", len(candidates))
    elif serpapi_key:
        candidates = _query_serpapi(name, company, serpapi_key, max_candidates)
        logger.info("serpapi returned %d candidates", len(candidates))
    else:
        raise RuntimeError(
            "No image search API key configured. Set BING_IMAGE_SEARCH_KEY or SERPAPI_KEY."
        )

    best: Optional[PhotoResult] = None
    best_score = 0.0

    for c in candidates:
        img = _download(c.url)
        if img is None:
            continue
        detection = _detect_best_face(img)
        if detection is None:
            logger.debug("no face in %s", c.url)
            continue
        conf, area_ratio = detection
        score = conf * area_ratio
        logger.info("candidate %s: conf=%.2f area=%.2f score=%.3f",
                    c.url, conf, area_ratio, score)
        if score > best_score:
            run_dir.mkdir(parents=True, exist_ok=True)
            target = run_dir / "photo_raw.jpg"
            img.convert("RGB").save(str(target), "JPEG", quality=92)
            best = PhotoResult(
                saved_path=target,
                source_url=c.url,
                confidence=conf,
                face_area_ratio=area_ratio,
            )
            best_score = score

    return best


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--company", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = find_photo(args.name, args.company, Path(args.run), args.max_candidates)
    if result is None:
        print("no frontal face found in any candidate", file=sys.stderr)
        return 1
    print(f"saved:      {result.saved_path}")
    print(f"source:     {result.source_url}")
    print(f"confidence: {result.confidence:.2f}")
    print(f"face_area:  {result.face_area_ratio:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
