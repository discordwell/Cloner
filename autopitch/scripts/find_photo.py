"""Find a frontal photo of a person via Bing Image Search + mediapipe face detection.

Behavior:
  1. Query `"{name}" "{company}"` via Bing Image Search API (or SerpAPI).
  2. Download the top N candidates.
  3. Run mediapipe face detection; pick the highest-confidence frontal face
     (face must occupy >=5% of the image area to reject thumbnail hits).
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

import requests
from PIL import Image

logger = logging.getLogger(__name__)

BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
DEFAULT_TIMEOUT = 15
MIN_FACE_AREA_RATIO = 0.05  # face bbox must be >= 5% of image area


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


def _detect_best_face(img: Image.Image) -> Optional[tuple[float, float]]:
    """Run mediapipe face detection. Return (confidence, face_area_ratio) or None."""
    # Import lazily — mediapipe is slow to load
    import mediapipe as mp
    import numpy as np

    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    total = h * w

    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4) as fd:
        res = fd.process(rgb)
        if not res.detections:
            return None
        best_conf = 0.0
        best_area = 0.0
        for det in res.detections:
            conf = float(det.score[0]) if det.score else 0.0
            bbox = det.location_data.relative_bounding_box
            area = max(0.0, bbox.width * bbox.height)
            # Score combines confidence and relative area so a large confident face wins
            # over a tiny confident one.
            if conf * area > best_conf * best_area:
                best_conf = conf
                best_area = area
        if best_area < MIN_FACE_AREA_RATIO:
            return None
        return best_conf, best_area


def _download(url: str) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT, headers={
            "User-Agent": "Mozilla/5.0 (compatible; autopitch/1.0)"
        })
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
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
