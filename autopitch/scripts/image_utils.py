"""Image processing utilities for autopitch.

Adapted from clawed-command/tools/asset_pipeline/scripts/image_utils.py.
Scope here is Pixar-style scene/portrait images (opaque, painted) rather than
transparent game sprites — background removal is only used when explicitly
requested (e.g. preparing a logo for overlay).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

logger = logging.getLogger(__name__)


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background via rembg. No-op if already transparent or rembg missing."""
    if img.mode == "RGBA":
        alpha = np.array(img.split()[-1])
        if alpha.min() < 250:
            return img

    if not HAS_REMBG:
        logger.warning("rembg not installed; returning image unchanged")
        return img.convert("RGBA")

    return rembg_remove(img).convert("RGBA")


def crop_to_content(img: Image.Image, padding: int = 2) -> Image.Image:
    """Crop to bounding box of non-transparent pixels."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = np.array(img.split()[-1])
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any() or not cols.any():
        return img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = max(0, rmin - padding)
    rmax = min(img.height - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(img.width - 1, cmax + padding)

    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


def validate_image_quality(img: Image.Image, min_size: Tuple[int, int] = (512, 512)) -> Tuple[bool, str]:
    """Validate a generated Pixar-style image.

    Checks:
      - Minimum pixel dimensions (ChatGPT sometimes returns tiny placeholders)
      - Not mostly-one-color (failed generation, loading placeholder)
      - Passes a Laplacian-variance sharpness floor

    Returns (passed, reason).
    """
    if img.width < min_size[0] or img.height < min_size[1]:
        return False, f"too small: {img.size} < {min_size}"

    arr = np.array(img.convert("RGB"))
    # Color-variance check — placeholders are often near-uniform
    total_var = float(arr.var())
    if total_var < 100:
        return False, f"near-uniform color (variance={total_var:.0f}) — likely placeholder"

    # Sharpness via Laplacian variance
    gray = np.mean(arr, axis=2).astype(np.float64)
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1] + padded[2:, 1:-1]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        - 4 * padded[1:-1, 1:-1]
    )
    edge_var = float(np.var(laplacian))
    if edge_var < 200:
        return False, f"blurry — Laplacian variance {edge_var:.0f} < 200"

    return True, f"ok (size={img.size}, sharpness={edge_var:.0f})"


def load_and_validate(raw_path: str, out_path: Optional[str] = None,
                       strip_bg: bool = False,
                       min_size: Tuple[int, int] = (512, 512),
                       min_bytes: int = 50_000) -> Tuple[bool, str]:
    """Load a downloaded image, QC, optionally strip bg, save.

    Args:
        raw_path: Path to the raw downloaded image.
        out_path: Where to save the processed result. If None, saves in-place.
        strip_bg: If True, run rembg (useful for logos we'll composite later).
        min_size: Minimum acceptable width/height.
        min_bytes: Reject files smaller than this (mid-generation grabs).

    Returns (success, reason) — caller persists/logs as needed.
    """
    raw = Path(raw_path)
    if not raw.exists():
        return False, f"raw file not found: {raw_path}"
    if raw.stat().st_size < min_bytes:
        return False, f"file too small: {raw.stat().st_size}B < {min_bytes}B"

    img = Image.open(str(raw)).convert("RGBA")

    passed, reason = validate_image_quality(img, min_size=min_size)
    if not passed:
        return False, f"QC fail: {reason}"

    if strip_bg:
        img = remove_background(img)
        img = crop_to_content(img, padding=4)

    dest = Path(out_path) if out_path else raw
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(dest))
    return True, f"ok ({img.size[0]}x{img.size[1]}, {reason})"
