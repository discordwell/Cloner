"""Thin CLI wrapper around cloner's AtlasClient for the autopitch agent.

Takes the cartoonified scene image + TTS audio and produces final.mp4.

Usage:
  python -m autopitch.scripts.lipsync --run autopitch/runs/jane-doe-acme
  python -m autopitch.scripts.lipsync --audio path/to/pitch.mp3 --image path/to/scene.png \
      --output path/to/final.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def lipsync(audio_path: Path, image_path: Path, output_path: Path,
             poll_interval: float = 2.0, timeout: float = 300.0) -> str:
    from src.video.atlas_client import AtlasClient

    with AtlasClient(poll_interval=poll_interval, timeout=timeout) as client:
        return client.generate_lipsync(
            audio_path=str(audio_path),
            image_path=str(image_path),
            output_path=str(output_path),
        )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="Run dir — defaults audio/image/output to run/{pitch.mp3,scene_cartoon.png,final.mp4}")
    p.add_argument("--audio", help="Audio file path (overrides --run)")
    p.add_argument("--image", help="Face/scene image path (overrides --run)")
    p.add_argument("--output", help="Output MP4 path (overrides --run)")
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.run:
        run_dir = Path(args.run)
        audio = Path(args.audio) if args.audio else run_dir / "pitch.mp3"
        image = Path(args.image) if args.image else run_dir / "scene_cartoon.png"
        output = Path(args.output) if args.output else run_dir / "final.mp4"
    else:
        if not (args.audio and args.image and args.output):
            print("either --run or all of --audio/--image/--output required", file=sys.stderr)
            return 2
        audio = Path(args.audio)
        image = Path(args.image)
        output = Path(args.output)

    for label, path in [("audio", audio), ("image", image)]:
        if not path.exists():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 2

    saved = lipsync(audio, image, output, args.poll_interval, args.timeout)
    print(f"saved: {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
