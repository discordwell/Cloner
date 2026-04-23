"""Thin CLI wrapper around cloner's ElevenLabsClient for the autopitch agent.

Subcommands:
  clone   — clone a voice from voice_sample.wav (or any list of audio files)
  speak   — read pitch.txt (or --text) and write pitch.mp3 via ElevenLabs TTS

Usage:
  python -m autopitch.scripts.tts clone --run autopitch/runs/jane-doe-acme --name "Jane Doe"
  python -m autopitch.scripts.tts speak --run autopitch/runs/jane-doe-acme --voice-id vXXXX
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Make cloner src importable (runs from repo root or subdir)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def cmd_clone(args: argparse.Namespace) -> int:
    from src.voice.elevenlabs_client import ElevenLabsClient

    run_dir = Path(args.run)
    audio_paths = [str(p) for p in (args.audio or [run_dir / "voice_sample.wav"])]
    for a in audio_paths:
        if not Path(a).exists():
            print(f"audio file missing: {a}", file=sys.stderr)
            return 2

    client = ElevenLabsClient()
    voice_id = client.clone_voice(
        name=args.name,
        audio_files=audio_paths,
        description=args.description,
    )
    (run_dir / "voice_id.txt").write_text(voice_id, encoding="utf-8")
    print(f"voice_id: {voice_id}")
    return 0


def cmd_speak(args: argparse.Namespace) -> int:
    from src.voice.elevenlabs_client import ElevenLabsClient

    run_dir = Path(args.run)
    if args.text:
        text = args.text
    else:
        pitch = run_dir / "pitch.txt"
        if not pitch.exists():
            print(f"no pitch.txt in {run_dir}", file=sys.stderr)
            return 2
        text = pitch.read_text(encoding="utf-8")

    voice_id = args.voice_id
    if not voice_id:
        vid_file = run_dir / "voice_id.txt"
        if vid_file.exists():
            voice_id = vid_file.read_text(encoding="utf-8").strip()
    if not voice_id:
        print("no --voice-id and no runs/voice_id.txt — specify one", file=sys.stderr)
        return 2

    out_path = Path(args.output) if args.output else run_dir / "pitch.mp3"
    client = ElevenLabsClient()
    saved = client.generate_speech(
        text=text,
        voice_id=voice_id,
        output_path=str(out_path),
        stability=args.stability,
        similarity_boost=args.similarity,
        style=args.style,
    )
    print(f"saved: {saved}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("clone")
    c.add_argument("--run", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--audio", nargs="+", help="Audio file paths (default: {run}/voice_sample.wav)")
    c.add_argument("--description", default=None)
    c.set_defaults(func=cmd_clone)

    s = sub.add_parser("speak")
    s.add_argument("--run", required=True)
    s.add_argument("--voice-id", help="Override voice_id (else uses {run}/voice_id.txt)")
    s.add_argument("--text", help="Override pitch text (else reads {run}/pitch.txt)")
    s.add_argument("--output", help="Override output path (else {run}/pitch.mp3)")
    s.add_argument("--stability", type=float, default=0.5)
    s.add_argument("--similarity", type=float, default=0.75)
    s.add_argument("--style", type=float, default=0.15)
    s.set_defaults(func=cmd_speak)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
