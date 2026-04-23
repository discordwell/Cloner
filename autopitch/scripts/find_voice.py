"""Find a voice sample for a target person (YouTube/podcast search) OR fall back to
an ElevenLabs library voice matching inferred demographics.

Pipeline:
  1. yt-dlp searches `"{name}" interview`, `"{name}" podcast`, `"{name}" keynote`.
  2. Download audio of the top candidate that passes duration gates.
  3. Diarize: pyannote if HUGGINGFACE_TOKEN is set, else a pydub-silence heuristic
     that picks the longest continuous non-silent chunk (assumes solo interviews).
  4. Extract ~target_s of that dominant speaker's speech.
  5. Save as voice_sample.wav.

On total failure, callers should use pick_library_voice().

Usage:
  python -m autopitch.scripts.find_voice --name "Jane Doe" --run autopitch/runs/jane-doe-acme
  python -m autopitch.scripts.find_voice --pick-library --gender female --region us \
      --run autopitch/runs/jane-doe-acme
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_QUERIES = ["interview", "podcast", "keynote", "talk"]
DEFAULT_TARGET_S = 90
DEFAULT_MIN_DURATION_S = 120
DEFAULT_MAX_DURATION_S = 1800
DEFAULT_MAX_CANDIDATES = 3


@dataclass
class VoiceResult:
    saved_path: Path
    source: str             # "cloned" | "library"
    detail: str             # YouTube URL, or library voice name
    voice_id: Optional[str] = None   # set when source=library
    duration_s: Optional[float] = None


@dataclass
class YouTubeCandidate:
    url: str
    title: str
    duration_s: float


# ── YouTube search via yt-dlp ──────────────────────────────────────

def search_youtube(query_suffix: str, name: str,
                    max_results: int = 5,
                    min_duration: int = DEFAULT_MIN_DURATION_S,
                    max_duration: int = DEFAULT_MAX_DURATION_S) -> List[YouTubeCandidate]:
    """Use yt-dlp's ytsearch to find candidate videos. Filters by duration."""
    from yt_dlp import YoutubeDL

    query = f'ytsearch{max_results}:"{name}" {query_suffix}'
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }
    out = []
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(query, download=False)
        for entry in info.get("entries", []) or []:
            dur = entry.get("duration") or 0
            if dur < min_duration or dur > max_duration:
                continue
            out.append(YouTubeCandidate(
                url=entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}",
                title=entry.get("title", ""),
                duration_s=float(dur),
            ))
    return out


def download_audio(url: str, out_wav: Path, fmt: str = "bestaudio[ext=m4a]/bestaudio") -> bool:
    """Download and transcode to mono 16kHz wav (ElevenLabs cloning likes wav)."""
    from yt_dlp import YoutubeDL

    with tempfile.TemporaryDirectory() as tmp:
        src_tmpl = str(Path(tmp) / "src.%(ext)s")
        opts = {
            "format": fmt,
            "outtmpl": src_tmpl,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            src = Path(ydl.prepare_filename(info))

        if not src.exists():
            return False

        out_wav.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-ac", "1", "-ar", "16000",
            str(out_wav),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.warning("ffmpeg transcode failed: %s", result.stderr.decode()[:300])
            return False
    return out_wav.exists()


# ── Speaker isolation ──────────────────────────────────────────────

def diarize_with_pyannote(wav_path: Path, hf_token: str) -> Optional[List[tuple[str, float, float]]]:
    """Return [(speaker_label, start_s, end_s)] from pyannote. None on failure."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        logger.warning("pyannote.audio not installed; skipping diarization")
        return None

    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        diarization = pipe(str(wav_path))
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((speaker, float(turn.start), float(turn.end)))
        return segments
    except Exception as e:
        logger.warning("pyannote diarization failed: %s", e)
        return None


def pick_dominant_speaker(segments: List[tuple[str, float, float]]) -> str:
    """Return the speaker label with the largest total duration."""
    totals: dict[str, float] = {}
    for spk, s, e in segments:
        totals[spk] = totals.get(spk, 0.0) + (e - s)
    return max(totals, key=totals.get)


def segments_for_speaker(segments: List[tuple[str, float, float]], speaker: str,
                          target_s: float) -> List[tuple[float, float]]:
    """Return contiguous segments for `speaker` until we've accumulated target_s seconds.

    Sorted by start time — keeps natural pacing rather than longest-first.
    """
    spk_segs = sorted([(s, e) for sp, s, e in segments if sp == speaker], key=lambda x: x[0])
    out: List[tuple[float, float]] = []
    acc = 0.0
    for s, e in spk_segs:
        seg_len = e - s
        if acc + seg_len <= target_s:
            out.append((s, e))
            acc += seg_len
        else:
            out.append((s, s + (target_s - acc)))
            break
    return out


def longest_speech_heuristic(wav_path: Path, target_s: float) -> List[tuple[float, float]]:
    """Fallback when pyannote isn't available: pick the longest continuous non-silent chunk.

    Doesn't dedup speakers; works best on solo interviews / monologue-style content.
    """
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    audio = AudioSegment.from_wav(str(wav_path))
    ranges_ms = detect_nonsilent(audio, min_silence_len=500, silence_thresh=audio.dBFS - 16)
    if not ranges_ms:
        return []

    ranges_ms.sort(key=lambda r: r[1] - r[0], reverse=True)
    picked: List[tuple[float, float]] = []
    acc = 0.0
    for start_ms, end_ms in ranges_ms:
        s, e = start_ms / 1000.0, end_ms / 1000.0
        seg_len = e - s
        if acc + seg_len <= target_s:
            picked.append((s, e))
            acc += seg_len
        else:
            picked.append((s, s + (target_s - acc)))
            break
    picked.sort(key=lambda r: r[0])
    return picked


def extract_segments(wav_path: Path, segments: List[tuple[float, float]], out_path: Path) -> bool:
    """Concatenate segments from wav_path into out_path via ffmpeg filtergraph."""
    if not segments:
        return False
    filter_parts = []
    for i, (s, e) in enumerate(segments):
        filter_parts.append(
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )
    concat_inputs = "".join(f"[a{i}]" for i in range(len(segments)))
    filter_complex = ";".join(filter_parts) + f";{concat_inputs}concat=n={len(segments)}:v=0:a=1[out]"

    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ac", "1", "-ar", "16000",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.error("ffmpeg extract failed: %s", result.stderr.decode()[:300])
        return False
    return out_path.exists()


# ── Library-voice fallback ─────────────────────────────────────────

def pick_library_voice(gender: Optional[str] = None, region: Optional[str] = None,
                        age: Optional[str] = None) -> Optional[tuple[str, str]]:
    """Pick the closest ElevenLabs library voice. Returns (voice_id, voice_name)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.voice.elevenlabs_client import ElevenLabsClient

    client = ElevenLabsClient()
    voices = client.list_voices()

    def score(v: dict) -> int:
        lbls = (v.get("description") or "").lower()
        name = (v.get("name") or "").lower()
        hay = f"{lbls} {name}"
        s = 0
        if gender and gender.lower() in hay:
            s += 3
        if region and region.lower() in hay:
            s += 2
        if age and age.lower() in hay:
            s += 1
        # Prefer premade/generated voices as they're high-quality defaults
        if v.get("category") in ("premade", "generated"):
            s += 1
        return s

    voices.sort(key=score, reverse=True)
    if not voices:
        return None
    best = voices[0]
    return best["voice_id"], best["name"]


# ── Top-level orchestration ────────────────────────────────────────

def find_voice(name: str, run_dir: Path,
                target_s: float = DEFAULT_TARGET_S,
                max_candidates: int = DEFAULT_MAX_CANDIDATES,
                search_queries: Optional[List[str]] = None,
                hf_token: Optional[str] = None) -> Optional[VoiceResult]:
    """Try to download a real voice sample. Returns None on failure."""
    run_dir.mkdir(parents=True, exist_ok=True)
    queries = search_queries or DEFAULT_SEARCH_QUERIES
    hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")

    all_candidates: List[YouTubeCandidate] = []
    for q in queries:
        try:
            hits = search_youtube(q, name, max_results=max_candidates)
            logger.info("search '%s': %d hits", q, len(hits))
            all_candidates.extend(hits)
        except Exception as e:
            logger.warning("yt search '%s' failed: %s", q, e)

    # Dedup by URL, prefer shorter (less multi-guest risk)
    seen = set()
    uniq: List[YouTubeCandidate] = []
    for c in sorted(all_candidates, key=lambda x: x.duration_s):
        if c.url in seen:
            continue
        seen.add(c.url)
        uniq.append(c)

    tmp_dir = run_dir / "_scratch"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for cand in uniq[: max_candidates * len(queries)]:
        raw_wav = tmp_dir / "candidate.wav"
        logger.info("trying %s (%ds): %s", cand.url, int(cand.duration_s), cand.title)
        if not download_audio(cand.url, raw_wav):
            continue

        segments = None
        if hf_token:
            diar = diarize_with_pyannote(raw_wav, hf_token)
            if diar:
                dominant = pick_dominant_speaker(diar)
                logger.info("pyannote dominant speaker=%s", dominant)
                segments = segments_for_speaker(diar, dominant, target_s)

        if segments is None:
            segments = longest_speech_heuristic(raw_wav, target_s)

        out = run_dir / "voice_sample.wav"
        if extract_segments(raw_wav, segments, out):
            total = sum(e - s for s, e in segments)
            logger.info("extracted %ds from %s", int(total), cand.url)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return VoiceResult(
                saved_path=out, source="cloned", detail=cand.url, duration_s=total,
            )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name")
    p.add_argument("--run", required=True)
    p.add_argument("--target-s", type=float, default=DEFAULT_TARGET_S)
    p.add_argument("--pick-library", action="store_true",
                   help="Skip search, just pick an ElevenLabs library voice matching demographics")
    p.add_argument("--gender", help="Hint for library fallback: male|female")
    p.add_argument("--region", help="Hint for library fallback: us|uk|midwest|...")
    p.add_argument("--age", help="Hint for library fallback: young|middle_aged|old")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.pick_library:
        picked = pick_library_voice(gender=args.gender, region=args.region, age=args.age)
        if picked is None:
            print("no library voice found", file=sys.stderr)
            return 1
        voice_id, voice_name = picked
        print(f"library_voice_id:   {voice_id}")
        print(f"library_voice_name: {voice_name}")
        return 0

    if not args.name:
        print("--name is required unless --pick-library", file=sys.stderr)
        return 2

    result = find_voice(args.name, Path(args.run), target_s=args.target_s)
    if result is None:
        print("no voice sample found; caller should pick --pick-library", file=sys.stderr)
        return 1
    print(f"saved:     {result.saved_path}")
    print(f"source:    {result.source}")
    print(f"detail:    {result.detail}")
    print(f"duration:  {result.duration_s:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
