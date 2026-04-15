"""
Integrated clone pipeline.

Combines voice cloning with lip-sync video generation to create talking head videos.

Two lip-sync paths:
  - Atlas (default): ElevenLabs TTS audio + face image -> Atlas API -> MP4
  - Viseme (legacy): ElevenLabs TTS with viseme timing -> local compositing -> MP4
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CloneProfile:
    """Complete profile for a cloned subject."""
    subject_id: str
    voice_id: Optional[str] = None
    viseme_library_path: Optional[str] = None
    reference_frame_path: Optional[str] = None
    created_at: float = 0.0

    def is_complete(self, lipsync_backend: str = "atlas") -> bool:
        """Check if profile has all required components for the given backend."""
        if not self.voice_id:
            return False
        if lipsync_backend == "atlas":
            return bool(
                self.reference_frame_path
                and Path(self.reference_frame_path).exists()
            )
        return bool(
            self.viseme_library_path
            and Path(self.viseme_library_path).exists()
        )


class ClonePipeline:
    """
    End-to-end pipeline for creating talking head videos.

    Workflow (Atlas path - default):
    1. Capture subject face image + audio sample
    2. Clone voice from audio via ElevenLabs
    3. Generate TTS audio for response text
    4. Send TTS audio + face image to Atlas -> lip-synced MP4

    Workflow (Viseme path - legacy):
    1. Capture subject video + audio
    2. Build viseme library from video
    3. Clone voice from audio
    4. Generate TTS with viseme timing -> local compositing
    """

    def __init__(
        self,
        elevenlabs_api_key: Optional[str] = None,
        atlas_api_key: Optional[str] = None,
        data_dir: str = "data",
        lipsync_backend: str = "atlas",
    ):
        """
        Initialize the clone pipeline.

        Args:
            elevenlabs_api_key: ElevenLabs API key (or env var)
            atlas_api_key: Atlas API key (or env var)
            data_dir: Base directory for data storage
            lipsync_backend: "atlas" (default) or "viseme" (legacy)
        """
        self.api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self.atlas_api_key = atlas_api_key or os.getenv("ATLAS_API_KEY")
        self.data_dir = Path(data_dir)
        self.profiles_dir = self.data_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.lipsync_backend = lipsync_backend

        # Lazy-loaded components
        self._voice_service = None
        self._tts_viseme = None
        self._viseme_builder = None
        self._atlas_client = None

    @property
    def voice_service(self):
        """Lazy-load voice cloning service."""
        if self._voice_service is None:
            from src.voice.voice_cloning_service import VoiceCloningService
            self._voice_service = VoiceCloningService(self.api_key)
        return self._voice_service

    @property
    def tts_viseme(self):
        """Lazy-load TTS with viseme output."""
        if self._tts_viseme is None:
            from src.viseme.tts_viseme import TTSWithVisemes
            self._tts_viseme = TTSWithVisemes(self.api_key)
        return self._tts_viseme

    @property
    def viseme_builder(self):
        """Lazy-load viseme library builder."""
        if self._viseme_builder is None:
            from src.viseme.viseme_library import VisemeLibraryBuilder
            self._viseme_builder = VisemeLibraryBuilder()
        return self._viseme_builder

    @property
    def atlas_client(self):
        """Lazy-load Atlas lip-sync client."""
        if self._atlas_client is None:
            from src.video.atlas_client import AtlasClient
            self._atlas_client = AtlasClient(api_key=self.atlas_api_key)
        return self._atlas_client

    def create_profile_from_video(
        self,
        subject_id: str,
        video_path: str,
        clone_voice: bool = True,
        voice_description: Optional[str] = None
    ) -> CloneProfile:
        """
        Create a complete clone profile from a video file.

        Args:
            subject_id: Unique identifier for the subject
            video_path: Path to video file with subject speaking
            clone_voice: Whether to clone voice from audio track
            voice_description: Optional description for cloned voice

        Returns:
            CloneProfile with voice_id and viseme library
        """
        logger.info(f"Creating profile for '{subject_id}' from {video_path}")
        start_time = time.time()

        profile = CloneProfile(
            subject_id=subject_id,
            created_at=time.time()
        )

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 1. Build viseme library from video
        logger.info("Building viseme library...")
        viseme_start = time.time()

        library_path = self.data_dir / "visemes" / subject_id
        library = self.viseme_builder.build_from_video(
            video_path=str(video_path),
            subject_id=subject_id,
            output_path=str(library_path),
            sample_interval=3
        )

        profile.viseme_library_path = str(library_path)
        viseme_time = time.time() - viseme_start
        logger.info(f"Viseme library built in {viseme_time:.1f}s: {len(library.templates)} visemes")

        # Save reference frame
        if library.neutral_frame is not None:
            import cv2
            ref_path = library_path / "reference_frame.png"
            cv2.imwrite(str(ref_path), library.neutral_frame)
            profile.reference_frame_path = str(ref_path)

        # 2. Clone voice from audio track
        if clone_voice and self.api_key:
            logger.info("Extracting audio and cloning voice...")
            voice_start = time.time()

            # Extract audio from video
            audio_path = self._extract_audio(video_path)

            if audio_path:
                try:
                    voice_id = self.voice_service.clone_voice_from_audio(
                        name=f"Clone_{subject_id}",
                        audio_files=[audio_path],
                        description=voice_description or f"Voice clone of {subject_id}",
                        validate=True,
                        auto_process=True
                    )
                    profile.voice_id = voice_id
                    voice_time = time.time() - voice_start
                    logger.info(f"Voice cloned in {voice_time:.1f}s: {voice_id}")
                except Exception as e:
                    logger.warning(f"Voice cloning failed: {e}")

        # Save profile metadata
        self._save_profile(profile)

        total_time = time.time() - start_time
        logger.info(f"Profile created in {total_time:.1f}s total")

        return profile

    def _extract_audio(self, video_path: Path) -> Optional[str]:
        """Extract audio track from video."""
        import subprocess

        audio_path = video_path.with_suffix('.wav')

        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',
                str(audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)

            if result.returncode == 0 and audio_path.exists():
                return str(audio_path)
            else:
                logger.warning(f"FFmpeg audio extraction failed: {result.stderr.decode()}")
                return None

        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            return None

    def generate_talking_video(
        self,
        subject_id: str,
        text: str,
        output_path: str,
        voice_settings: Optional[Dict[str, float]] = None,
        lipsync_backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a talking head video from text.

        Args:
            subject_id: Subject profile to use
            text: Text for the subject to speak
            output_path: Path for output video
            voice_settings: Optional voice settings (stability, similarity_boost)
            lipsync_backend: Override lip-sync backend ("atlas" or "viseme")

        Returns:
            Dictionary with output paths and timing info
        """
        backend = lipsync_backend or self.lipsync_backend

        if backend == "atlas":
            return self._generate_talking_video_atlas(
                subject_id, text, output_path, voice_settings
            )
        else:
            return self._generate_talking_video_viseme(
                subject_id, text, output_path, voice_settings
            )

    def _generate_talking_video_atlas(
        self,
        subject_id: str,
        text: str,
        output_path: str,
        voice_settings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Generate talking video using Atlas API (audio + face image -> MP4)."""
        logger.info(f"Generating talking video for '{subject_id}' via Atlas")
        start_time = time.time()

        profile = self._load_profile(subject_id)
        if not profile:
            raise ValueError(f"Profile not found: {subject_id}")
        if not profile.voice_id:
            raise ValueError(f"Profile has no voice clone: {subject_id}")
        if not profile.reference_frame_path or not Path(profile.reference_frame_path).exists():
            raise ValueError(
                f"Profile has no reference face image: {subject_id}. "
                "Atlas requires a face image. Set reference_frame_path on the profile."
            )

        result = {
            "subject_id": subject_id,
            "text": text,
            "lipsync_backend": "atlas",
            "timings": {},
        }

        # 1. Generate TTS audio
        logger.info("Generating TTS audio...")
        tts_start = time.time()

        output_path = Path(output_path)
        audio_path = output_path.with_suffix(".mp3")

        voice_settings = voice_settings or {}
        from src.voice.elevenlabs_client import ElevenLabsClient
        tts_client = ElevenLabsClient(api_key=self.api_key)
        tts_client.generate_speech(
            text=text,
            voice_id=profile.voice_id,
            output_path=str(audio_path),
            stability=voice_settings.get("stability", 0.5),
            similarity_boost=voice_settings.get("similarity_boost", 0.75),
        )

        result["audio_path"] = str(audio_path)
        result["timings"]["tts"] = time.time() - tts_start

        # 2. Send to Atlas for lip-sync video
        logger.info("Generating lip-sync video via Atlas...")
        atlas_start = time.time()

        video_path = self.atlas_client.generate_lipsync(
            audio_path=str(audio_path),
            image_path=profile.reference_frame_path,
            output_path=str(output_path),
        )

        result["video_path"] = video_path
        result["timings"]["atlas"] = time.time() - atlas_start
        result["timings"]["total"] = time.time() - start_time

        logger.info(f"Video generated in {result['timings']['total']:.1f}s: {video_path}")
        return result

    def _generate_talking_video_viseme(
        self,
        subject_id: str,
        text: str,
        output_path: str,
        voice_settings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Generate talking video using legacy viseme compositing."""
        logger.info(f"Generating talking video for '{subject_id}' via viseme compositing")
        start_time = time.time()

        profile = self._load_profile(subject_id)
        if not profile:
            raise ValueError(f"Profile not found: {subject_id}")
        if not profile.voice_id:
            raise ValueError(f"Profile has no voice clone: {subject_id}")

        result = {
            "subject_id": subject_id,
            "text": text,
            "lipsync_backend": "viseme",
            "timings": {},
        }

        # 1. Generate TTS with viseme timing
        logger.info("Generating speech with viseme timing...")
        tts_start = time.time()

        output_path = Path(output_path)
        audio_path = output_path.with_suffix('.mp3')

        voice_settings = voice_settings or {}
        tts_result = self.tts_viseme.generate_with_visemes(
            text=text,
            voice_id=profile.voice_id,
            output_path=str(audio_path),
            stability=voice_settings.get('stability', 0.5),
            similarity_boost=voice_settings.get('similarity_boost', 0.75)
        )

        result["audio_path"] = tts_result.audio_path
        result["audio_duration_ms"] = tts_result.audio_duration_ms
        result["viseme_count"] = len(tts_result.viseme_events)
        result["timings"]["tts"] = time.time() - tts_start

        # 2. Load viseme library
        from src.viseme.viseme_library import VisemeLibrary
        library = VisemeLibrary.load(profile.viseme_library_path)

        if library.neutral_frame is None:
            raise ValueError("Viseme library has no reference frame")

        # 3. Composite video
        logger.info("Compositing video...")
        composite_start = time.time()

        from src.viseme.realtime_compositor import RealtimeVisemeCompositor
        compositor = RealtimeVisemeCompositor(library)

        video_path = compositor.render_to_video(
            base_frame=library.neutral_frame,
            tts_result=tts_result,
            output_path=str(output_path),
            include_audio=True
        )

        result["video_path"] = video_path
        result["timings"]["composite"] = time.time() - composite_start
        result["timings"]["total"] = time.time() - start_time

        logger.info(f"Video generated in {result['timings']['total']:.1f}s: {video_path}")
        return result

    def generate_response_realtime(
        self,
        subject_id: str,
        text: str,
        frame_callback,
        audio_callback=None
    ):
        """
        Generate response with real-time frame output.

        For streaming/live scenarios where you need frames as they're generated.

        Args:
            subject_id: Subject profile to use
            text: Text to speak
            frame_callback: Called with each video frame
            audio_callback: Called when audio is ready
        """
        profile = self._load_profile(subject_id)
        if not profile or not profile.voice_id:
            raise ValueError(f"Invalid profile: {subject_id}")

        # Generate TTS with timing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            audio_path = f.name

        tts_result = self.tts_viseme.generate_with_visemes(
            text=text,
            voice_id=profile.voice_id,
            output_path=audio_path
        )

        if audio_callback:
            audio_callback(audio_path)

        # Load library and stream frames
        from src.viseme.viseme_library import VisemeLibrary
        from src.viseme.realtime_compositor import RealtimeVisemeCompositor

        library = VisemeLibrary.load(profile.viseme_library_path)
        compositor = RealtimeVisemeCompositor(library)

        for frame in compositor.generate_frames(
            library.neutral_frame,
            tts_result.viseme_events,
            tts_result.audio_duration_ms
        ):
            frame_callback(frame)

    def _save_profile(self, profile: CloneProfile):
        """Save profile to disk."""
        import json

        profile_path = self.profiles_dir / f"{profile.subject_id}.json"
        data = {
            "subject_id": profile.subject_id,
            "voice_id": profile.voice_id,
            "viseme_library_path": profile.viseme_library_path,
            "reference_frame_path": profile.reference_frame_path,
            "created_at": profile.created_at
        }

        with open(profile_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_profile(self, subject_id: str) -> Optional[CloneProfile]:
        """Load profile from disk."""
        import json

        profile_path = self.profiles_dir / f"{subject_id}.json"
        if not profile_path.exists():
            return None

        with open(profile_path, 'r') as f:
            data = json.load(f)

        return CloneProfile(**data)

    def list_profiles(self) -> List[str]:
        """List all available profiles."""
        return [p.stem for p in self.profiles_dir.glob("*.json")]

    def benchmark_pipeline(self, text: str = "Hello, this is a test.") -> Dict[str, float]:
        """
        Benchmark the pipeline latency without actually generating.

        Returns:
            Dictionary with timing for each stage
        """
        from src.viseme import PhonemeToVisemeMapper

        timings = {}

        # Viseme mapping (local, fast)
        mapper = PhonemeToVisemeMapper()
        start = time.perf_counter()
        frames = mapper.text_to_visemes(text, 2.0)
        timings["viseme_mapping_ms"] = (time.perf_counter() - start) * 1000

        # Estimate TTS latency (based on typical ElevenLabs performance)
        timings["tts_estimated_ms"] = 500 + len(text) * 10  # ~500ms TTFB + ~10ms/char

        # Frame generation (depends on duration)
        duration_s = 2.0
        fps = 30
        total_frames = int(duration_s * fps)
        # Estimate ~1ms per frame for compositing on M4
        timings["composite_estimated_ms"] = total_frames * 1

        timings["total_estimated_ms"] = sum([
            timings["viseme_mapping_ms"],
            timings["tts_estimated_ms"],
            timings["composite_estimated_ms"]
        ])

        return timings


def demo():
    """Demo the pipeline."""
    print("Clone Pipeline Demo")
    print("=" * 50)

    pipeline = ClonePipeline()

    # Benchmark
    print("\nBenchmarking pipeline latency...")
    timings = pipeline.benchmark_pipeline("Hello, how are you today?")

    print(f"\nLatency estimates:")
    for stage, ms in timings.items():
        print(f"  {stage}: {ms:.1f}ms")

    print(f"\nTotal estimated latency: {timings['total_estimated_ms']:.0f}ms")
    print("(Actual TTS latency depends on network and API load)")


if __name__ == "__main__":
    demo()
