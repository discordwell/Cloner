"""
Real-time clone response system.

End-to-end pipeline:
1. Capture phase: Build viseme library + clone voice from video
2. Response phase: Question → LLM → TTS → Composite → Play

Optimized for speed (~4-6s latency from question to response).
"""

import os
import sys
import time
import json
import tempfile
import subprocess
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CloneSession:
    """Active clone session with loaded resources."""
    subject_id: str
    viseme_library_path: str
    voice_id: Optional[str] = None  # ElevenLabs voice ID
    use_local_tts: bool = True  # Use macOS 'say' if no voice_id


class RealtimeCloneSystem:
    """
    Real-time clone response system.

    Usage:
        system = RealtimeCloneSystem()

        # Setup phase (~10s)
        session = system.setup_from_video("path/to/video.mp4", "subject_name")

        # Response phase (~4-6s per response)
        system.respond("Hello, how are you?", play_immediately=True)
    """

    def __init__(
        self,
        elevenlabs_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        data_dir: str = "data"
    ):
        self.elevenlabs_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self.anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.data_dir = Path(data_dir)

        self.session: Optional[CloneSession] = None
        self._library = None
        self._compositor = None
        self._tts = None

        # Response queue for async generation
        self._response_queue = []

    def setup_from_video(
        self,
        video_path: str,
        subject_id: str,
        clone_voice: bool = True,
        enhance_lighting: bool = True
    ) -> CloneSession:
        """
        Setup clone session from video file.

        Args:
            video_path: Path to video of subject speaking
            subject_id: Unique identifier
            clone_voice: Whether to clone voice via ElevenLabs
            enhance_lighting: Apply CLAHE lighting enhancement

        Returns:
            CloneSession ready for responses
        """
        logger.info(f"Setting up clone session for '{subject_id}'")
        start_time = time.time()

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 1. Build viseme library
        logger.info("Building viseme library...")
        lib_start = time.time()

        from src.viseme.viseme_library import VisemeLibraryBuilder
        builder = VisemeLibraryBuilder()

        library_path = self.data_dir / "visemes" / subject_id
        library = builder.build_from_video(
            video_path=str(video_path),
            subject_id=subject_id,
            output_path=str(library_path),
            sample_interval=2
        )

        logger.info(f"Library built in {time.time() - lib_start:.1f}s")

        # 2. Enhance lighting if requested
        if enhance_lighting:
            logger.info("Enhancing lighting...")
            self._enhance_library_lighting(library_path)
            library_path = self.data_dir / "visemes" / f"{subject_id}_enhanced"

        # 3. Clone voice if requested and API key available
        voice_id = None
        if clone_voice and self.elevenlabs_key:
            logger.info("Cloning voice...")
            voice_id = self._clone_voice_from_video(video_path, subject_id)

        # Create session
        self.session = CloneSession(
            subject_id=subject_id,
            viseme_library_path=str(library_path),
            voice_id=voice_id,
            use_local_tts=(voice_id is None)
        )

        # Pre-load resources
        self._load_session_resources()

        total_time = time.time() - start_time
        logger.info(f"Setup complete in {total_time:.1f}s")

        return self.session

    def setup_from_webcam(
        self,
        subject_id: str,
        duration: int = 6,
        clone_voice: bool = True
    ) -> CloneSession:
        """
        Setup clone session by capturing from webcam.

        Args:
            subject_id: Unique identifier
            duration: Capture duration in seconds
            clone_voice: Whether to clone voice

        Returns:
            CloneSession ready for responses
        """
        logger.info(f"Capturing {duration}s from webcam...")

        # Capture video
        capture_path = self.data_dir / "captures" / f"{subject_id}_{int(time.time())}.mp4"
        capture_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            'ffmpeg', '-y',
            '-f', 'avfoundation',
            '-framerate', '30',
            '-video_size', '1280x720',
            '-i', '0:0',
            '-t', str(duration),
            str(capture_path)
        ]

        print(f"Recording in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("RECORDING - Speak now!")

        subprocess.run(cmd, capture_output=True)

        print("Done capturing.")

        return self.setup_from_video(
            str(capture_path),
            subject_id,
            clone_voice=clone_voice
        )

    def respond(
        self,
        text: str,
        play_immediately: bool = True,
        output_path: Optional[str] = None,
        on_complete: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Generate and play a response.

        Args:
            text: Text for the clone to speak
            play_immediately: Play video when ready
            output_path: Custom output path (default: temp file)
            on_complete: Callback when video is ready

        Returns:
            Dict with timing info and output path
        """
        if not self.session:
            raise RuntimeError("No active session. Call setup_from_video first.")

        result = {"text": text, "timings": {}}
        start_time = time.time()

        # 1. Generate TTS with visemes
        logger.info("Generating TTS...")
        tts_start = time.time()

        audio_path = output_path or tempfile.mktemp(suffix='.wav')
        if self.session.use_local_tts:
            tts_result = self._tts.generate_with_visemes(text, audio_path)
        else:
            tts_result = self._tts.generate_with_visemes(
                text, self.session.voice_id, audio_path
            )

        result["timings"]["tts"] = time.time() - tts_start
        result["audio_duration_ms"] = tts_result.audio_duration_ms

        # 2. Composite video
        logger.info("Compositing video...")
        comp_start = time.time()

        video_path = output_path or tempfile.mktemp(suffix='.mp4')
        video_path = self._compositor.render_to_video(
            base_frame=self._library.neutral_frame,
            tts_result=tts_result,
            output_path=video_path,
            include_audio=True
        )

        result["timings"]["composite"] = time.time() - comp_start
        result["video_path"] = video_path
        result["timings"]["total"] = time.time() - start_time

        logger.info(f"Response generated in {result['timings']['total']:.2f}s")

        # 3. Play if requested
        if play_immediately:
            self.play_video(video_path)

        # 4. Callback if provided
        if on_complete:
            on_complete(video_path)

        return result

    def respond_to_question(
        self,
        question: str,
        context: str = "",
        play_immediately: bool = True,
        max_tokens: int = 150
    ) -> Dict[str, Any]:
        """
        Generate LLM response to question, then speak it.

        Args:
            question: The question to answer
            context: Additional context for the LLM
            play_immediately: Play video when ready
            max_tokens: Max response length

        Returns:
            Dict with response text, timing info, and paths
        """
        result = {"question": question, "timings": {}}
        start_time = time.time()

        # 1. Generate LLM response
        logger.info("Generating LLM response...")
        llm_start = time.time()

        response_text = self._generate_llm_response(question, context, max_tokens)

        result["timings"]["llm"] = time.time() - llm_start
        result["response_text"] = response_text

        # 2. Generate video response
        video_result = self.respond(response_text, play_immediately=play_immediately)

        result.update(video_result)
        result["timings"]["total"] = time.time() - start_time

        return result

    def play_video(self, video_path: str):
        """Play video using system player."""
        subprocess.run(['open', video_path], capture_output=True)

    def _load_session_resources(self):
        """Pre-load library and compositor for fast responses."""
        from src.viseme.viseme_library import VisemeLibrary
        from src.viseme.enhanced_compositor import (
            EnhancedVisemeCompositor,
            EnhancedCompositorConfig,
            MotionConfig,
            BlendConfig
        )

        # Load library
        self._library = VisemeLibrary.load(self.session.viseme_library_path)

        # Setup compositor
        config = EnhancedCompositorConfig(
            fps=30,
            motion=MotionConfig(
                breathing_enabled=True,
                breathing_amplitude=0.004,
                sway_enabled=True,
                sway_amplitude_x=2.0,
                sway_amplitude_y=1.5,
                blink_enabled=True,
            ),
            blend=BlendConfig(
                feather_radius=18,
                color_match=True,
                brightness_match=True,
                edge_blur=9,
            )
        )
        self._compositor = EnhancedVisemeCompositor(self._library, config)

        # Setup TTS
        if self.session.use_local_tts:
            from src.viseme.local_tts import LocalTTS
            self._tts = LocalTTS()
        else:
            from src.viseme.tts_viseme import TTSWithVisemes
            self._tts = TTSWithVisemes(self.elevenlabs_key)

    def _enhance_library_lighting(self, library_path: Path):
        """Enhance lighting in library images."""
        import cv2
        import numpy as np
        import shutil

        src_dir = Path(library_path)
        dst_dir = src_dir.parent / f"{src_dir.name}_enhanced"

        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True)

        def enhance(img):
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            return np.clip(cv2.addWeighted(enhanced, 1.1, np.zeros_like(enhanced), 0, 8), 0, 255).astype(np.uint8)

        for item in src_dir.iterdir():
            if item.is_dir() and item.name.startswith('viseme_'):
                vis_dst = dst_dir / item.name
                vis_dst.mkdir()
                for img_file in item.glob('*.png'):
                    img = cv2.imread(str(img_file))
                    cv2.imwrite(str(vis_dst / img_file.name), enhance(img))
            elif item.suffix == '.png':
                img = cv2.imread(str(item))
                cv2.imwrite(str(dst_dir / item.name), enhance(img))
            elif item.suffix == '.json':
                shutil.copy(item, dst_dir / item.name)

    def _clone_voice_from_video(self, video_path: Path, subject_id: str) -> Optional[str]:
        """Extract audio and clone voice."""
        try:
            # Extract audio
            audio_path = self.data_dir / "audio" / f"{subject_id}_source.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '1',
                str(audio_path)
            ]
            subprocess.run(cmd, capture_output=True)

            if not audio_path.exists():
                return None

            # Clone voice
            from src.voice.voice_cloning_service import VoiceCloningService
            service = VoiceCloningService(self.elevenlabs_key)

            voice_id = service.clone_voice_from_audio(
                name=f"Clone_{subject_id}",
                audio_files=[str(audio_path)],
                validate=True,
                auto_process=True
            )

            return voice_id

        except Exception as e:
            logger.warning(f"Voice cloning failed: {e}")
            return None

    def _generate_llm_response(
        self,
        question: str,
        context: str,
        max_tokens: int
    ) -> str:
        """Generate response using available LLM."""

        prompt = f"""You are being interviewed. Answer the following question concisely and naturally, as if speaking aloud. Keep your response to 1-3 sentences.

{f"Context: {context}" if context else ""}

Question: {question}

Answer:"""

        # Try Anthropic first
        if self.anthropic_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.anthropic_key)
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            except Exception as e:
                logger.warning(f"Anthropic failed: {e}")

        # Try OpenAI
        if self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"OpenAI failed: {e}")

        # Fallback to simple response
        return f"That's a great question about {question[:50]}. Let me think about that."


def demo():
    """Demo the realtime clone system."""
    print("=" * 50)
    print("Realtime Clone System Demo")
    print("=" * 50)

    system = RealtimeCloneSystem()

    # Check for existing library
    existing = Path("data/visemes/user_lit_enhanced")
    if existing.exists():
        print(f"\nUsing existing library: {existing}")
        system.session = CloneSession(
            subject_id="user_lit",
            viseme_library_path=str(existing),
            use_local_tts=True
        )
        system._load_session_resources()
    else:
        print("\nNo existing library. Run setup_from_webcam first.")
        return

    # Test response
    print("\nGenerating test response...")
    result = system.respond(
        "Hello! This is a test of the realtime clone system. It should be pretty fast.",
        play_immediately=True
    )

    print(f"\nTimings:")
    for k, v in result["timings"].items():
        print(f"  {k}: {v:.2f}s")


if __name__ == "__main__":
    demo()
