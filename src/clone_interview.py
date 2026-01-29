"""
Clone Interview System - Integrated meeting capture with real-time clone responses.

Combines:
1. Meeting capture (Google Meet/Zoom join + audio monitoring)
2. Speech detection & transcription (Whisper/Deepgram)
3. Real-time clone response generation (viseme-based)

Flow:
    [Setup] Video capture → Build viseme library → Clone voice
    [Runtime] Meeting audio → Speech detection → Transcribe → LLM → TTS → Composite → Play

Usage:
    # Setup from existing video
    interview = CloneInterview()
    interview.setup_from_video("training.mp4", "subject_name")

    # Or capture live
    interview.setup_from_webcam("subject_name", duration=10)

    # Start interview mode
    interview.start_listening()
"""

import os
import sys
import time
import wave
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from collections import deque
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ListenerConfig:
    """Configuration for audio listener."""
    sample_rate: int = 48000
    channels: int = 1
    chunk_size: int = 1024

    # Voice activity detection
    silence_threshold: float = 500
    speech_threshold: float = 800
    min_speech_duration: float = 0.5
    silence_after_speech: float = 1.5
    max_recording_duration: float = 30.0


class AudioListener:
    """
    Cross-platform audio listener with voice activity detection.

    Detects speech, records it, and triggers transcription.
    """

    def __init__(
        self,
        config: Optional[ListenerConfig] = None,
        on_speech_callback: Optional[Callable[[str], None]] = None
    ):
        self.config = config or ListenerConfig()
        self.on_speech_callback = on_speech_callback

        self._is_listening = False
        self._stream = None
        self._listen_thread = None

        # VAD state
        self._is_speaking = False
        self._speech_start_time = None
        self._silence_start_time = None
        self._audio_buffer = deque(maxlen=int(
            self.config.sample_rate * self.config.max_recording_duration / self.config.chunk_size
        ))

        # Transcription client
        self._transcription_client = None

    def _init_transcription(self):
        """Initialize transcription client."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI
                self._transcription_client = OpenAI(api_key=openai_key)
                logger.info("Using OpenAI Whisper for transcription")
            except ImportError:
                logger.warning("OpenAI not installed, transcription disabled")
        else:
            logger.warning("No OPENAI_API_KEY, transcription disabled")

    def list_audio_devices(self) -> List[Dict]:
        """List available audio input devices."""
        try:
            import pyaudio
            audio = pyaudio.PyAudio()
            devices = []

            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate'])
                    })

            audio.terminate()
            return devices

        except ImportError:
            logger.error("pyaudio not installed")
            return []

    def find_loopback_device(self) -> Optional[int]:
        """Find a loopback/system audio capture device."""
        devices = self.list_audio_devices()

        # macOS: Look for BlackHole, Soundflower, or similar
        mac_keywords = ['blackhole', 'soundflower', 'loopback', 'multi-output']

        # Windows: Look for Stereo Mix, VB-Cable
        win_keywords = ['stereo mix', 'cable output', 'vb-audio', 'loopback', 'what u hear']

        keywords = mac_keywords + win_keywords

        for device in devices:
            name_lower = device['name'].lower()
            for keyword in keywords:
                if keyword in name_lower:
                    logger.info(f"Found loopback device: {device['name']}")
                    return device['index']

        return None

    def start_listening(self, device_index: Optional[int] = None):
        """Start listening for speech."""
        if self._is_listening:
            return

        self._init_transcription()

        try:
            import pyaudio

            self._audio = pyaudio.PyAudio()

            # Use specified device or try to find loopback
            if device_index is None:
                device_index = self.find_loopback_device()

            # Get device info
            if device_index is not None:
                dev_info = self._audio.get_device_info_by_index(device_index)
                sample_rate = int(dev_info['defaultSampleRate'])
                channels = min(int(dev_info['maxInputChannels']), 2)
                logger.info(f"Using device: {dev_info['name']} ({sample_rate}Hz, {channels}ch)")
            else:
                sample_rate = self.config.sample_rate
                channels = self.config.channels
                logger.info("Using default audio device")

            self._actual_sample_rate = sample_rate
            self._actual_channels = channels

            self._stream = self._audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.config.chunk_size
            )

            self._is_listening = True
            self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._listen_thread.start()

            logger.info("Audio listener started")

        except ImportError:
            logger.error("pyaudio not installed. Install with: pip install pyaudio")
            raise
        except Exception as e:
            logger.error(f"Failed to start audio listener: {e}")
            raise

    def stop_listening(self):
        """Stop listening."""
        self._is_listening = False

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if hasattr(self, '_audio') and self._audio:
            self._audio.terminate()

        logger.info("Audio listener stopped")

    def _get_rms(self, audio_data: bytes) -> float:
        """Calculate RMS (volume level) of audio data."""
        data = np.frombuffer(audio_data, dtype=np.int16)
        return float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

    def _listen_loop(self):
        """Main listening loop with voice activity detection."""
        recording_frames = []

        while self._is_listening:
            try:
                audio_data = self._stream.read(
                    self.config.chunk_size,
                    exception_on_overflow=False
                )
                rms = self._get_rms(audio_data)
                current_time = time.time()

                # Voice activity detection
                if rms > self.config.speech_threshold:
                    # Speech detected
                    if not self._is_speaking:
                        self._is_speaking = True
                        self._speech_start_time = current_time
                        recording_frames = list(self._audio_buffer)
                        logger.debug("Speech started")

                    self._silence_start_time = None
                    recording_frames.append(audio_data)

                elif self._is_speaking:
                    # Currently in speech mode but low volume
                    recording_frames.append(audio_data)

                    if rms < self.config.silence_threshold:
                        if self._silence_start_time is None:
                            self._silence_start_time = current_time
                        elif current_time - self._silence_start_time > self.config.silence_after_speech:
                            # Silence long enough - process the recording
                            speech_duration = current_time - self._speech_start_time
                            if speech_duration >= self.config.min_speech_duration:
                                logger.info(f"Speech ended ({speech_duration:.1f}s)")
                                self._process_recording(recording_frames)
                            else:
                                logger.debug("Speech too short, ignoring")

                            # Reset state
                            self._is_speaking = False
                            self._speech_start_time = None
                            self._silence_start_time = None
                            recording_frames = []
                    else:
                        self._silence_start_time = None

                # Keep a rolling buffer for pre-speech capture
                self._audio_buffer.append(audio_data)

            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                time.sleep(0.1)

    def _process_recording(self, frames: List[bytes]):
        """Save recording and transcribe."""
        if not frames:
            return

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            filepath = f.name

        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self._actual_channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._actual_sample_rate)
            wf.writeframes(b''.join(frames))

        logger.debug(f"Saved audio: {filepath}")

        # Transcribe
        if self._transcription_client:
            try:
                transcription = self._transcribe(filepath)
                if transcription and transcription.strip():
                    logger.info(f"Transcribed: \"{transcription}\"")
                    if self.on_speech_callback:
                        self.on_speech_callback(transcription)
            except Exception as e:
                logger.error(f"Transcription error: {e}")

        # Cleanup
        try:
            os.unlink(filepath)
        except:
            pass

    def _transcribe(self, audio_file: str) -> str:
        """Transcribe audio using OpenAI Whisper."""
        with open(audio_file, 'rb') as f:
            response = self._transcription_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en"
            )
        return response.text


class CloneInterview:
    """
    Integrated clone interview system.

    Combines meeting capture, speech detection, and real-time clone responses.
    """

    def __init__(
        self,
        data_dir: str = "data",
        elevenlabs_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)

        # API keys
        self.elevenlabs_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self.anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Components
        self._clone_system = None
        self._listener = None
        self._meeting_capture = None

        # State
        self._is_running = False
        self._context = ""
        self._response_callback = None

    def setup_from_video(
        self,
        video_path: str,
        subject_id: str,
        clone_voice: bool = True,
        context: str = ""
    ) -> bool:
        """
        Setup clone session from video file.

        Args:
            video_path: Path to video of subject speaking
            subject_id: Unique identifier
            clone_voice: Whether to clone voice via ElevenLabs
            context: Background context for LLM responses

        Returns:
            True if setup successful
        """
        logger.info(f"Setting up clone interview from video: {video_path}")

        from src.realtime_clone import RealtimeCloneSystem

        self._clone_system = RealtimeCloneSystem(
            elevenlabs_api_key=self.elevenlabs_key,
            anthropic_api_key=self.anthropic_key,
            openai_api_key=self.openai_key,
            data_dir=str(self.data_dir)
        )

        # Setup session
        self._clone_system.setup_from_video(
            video_path=video_path,
            subject_id=subject_id,
            clone_voice=clone_voice
        )

        self._context = context
        logger.info("Clone interview setup complete")
        return True

    def setup_from_webcam(
        self,
        subject_id: str,
        duration: int = 10,
        clone_voice: bool = True,
        context: str = ""
    ) -> bool:
        """
        Setup clone session by capturing from webcam.

        Args:
            subject_id: Unique identifier
            duration: Capture duration in seconds
            clone_voice: Whether to clone voice
            context: Background context for LLM responses

        Returns:
            True if setup successful
        """
        logger.info(f"Setting up clone interview from webcam ({duration}s capture)")

        from src.realtime_clone import RealtimeCloneSystem

        self._clone_system = RealtimeCloneSystem(
            elevenlabs_api_key=self.elevenlabs_key,
            anthropic_api_key=self.anthropic_key,
            openai_api_key=self.openai_key,
            data_dir=str(self.data_dir)
        )

        # Setup from webcam
        self._clone_system.setup_from_webcam(
            subject_id=subject_id,
            duration=duration,
            clone_voice=clone_voice
        )

        self._context = context
        logger.info("Clone interview setup complete")
        return True

    def setup_from_existing(
        self,
        subject_id: str,
        library_path: Optional[str] = None,
        context: str = ""
    ) -> bool:
        """
        Setup from an existing viseme library.

        Args:
            subject_id: Subject identifier
            library_path: Path to library (default: data/visemes/{subject_id}_enhanced)
            context: Background context for LLM responses

        Returns:
            True if setup successful
        """
        from src.realtime_clone import RealtimeCloneSystem, CloneSession

        library_path = library_path or str(
            self.data_dir / "visemes" / f"{subject_id}_enhanced"
        )

        if not Path(library_path).exists():
            # Try without _enhanced suffix
            library_path = str(self.data_dir / "visemes" / subject_id)
            if not Path(library_path).exists():
                raise FileNotFoundError(f"No library found for {subject_id}")

        logger.info(f"Loading existing library: {library_path}")

        self._clone_system = RealtimeCloneSystem(
            elevenlabs_api_key=self.elevenlabs_key,
            anthropic_api_key=self.anthropic_key,
            openai_api_key=self.openai_key,
            data_dir=str(self.data_dir)
        )

        # Create session directly
        self._clone_system.session = CloneSession(
            subject_id=subject_id,
            viseme_library_path=library_path,
            use_local_tts=True
        )
        self._clone_system._load_session_resources()

        self._context = context
        logger.info("Clone interview setup complete")
        return True

    def _on_speech_detected(self, text: str):
        """Callback when speech is transcribed."""
        logger.info(f"[Interviewer]: {text}")

        if not self._clone_system or not self._clone_system.session:
            logger.warning("Clone system not ready")
            return

        # Generate response
        logger.info("[Clone thinking...]")
        result = self._clone_system.respond_to_question(
            question=text,
            context=self._context,
            play_immediately=True
        )

        response_text = result.get("response_text", "")
        logger.info(f"[Clone]: {response_text}")

        # Call external callback if set
        if self._response_callback:
            self._response_callback(text, response_text, result)

    def start_listening(
        self,
        device_index: Optional[int] = None,
        on_response: Optional[Callable[[str, str, Dict], None]] = None
    ):
        """
        Start listening for interview questions.

        Args:
            device_index: Audio device index (None = auto-detect loopback)
            on_response: Callback(question, response, result_dict)
        """
        if self._is_running:
            logger.warning("Already listening")
            return

        if not self._clone_system or not self._clone_system.session:
            raise RuntimeError("Clone system not set up. Call setup_from_* first.")

        self._response_callback = on_response

        # Create listener
        self._listener = AudioListener(
            on_speech_callback=self._on_speech_detected
        )

        self._listener.start_listening(device_index)
        self._is_running = True

        logger.info("Clone interview listening started")
        logger.info("Speak to test, or press Ctrl+C to stop")

    def stop_listening(self):
        """Stop listening."""
        if self._listener:
            self._listener.stop_listening()
            self._listener = None

        self._is_running = False
        logger.info("Clone interview stopped")

    def respond_to_text(self, text: str, play: bool = True) -> Dict[str, Any]:
        """
        Manually generate a clone response to text.

        Args:
            text: Question or text to respond to
            play: Whether to play the video immediately

        Returns:
            Response result dict
        """
        if not self._clone_system or not self._clone_system.session:
            raise RuntimeError("Clone system not set up")

        return self._clone_system.respond_to_question(
            question=text,
            context=self._context,
            play_immediately=play
        )

    def say(self, text: str, play: bool = True) -> Dict[str, Any]:
        """
        Make the clone speak specific text.

        Args:
            text: Text for the clone to speak
            play: Whether to play the video immediately

        Returns:
            Response result dict
        """
        if not self._clone_system or not self._clone_system.session:
            raise RuntimeError("Clone system not set up")

        return self._clone_system.respond(
            text=text,
            play_immediately=play
        )


def demo():
    """Demo the clone interview system."""
    print("=" * 60)
    print("Clone Interview System Demo")
    print("=" * 60)

    # Check for existing library
    existing = Path("data/visemes/user_lit_enhanced")
    if not existing.exists():
        existing = Path("data/visemes/user_lit")

    if not existing.exists():
        print("\nNo existing library found.")
        print("Run: python -m src.realtime_clone")
        print("Or create one with setup_from_webcam()")
        return

    print(f"\nUsing library: {existing}")

    # Create interview system
    interview = CloneInterview()
    interview.setup_from_existing("user_lit")

    # Test with manual input
    print("\n--- Testing manual response ---")
    result = interview.say("Hello! I'm ready to answer your questions.")
    print(f"Response generated in {result['timings']['total']:.2f}s")

    print("\n--- Testing question response ---")
    result = interview.respond_to_text("Tell me about yourself.")
    print(f"Response: {result.get('response_text', 'N/A')}")
    print(f"Total time: {result['timings']['total']:.2f}s")

    # Start listening mode
    print("\n" + "=" * 60)
    print("Starting listening mode...")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    try:
        interview.start_listening()

        # Keep running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        interview.stop_listening()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
