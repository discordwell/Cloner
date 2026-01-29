"""
Local TTS using macOS 'say' command.

No API keys required - uses system speech synthesis.
"""

import subprocess
import time
import wave
import os
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .phoneme_mapper import PhonemeToVisemeMapper
from .tts_viseme import VisemeEvent, TTSResult


@dataclass
class LocalTTSConfig:
    """Configuration for local TTS."""
    voice: str = "Samantha"  # Default macOS voice
    rate: int = 175  # Words per minute (default ~175-200)


class LocalTTS:
    """
    Local text-to-speech using macOS 'say' command.

    Generates audio and estimates viseme timing based on text.
    """

    # Average speech rates for timing estimation
    CHARS_PER_SECOND = 15  # ~150 WPM, ~5 chars/word

    def __init__(self, config: Optional[LocalTTSConfig] = None):
        """Initialize local TTS."""
        self.config = config or LocalTTSConfig()
        self.mapper = PhonemeToVisemeMapper()

        # Verify 'say' command exists
        result = subprocess.run(['which', 'say'], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("macOS 'say' command not found")

    def list_voices(self) -> List[str]:
        """List available voices."""
        result = subprocess.run(
            ['say', '-v', '?'],
            capture_output=True,
            text=True
        )
        voices = []
        for line in result.stdout.strip().split('\n'):
            if line:
                voice_name = line.split()[0]
                voices.append(voice_name)
        return voices

    def generate_with_visemes(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        rate: Optional[int] = None
    ) -> TTSResult:
        """
        Generate speech with viseme timing data.

        Args:
            text: Text to synthesize
            output_path: Path for output audio (will be .aiff, converted to .wav)
            voice: Voice name (default: Samantha)
            rate: Speech rate in WPM (default: 175)

        Returns:
            TTSResult with audio path and viseme events
        """
        voice = voice or self.config.voice
        rate = rate or self.config.rate

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate AIFF first (say's native format)
        aiff_path = output_path.with_suffix('.aiff')

        cmd = [
            'say',
            '-v', voice,
            '-r', str(rate),
            '-o', str(aiff_path),
            text
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True)
        generation_time = time.time() - start_time

        if result.returncode != 0:
            raise RuntimeError(f"TTS failed: {result.stderr.decode()}")

        # Convert to WAV for broader compatibility
        wav_path = output_path.with_suffix('.wav')
        self._convert_aiff_to_wav(aiff_path, wav_path)

        # Get actual duration from audio file
        duration_ms = self._get_audio_duration_ms(wav_path)

        # Generate viseme timing
        viseme_events = self._generate_viseme_timing(text, duration_ms)

        # Clean up AIFF
        aiff_path.unlink(missing_ok=True)

        return TTSResult(
            audio_path=str(wav_path),
            audio_duration_ms=duration_ms,
            viseme_events=viseme_events,
            text=text
        )

    def _convert_aiff_to_wav(self, aiff_path: Path, wav_path: Path):
        """Convert AIFF to WAV using ffmpeg or afconvert."""
        # Try ffmpeg first
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', str(aiff_path), str(wav_path)],
            capture_output=True
        )

        if result.returncode != 0:
            # Fall back to macOS afconvert
            result = subprocess.run(
                ['afconvert', '-f', 'WAVE', '-d', 'LEI16',
                 str(aiff_path), str(wav_path)],
                capture_output=True
            )

            if result.returncode != 0:
                raise RuntimeError("Could not convert audio format")

    def _get_audio_duration_ms(self, wav_path: Path) -> int:
        """Get duration of WAV file in milliseconds."""
        try:
            with wave.open(str(wav_path), 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration_s = frames / float(rate)
                return int(duration_s * 1000)
        except Exception:
            # Estimate from text if wave parsing fails
            return 2000  # Default 2 seconds

    def _generate_viseme_timing(
        self,
        text: str,
        duration_ms: int
    ) -> List[VisemeEvent]:
        """Generate viseme events with timing."""
        # Use the phoneme mapper to get viseme sequence
        duration_s = duration_ms / 1000.0
        frames = self.mapper.text_to_visemes(text, duration_s)

        # Convert to VisemeEvent format
        events = []
        for frame in frames:
            events.append(VisemeEvent(
                viseme_id=frame.viseme_id,
                start_time_ms=int(frame.start_time * 1000),
                end_time_ms=int(frame.end_time * 1000),
                character="",
                intensity=frame.intensity
            ))

        return events

    def quick_test(self, text: str = "Hello, this is a test."):
        """Quick test of local TTS."""
        print(f"Generating speech: '{text}'")

        result = self.generate_with_visemes(
            text=text,
            output_path="/tmp/local_tts_test.wav"
        )

        print(f"Audio: {result.audio_path}")
        print(f"Duration: {result.audio_duration_ms}ms")
        print(f"Visemes: {len(result.viseme_events)}")

        # Play the audio
        subprocess.run(['afplay', result.audio_path])

        return result


def demo():
    """Demo local TTS."""
    print("Local TTS Demo (macOS)")
    print("=" * 40)

    tts = LocalTTS()

    print("\nAvailable voices:")
    voices = tts.list_voices()[:10]
    for v in voices:
        print(f"  - {v}")

    print("\nGenerating speech...")
    result = tts.quick_test("Hello! This is a test of local text to speech.")

    print("\nFirst 10 viseme events:")
    for e in result.viseme_events[:10]:
        print(f"  {e.start_time_ms:4d}ms: Viseme {e.viseme_id}")


if __name__ == "__main__":
    demo()
