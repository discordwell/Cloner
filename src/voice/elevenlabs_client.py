"""ElevenLabs API client for voice cloning and TTS generation."""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from elevenlabs import ElevenLabs
from elevenlabs.types import VoiceSettings
import logging

logger = logging.getLogger(__name__)


class ElevenLabsClient:
    """
    Client for interacting with ElevenLabs API.

    Handles voice cloning, text-to-speech generation, and voice management.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ElevenLabs client.

        Args:
            api_key: ElevenLabs API key. If None, reads from ELEVENLABS_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")

        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key not provided. "
                "Set ELEVENLABS_API_KEY environment variable or pass api_key parameter."
            )

        self.client = ElevenLabs(api_key=self.api_key)
        logger.info("ElevenLabs client initialized successfully")

    def list_voices(self) -> List[Dict[str, Any]]:
        """
        List all available voices.

        Returns:
            List of voice dictionaries with name, voice_id, and other metadata
        """
        try:
            voices = self.client.voices.get_all()
            voice_list = []

            for voice in voices.voices:
                voice_list.append({
                    "name": voice.name,
                    "voice_id": voice.voice_id,
                    "category": voice.category if hasattr(voice, 'category') else None,
                    "description": voice.description if hasattr(voice, 'description') else None,
                })

            logger.info(f"Retrieved {len(voice_list)} voices")
            return voice_list

        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            raise

    def clone_voice(
        self,
        name: str,
        audio_files: List[str],
        description: Optional[str] = None
    ) -> str:
        """
        Clone a voice from audio samples using instant voice cloning (IVC).

        Args:
            name: Name for the cloned voice
            audio_files: List of paths to audio files (ideally 1-5 minutes total)
            description: Optional description of the voice

        Returns:
            voice_id: ID of the cloned voice
        """
        try:
            for audio_file in audio_files:
                if not Path(audio_file).exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_file}")

            logger.info(f"Cloning voice '{name}' from {len(audio_files)} audio files")

            files = []
            for audio_path in audio_files:
                files.append(open(audio_path, 'rb'))

            try:
                result = self.client.voices.ivc.create(
                    name=name,
                    files=files,
                    description=description or f"Cloned voice: {name}",
                )

                voice_id = result.voice_id
                logger.info(f"Voice cloned successfully: {voice_id}")
                return voice_id

            finally:
                for f in files:
                    f.close()

        except Exception as e:
            logger.error(f"Failed to clone voice: {e}")
            raise

    def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_path: str,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
        output_format: str = "mp3_44100_128",
    ) -> str:
        """
        Generate speech from text using a specific voice.

        Args:
            text: Text to convert to speech
            voice_id: ID of the voice to use
            output_path: Path to save the generated audio
            stability: Voice stability (0.0 to 1.0)
            similarity_boost: Clarity + similarity (0.0 to 1.0)
            style: Style exaggeration (0.0 to 1.0)
            use_speaker_boost: Enable speaker boost for better quality
            output_format: Audio format (default mp3_44100_128)

        Returns:
            Path to the generated audio file
        """
        try:
            logger.info(f"Generating speech for {len(text)} characters using voice {voice_id}")

            audio = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                output_format=output_format,
                voice_settings=VoiceSettings(
                    stability=stability,
                    similarity_boost=similarity_boost,
                    style=style,
                    use_speaker_boost=use_speaker_boost,
                ),
            )

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                for chunk in audio:
                    f.write(chunk)

            logger.info(f"Speech generated successfully: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            raise

    def generate_speech_with_voice_name(
        self,
        text: str,
        voice_name: str,
        output_path: str,
        **kwargs
    ) -> str:
        """
        Generate speech using voice name instead of ID.

        Args:
            text: Text to convert to speech
            voice_name: Name of the voice to use
            output_path: Path to save the generated audio
            **kwargs: Additional arguments passed to generate_speech()

        Returns:
            Path to the generated audio file
        """
        voices = self.list_voices()
        voice_id = None

        for voice in voices:
            if voice['name'].lower() == voice_name.lower():
                voice_id = voice['voice_id']
                break

        if not voice_id:
            raise ValueError(f"Voice not found: {voice_name}")

        return self.generate_speech(text, voice_id, output_path, **kwargs)

    def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a cloned voice.

        Args:
            voice_id: ID of the voice to delete

        Returns:
            True if successful
        """
        try:
            self.client.voices.delete(voice_id)
            logger.info(f"Voice deleted: {voice_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete voice: {e}")
            raise

    def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """
        Get information about a specific voice.

        Args:
            voice_id: ID of the voice

        Returns:
            Dictionary with voice information
        """
        try:
            voice = self.client.voices.get(voice_id)

            return {
                "name": voice.name,
                "voice_id": voice.voice_id,
                "category": voice.category if hasattr(voice, 'category') else None,
                "description": voice.description if hasattr(voice, 'description') else None,
                "labels": voice.labels if hasattr(voice, 'labels') else None,
            }

        except Exception as e:
            logger.error(f"Failed to get voice info: {e}")
            raise
