"""
Streaming TTS worker for the pipeline.

Pulls SentenceChunk objects from a queue, generates audio via ElevenLabs
streaming TTS, and pushes AudioChunk objects downstream.
"""

import os
import queue
import logging
import threading
from typing import Optional

from elevenlabs import ElevenLabs
from elevenlabs.types import VoiceSettings

from src.streaming.types import SentenceChunk, AudioChunk

logger = logging.getLogger(__name__)


class TTSStreamer:
    """
    Streaming TTS worker. Converts sentences to audio bytes.

    Pulls SentenceChunk from input queue, calls ElevenLabs streaming TTS,
    pushes AudioChunk with complete audio bytes into output queue.

    Usage:
        streamer = TTSStreamer(voice_id="nf18MnSL81anCHgQgL1A")
        # Run in a thread:
        streamer.run(input_queue, output_queue, stop_event)
    """

    def __init__(
        self,
        voice_id: str,
        api_key: Optional[str] = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        output_format: str = "mp3_44100_128",
    ):
        """
        Args:
            voice_id: ElevenLabs voice ID.
            api_key: API key. Falls back to ELEVENLABS_API_KEY env var.
            stability: Voice stability (0-1).
            similarity_boost: Clarity + similarity (0-1).
            output_format: Audio output format.
        """
        self.voice_id = voice_id
        self.output_format = output_format
        self._api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self._api_key:
            raise ValueError(
                "ElevenLabs API key not provided. "
                "Set ELEVENLABS_API_KEY or pass api_key parameter."
            )
        self._client = ElevenLabs(api_key=self._api_key)
        self._voice_settings = VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost,
        )

    def generate_audio(self, text: str) -> bytes:
        """
        Generate complete audio bytes for a text string.

        Returns:
            Audio bytes (MP3).
        """
        audio_iter = self._client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            output_format=self.output_format,
            voice_settings=self._voice_settings,
        )

        chunks = []
        for chunk in audio_iter:
            chunks.append(chunk)
        return b"".join(chunks)

    def run(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        """
        Worker loop: pull SentenceChunks, generate audio, push AudioChunks.

        Blocks until is_final chunk is received or stop_event is set.
        Run this in a worker thread.
        """
        logger.info("[TTS] Worker started")

        while not stop_event.is_set():
            try:
                chunk: SentenceChunk = input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk.is_final:
                # Forward the completion signal
                output_queue.put(AudioChunk(
                    seq=chunk.seq, audio_bytes=b"",
                    sentence_text="", is_final=True,
                ))
                logger.info("[TTS] Worker finished (final chunk)")
                break

            if not chunk.text.strip():
                continue

            try:
                logger.info(f"[TTS] Generating audio for sentence {chunk.seq}: {chunk.text[:50]}...")
                audio_bytes = self.generate_audio(chunk.text)

                output_queue.put(AudioChunk(
                    seq=chunk.seq,
                    audio_bytes=audio_bytes,
                    sentence_text=chunk.text,
                    is_final=False,
                ))
                logger.info(
                    f"[TTS] Sentence {chunk.seq} done: {len(audio_bytes)} bytes"
                )

            except Exception as e:
                logger.error(f"[TTS] Error on sentence {chunk.seq}: {e}")
                # Skip this sentence but continue processing

        logger.info("[TTS] Worker stopped")
