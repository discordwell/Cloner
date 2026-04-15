"""
Streaming pipeline orchestrator.

Wires LLM streaming -> sentence splitting -> TTS -> Atlas Realtime -> playback
into a parallel pipeline with minimal latency.
"""

import os
import queue
import logging
import threading
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field

from src.streaming.types import SentenceChunk, AudioChunk
from src.streaming.llm_streamer import LLMStreamer
from src.streaming.tts_streamer import TTSStreamer
from src.streaming.atlas_realtime import AtlasRealtimeSession

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the streaming pipeline."""
    # LLM
    llm_provider: str = "openai"
    llm_model: Optional[str] = None
    llm_max_tokens: int = 300
    llm_temperature: float = 0.8
    system_prompt: str = (
        "You are being interviewed. Answer concisely and naturally, "
        "as if speaking aloud. Keep responses to 2-4 sentences."
    )

    # TTS
    voice_id: str = ""
    tts_stability: float = 0.5
    tts_similarity_boost: float = 0.75

    # Atlas
    face_image_path: str = ""


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    question: str = ""
    full_response: str = ""
    sentences: list = field(default_factory=list)
    video_frames: int = 0
    time_to_first_sentence: float = 0.0
    time_to_first_audio: float = 0.0
    time_to_first_frame: float = 0.0
    total_time: float = 0.0


class StreamingPipeline:
    """
    Orchestrates the full streaming pipeline:
    LLM -> sentence split -> TTS -> Atlas Realtime -> video frames.

    Usage:
        pipeline = StreamingPipeline(config)
        pipeline.start_session()

        result = pipeline.respond("What do you think about AI?")

        pipeline.end_session()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        self._llm = LLMStreamer(
            provider=config.llm_provider,
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
        )
        self._tts = TTSStreamer(
            voice_id=config.voice_id,
            stability=config.tts_stability,
            similarity_boost=config.tts_similarity_boost,
        )
        self._atlas_session: Optional[AtlasRealtimeSession] = None

        # Inter-stage queues
        self._tts_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()

        # Control
        self._stop_event = threading.Event()
        self._workers: list = []

        # Callbacks
        self._on_video_frame: Optional[Callable] = None
        self._on_first_frame: Optional[Callable] = None
        self._on_sentence: Optional[Callable] = None
        self._on_complete: Optional[Callable] = None

        # Metrics
        self._result: Optional[PipelineResult] = None
        self._start_time: float = 0

    @property
    def is_session_active(self) -> bool:
        return self._atlas_session is not None and self._atlas_session.is_connected

    def start_session(self, face_image_path: Optional[str] = None) -> str:
        """
        Create an Atlas Realtime session. Call once before respond().

        Args:
            face_image_path: Override face image (or uses config default).

        Returns:
            session_id
        """
        face = face_image_path or self.config.face_image_path
        if not face:
            raise ValueError("No face_image_path provided")

        self._first_frame_fired = False

        self._atlas_session = AtlasRealtimeSession(
            on_video_frame=self._handle_video_frame,
            on_disconnected=self._handle_disconnect,
        )
        return self._atlas_session.connect(face)

    def respond(
        self,
        question: str,
        context: str = "",
        on_video_frame: Optional[Callable] = None,
        on_first_frame: Optional[Callable] = None,
        on_sentence: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[PipelineResult], None]] = None,
    ) -> PipelineResult:
        """
        Generate a streaming response. Blocks until complete.

        Starts LLM streaming, sentence splitting, TTS, and audio publishing
        in parallel threads. Video frames arrive via callbacks.

        Args:
            question: The question to answer.
            context: Additional context for the LLM.
            on_video_frame: Called for each video frame.
            on_first_frame: Called when the first video frame arrives.
            on_sentence: Called when each sentence is generated.
            on_complete: Called when the full response is done.

        Returns:
            PipelineResult with timings and metrics.
        """
        if not self.is_session_active:
            raise RuntimeError("No active session. Call start_session() first.")

        self._on_video_frame = on_video_frame
        self._on_first_frame = on_first_frame
        self._on_sentence = on_sentence
        self._on_complete = on_complete
        self._first_frame_fired = False
        self._stop_event.clear()

        # Clear queues
        self._drain_queue(self._tts_queue)
        self._drain_queue(self._audio_queue)

        # Initialize result tracking
        self._start_time = time.time()
        self._result = PipelineResult(question=question)

        # Build messages
        prompt = self.config.system_prompt
        if context:
            prompt += f"\n\nContext: {context}"

        messages = [{"role": "user", "content": question}]

        # Start TTS worker thread
        tts_thread = threading.Thread(
            target=self._tts.run,
            args=(self._tts_queue, self._audio_queue, self._stop_event),
            daemon=True,
        )
        tts_thread.start()

        # Start audio publisher thread
        pub_thread = threading.Thread(
            target=self._audio_publisher_worker,
            daemon=True,
        )
        pub_thread.start()

        # Run LLM streamer in current thread (blocks until done)
        def on_llm_sentence(text):
            elapsed = time.time() - self._start_time
            if not self._result.sentences:
                self._result.time_to_first_sentence = elapsed
            self._result.sentences.append(text)
            if self._on_sentence:
                self._on_sentence(text)

        full_text = self._llm.stream_response(
            messages=messages,
            system_prompt=prompt,
            output_queue=self._tts_queue,
            stop_event=self._stop_event,
            on_sentence=on_llm_sentence,
        )

        self._result.full_response = full_text

        # Wait for TTS and audio publisher to finish
        tts_thread.join(timeout=60)
        pub_thread.join(timeout=60)

        self._result.total_time = time.time() - self._start_time

        if self._on_complete:
            self._on_complete(self._result)

        logger.info(
            f"[PIPELINE] Done: {len(self._result.sentences)} sentences, "
            f"{self._result.video_frames} frames, "
            f"first_frame={self._result.time_to_first_frame:.2f}s, "
            f"total={self._result.total_time:.2f}s"
        )

        return self._result

    def interrupt(self):
        """Cancel the current response mid-stream."""
        logger.info("[PIPELINE] Interrupting...")
        self._stop_event.set()
        self._drain_queue(self._tts_queue)
        self._drain_queue(self._audio_queue)

    def end_session(self):
        """Tear down the Atlas Realtime session."""
        self.interrupt()
        if self._atlas_session:
            self._atlas_session.disconnect()
            self._atlas_session = None

    # ── Worker threads ──────────────────────────────────────────────

    def _audio_publisher_worker(self):
        """Pull AudioChunks from queue and publish to Atlas Realtime."""
        logger.info("[PIPELINE] Audio publisher started")

        while not self._stop_event.is_set():
            try:
                chunk: AudioChunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk.is_final:
                logger.info("[PIPELINE] Audio publisher: final chunk")
                break

            if not chunk.audio_bytes:
                continue

            try:
                elapsed = time.time() - self._start_time
                if self._result and not self._result.time_to_first_audio:
                    self._result.time_to_first_audio = elapsed
                    logger.info(f"[PIPELINE] First audio at {elapsed:.2f}s")

                # Decode MP3 to PCM and publish
                pcm = AtlasRealtimeSession._decode_to_pcm_from_bytes(
                    chunk.audio_bytes
                )
                self._atlas_session.publish_audio(pcm)

            except Exception as e:
                logger.error(f"[PIPELINE] Audio publish error: {e}")

        logger.info("[PIPELINE] Audio publisher stopped")

    # ── Callbacks ───────────────────────────────────────────────────

    def _handle_video_frame(self, event):
        """Called from LiveKit thread when a video frame arrives."""
        if self._result:
            self._result.video_frames += 1

            if not self._first_frame_fired:
                self._first_frame_fired = True
                elapsed = time.time() - self._start_time
                if self._result:
                    self._result.time_to_first_frame = elapsed
                logger.info(f"[PIPELINE] First video frame at {elapsed:.2f}s")
                if self._on_first_frame:
                    self._on_first_frame(event)

        if self._on_video_frame:
            self._on_video_frame(event)

    def _handle_disconnect(self, reason):
        """Called when Atlas session disconnects."""
        logger.warning(f"[PIPELINE] Atlas disconnected: {reason}")

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _drain_queue(q: queue.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
