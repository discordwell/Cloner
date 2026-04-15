"""
Atlas Realtime session manager via LiveKit WebRTC.

Creates a persistent WebRTC session with Atlas, publishes TTS audio,
and receives lip-synced video frames with sub-second latency.
"""

import os
import asyncio
import logging
import threading
import struct
import io
from typing import Optional, Callable

from livekit import rtc

from src.video.atlas_client import AtlasClient

logger = logging.getLogger(__name__)


class AtlasRealtimeSession:
    """
    Manages a persistent Atlas Realtime WebRTC session.

    Lifecycle:
        1. connect(face_image_path) -> creates Atlas session, connects LiveKit
        2. publish_audio(pcm_bytes) -> pushes audio into the WebRTC track
        3. Video frames arrive via on_video_frame callback
        4. disconnect() -> tears down everything

    The LiveKit event loop runs in a background thread. All public methods
    are thread-safe and can be called from any thread.
    """

    SAMPLE_RATE = 48000
    NUM_CHANNELS = 1
    SAMPLES_PER_FRAME = 480  # 10ms at 48kHz

    def __init__(
        self,
        atlas_api_key: Optional[str] = None,
        on_video_frame: Optional[Callable] = None,
        on_connected: Optional[Callable] = None,
        on_disconnected: Optional[Callable] = None,
    ):
        """
        Args:
            atlas_api_key: Atlas API key. Falls back to ATLAS_API_KEY env var.
            on_video_frame: Callback(VideoFrameEvent) when video frame arrives.
            on_connected: Callback() when session is connected.
            on_disconnected: Callback(reason) when session disconnects.
        """
        self._atlas = AtlasClient(api_key=atlas_api_key)
        self._on_video_frame = on_video_frame
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected

        self._session_id: Optional[str] = None
        self._room: Optional[rtc.Room] = None
        self._audio_source: Optional[rtc.AudioSource] = None
        self._audio_track: Optional[rtc.LocalAudioTrack] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._connected = threading.Event()

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def connect(self, face_image_path: str) -> str:
        """
        Create an Atlas Realtime session and connect to LiveKit.

        Args:
            face_image_path: Path to face image for the avatar.

        Returns:
            session_id
        """
        logger.info("[ATLAS-RT] Creating realtime session...")

        # Create Atlas session via HTTP
        session = self._atlas._request(
            "POST", "/v1/realtime/session",
            files={"face": (
                os.path.basename(face_image_path),
                open(face_image_path, "rb"),
                "image/jpeg",
            )},
            data={"mode": "passthrough"},
        )

        self._session_id = session["session_id"]
        livekit_url = session["livekit_url"]
        token = session["token"]

        logger.info(f"[ATLAS-RT] Session created: {self._session_id}")
        logger.info(f"[ATLAS-RT] LiveKit URL: {livekit_url}")

        # Start asyncio event loop in background thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            args=(livekit_url, token),
            daemon=True,
        )
        self._loop_thread.start()

        # Wait for connection
        if not self._connected.wait(timeout=15):
            raise TimeoutError("[ATLAS-RT] Failed to connect to LiveKit within 15s")

        logger.info("[ATLAS-RT] Connected and publishing audio track")
        return self._session_id

    def publish_audio(self, pcm_bytes: bytes):
        """
        Publish raw PCM audio bytes to the LiveKit audio track.

        Args:
            pcm_bytes: Raw 16-bit PCM audio at 48kHz mono.
                       Length must be a multiple of 2 (16-bit samples).
        """
        if not self.is_connected or not self._audio_source:
            logger.warning("[ATLAS-RT] Cannot publish audio: not connected")
            return

        asyncio.run_coroutine_threadsafe(
            self._capture_audio(pcm_bytes), self._loop
        )

    def publish_audio_file(self, audio_path: str):
        """
        Publish an audio file (MP3/WAV) to the LiveKit track.

        Decodes to PCM and publishes in 10ms frames. Blocks until complete.

        Args:
            audio_path: Path to audio file.
        """
        pcm_bytes = self._decode_to_pcm(audio_path)
        self.publish_audio(pcm_bytes)

    def swap_face(self, new_image_path: str):
        """Hot-swap the avatar face mid-session."""
        if not self._session_id:
            raise RuntimeError("No active session")

        logger.info(f"[ATLAS-RT] Swapping face: {new_image_path}")
        self._atlas._request(
            "PATCH", f"/v1/realtime/session/{self._session_id}",
            files={"face": (
                os.path.basename(new_image_path),
                open(new_image_path, "rb"),
                "image/jpeg",
            )},
        )

    def get_session_status(self) -> dict:
        """Get current session status from Atlas."""
        if not self._session_id:
            return {"status": "no_session"}
        return self._atlas._request(
            "GET", f"/v1/realtime/session/{self._session_id}"
        )

    def disconnect(self):
        """Tear down the session and LiveKit connection."""
        logger.info("[ATLAS-RT] Disconnecting...")
        self._connected.clear()

        # Disconnect LiveKit
        if self._loop and self._room:
            future = asyncio.run_coroutine_threadsafe(
                self._room.disconnect(), self._loop
            )
            try:
                future.result(timeout=5)
            except Exception as e:
                logger.warning(f"[ATLAS-RT] Error disconnecting room: {e}")

        # Cancel pending tasks and stop event loop
        if self._loop:
            for task in asyncio.all_tasks(self._loop):
                task.cancel()
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread:
            self._loop_thread.join(timeout=5)
        if self._loop and not self._loop.is_closed():
            self._loop.close()

        # End Atlas session
        if self._session_id:
            try:
                result = self._atlas._request(
                    "DELETE", f"/v1/realtime/session/{self._session_id}"
                )
                cost = result.get("estimated_cost", "unknown")
                duration = result.get("duration_seconds", 0)
                logger.info(
                    f"[ATLAS-RT] Session ended: {duration:.1f}s, cost: {cost}"
                )
            except Exception as e:
                logger.warning(f"[ATLAS-RT] Error ending session: {e}")

        self._session_id = None
        self._room = None
        self._audio_source = None
        self._audio_track = None
        logger.info("[ATLAS-RT] Disconnected")

    # ── Internal async methods ──────────────────────────────────────

    def _run_loop(self, livekit_url: str, token: str):
        """Run the asyncio event loop (in background thread)."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_run(livekit_url, token))

    async def _connect_and_run(self, livekit_url: str, token: str):
        """Connect to LiveKit and set up tracks."""
        self._room = rtc.Room()

        # Register event handlers
        self._room.on("track_subscribed", self._on_track_subscribed)
        self._room.on("disconnected", self._on_room_disconnected)
        self._room.on("reconnected", self._on_room_reconnected)

        # Connect to room
        await self._room.connect(livekit_url, token)
        logger.info(f"[ATLAS-RT] Connected to room: {self._room.name}")

        # Create and publish audio source
        self._audio_source = rtc.AudioSource(
            sample_rate=self.SAMPLE_RATE,
            num_channels=self.NUM_CHANNELS,
        )
        self._audio_track = rtc.LocalAudioTrack.create_audio_track(
            "tts-audio", self._audio_source
        )
        await self._room.local_participant.publish_track(self._audio_track)

        self._connected.set()
        if self._on_connected:
            self._on_connected()

        # Keep the loop running until disconnected
        try:
            while self._connected.is_set():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    async def _capture_audio(self, pcm_bytes: bytes):
        """Capture PCM audio into the LiveKit audio source in 10ms frames."""
        if not self._audio_source:
            return

        # Split into 10ms frames (480 samples at 48kHz, 2 bytes per sample)
        frame_size = self.SAMPLES_PER_FRAME * 2  # 16-bit = 2 bytes/sample
        offset = 0

        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset:offset + frame_size]

            # Pad last frame if needed
            if len(chunk) < frame_size:
                chunk = chunk + b'\x00' * (frame_size - len(chunk))

            frame = rtc.AudioFrame.create(
                self.SAMPLE_RATE, self.NUM_CHANNELS, self.SAMPLES_PER_FRAME
            )
            # Copy PCM data into the frame
            frame.data[:len(chunk)] = chunk

            await self._audio_source.capture_frame(frame)
            offset += frame_size

    # ── Event handlers ──────────────────────────────────────────────

    def _on_track_subscribed(self, track, publication, participant):
        """Called when we subscribe to a track (Atlas's video output)."""
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.info(f"[ATLAS-RT] Video track subscribed: {track.name}")
            if self._on_video_frame:
                # Set up a video stream to receive frames
                asyncio.ensure_future(
                    self._video_frame_loop(track),
                    loop=self._loop,
                )
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"[ATLAS-RT] Audio track subscribed: {track.name}")

    async def _video_frame_loop(self, track):
        """Receive video frames from Atlas and forward to callback."""
        video_stream = rtc.VideoStream(track)
        async for event in video_stream:
            if self._on_video_frame:
                self._on_video_frame(event)

    def _on_room_disconnected(self, reason):
        """Called when disconnected from the room."""
        logger.warning(f"[ATLAS-RT] Disconnected: {reason}")
        if self._on_disconnected:
            self._on_disconnected(reason)

    def _on_room_reconnected(self):
        """Called when reconnected to the room."""
        logger.info("[ATLAS-RT] Reconnected")

    # ── Audio decoding ──────────────────────────────────────────────

    @staticmethod
    def _decode_to_pcm(audio_path: str) -> bytes:
        """Decode an audio file to raw 48kHz 16-bit mono PCM."""
        import subprocess

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", "48000",
                "-ac", "1",
                "-",
            ],
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg decode failed: {result.stderr.decode()[:200]}"
            )

        return result.stdout

    @staticmethod
    def _decode_to_pcm_from_bytes(audio_bytes: bytes) -> bytes:
        """Decode in-memory audio bytes (MP3) to raw 48kHz 16-bit mono PCM."""
        import subprocess

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", "pipe:0",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", "48000",
                "-ac", "1",
                "pipe:1",
            ],
            input=audio_bytes,
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg decode failed: {result.stderr.decode()[:200]}"
            )

        return result.stdout
