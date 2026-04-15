"""
Video playback module for the streaming pipeline.

Accumulates video frames from Atlas Realtime into an MP4 file,
then plays through OBS by swapping the IdleVideo source.
"""

import os
import sys
import time
import logging
import tempfile
import subprocess
import threading
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class OBSVideoPlayback:
    """
    Accumulates video frames into an MP4, then plays through OBS.

    Usage:
        playback = OBSVideoPlayback(obs_host, obs_port, obs_password)
        playback.start_recording()        # Begin accumulating frames
        playback.push_frame(frame_event)   # Called from Atlas RT callback
        playback.stop_and_play()           # Finalize MP4 and play through OBS
    """

    def __init__(
        self,
        obs_host: str = "localhost",
        obs_port: int = 4455,
        obs_password: str = "slopifywins",
        idle_video_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        source_name: str = "IdleVideo",
    ):
        self._obs_host = obs_host
        self._obs_port = obs_port
        self._obs_password = obs_password
        self._source_name = source_name
        self._idle_video_path = idle_video_path
        self._output_dir = output_dir or tempfile.gettempdir()

        self._frames: list = []
        self._recording = False
        self._lock = threading.Lock()
        self._current_video: Optional[str] = None

    def start_recording(self):
        """Begin accumulating video frames."""
        with self._lock:
            self._frames = []
            self._recording = True
        logger.info("[PLAYBACK] Recording started")

    def push_frame(self, frame_event):
        """
        Push a video frame from Atlas Realtime.

        Args:
            frame_event: LiveKit VideoFrameEvent from the track subscription.
        """
        if not self._recording:
            return
        with self._lock:
            self._frames.append(frame_event)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def stop_and_play(self, audio_path: Optional[str] = None) -> Optional[str]:
        """
        Stop recording, write frames to MP4, and play through OBS.

        Args:
            audio_path: Optional audio file to mux into the video.

        Returns:
            Path to the generated video, or None if no frames.
        """
        with self._lock:
            self._recording = False
            frames = list(self._frames)
            self._frames = []

        if not frames:
            logger.warning("[PLAYBACK] No frames to play")
            return None

        # Write frames to MP4
        video_path = self._write_frames_to_mp4(frames, audio_path)
        if not video_path:
            return None

        self._current_video = video_path

        # Play through OBS
        duration = self._play_through_obs(video_path)

        # Wait for playback to finish
        if duration > 0:
            time.sleep(duration + 0.5)

        # Restore idle loop
        self._restore_idle()

        return video_path

    def stop_recording(self) -> list:
        """Stop recording and return accumulated frames without playing."""
        with self._lock:
            self._recording = False
            frames = list(self._frames)
            self._frames = []
        return frames

    def _write_frames_to_mp4(
        self, frames: list, audio_path: Optional[str] = None
    ) -> Optional[str]:
        """Write accumulated ARGB frames to an MP4 file via ffmpeg."""
        if not frames:
            return None

        # Extract frame dimensions from first frame
        first = frames[0]
        buffer = first.frame
        width = buffer.width
        height = buffer.height

        video_path = os.path.join(
            self._output_dir, f"stream_{int(time.time())}.mp4"
        )

        logger.info(
            f"[PLAYBACK] Writing {len(frames)} frames ({width}x{height}) to {video_path}"
        )

        # Estimate FPS from frame count and timing
        fps = 25  # Atlas default

        # Build ffmpeg command for raw ARGB input
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "argb",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "pipe:0",
        ]

        if audio_path and os.path.exists(audio_path):
            cmd.extend(["-i", audio_path, "-shortest"])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
        ])

        if audio_path and os.path.exists(audio_path):
            cmd.extend(["-c:a", "aac"])

        cmd.append(video_path)

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Write each frame's raw data
            for frame_event in frames:
                buf = frame_event.frame
                # Convert ARGB to bytes
                argb_data = bytes(buf.data)
                proc.stdin.write(argb_data)

            proc.stdin.close()
            proc.wait(timeout=30)

            if proc.returncode != 0:
                stderr = proc.stderr.read().decode()[:300]
                logger.error(f"[PLAYBACK] ffmpeg error: {stderr}")
                return None

            size = os.path.getsize(video_path)
            duration = len(frames) / fps
            logger.info(
                f"[PLAYBACK] Wrote {size // 1024}KB, {duration:.1f}s video"
            )
            return video_path

        except Exception as e:
            logger.error(f"[PLAYBACK] Failed to write MP4: {e}")
            return None

    def _play_through_obs(self, video_path: str) -> float:
        """Swap OBS IdleVideo source to the generated video. Returns duration."""
        try:
            import obsws_python as obs

            cl = obs.ReqClient(
                host=self._obs_host,
                port=self._obs_port,
                password=self._obs_password,
            )

            # Convert path for OBS
            if sys.platform == "win32" or "/mnt/c/" in video_path:
                obs_path = video_path.replace("/mnt/c/", "C:\\").replace("/", "\\")
            else:
                obs_path = video_path

            cl.set_input_settings(
                name=self._source_name,
                settings={
                    "local_file": obs_path,
                    "looping": False,
                    "restart_on_activate": True,
                    "clear_on_media_end": False,
                },
                overlay=True,
            )
            cl.trigger_media_input_action(
                self._source_name,
                "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART",
            )

            # Estimate duration from file
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", video_path,
                ],
                capture_output=True, text=True, timeout=10,
            )
            import json
            duration = float(json.loads(probe.stdout)["format"]["duration"])
            logger.info(f"[PLAYBACK] Playing {duration:.1f}s through OBS")
            return duration

        except Exception as e:
            logger.error(f"[PLAYBACK] OBS playback error: {e}")
            return 0

    def _restore_idle(self):
        """Restore the idle loop video in OBS."""
        if not self._idle_video_path:
            return

        try:
            import obsws_python as obs

            cl = obs.ReqClient(
                host=self._obs_host,
                port=self._obs_port,
                password=self._obs_password,
            )

            if sys.platform == "win32" or "/mnt/c/" in self._idle_video_path:
                obs_path = self._idle_video_path.replace("/mnt/c/", "C:\\").replace("/", "\\")
            else:
                obs_path = self._idle_video_path

            cl.set_input_settings(
                name=self._source_name,
                settings={
                    "local_file": obs_path,
                    "looping": True,
                    "restart_on_activate": True,
                    "clear_on_media_end": False,
                },
                overlay=True,
            )
            cl.trigger_media_input_action(
                self._source_name,
                "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART",
            )
            logger.info("[PLAYBACK] Restored idle loop")

        except Exception as e:
            logger.error(f"[PLAYBACK] Failed to restore idle: {e}")

    def cleanup(self, keep_last: int = 3):
        """Remove old generated video files."""
        try:
            files = sorted(
                [
                    f for f in os.listdir(self._output_dir)
                    if f.startswith("stream_") and f.endswith(".mp4")
                ],
                key=lambda x: os.path.getmtime(
                    os.path.join(self._output_dir, x)
                ),
            )
            for f in files[:-keep_last]:
                try:
                    os.unlink(os.path.join(self._output_dir, f))
                except OSError:
                    pass
        except Exception:
            pass
