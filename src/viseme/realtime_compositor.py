"""
Real-time viseme compositor.

Composites mouth images onto a base video/frame based on
viseme timing data, enabling real-time lip sync.
"""

import time
import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Generator, Callable
from dataclasses import dataclass
from queue import Queue
import numpy as np
import cv2

from .tts_viseme import VisemeEvent, TTSResult
from .viseme_library import VisemeLibrary

logger = logging.getLogger(__name__)


@dataclass
class CompositorConfig:
    """Configuration for the compositor."""
    blend_frames: int = 3          # Frames to blend between visemes
    mouth_scale: float = 1.0       # Scale factor for mouth overlay
    feather_pixels: int = 5        # Feathering at mouth edges
    fps: int = 30                  # Output frame rate
    use_poisson_blend: bool = True # Use Poisson blending for seamless composite


class RealtimeVisemeCompositor:
    """
    Composites viseme mouth images onto video in real-time.

    Supports:
    - Live compositing with audio playback sync
    - Pre-rendering to video file
    - Frame-by-frame generation for streaming
    """

    def __init__(
        self,
        library: VisemeLibrary,
        config: Optional[CompositorConfig] = None
    ):
        """
        Initialize compositor.

        Args:
            library: VisemeLibrary with mouth images
            config: Compositor configuration
        """
        self.library = library
        self.config = config or CompositorConfig()

        # State for real-time mode
        self._current_viseme = 0
        self._prev_viseme = 0
        self._blend_progress = 1.0
        self._playback_start_time = 0.0

    def composite_frame(
        self,
        base_frame: np.ndarray,
        viseme_id: int,
        prev_viseme_id: Optional[int] = None,
        blend_factor: float = 1.0,
        mouth_position: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Composite a single frame with mouth overlay.

        Args:
            base_frame: Background frame (BGR)
            viseme_id: Current viseme to display
            prev_viseme_id: Previous viseme for blending
            blend_factor: 0.0 = prev viseme, 1.0 = current viseme
            mouth_position: (x, y) center of mouth, or None to auto-detect

        Returns:
            Composited frame
        """
        # Get mouth image (with blending if transitioning)
        if prev_viseme_id is not None and blend_factor < 1.0:
            mouth_img = self.library.get_blended_image(
                viseme_id, prev_viseme_id, blend_factor
            )
        else:
            mouth_img = self.library.get_viseme_image(viseme_id)

        if mouth_img is None:
            return base_frame

        # Determine mouth position
        if mouth_position is None:
            mouth_position = self._detect_mouth_position(base_frame)

        if mouth_position is None:
            # Use stored position from library
            if self.library.face_bbox:
                x, y, w, h = self.library.face_bbox
                # Estimate mouth center (bottom third of face)
                mouth_position = (x + w // 2, y + int(h * 0.75))
            else:
                return base_frame

        # Apply scaling
        if self.config.mouth_scale != 1.0:
            new_w = int(mouth_img.shape[1] * self.config.mouth_scale)
            new_h = int(mouth_img.shape[0] * self.config.mouth_scale)
            mouth_img = cv2.resize(mouth_img, (new_w, new_h))

        # Composite
        result = self._blend_mouth(base_frame, mouth_img, mouth_position)

        return result

    def _detect_mouth_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect mouth center position in frame."""
        # Use stored position from library instead of runtime detection
        # This is faster and avoids MediaPipe API issues
        return None

    def _blend_mouth(
        self,
        base: np.ndarray,
        mouth: np.ndarray,
        center: Tuple[int, int]
    ) -> np.ndarray:
        """Blend mouth onto base frame."""
        result = base.copy()
        mh, mw = mouth.shape[:2]
        bh, bw = base.shape[:2]

        # Calculate placement
        x1 = center[0] - mw // 2
        y1 = center[1] - mh // 2
        x2 = x1 + mw
        y2 = y1 + mh

        # Clip to bounds
        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = mw - max(0, x2 - bw)
        src_y2 = mh - max(0, y2 - bh)

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(bw, x2)
        dst_y2 = min(bh, y2)

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return result

        # Extract regions
        mouth_region = mouth[src_y1:src_y2, src_x1:src_x2]
        base_region = base[dst_y1:dst_y2, dst_x1:dst_x2]

        if self.config.use_poisson_blend:
            # Poisson blending for seamless edges
            try:
                mask = self._create_feathered_mask(mouth_region.shape[:2])
                center_in_result = (
                    (dst_x1 + dst_x2) // 2,
                    (dst_y1 + dst_y2) // 2
                )
                blended = cv2.seamlessClone(
                    mouth_region, result, mask,
                    center_in_result, cv2.NORMAL_CLONE
                )
                return blended
            except cv2.error:
                # Fall back to alpha blending if Poisson fails
                pass

        # Alpha blending with feathered edges
        mask = self._create_feathered_mask(mouth_region.shape[:2])
        mask_3ch = np.stack([mask, mask, mask], axis=-1) / 255.0

        blended_region = (
            mouth_region.astype(float) * mask_3ch +
            base_region.astype(float) * (1 - mask_3ch)
        ).astype(np.uint8)

        result[dst_y1:dst_y2, dst_x1:dst_x2] = blended_region

        return result

    def _create_feathered_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create elliptical mask with feathered edges."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw filled ellipse
        center = (w // 2, h // 2)
        axes = (w // 2 - self.config.feather_pixels,
                h // 2 - self.config.feather_pixels)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # Apply Gaussian blur for feathering
        if self.config.feather_pixels > 0:
            ksize = self.config.feather_pixels * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return mask

    def generate_frames(
        self,
        base_frame: np.ndarray,
        viseme_events: List[VisemeEvent],
        total_duration_ms: int
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate composited frames for a viseme sequence.

        Args:
            base_frame: Static background frame
            viseme_events: List of VisemeEvent with timing
            total_duration_ms: Total duration in milliseconds

        Yields:
            Composited frames at configured FPS
        """
        frame_duration_ms = 1000 / self.config.fps
        total_frames = int(total_duration_ms / frame_duration_ms)

        # Build time-indexed viseme lookup
        viseme_at_time = self._build_viseme_timeline(viseme_events, total_duration_ms)

        prev_viseme = 0

        for frame_idx in range(total_frames):
            current_time_ms = frame_idx * frame_duration_ms

            # Get current viseme
            current_viseme = viseme_at_time.get(int(current_time_ms), 0)

            # Calculate blend factor for smooth transitions
            blend_factor = 1.0
            if current_viseme != prev_viseme:
                # Find how far into this viseme we are
                for event in viseme_events:
                    if (event.start_time_ms <= current_time_ms < event.end_time_ms
                        and event.viseme_id == current_viseme):
                        progress = (current_time_ms - event.start_time_ms) / max(1, self.config.blend_frames * frame_duration_ms)
                        blend_factor = min(1.0, progress)
                        break

            # Generate frame
            frame = self.composite_frame(
                base_frame,
                current_viseme,
                prev_viseme if blend_factor < 1.0 else None,
                blend_factor
            )

            yield frame

            if blend_factor >= 1.0:
                prev_viseme = current_viseme

    def _build_viseme_timeline(
        self,
        events: List[VisemeEvent],
        total_ms: int
    ) -> dict:
        """Build millisecond -> viseme_id mapping."""
        timeline = {}

        for event in events:
            for t in range(event.start_time_ms, min(event.end_time_ms, total_ms)):
                timeline[t] = event.viseme_id

        return timeline

    def render_to_video(
        self,
        base_frame: np.ndarray,
        tts_result: TTSResult,
        output_path: str,
        include_audio: bool = True
    ) -> str:
        """
        Render viseme animation to video file.

        Args:
            base_frame: Background frame
            tts_result: TTS result with audio and viseme data
            output_path: Path for output video
            include_audio: Whether to mux audio into video

        Returns:
            Path to output video
        """
        logger.info(f"Rendering video to {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create video writer
        h, w = base_frame.shape[:2]
        temp_video = str(output_path.with_suffix('.temp.mp4'))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, self.config.fps, (w, h))

        # Generate and write frames
        frame_count = 0
        for frame in self.generate_frames(
            base_frame,
            tts_result.viseme_events,
            tts_result.audio_duration_ms
        ):
            writer.write(frame)
            frame_count += 1

        writer.release()

        logger.info(f"Wrote {frame_count} frames")

        # Mux audio if requested
        if include_audio and Path(tts_result.audio_path).exists():
            import subprocess

            # Use ffmpeg to combine video and audio
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', tts_result.audio_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-shortest',
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                Path(temp_video).unlink()  # Remove temp file
                logger.info(f"Created video with audio: {output_path}")
            else:
                logger.warning(f"FFmpeg failed, keeping video without audio")
                Path(temp_video).rename(output_path)
        else:
            Path(temp_video).rename(output_path)

        return str(output_path)

    def start_realtime_playback(
        self,
        base_frame: np.ndarray,
        viseme_events: List[VisemeEvent],
        display_callback: Callable[[np.ndarray], None],
        audio_start_callback: Optional[Callable[[], None]] = None
    ):
        """
        Start real-time playback with frame callback.

        Args:
            base_frame: Background frame
            viseme_events: Viseme sequence
            display_callback: Called with each frame to display
            audio_start_callback: Called when playback starts (for audio sync)
        """
        self._playback_start_time = time.time()

        if audio_start_callback:
            audio_start_callback()

        # Build timeline
        if not viseme_events:
            return

        total_ms = max(e.end_time_ms for e in viseme_events)
        timeline = self._build_viseme_timeline(viseme_events, total_ms)

        prev_viseme = 0
        frame_duration = 1.0 / self.config.fps

        while True:
            elapsed_ms = (time.time() - self._playback_start_time) * 1000

            if elapsed_ms >= total_ms:
                break

            current_viseme = timeline.get(int(elapsed_ms), 0)

            # Generate frame
            frame = self.composite_frame(
                base_frame,
                current_viseme,
                prev_viseme if current_viseme != prev_viseme else None,
                1.0
            )

            display_callback(frame)
            prev_viseme = current_viseme

            # Maintain frame rate
            next_frame_time = self._playback_start_time + (int(elapsed_ms / (frame_duration * 1000)) + 1) * frame_duration
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)


class StreamingVisemeCompositor:
    """
    Compositor optimized for streaming scenarios.

    Accepts viseme events as they arrive and generates
    frames with minimal latency.
    """

    def __init__(
        self,
        library: VisemeLibrary,
        base_frame: np.ndarray,
        config: Optional[CompositorConfig] = None
    ):
        """
        Initialize streaming compositor.

        Args:
            library: VisemeLibrary with mouth images
            base_frame: Static background frame
            config: Compositor configuration
        """
        self.compositor = RealtimeVisemeCompositor(library, config)
        self.base_frame = base_frame
        self.config = config or CompositorConfig()

        self._event_queue: Queue = Queue()
        self._current_viseme = 0
        self._running = False
        self._frame_thread = None

    def push_viseme(self, event: VisemeEvent):
        """Push a new viseme event to the stream."""
        self._event_queue.put(event)

    def start(self, frame_callback: Callable[[np.ndarray], None]):
        """
        Start streaming compositor.

        Args:
            frame_callback: Called with each generated frame
        """
        self._running = True
        self._frame_thread = threading.Thread(
            target=self._frame_loop,
            args=(frame_callback,)
        )
        self._frame_thread.start()

    def stop(self):
        """Stop the compositor."""
        self._running = False
        if self._frame_thread:
            self._frame_thread.join()

    def _frame_loop(self, callback: Callable[[np.ndarray], None]):
        """Main frame generation loop."""
        frame_duration = 1.0 / self.config.fps
        last_frame_time = time.time()
        prev_viseme = 0

        while self._running:
            # Check for new viseme events
            while not self._event_queue.empty():
                event = self._event_queue.get_nowait()
                self._current_viseme = event.viseme_id

            # Generate frame
            frame = self.compositor.composite_frame(
                self.base_frame,
                self._current_viseme,
                prev_viseme if self._current_viseme != prev_viseme else None,
                1.0
            )

            callback(frame)
            prev_viseme = self._current_viseme

            # Maintain frame rate
            elapsed = time.time() - last_frame_time
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_frame_time = time.time()


def demo():
    """Demo the compositor."""
    import sys

    # Check for library path
    if len(sys.argv) < 2:
        print("Usage: python realtime_compositor.py <library_path> [base_image]")
        print("\nRunning synthetic demo...")

        # Create synthetic demo
        from .phoneme_mapper import PhonemeToVisemeMapper

        # Create fake library with colored rectangles for each viseme
        class FakeLibrary:
            def __init__(self):
                self.face_bbox = (100, 100, 200, 200)
                colors = [
                    (50, 50, 50),    # 0: silence - dark gray
                    (0, 100, 200),   # 1: open mid - orange
                    (0, 50, 255),    # 2: open wide - red
                    (100, 100, 200), # etc...
                ]
                self.images = {}
                for i in range(22):
                    color = colors[i % len(colors)]
                    img = np.full((60, 100, 3), color, dtype=np.uint8)
                    self.images[i] = img

            def get_viseme_image(self, viseme_id, variant=0):
                return self.images.get(viseme_id, self.images[0])

            def get_blended_image(self, v1, v2, blend):
                img1 = self.get_viseme_image(v1)
                img2 = self.get_viseme_image(v2)
                return cv2.addWeighted(img2, 1-blend, img1, blend, 0)

        # Create base frame (gray background with face outline)
        base = np.full((400, 600, 3), (200, 200, 200), dtype=np.uint8)
        cv2.rectangle(base, (100, 100), (300, 300), (150, 150, 150), -1)  # Face
        cv2.circle(base, (150, 170), 20, (100, 100, 100), -1)  # Left eye
        cv2.circle(base, (250, 170), 20, (100, 100, 100), -1)  # Right eye

        library = FakeLibrary()
        compositor = RealtimeVisemeCompositor(library)

        # Generate visemes for test text
        mapper = PhonemeToVisemeMapper()
        text = "Hello world, this is a test"
        frames_data = mapper.text_to_visemes(text, duration=3.0)

        # Convert to VisemeEvent
        from .tts_viseme import VisemeEvent
        events = [
            VisemeEvent(
                viseme_id=f.viseme_id,
                start_time_ms=int(f.start_time * 1000),
                end_time_ms=int(f.end_time * 1000),
                character=""
            )
            for f in frames_data
        ]

        print(f"Text: {text}")
        print(f"Generated {len(events)} viseme events")
        print("\nPlaying animation (press 'q' to quit)...")

        # Play animation
        for frame in compositor.generate_frames(base, events, 3000):
            cv2.imshow("Viseme Demo", frame)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        return

    # Load real library
    library_path = sys.argv[1]
    library = VisemeLibrary.load(library_path)

    if len(sys.argv) > 2:
        base_frame = cv2.imread(sys.argv[2])
    elif library.neutral_frame is not None:
        base_frame = library.neutral_frame
    else:
        print("No base frame available")
        return

    compositor = RealtimeVisemeCompositor(library)

    # Demo with test visemes
    print("Testing compositor with library...")

    for viseme_id in range(min(10, len(library.templates))):
        frame = compositor.composite_frame(base_frame, viseme_id)
        cv2.imshow(f"Viseme {viseme_id}", frame)
        cv2.waitKey(500)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo()
