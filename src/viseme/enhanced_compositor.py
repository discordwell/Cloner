"""
Enhanced viseme compositor with natural motion and better blending.

Addresses the "uncanny valley" effect of static face + moving mouth by:
1. Better edge blending (color matching, soft feathering)
2. Subtle micro-movements (breathing, head sway)
3. Blink animation
4. Color/brightness matching
"""

import math
import random
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Generator
from dataclasses import dataclass, field
import numpy as np
import cv2

from .tts_viseme import VisemeEvent, TTSResult
from .viseme_library import VisemeLibrary

logger = logging.getLogger(__name__)


@dataclass
class MotionConfig:
    """Configuration for subtle face motion."""
    # Breathing motion (subtle scale)
    breathing_enabled: bool = True
    breathing_amplitude: float = 0.003  # ~0.3% scale change
    breathing_rate: float = 0.25  # Hz (one breath per 4 seconds)

    # Head sway (subtle rotation/translation)
    sway_enabled: bool = True
    sway_amplitude_x: float = 2.0  # pixels
    sway_amplitude_y: float = 1.0  # pixels
    sway_rate: float = 0.15  # Hz

    # Micro head rotation
    rotation_enabled: bool = True
    rotation_amplitude: float = 0.3  # degrees
    rotation_rate: float = 0.1  # Hz

    # Blink animation
    blink_enabled: bool = True
    blink_interval_mean: float = 4.0  # seconds between blinks
    blink_interval_std: float = 1.5  # randomness
    blink_duration: float = 0.15  # seconds per blink


@dataclass
class BlendConfig:
    """Configuration for mouth blending."""
    feather_radius: int = 15  # Soft edge radius
    color_match: bool = True  # Match mouth color to face
    brightness_match: bool = True  # Match brightness
    use_seamless_clone: bool = False  # Poisson blending (slower but better)
    edge_blur: int = 7  # Gaussian blur on edges


@dataclass
class EnhancedCompositorConfig:
    """Full compositor configuration."""
    fps: int = 30
    mouth_scale: float = 1.0
    motion: MotionConfig = field(default_factory=MotionConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)


class EnhancedVisemeCompositor:
    """
    Enhanced compositor with natural motion and better blending.
    """

    def __init__(
        self,
        library: VisemeLibrary,
        config: Optional[EnhancedCompositorConfig] = None
    ):
        self.library = library
        self.config = config or EnhancedCompositorConfig()

        # Blink state
        self._next_blink_time = random.uniform(2.0, 5.0)
        self._blink_progress = 1.0  # 1.0 = open, 0.0 = closed
        self._in_blink = False

        # Pre-compute blink overlay if we have eye region
        self._blink_overlay = None

    def generate_frames(
        self,
        base_frame: np.ndarray,
        viseme_events: List[VisemeEvent],
        total_duration_ms: int
    ) -> Generator[np.ndarray, None, None]:
        """Generate enhanced composited frames."""

        frame_duration_ms = 1000 / self.config.fps
        total_frames = int(total_duration_ms / frame_duration_ms)

        # Build viseme timeline
        viseme_at_time = self._build_viseme_timeline(viseme_events, total_duration_ms)

        prev_viseme = 0

        for frame_idx in range(total_frames):
            current_time_ms = frame_idx * frame_duration_ms
            current_time_s = current_time_ms / 1000.0

            # 1. Apply subtle motion to base frame
            animated_base = self._apply_motion(base_frame, current_time_s)

            # 2. Update blink state
            self._update_blink(current_time_s, frame_duration_ms / 1000.0)

            # 3. Get current viseme
            current_viseme = viseme_at_time.get(int(current_time_ms), 0)

            # 4. Composite mouth with enhanced blending
            frame = self._composite_mouth(
                animated_base,
                current_viseme,
                prev_viseme,
                current_time_ms,
                viseme_events
            )

            # 5. Apply blink if active
            if self._blink_progress < 1.0:
                frame = self._apply_blink(frame)

            yield frame

            prev_viseme = current_viseme

    def _apply_motion(self, frame: np.ndarray, time_s: float) -> np.ndarray:
        """Apply subtle motion to make the face feel alive."""

        if not any([
            self.config.motion.breathing_enabled,
            self.config.motion.sway_enabled,
            self.config.motion.rotation_enabled
        ]):
            return frame.copy()

        h, w = frame.shape[:2]
        center = (w // 2, h // 2)

        # Calculate motion components
        scale = 1.0
        tx, ty = 0.0, 0.0
        angle = 0.0

        if self.config.motion.breathing_enabled:
            # Breathing: subtle scale oscillation
            breath_phase = 2 * math.pi * self.config.motion.breathing_rate * time_s
            scale = 1.0 + self.config.motion.breathing_amplitude * math.sin(breath_phase)

        if self.config.motion.sway_enabled:
            # Head sway: gentle side-to-side and up-down
            sway_phase = 2 * math.pi * self.config.motion.sway_rate * time_s
            tx = self.config.motion.sway_amplitude_x * math.sin(sway_phase)
            ty = self.config.motion.sway_amplitude_y * math.sin(sway_phase * 1.3)  # Slightly different rate

        if self.config.motion.rotation_enabled:
            # Micro rotation
            rot_phase = 2 * math.pi * self.config.motion.rotation_rate * time_s
            angle = self.config.motion.rotation_amplitude * math.sin(rot_phase * 0.7)

        # Build transformation matrix
        # Rotation + scale around center, then translation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        # Apply transformation
        result = cv2.warpAffine(
            frame, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return result

    def _update_blink(self, time_s: float, dt: float):
        """Update blink animation state."""

        if not self.config.motion.blink_enabled:
            return

        blink_speed = 1.0 / self.config.motion.blink_duration

        if self._in_blink:
            # Closing or opening
            if self._blink_progress > 0:
                # Closing
                self._blink_progress -= dt * blink_speed * 2  # Close faster
                if self._blink_progress <= 0:
                    self._blink_progress = 0
            else:
                # Opening
                self._blink_progress += dt * blink_speed
                if self._blink_progress >= 1.0:
                    self._blink_progress = 1.0
                    self._in_blink = False
                    # Schedule next blink
                    interval = random.gauss(
                        self.config.motion.blink_interval_mean,
                        self.config.motion.blink_interval_std
                    )
                    self._next_blink_time = time_s + max(1.0, interval)
        else:
            # Check if it's time to blink
            if time_s >= self._next_blink_time:
                self._in_blink = True
                self._blink_progress = 1.0

    def _apply_blink(self, frame: np.ndarray) -> np.ndarray:
        """Apply blink effect by darkening eye region."""

        if self.library.face_bbox is None:
            return frame

        result = frame.copy()
        x, y, w, h = self.library.face_bbox

        # Estimate eye region (upper third of face bbox)
        eye_y1 = int(y + h * 0.15)
        eye_y2 = int(y + h * 0.35)
        eye_x1 = int(x + w * 0.1)
        eye_x2 = int(x + w * 0.9)

        # Darken eyes based on blink progress
        darkness = 1.0 - (1.0 - self._blink_progress) * 0.7  # Don't go fully black

        if 0 <= eye_y1 < frame.shape[0] and 0 <= eye_y2 < frame.shape[0]:
            eye_region = result[eye_y1:eye_y2, eye_x1:eye_x2]
            result[eye_y1:eye_y2, eye_x1:eye_x2] = (eye_region * darkness).astype(np.uint8)

        return result

    def _composite_mouth(
        self,
        base_frame: np.ndarray,
        viseme_id: int,
        prev_viseme_id: int,
        current_time_ms: float,
        viseme_events: List[VisemeEvent]
    ) -> np.ndarray:
        """Composite mouth with enhanced blending."""

        # Get mouth image
        mouth_img = self.library.get_viseme_image(viseme_id)
        if mouth_img is None:
            mouth_img = self.library.get_viseme_image(0)  # Fallback to neutral
        if mouth_img is None:
            return base_frame

        result = base_frame.copy()

        # Determine mouth position from face bbox
        if self.library.face_bbox is None:
            return result

        fx, fy, fw, fh = self.library.face_bbox
        # Mouth is typically at bottom third of face
        mouth_center_x = fx + fw // 2
        mouth_center_y = fy + int(fh * 0.75)

        # Scale mouth if needed
        if self.config.mouth_scale != 1.0:
            new_w = int(mouth_img.shape[1] * self.config.mouth_scale)
            new_h = int(mouth_img.shape[0] * self.config.mouth_scale)
            mouth_img = cv2.resize(mouth_img, (new_w, new_h))

        # Color/brightness matching
        if self.config.blend.color_match or self.config.blend.brightness_match:
            mouth_img = self._match_colors(mouth_img, base_frame, mouth_center_x, mouth_center_y)

        # Calculate placement
        mh, mw = mouth_img.shape[:2]
        x1 = mouth_center_x - mw // 2
        y1 = mouth_center_y - mh // 2
        x2 = x1 + mw
        y2 = y1 + mh

        # Clip to frame bounds
        bh, bw = base_frame.shape[:2]
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

        # Get regions
        mouth_region = mouth_img[src_y1:src_y2, src_x1:src_x2]

        # Create soft elliptical mask
        mask = self._create_soft_mask(mouth_region.shape[:2])

        # Apply edge blur for softer transition
        if self.config.blend.edge_blur > 0:
            ksize = self.config.blend.edge_blur | 1  # Ensure odd
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        # Blend
        mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(np.float32) / 255.0

        base_region = result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
        mouth_float = mouth_region.astype(np.float32)

        blended = mouth_float * mask_3ch + base_region * (1 - mask_3ch)
        result[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)

        return result

    def _match_colors(
        self,
        mouth_img: np.ndarray,
        base_frame: np.ndarray,
        cx: int,
        cy: int
    ) -> np.ndarray:
        """Match mouth colors/brightness to surrounding face region."""

        h, w = base_frame.shape[:2]
        mh, mw = mouth_img.shape[:2]

        # Sample region around where mouth will be placed
        sample_pad = 20
        sx1 = max(0, cx - mw//2 - sample_pad)
        sy1 = max(0, cy - mh//2 - sample_pad)
        sx2 = min(w, cx + mw//2 + sample_pad)
        sy2 = min(h, cy + mh//2 + sample_pad)

        face_sample = base_frame[sy1:sy2, sx1:sx2]

        if face_sample.size == 0:
            return mouth_img

        # Calculate mean color/brightness
        face_mean = np.mean(face_sample, axis=(0, 1))
        mouth_mean = np.mean(mouth_img, axis=(0, 1))

        # Adjust mouth to match
        result = mouth_img.astype(np.float32)

        if self.config.blend.brightness_match:
            # Match overall brightness
            face_brightness = np.mean(face_mean)
            mouth_brightness = np.mean(mouth_mean)
            if mouth_brightness > 0:
                brightness_ratio = face_brightness / mouth_brightness
                # Clamp to avoid extreme adjustments
                brightness_ratio = np.clip(brightness_ratio, 0.7, 1.4)
                result = result * brightness_ratio

        if self.config.blend.color_match:
            # Subtle color shift toward face tone
            for c in range(3):
                if mouth_mean[c] > 0:
                    ratio = face_mean[c] / mouth_mean[c]
                    ratio = np.clip(ratio, 0.8, 1.2)
                    result[:, :, c] = result[:, :, c] * ratio

        return np.clip(result, 0, 255).astype(np.uint8)

    def _create_soft_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create soft elliptical mask with feathered edges."""

        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw filled ellipse
        center = (w // 2, h // 2)
        axes = (
            w // 2 - self.config.blend.feather_radius,
            h // 2 - self.config.blend.feather_radius
        )

        if axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # Apply significant blur for soft edges
        blur_size = self.config.blend.feather_radius * 2 + 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return mask

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
        """Render to video file with audio."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        h, w = base_frame.shape[:2]
        temp_video = str(output_path.with_suffix('.temp.mp4'))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, self.config.fps, (w, h))

        frame_count = 0
        for frame in self.generate_frames(
            base_frame,
            tts_result.viseme_events,
            tts_result.audio_duration_ms
        ):
            writer.write(frame)
            frame_count += 1

        writer.release()

        # Mux audio
        if include_audio and Path(tts_result.audio_path).exists():
            import subprocess
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
                Path(temp_video).unlink()
            else:
                Path(temp_video).rename(output_path)
        else:
            Path(temp_video).rename(output_path)

        return str(output_path)


def demo():
    """Demo enhanced compositor."""
    print("Enhanced Compositor Demo")
    print("Run with a viseme library to see the difference.")


if __name__ == "__main__":
    demo()
