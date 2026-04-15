"""
Atlas Client - Lip-sync video generation via North Model Labs Atlas API.

Takes audio + face image and produces a lip-synced MP4 video.
Replaces the viseme-based compositing pipeline with a single API call.

Usage:
    from src.video.atlas_client import AtlasClient

    client = AtlasClient()
    output = client.generate_lipsync(
        audio_path="speech.mp3",
        image_path="face.jpg",
        output_path="output.mp4",
    )

    # Or as context manager:
    with AtlasClient() as client:
        client.generate_lipsync(...)
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.atlasv1.com"

# Status codes that are transient and worth retrying during polling
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503}
_MAX_POLL_RETRIES = 3

# MIME type lookup for audio/image files
AUDIO_MIMES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".mpeg": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
}

IMAGE_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class AtlasError(Exception):
    """Base exception for Atlas API errors."""

    def __init__(self, status_code: int, error: str, message: str):
        self.status_code = status_code
        self.error = error
        self.message = message
        super().__init__(f"Atlas API {status_code}: {error} - {message}")


class AtlasJobFailed(AtlasError):
    """Raised when an Atlas generation job fails."""
    pass


class AtlasTimeout(Exception):
    """Raised when polling for job completion exceeds timeout."""
    pass


@dataclass
class AtlasJob:
    """Represents an Atlas generation job."""
    job_id: str
    status: str
    error: Optional[str] = None
    error_code: Optional[str] = None
    latency_ms: Optional[int] = None
    url: Optional[str] = None


class AtlasClient:
    """
    Client for Atlas lip-sync video generation API.

    Offline flow: submit audio + image -> poll job -> download MP4.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ):
        """
        Args:
            api_key: Atlas API key. Falls back to ATLAS_API_KEY env var.
            poll_interval: Seconds between poll requests (default 2s).
            timeout: Max seconds to wait for job completion (default 300s).
        """
        self.api_key = api_key or os.getenv("ATLAS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Atlas API key not provided. "
                "Set ATLAS_API_KEY environment variable or pass api_key parameter."
            )
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self.api_key}"

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an authenticated request to the Atlas API."""
        url = f"{BASE_URL}{path}"
        resp = self._session.request(method, url, **kwargs)

        if resp.status_code >= 400:
            try:
                body = resp.json()
            except ValueError:
                raise AtlasError(resp.status_code, "unknown", resp.text)
            raise AtlasError(
                resp.status_code,
                body.get("error", "unknown"),
                body.get("message", resp.text),
            )

        return resp.json()

    def _mime_for_audio(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        return AUDIO_MIMES.get(ext, "audio/mpeg")

    def _mime_for_image(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        return IMAGE_MIMES.get(ext, "image/png")

    # ── Offline generation ──────────────────────────────────────────

    def submit(
        self,
        audio_path: str,
        image_path: str,
        callback_url: Optional[str] = None,
    ) -> AtlasJob:
        """
        Submit a lip-sync generation job.

        Args:
            audio_path: Path to audio file (WAV, MP3, OGG, WebM).
            image_path: Path to face image (PNG, JPEG, WebP).
            callback_url: Optional webhook URL for async notification.

        Returns:
            AtlasJob with job_id and initial status.
        """
        audio_path = Path(audio_path)
        image_path = Path(image_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"[ATLAS] Submitting job: audio={audio_path.name}, image={image_path.name}")

        headers = {}
        if callback_url:
            headers["X-Callback-URL"] = callback_url

        with open(audio_path, "rb") as af, open(image_path, "rb") as imf:
            files = {
                "audio": (audio_path.name, af, self._mime_for_audio(str(audio_path))),
                "image": (image_path.name, imf, self._mime_for_image(str(image_path))),
            }
            data = self._request("POST", "/v1/generate", files=files, headers=headers)

        job = AtlasJob(job_id=data["job_id"], status=data.get("status", "pending"))
        logger.info(f"[ATLAS] Job submitted: {job.job_id}")
        return job

    def poll(self, job_id: str) -> AtlasJob:
        """
        Poll the status of a generation job.

        Args:
            job_id: The job ID to check.

        Returns:
            AtlasJob with current status.
        """
        data = self._request("GET", f"/v1/jobs/{job_id}")

        return AtlasJob(
            job_id=data["job_id"],
            status=data["status"],
            error=data.get("error"),
            error_code=data.get("error_code"),
            latency_ms=data.get("timing", {}).get("latency_ms"),
            url=data.get("url"),
        )

    def wait_for_completion(self, job_id: str) -> AtlasJob:
        """
        Poll until job completes or fails. Retries on transient API errors.

        Args:
            job_id: The job ID to wait on.

        Returns:
            AtlasJob with final status.

        Raises:
            AtlasJobFailed: If job status is 'failed'.
            AtlasTimeout: If polling exceeds timeout.
        """
        start = time.time()
        consecutive_errors = 0

        while True:
            try:
                job = self.poll(job_id)
                consecutive_errors = 0
            except AtlasError as e:
                if e.status_code in _TRANSIENT_STATUS_CODES:
                    consecutive_errors += 1
                    if consecutive_errors > _MAX_POLL_RETRIES:
                        raise
                    backoff = self.poll_interval * (2 ** (consecutive_errors - 1))
                    logger.warning(
                        f"[ATLAS] Transient error polling {job_id} "
                        f"({e.status_code}), retry {consecutive_errors}/{_MAX_POLL_RETRIES} "
                        f"in {backoff:.0f}s"
                    )
                    time.sleep(backoff)
                    continue
                raise

            if job.status == "completed":
                logger.info(
                    f"[ATLAS] Job {job_id} completed in {job.latency_ms}ms"
                )
                return job

            if job.status == "failed":
                raise AtlasJobFailed(
                    502,
                    job.error_code or "generation_failed",
                    job.error or "Job failed",
                )

            elapsed = time.time() - start
            if elapsed > self.timeout:
                raise AtlasTimeout(
                    f"Job {job_id} did not complete within {self.timeout}s "
                    f"(last status: {job.status})"
                )

            logger.debug(
                f"[ATLAS] Job {job_id}: {job.status} ({elapsed:.0f}s elapsed)"
            )
            time.sleep(self.poll_interval)

    def get_result_url(self, job_id: str) -> str:
        """
        Get the presigned download URL for a completed job.

        Args:
            job_id: Completed job ID.

        Returns:
            Presigned URL string (valid 24 hours).
        """
        data = self._request("GET", f"/v1/jobs/{job_id}/result")
        return data["url"]

    def download(self, job_id: str, output_path: str) -> str:
        """
        Download the result video for a completed job (streamed).

        Args:
            job_id: Completed job ID.
            output_path: Where to save the MP4.

        Returns:
            Path to saved file.
        """
        url = self.get_result_url(job_id)

        logger.info(f"[ATLAS] Downloading result to {output_path}")
        resp = self._session.get(url, stream=True)
        resp.raise_for_status()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        size = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                size += len(chunk)

        logger.info(f"[ATLAS] Saved {size} bytes to {output_path}")
        return str(output_path)

    def generate_lipsync(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        callback_url: Optional[str] = None,
    ) -> str:
        """
        End-to-end: submit audio + image, wait for completion, download MP4.

        Args:
            audio_path: Path to audio file.
            image_path: Path to face image.
            output_path: Where to save the output MP4.
            callback_url: Optional webhook URL.

        Returns:
            Path to saved MP4 file.
        """
        start = time.time()

        job = self.submit(audio_path, image_path, callback_url)
        job = self.wait_for_completion(job.job_id)
        result_path = self.download(job.job_id, output_path)

        total = time.time() - start
        logger.info(f"[ATLAS] Total generate_lipsync: {total:.1f}s")

        return result_path

    def generate_lipsync_from_bytes(
        self,
        audio_bytes: bytes,
        audio_filename: str,
        image_path: str,
        output_path: str,
    ) -> str:
        """
        Generate lip-sync video from in-memory audio bytes.

        Useful when TTS returns audio bytes directly (e.g. ElevenLabs streaming).

        Args:
            audio_bytes: Raw audio data.
            audio_filename: Filename for MIME type detection (e.g. "speech.mp3").
            image_path: Path to face image.
            output_path: Where to save output MP4.

        Returns:
            Path to saved MP4 file.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(
            f"[ATLAS] Submitting from bytes: audio={audio_filename} "
            f"({len(audio_bytes)} bytes), image={image_path.name}"
        )

        with open(image_path, "rb") as imf:
            files = {
                "audio": (audio_filename, audio_bytes, self._mime_for_audio(audio_filename)),
                "image": (image_path.name, imf, self._mime_for_image(str(image_path))),
            }
            data = self._request("POST", "/v1/generate", files=files)

        job_id = data["job_id"]
        self.wait_for_completion(job_id)
        return self.download(job_id, output_path)

    # ── Job management ──────────────────────────────────────────────

    def list_jobs(self, limit: int = 20, offset: int = 0) -> dict:
        """List recent jobs."""
        return self._request(
            "GET", "/v1/jobs", params={"limit": limit, "offset": offset}
        )

    # ── Account ─────────────────────────────────────────────────────

    def me(self) -> dict:
        """Get API key info, rate limits, and plan details."""
        return self._request("GET", "/v1/me")

    def status(self) -> dict:
        """Check system status."""
        return self._request("GET", "/v1/status")

    def health(self) -> dict:
        """Health check (no auth required)."""
        url = f"{BASE_URL}/v1/health"
        return requests.get(url).json()
