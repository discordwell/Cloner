"""Tests for Atlas lip-sync client and video backend integration."""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_streaming_response(data: bytes, status_code=200):
    """Create a mock response that supports iter_content (streaming)."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.raise_for_status = MagicMock()
    mock.iter_content = MagicMock(return_value=[data])
    return mock


class TestAtlasClientInit:
    """Test AtlasClient initialization."""

    def test_requires_api_key(self):
        """Should raise if no API key provided."""
        old = os.environ.pop("ATLAS_API_KEY", None)
        try:
            from src.video.atlas_client import AtlasClient
            with pytest.raises(ValueError, match="Atlas API key not provided"):
                AtlasClient()
        finally:
            if old:
                os.environ["ATLAS_API_KEY"] = old

    def test_accepts_env_var(self):
        os.environ["ATLAS_API_KEY"] = "test_key_123"
        try:
            from src.video.atlas_client import AtlasClient
            client = AtlasClient()
            assert client.api_key == "test_key_123"
        finally:
            del os.environ["ATLAS_API_KEY"]

    def test_accepts_explicit_key(self):
        from src.video.atlas_client import AtlasClient
        client = AtlasClient(api_key="explicit_key")
        assert client.api_key == "explicit_key"

    def test_custom_poll_interval_and_timeout(self):
        from src.video.atlas_client import AtlasClient
        client = AtlasClient(api_key="k", poll_interval=5.0, timeout=60.0)
        assert client.poll_interval == 5.0
        assert client.timeout == 60.0

    def test_context_manager_closes_session(self):
        from src.video.atlas_client import AtlasClient
        with AtlasClient(api_key="k") as client:
            mock_close = MagicMock()
            client._session.close = mock_close
        mock_close.assert_called_once()

    def test_close_method(self):
        from src.video.atlas_client import AtlasClient
        client = AtlasClient(api_key="k")
        with patch.object(client._session, "close") as mock_close:
            client.close()
        mock_close.assert_called_once()


class TestAtlasClientSubmit:
    """Test job submission."""

    def test_submit_sends_files(self, tmp_path):
        from src.video.atlas_client import AtlasClient

        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"fake_audio_data")
        image = tmp_path / "face.jpg"
        image.write_bytes(b"fake_image_data")

        client = AtlasClient(api_key="test_key")

        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_resp.json.return_value = {"job_id": "job_123", "status": "pending"}

        with patch.object(client._session, "request", return_value=mock_resp):
            job = client.submit(str(audio), str(image))

        assert job.job_id == "job_123"
        assert job.status == "pending"

    def test_submit_raises_on_missing_audio(self, tmp_path):
        from src.video.atlas_client import AtlasClient

        image = tmp_path / "face.jpg"
        image.write_bytes(b"data")

        client = AtlasClient(api_key="test_key")

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            client.submit("/nonexistent/audio.mp3", str(image))

    def test_submit_raises_on_missing_image(self, tmp_path):
        from src.video.atlas_client import AtlasClient

        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"data")

        client = AtlasClient(api_key="test_key")

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            client.submit(str(audio), "/nonexistent/face.jpg")

    def test_submit_with_callback_url(self, tmp_path):
        from src.video.atlas_client import AtlasClient

        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"data")
        image = tmp_path / "face.jpg"
        image.write_bytes(b"data")

        client = AtlasClient(api_key="test_key")

        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_resp.json.return_value = {"job_id": "job_456", "status": "pending"}

        with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
            client.submit(str(audio), str(image), callback_url="https://example.com/hook")

        # Verify X-Callback-URL was included in headers
        call_kwargs = mock_req.call_args
        assert "X-Callback-URL" in call_kwargs.kwargs.get("headers", {})


class TestAtlasClientPoll:
    """Test job status polling."""

    def test_poll_returns_job_status(self):
        from src.video.atlas_client import AtlasClient

        client = AtlasClient(api_key="test_key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "job_id": "job_123",
            "status": "processing",
            "error": None,
            "error_code": None,
            "timing": {"latency_ms": None},
        }

        with patch.object(client._session, "request", return_value=mock_resp):
            job = client.poll("job_123")

        assert job.job_id == "job_123"
        assert job.status == "processing"

    def test_poll_completed_includes_url(self):
        from src.video.atlas_client import AtlasClient

        client = AtlasClient(api_key="test_key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "job_id": "job_123",
            "status": "completed",
            "error": None,
            "error_code": None,
            "timing": {"latency_ms": 43000},
            "url": "https://storage.example.com/output.mp4",
        }

        with patch.object(client._session, "request", return_value=mock_resp):
            job = client.poll("job_123")

        assert job.status == "completed"
        assert job.latency_ms == 43000
        assert job.url == "https://storage.example.com/output.mp4"


class TestAtlasClientWait:
    """Test wait_for_completion polling loop."""

    def test_wait_returns_on_completed(self):
        from src.video.atlas_client import AtlasClient

        client = AtlasClient(api_key="test_key", poll_interval=0.01)

        responses = [
            {"job_id": "j1", "status": "pending", "error": None, "error_code": None, "timing": {}},
            {"job_id": "j1", "status": "processing", "error": None, "error_code": None, "timing": {}},
            {"job_id": "j1", "status": "completed", "error": None, "error_code": None, "timing": {"latency_ms": 5000}},
        ]

        mock_resps = []
        for r in responses:
            m = MagicMock()
            m.status_code = 200
            m.json.return_value = r
            mock_resps.append(m)

        with patch.object(client._session, "request", side_effect=mock_resps):
            job = client.wait_for_completion("j1")

        assert job.status == "completed"

    def test_wait_raises_on_failed(self):
        from src.video.atlas_client import AtlasClient, AtlasJobFailed

        client = AtlasClient(api_key="test_key", poll_interval=0.01)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "job_id": "j1",
            "status": "failed",
            "error": "Processing failed",
            "error_code": "generation_failed",
            "timing": {},
        }

        with patch.object(client._session, "request", return_value=mock_resp):
            with pytest.raises(AtlasJobFailed, match="generation_failed"):
                client.wait_for_completion("j1")

    def test_wait_raises_on_timeout(self):
        from src.video.atlas_client import AtlasClient, AtlasTimeout

        client = AtlasClient(api_key="test_key", poll_interval=0.01, timeout=0.05)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "job_id": "j1",
            "status": "processing",
            "error": None,
            "error_code": None,
            "timing": {},
        }

        with patch.object(client._session, "request", return_value=mock_resp):
            with pytest.raises(AtlasTimeout, match="did not complete"):
                client.wait_for_completion("j1")

    def test_wait_retries_on_transient_error(self):
        """Transient 503 during polling should be retried, not crash."""
        from src.video.atlas_client import AtlasClient

        client = AtlasClient(api_key="test_key", poll_interval=0.01)

        # First poll: 503 (transient), second poll: completed
        error_resp = MagicMock(status_code=503)
        error_resp.json.return_value = {
            "error": "no_capacity",
            "message": "All GPU pods busy",
        }

        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = {
            "job_id": "j1", "status": "completed",
            "error": None, "error_code": None,
            "timing": {"latency_ms": 50000},
        }

        with patch.object(client._session, "request", side_effect=[error_resp, ok_resp]):
            job = client.wait_for_completion("j1")

        assert job.status == "completed"

    def test_wait_gives_up_after_max_retries(self):
        """Should raise after exceeding max transient retries."""
        from src.video.atlas_client import AtlasClient, AtlasError

        client = AtlasClient(api_key="test_key", poll_interval=0.01)

        error_resp = MagicMock(status_code=502)
        error_resp.json.return_value = {
            "error": "generation_failed",
            "message": "Generation failed after retries",
        }

        # 4 consecutive 502s should exceed _MAX_POLL_RETRIES (3)
        with patch.object(client._session, "request", return_value=error_resp):
            with pytest.raises(AtlasError) as exc_info:
                client.wait_for_completion("j1")
            assert exc_info.value.status_code == 502

    def test_wait_does_not_retry_non_transient_error(self):
        """Non-transient errors (e.g. 401) should raise immediately."""
        from src.video.atlas_client import AtlasClient, AtlasError

        client = AtlasClient(api_key="test_key", poll_interval=0.01)

        error_resp = MagicMock(status_code=401)
        error_resp.json.return_value = {
            "error": "unauthorized",
            "message": "Invalid API key",
        }

        with patch.object(client._session, "request", return_value=error_resp):
            with pytest.raises(AtlasError) as exc_info:
                client.wait_for_completion("j1")
            assert exc_info.value.status_code == 401


class TestAtlasClientDownload:
    """Test result download."""

    def test_download_saves_file_streaming(self, tmp_path):
        from src.video.atlas_client import AtlasClient

        client = AtlasClient(api_key="test_key")
        output = tmp_path / "result.mp4"

        # Mock get_result_url response
        mock_result_resp = MagicMock()
        mock_result_resp.status_code = 200
        mock_result_resp.json.return_value = {
            "url": "https://storage.example.com/output.mp4",
            "content_type": "video/mp4",
            "expires_in": 86400,
        }

        # Mock streaming download via session.get
        mock_download_resp = _make_streaming_response(b"fake_mp4_video_data")

        # First call is _request (get result URL), second is session.get (download)
        with patch.object(
            client._session, "request", return_value=mock_result_resp
        ):
            with patch.object(
                client._session, "get", return_value=mock_download_resp
            ):
                path = client.download("job_123", str(output))

        assert Path(path).exists()
        assert Path(path).read_bytes() == b"fake_mp4_video_data"

    def test_download_uses_session_not_bare_requests(self, tmp_path):
        """Download should use self._session.get, not requests.get."""
        from src.video.atlas_client import AtlasClient

        client = AtlasClient(api_key="test_key")
        output = tmp_path / "result.mp4"

        mock_result_resp = MagicMock(status_code=200)
        mock_result_resp.json.return_value = {
            "url": "https://storage.example.com/output.mp4",
            "content_type": "video/mp4",
            "expires_in": 86400,
        }

        mock_download_resp = _make_streaming_response(b"data")

        with patch.object(client._session, "request", return_value=mock_result_resp):
            with patch.object(client._session, "get", return_value=mock_download_resp) as mock_get:
                client.download("job_123", str(output))

        mock_get.assert_called_once()
        # Verify stream=True was passed
        assert mock_get.call_args.kwargs.get("stream") is True


class TestAtlasClientEndToEnd:
    """Test the full generate_lipsync flow."""

    def test_generate_lipsync_full_flow(self, tmp_path):
        from src.video.atlas_client import AtlasClient

        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"audio_data")
        image = tmp_path / "face.jpg"
        image.write_bytes(b"image_data")
        output = tmp_path / "output.mp4"

        client = AtlasClient(api_key="test_key", poll_interval=0.01)

        submit_resp = MagicMock(status_code=202)
        submit_resp.json.return_value = {"job_id": "j1", "status": "pending"}

        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {
            "job_id": "j1", "status": "completed",
            "error": None, "error_code": None,
            "timing": {"latency_ms": 42000},
        }

        result_resp = MagicMock(status_code=200)
        result_resp.json.return_value = {
            "url": "https://storage.example.com/output.mp4",
            "content_type": "video/mp4",
            "expires_in": 86400,
        }

        mock_download_resp = _make_streaming_response(b"mp4_video_bytes")

        with patch.object(
            client._session, "request",
            side_effect=[submit_resp, poll_resp, result_resp]
        ):
            with patch.object(client._session, "get", return_value=mock_download_resp):
                path = client.generate_lipsync(
                    str(audio), str(image), str(output)
                )

        assert path == str(output)
        assert output.read_bytes() == b"mp4_video_bytes"

    def test_generate_lipsync_from_bytes(self, tmp_path):
        from src.video.atlas_client import AtlasClient

        image = tmp_path / "face.jpg"
        image.write_bytes(b"image_data")
        output = tmp_path / "output.mp4"

        client = AtlasClient(api_key="test_key", poll_interval=0.01)

        submit_resp = MagicMock(status_code=202)
        submit_resp.json.return_value = {"job_id": "j2", "status": "pending"}

        poll_resp = MagicMock(status_code=200)
        poll_resp.json.return_value = {
            "job_id": "j2", "status": "completed",
            "error": None, "error_code": None,
            "timing": {"latency_ms": 35000},
        }

        result_resp = MagicMock(status_code=200)
        result_resp.json.return_value = {
            "url": "https://storage.example.com/j2.mp4",
            "content_type": "video/mp4",
            "expires_in": 86400,
        }

        mock_download_resp = _make_streaming_response(b"video_data")

        with patch.object(
            client._session, "request",
            side_effect=[submit_resp, poll_resp, result_resp]
        ):
            with patch.object(client._session, "get", return_value=mock_download_resp):
                path = client.generate_lipsync_from_bytes(
                    audio_bytes=b"tts_audio_bytes",
                    audio_filename="speech.mp3",
                    image_path=str(image),
                    output_path=str(output),
                )

        assert path == str(output)
        assert output.read_bytes() == b"video_data"


class TestAtlasClientErrors:
    """Test error handling."""

    def test_api_error_raises_atlas_error(self):
        from src.video.atlas_client import AtlasClient, AtlasError

        client = AtlasClient(api_key="test_key")

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {
            "error": "unauthorized",
            "message": "Missing Authorization header.",
        }

        with patch.object(client._session, "request", return_value=mock_resp):
            with pytest.raises(AtlasError, match="unauthorized"):
                client.me()

    def test_rate_limit_raises_atlas_error(self):
        from src.video.atlas_client import AtlasClient, AtlasError

        client = AtlasClient(api_key="test_key")

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {
            "error": "rate_limit_exceeded",
            "message": "Slow down!",
            "retry_after_seconds": 12,
        }

        with patch.object(client._session, "request", return_value=mock_resp):
            with pytest.raises(AtlasError) as exc_info:
                client.list_jobs()
            assert exc_info.value.status_code == 429

    def test_non_json_error_response(self):
        """Non-JSON error body should still raise AtlasError."""
        from src.video.atlas_client import AtlasClient, AtlasError

        client = AtlasClient(api_key="test_key")

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.side_effect = ValueError("No JSON")
        mock_resp.text = "Internal Server Error"

        with patch.object(client._session, "request", return_value=mock_resp):
            with pytest.raises(AtlasError) as exc_info:
                client.status()
            assert exc_info.value.status_code == 500
            assert "Internal Server Error" in str(exc_info.value)


class TestAtlasBackendFactory:
    """Test Atlas integration with video_backend factory."""

    def test_atlas_backend_returns_client(self):
        from src.video.video_backend import get_video_backend
        from src.video.atlas_client import AtlasClient

        config = {"atlas": {"api_key": "factory_test_key"}}
        backend = get_video_backend("atlas", config)
        assert isinstance(backend, AtlasClient)
        assert backend.api_key == "factory_test_key"

    def test_atlas_backend_uses_config(self):
        from src.video.video_backend import get_video_backend

        config = {
            "atlas": {
                "api_key": "cfg_key",
                "poll_interval": 5.0,
                "timeout": 120.0,
            }
        }
        backend = get_video_backend("atlas", config)
        assert backend.poll_interval == 5.0
        assert backend.timeout == 120.0


class TestAtlasAccountEndpoints:
    """Test account/status endpoints."""

    def test_me_returns_account_info(self):
        from src.video.atlas_client import AtlasClient

        client = AtlasClient(api_key="test_key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "authenticated": True,
            "tier": "starter",
            "requests_used": 14,
        }

        with patch.object(client._session, "request", return_value=mock_resp):
            info = client.me()

        assert info["authenticated"] is True

    def test_health_no_auth(self):
        from src.video.atlas_client import AtlasClient

        client = AtlasClient(api_key="test_key")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "all systems go"}

        with patch("src.video.atlas_client.requests.get", return_value=mock_resp):
            health = client.health()

        assert health["status"] == "all systems go"


class TestCloneProfileIsComplete:
    """Test CloneProfile.is_complete() with both backends."""

    def test_atlas_complete_with_reference_frame(self, tmp_path):
        from src.clone_pipeline import CloneProfile

        ref = tmp_path / "face.png"
        ref.write_bytes(b"image")

        profile = CloneProfile(
            subject_id="test",
            voice_id="v123",
            reference_frame_path=str(ref),
        )
        assert profile.is_complete("atlas") is True

    def test_atlas_incomplete_without_reference_frame(self):
        from src.clone_pipeline import CloneProfile

        profile = CloneProfile(
            subject_id="test",
            voice_id="v123",
        )
        assert profile.is_complete("atlas") is False

    def test_viseme_complete_with_library(self, tmp_path):
        from src.clone_pipeline import CloneProfile

        lib = tmp_path / "visemes"
        lib.mkdir()

        profile = CloneProfile(
            subject_id="test",
            voice_id="v123",
            viseme_library_path=str(lib),
        )
        assert profile.is_complete("viseme") is True

    def test_incomplete_without_voice_id(self, tmp_path):
        from src.clone_pipeline import CloneProfile

        ref = tmp_path / "face.png"
        ref.write_bytes(b"image")

        profile = CloneProfile(
            subject_id="test",
            reference_frame_path=str(ref),
        )
        assert profile.is_complete("atlas") is False
