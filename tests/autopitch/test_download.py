"""Tests for autopitch.scripts.download — the capped streaming fetch helpers.

The property under test: the byte cap is enforced against the *streamed* size,
so a missing or under-reported Content-Length header can't make us read an
unbounded body into memory.
"""

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts import download
from autopitch.scripts.download import fetch_bytes


class FakeResponse:
    """Minimal stand-in for a streaming requests.Response."""

    def __init__(self, chunks, headers=None, status=200):
        self._chunks = list(chunks)
        self.headers = headers or {}
        self._status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self._status >= 400:
            raise requests.HTTPError(f"status {self._status}")

    def iter_content(self, chunk_size=8192):
        yield from self._chunks


def _patch_get(monkeypatch, response, captured=None):
    def fake_get(url, **kwargs):
        if captured is not None:
            captured.update(kwargs)
            captured["url"] = url
        return response
    monkeypatch.setattr(download.requests, "get", fake_get)


class TestFetchBytes:
    def test_joins_chunks(self, monkeypatch):
        _patch_get(monkeypatch, FakeResponse([b"abc", b"def"]))
        assert fetch_bytes("http://x") == b"abcdef"

    def test_passes_stream_timeout_and_headers(self, monkeypatch):
        cap = {}
        _patch_get(monkeypatch, FakeResponse([b"ok"]), cap)
        fetch_bytes("http://x", headers={"User-Agent": "ua"}, timeout=7)
        assert cap["url"] == "http://x"
        assert cap["stream"] is True
        assert cap["timeout"] == 7
        assert cap["headers"] == {"User-Agent": "ua"}

    def test_caps_when_no_content_length_header(self, monkeypatch):
        # The regression: no content-length at all → must still cap by streamed size.
        ten_k = [b"x" * 1000] * 10
        _patch_get(monkeypatch, FakeResponse(ten_k))  # no headers
        with pytest.raises(ValueError, match="exceeds"):
            fetch_bytes("http://x", max_bytes=5000)

    def test_rejects_oversize_content_length_before_streaming(self, monkeypatch):
        # Header alone is over the cap → reject without reading the (small) body.
        resp = FakeResponse([b"x"], headers={"content-length": "999999"})
        _patch_get(monkeypatch, resp)
        with pytest.raises(ValueError, match="declared size"):
            fetch_bytes("http://x", max_bytes=1000)

    def test_ignores_non_numeric_content_length(self, monkeypatch):
        # A junk header must not crash the int() pre-check; cap still enforced by stream.
        _patch_get(monkeypatch, FakeResponse([b"hello"], headers={"content-length": "weird"}))
        assert fetch_bytes("http://x", max_bytes=1000) == b"hello"

    def test_allows_exactly_at_cap(self, monkeypatch):
        _patch_get(monkeypatch, FakeResponse([b"x" * 1000]))
        assert fetch_bytes("http://x", max_bytes=1000) == b"x" * 1000

    def test_uses_session_get_not_requests_get(self, monkeypatch):
        seen = {}

        class FakeSession:
            def get(self, url, **kwargs):
                seen["url"] = url
                seen["stream"] = kwargs.get("stream")
                return FakeResponse([b"sess"])

        def forbidden(*a, **k):
            raise AssertionError("requests.get must not be used when a session is given")

        monkeypatch.setattr(download.requests, "get", forbidden)
        assert fetch_bytes("http://x", session=FakeSession()) == b"sess"
        assert seen == {"url": "http://x", "stream": True}

    def test_raises_for_http_error(self, monkeypatch):
        _patch_get(monkeypatch, FakeResponse([], status=404))
        with pytest.raises(requests.HTTPError):
            fetch_bytes("http://x")


class TestDownloadToFile:
    def test_writes_file_and_returns_size(self, monkeypatch, tmp_path):
        _patch_get(monkeypatch, FakeResponse([b"ab", b"cd"]))
        out = tmp_path / "nested" / "f.bin"
        n = download.download("http://x", out)
        assert n == 4
        assert out.read_bytes() == b"abcd"

    def test_caps_oversize_without_content_length(self, monkeypatch, tmp_path):
        _patch_get(monkeypatch, FakeResponse([b"x" * 1000] * 10))
        out = tmp_path / "f.bin"
        with pytest.raises(ValueError, match="exceeds"):
            download.download("http://x", out, max_bytes=5000)
