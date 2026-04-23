"""Tests for autopitch.scripts.chatgpt_mcp (JS string builders)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts.chatgpt_mcp import (
    CLICK_SEND_JS,
    COOLDOWN_DETECT_JS,
    IMAGE_LOADED_JS,
    IMAGE_SIGNATURE_JS,
    IMAGE_SRC_JS,
    fill_prompt_js,
    upload_file_js,
)


class TestFillPromptJs:
    def test_embeds_text_as_valid_json(self):
        js = fill_prompt_js("Hello world")
        assert "Hello world" in js
        # Must be valid JSON — no unescaped quotes leaking
        assert '"Hello world"' in js

    def test_escapes_quotes_and_newlines(self):
        js = fill_prompt_js('He said "hi"\nthen left')
        # The text should appear as a JSON-encoded string inside the JS
        encoded = json.dumps('He said "hi"\nthen left')
        assert encoded in js


class TestUploadFileJs:
    def test_embeds_base64_content(self, tmp_path):
        f = tmp_path / "tiny.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")
        js = upload_file_js(str(f), "image/png")
        assert "atob" in js
        assert "DataTransfer" in js
        assert "tiny.png" in js
        assert "image/png" in js


class TestStaticSnippets:
    def test_click_send_has_known_selectors(self):
        assert "data-testid" in CLICK_SEND_JS
        assert "send-button" in CLICK_SEND_JS

    def test_image_snippets_target_generated_image(self):
        for snippet in (IMAGE_SRC_JS, IMAGE_LOADED_JS, IMAGE_SIGNATURE_JS):
            assert 'alt="Generated image"' in snippet

    def test_cooldown_checks_common_markers(self):
        assert "rate limit" in COOLDOWN_DETECT_JS
        assert "try again later" in COOLDOWN_DETECT_JS
