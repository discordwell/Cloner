"""JavaScript snippets for driving ChatGPT's web UI via Claude's MCP chrome tools.

These are pure strings the autopitch agent passes to
`mcp__claude-in-chrome__javascript_tool`. Pattern ported from
clawed-command/tools/asset_pipeline/scripts/providers/chatgpt.py.

IMPORTANT: Avoid ChatGPT Pro ("extended thinking") mode — it rewrites prompts
aggressively and takes 60+s. Make sure the chat is in standard mode before
submitting a cartoonify prompt.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

# ── DOM helpers ─────────────────────────────────────────────────────

# Fill #prompt-textarea (contenteditable). Use execCommand to preserve React state.
FILL_PROMPT_JS_TEMPLATE = """
(function() {
    var el = document.querySelector('#prompt-textarea') ||
             document.querySelector('textarea[data-testid="prompt-textarea"]') ||
             document.querySelector('div[contenteditable="true"]');
    if (!el) return "no_prompt_box";
    el.focus();
    try { document.execCommand('selectAll', false, null); } catch (e) {}
    try { document.execCommand('insertText', false, %(text_json)s); } catch (e) {
        el.textContent = %(text_json)s;
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
    return "filled";
})()
"""


def fill_prompt_js(text: str) -> str:
    """JS that types `text` into the prompt box. Returns 'filled' or 'no_prompt_box'."""
    return FILL_PROMPT_JS_TEMPLATE % {"text_json": json.dumps(text)}


CLICK_SEND_JS = """
(function() {
    var btn = document.querySelector('button[data-testid="send-button"]') ||
              document.querySelector('button[aria-label="Send prompt"]') ||
              document.querySelector('button[aria-label^="Send"]');
    if (!btn) return "no_send_button";
    if (btn.disabled) return "send_disabled";
    btn.click();
    return "sent";
})()
"""


# Return count of generated images on the page.
IMAGE_COUNT_JS = """
(function() {
    return document.querySelectorAll('img[alt="Generated image"]').length.toString();
})()
"""


# Return the current src URL of the most recent generated image.
IMAGE_SRC_JS = """
(function() {
    var imgs = document.querySelectorAll('img[alt="Generated image"]');
    if (imgs.length === 0) return "none";
    var img = imgs[imgs.length - 1];
    return img.src;
})()
"""


# Check if the latest generated image has fully loaded.
IMAGE_LOADED_JS = """
(function() {
    var imgs = document.querySelectorAll('img[alt="Generated image"]');
    if (imgs.length === 0) return "no_img";
    var img = imgs[imgs.length - 1];
    if (img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) return "loaded";
    return "loading";
})()
"""


# Rate-limit / cooldown detection — look for telltale phrases in visible text.
COOLDOWN_DETECT_JS = """
(function() {
    var body = (document.body.innerText || "").toLowerCase();
    var markers = [
        "rate limit", "too many", "try again later", "usage cap",
        "slow down", "temporarily unavailable", "please wait"
    ];
    for (var i = 0; i < markers.length; i++) {
        if (body.indexOf(markers[i]) !== -1) return markers[i];
    }
    return "ok";
})()
"""


# Extract the latest generated image as a base64 PNG via canvas. Returns a data URL.
# Use image_src_js() if you can just fetch the URL directly — it's much cheaper.
IMAGE_TO_BASE64_JS = """
(async function() {
    var imgs = document.querySelectorAll('img[alt="Generated image"]');
    if (imgs.length === 0) return "no_img";
    var img = imgs[imgs.length - 1];
    if (!(img.complete && img.naturalWidth > 0)) return "not_loaded";
    try {
        var canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        return canvas.toDataURL('image/png');
    } catch (e) {
        return "error:" + e.message;
    }
})()
"""


# Upload a file into ChatGPT's hidden file input (for reference image attachment).
UPLOAD_FILE_JS_TEMPLATE = """
(function() {
    var input = document.querySelector('input[type="file"]');
    if (!input) return "no_file_input";
    var data = atob(%(b64_json)s);
    var bytes = new Uint8Array(data.length);
    for (var i = 0; i < data.length; i++) bytes[i] = data.charCodeAt(i);
    var blob = new Blob([bytes], { type: %(mime_json)s });
    var file = new File([blob], %(name_json)s, { type: %(mime_json)s });
    var dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return "attached";
})()
"""


def upload_file_js(file_path: str, mime: str = "image/jpeg") -> str:
    """JS that attaches `file_path` into ChatGPT's file input."""
    p = Path(file_path)
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return UPLOAD_FILE_JS_TEMPLATE % {
        "b64_json": json.dumps(b64),
        "mime_json": json.dumps(mime),
        "name_json": json.dumps(p.name),
    }


# Stabilization marker: return a combined signature (src + naturalWidth) used to
# detect when the latest generated image has stopped changing across polls.
IMAGE_SIGNATURE_JS = """
(function() {
    var imgs = document.querySelectorAll('img[alt="Generated image"]');
    if (imgs.length === 0) return "none";
    var img = imgs[imgs.length - 1];
    return (img.src || "") + "|" + img.naturalWidth + "x" + img.naturalHeight;
})()
"""
