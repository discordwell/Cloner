# Claudepad — Cloner Project

## Session Summaries

### 2026-04-23T12:55Z — Added autopitch subproject

Built `autopitch/` — a self-contained subproject that procedurally generates
~60s Pixar-style cartoon pitch videos from `{name, company_url}`.

**Architecture chosen:** blueprint (per-prospect YAML) + Python scripts +
Claude Code skills + orchestrator subagent. The agent drives ChatGPT via MCP
chrome tools for cartoonify (clawed-command pattern), and calls Python scripts
for everything else.

**Stages (10):** scrape_site → find_photo (Bing + mediapipe face) →
find_voice (yt-dlp + pyannote diarization, library-voice fallback) →
clone_voice (ElevenLabs, skipped when library fallback) → cartoonify_portrait
(ChatGPT web) → cartoonify_scene (ChatGPT web + logo attached) → analyze_site
(GPT-5.4) → write_pitch (GPT-5.4) → tts (ElevenLabs) → lipsync (Atlas).

**New code paths:**
- `autopitch/scripts/*.py` (13 modules)
- `autopitch/prompts/*.txt` (4 templates)
- `.claude/agents/autopitch.md` (orchestrator)
- `.claude/skills/autopitch-{analyze-site,write-pitch,cartoonify-prompt}/SKILL.md`
- `tests/autopitch/test_*.py` — 43 tests passing, 1 conditionally skipped (mediapipe)

**Reuse:** directly wraps `src/voice/elevenlabs_client.py` (clone + TTS) and
`src/video/atlas_client.py` (lip-sync). Ported `image_utils.py` from
clawed-command.

**Review + fixes applied:**
- Replaced agent's shell `curl` with `autopitch/scripts/download.py` to
  eliminate URL-shell-quoting risk.
- Refactored `build_blueprint` to use `yaml.safe_dump` — previously
  hand-formatted, broke on names containing colons/quotes.
- Added tests for `_llm` (missing-key path) and `find_photo._detect_best_face`
  (synthetic no-face images; skipped when mediapipe not installed).

**Config + deps:**
- `config/config.yaml` has an `autopitch:` block (documentation only; scripts
  read env vars directly — wire up later if a script grows knobs).
- `requirements.txt` adds beautifulsoup4, yt-dlp, rembg. `pyannote.audio`
  listed as optional install-by-hand.
- `.env.example` adds BING_IMAGE_SEARCH_KEY, HUGGINGFACE_TOKEN.
- `.gitignore` excludes `autopitch/runs/*` (per-prospect artifacts).

**Not yet done:** end-to-end wet test against a real prospect. That requires
the user to have ChatGPT logged in in Chrome and API keys in `.env`.

## Key Findings

### Autopitch — cartoonify method
Use MCP chrome tools (`mcp__claude-in-chrome__*`) to drive `chatgpt.com`, not
Playwright. This matches clawed-command's asset pipeline and avoids a second
browser-automation stack. Model selector must be **standard GPT-4o/5, NOT
Pro/extended-thinking** — Pro rewrites prompts and produces grids, breaking
the "generate ONLY this one image" guardrail. This guardrail is baked into
the prompt text and repeated in the agent spec.

### Autopitch — voice search
yt-dlp `ytsearch3:"{name}" interview|podcast|keynote|talk` → ffmpeg transcode
to mono 16kHz WAV → pyannote diarization (optional; HF-gated) → pick the
speaker with most airtime (solo-interview heuristic) → extract sorted
segments until target_s reached. Fallback when pyannote unavailable: longest
continuous non-silent chunk via `pydub.silence.detect_nonsilent`. Last
resort: `pick_library_voice()` scores ElevenLabs voices by matching
gender/region/age hints in description text.

### Autopitch — Atlas lip-sync requirements
Atlas needs the face clearly visible and the mouth unobstructed. The
`cartoonify_scene.txt` prompt explicitly requires face ≥25% of image height
and mouth unobstructed, because Atlas modifies only the mouth region of the
source image.
