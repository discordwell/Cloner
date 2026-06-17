# Claudepad — Cloner Project

## Session Summaries

### 2026-06-17T09:30Z — Autopitch robustness: find_voice zero-length segment bug, write_pitch hardening

Follow-on maintenance pass on autopitch. Repo green at 182 tests (was 173).

- **find_voice zero-length segment (real bug):** `segments_for_speaker` and
  `longest_speech_heuristic` shared an accumulation loop that, when a span filled
  `target_s` *exactly*, appended the next span as a zero-length `(s, s)` trim.
  That becomes a degenerate `atrim=start=s:end=s` (empty input) in
  `extract_segments`' ffmpeg concat filtergraph. Extracted a shared
  `_take_until_target()` helper that stops cleanly when the remaining budget is
  ≤ a 1ms floor (matching ffmpeg's `:.3f` precision) and also skips degenerate
  input spans. Both functions now route through it. Added 6 tests (regression +
  helper coverage).
- **write_pitch hardening:** finished the thread the prior pass flagged —
  `build_prompt` used `str.format()`, which is one template-edit away from the
  same `KeyError` that broke `analyze_site`. Switched to the identical
  single-pass placeholder-substitution approach (regex over the 8 known names),
  so example braces in the template and braces in the LLM hypothesis pass through
  untouched. Also fixed an `IndexError` on an empty `name` (`name.split()[0]`).
  Verified happy-path output is byte-identical to the old `str.format`. Added 3
  robustness tests.

### 2026-06-17T08:38Z — Autopitch maintenance: face-detect modernization, analyze_site fix, test hygiene

Maintenance pass on the autopitch pipeline. Repo green at 173 tests.

- **find_photo face detection** (landed in-progress work): mediapipe ≥0.10.21
  removed the legacy `mp.solutions` API that `_detect_best_face` used, so on the
  installed 0.10.32 the stage was broken. Reworked to the mediapipe Tasks API
  (BlazeFace, model cached in `data/models/`, downloaded on first use) with an
  OpenCV Haar-cascade fallback for offline/unavailable cases. Added tests for the
  area threshold, conf×area picking, the mediapipe→opencv fallback boundary, and
  the model cache.
- **analyze_site crash (critical):** `analyze()` did `template.format(...)` on a
  prompt whose example skeleton has literal braces (`{short title}`,
  `{2-3 sentences …}`) → `KeyError: 'short title'` every run. Extracted a pure
  `build_prompt()` that substitutes only `{company}/{url}/{site_text}` via a
  single-pass regex; immune to braces in scraped text. Added regression tests.
- **Build hygiene:** a bare `pytest` at the repo root aborted because the manual
  smoke scripts `scripts/test_{sora,person_descriptor}.py` (live-service scripts
  named `test_*.py`) were collected and `test_sora.py` failed on missing
  playwright. Added `pytest.ini` with `testpaths = tests`.

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

### Autopitch — never `str.format()` a prompt template that shows example braces
`analyze_site.txt` (and any prompt that gives the LLM an output skeleton) embeds
literal `{…}` to mean "put your text here". `template.format(**kw)` treats those
as fields and raises `KeyError`, and it also breaks on `{ }` in scraped/LLM text
passed as values. Substitute only the known placeholders (single-pass regex),
leaving other braces literal. Both prompt-fillers now do this:
`analyze_site.build_prompt` and `write_pitch.build_prompt` each compile a
`_PLACEHOLDERS` regex of just their known field names and `re.sub` once — so the
pattern is consistent and a future example brace in either template is safe. Any
new prompt-assembly script must follow the same rule; do **not** reintroduce
`str.format()` on a template that doubles as an output skeleton.

### Autopitch — mediapipe ≥0.10.21 dropped `mp.solutions`
The legacy `mp.solutions.face_detection` / `face_mesh` API is gone in modern
mediapipe (0.10.32 installed). Use the Tasks API (`mediapipe.tasks.python.vision`),
which needs an explicit `.tflite` model asset (not bundled) — `find_photo.py`
caches BlazeFace under `data/models/` and downloads on first use, falling back to
an OpenCV Haar cascade. The legacy `src/viseme/` still uses `mp.solutions` but is
guarded by try/except (legacy path, Atlas is preferred).

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
