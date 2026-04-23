# Autopitch

Procedurally generate a ~60-second Pixar-style cartoon video of a prospect
pitching AI consulting services to themselves, from just a name + company URL.

## What it does

Given `{name, url, [extras]}` for a prospect, the pipeline produces:

- A cartoon portrait of them (Pixar style, via ChatGPT image generation)
- A cartoon scene of them standing in front of their business (with their logo)
- A pitch script specific to their company (website analysis → AI-opportunity hypothesis → script)
- A cloned voice (from YouTube/podcast audio if findable, else a library voice)
- A final MP4 (Atlas lip-sync puts the cloned voice onto the cartoon scene)

Self-evidently synthetic — the recipient is also the subject, so they
immediately see it's an AI cartoon, not a deepfake of them.

## Architecture

- **Blueprint** (`runs/{slug}/blueprint.md`): per-prospect state file with YAML
  frontmatter. All assets and stage statuses live here.
- **Scripts** (`autopitch/scripts/`): Python modules that do the deterministic
  work — scraping, image search, voice download/diarization, TTS, Atlas lip-sync.
- **Skills** (`.claude/skills/autopitch-*/`): reusable LLM prompt modules.
- **Agent** (`.claude/agents/autopitch.md`): a Claude Code subagent that
  orchestrates the pipeline and drives ChatGPT via MCP chrome tools for the
  cartoonify stages.

## Setup

1. Install new deps from the repo root:
   ```bash
   pip install -r requirements.txt
   ```
   For higher-quality voice isolation (optional), also install pyannote:
   ```bash
   pip install "pyannote.audio>=3.1.1"
   ```
   and set `HUGGINGFACE_TOKEN` in `.env` after accepting the model licenses on
   HuggingFace.

2. Add API keys to `.env` (see `.env.example`):
   ```
   BING_IMAGE_SEARCH_KEY=...          # or SERPAPI_KEY
   HUGGINGFACE_TOKEN=...               # optional, for diarization
   OPENAI_API_KEY=...                  # already present
   ELEVENLABS_API_KEY=...              # already present
   ATLAS_API_KEY=...                   # already present
   ```

3. Open `https://chatgpt.com` in your Chrome and log in — the autopitch agent
   will use your existing session via Claude's MCP chrome tools. Make sure the
   model selector shows standard GPT-4o / GPT-5 (NOT "Pro" / extended
   thinking).

## Usage

### Step 1: create the blueprint

```bash
python -m autopitch.scripts.run \
    --name "Jane Doe" \
    --url https://acme.example.com
```

With extras:

```bash
python -m autopitch.scripts.run \
    --name "Jane Doe" \
    --url https://acme.example.com \
    --company "Acme Widgets" \
    --linkedin https://linkedin.com/in/janedoe \
    --role CEO \
    --region midwest_us \
    --duration 45
```

Creates `autopitch/runs/jane-doe-acme/blueprint.md`.

### Step 2: run the agent

In a Claude Code session, just say:

> Run autopitch for `autopitch/runs/jane-doe-acme`.

The agent will walk all ten stages, updating `blueprint.md` as it goes,
driving ChatGPT for cartoonify, and finishing with
`autopitch/runs/jane-doe-acme/final.mp4`.

Or non-interactively:

```bash
claude --agent autopitch "Run autopitch for autopitch/runs/jane-doe-acme"
```

## Directory layout

```
autopitch/
├── README.md                (this file)
├── __init__.py
├── scripts/                 (importable + CLI-invokable)
│   ├── run.py               — blueprint creator
│   ├── scrape_site.py       — homepage → site.txt, logo.png
│   ├── find_photo.py        — Bing/SerpAPI → photo_raw.jpg
│   ├── find_voice.py        — yt-dlp + diarization → voice_sample.wav
│   ├── analyze_site.py      — GPT → hypothesis.md
│   ├── write_pitch.py       — GPT → pitch.txt
│   ├── tts.py               — ElevenLabs clone + speak
│   ├── lipsync.py           — Atlas lip-sync
│   ├── image_utils.py       — rembg, crop, QC (ported from clawed-command)
│   ├── chatgpt_mcp.py       — JS snippets for MCP chrome automation
│   └── _llm.py              — shared OpenAI wrapper
├── prompts/
│   ├── cartoonify_portrait.txt
│   ├── cartoonify_scene.txt
│   ├── analyze_site.txt
│   └── pitch_script.txt
└── runs/
    └── {slug}/              — per-prospect artifacts (gitignored)
```

## Running individual stages

Every script is CLI-invokable — useful for debugging one stage at a time:

```bash
python -m autopitch.scripts.scrape_site --url https://acme.com --run autopitch/runs/jane-doe-acme
python -m autopitch.scripts.find_photo  --name "Jane Doe" --company Acme --run autopitch/runs/jane-doe-acme
python -m autopitch.scripts.find_voice  --name "Jane Doe" --run autopitch/runs/jane-doe-acme
python -m autopitch.scripts.analyze_site --run autopitch/runs/jane-doe-acme --company Acme --url https://acme.com
python -m autopitch.scripts.write_pitch  --run autopitch/runs/jane-doe-acme --name "Jane Doe" --company Acme
python -m autopitch.scripts.tts clone   --run autopitch/runs/jane-doe-acme --name "Jane Doe"
python -m autopitch.scripts.tts speak   --run autopitch/runs/jane-doe-acme
python -m autopitch.scripts.lipsync     --run autopitch/runs/jane-doe-acme
```

## Troubleshooting

- **"no frontal face found"** — the Bing search didn't return an image with a
  clearly-visible face. Try passing `--company` explicitly (narrows the query),
  or download a photo manually to `{run_dir}/photo_raw.jpg` and re-run from
  the cartoonify stage.

- **"no voice sample found"** — no interview/podcast match on YouTube. The
  pipeline will fall back to an ElevenLabs library voice automatically; you
  can bias the pick with `--gender`, `--region`, `--age`.

- **ChatGPT produces a photorealistic image instead of Pixar** — the prompt
  explicitly asks for Pixar style. If it drifts, check you're not in Pro mode
  and re-run the cartoonify stage.

- **Atlas lip-sync looks off** — the scene image may have the face too small
  or obstructed. Regenerate the scene with a tighter crop; Atlas prefers
  head-and-shoulders framing with the mouth unambiguous.

## Tests

```bash
pytest tests/autopitch/ -v
```

Tests cover pure logic (slug, blueprint, HTML parsing, speaker selection,
prompt assembly, image QC, JS snippet assembly). End-to-end runs hit live
APIs and are done manually via the wet-test protocol in `CLAUDE.md`.
