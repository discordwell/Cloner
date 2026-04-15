# Cloner Project

## Overview
Real-time human clone system: captures a subject's voice + appearance, then generates talking-head video responses on demand.

## Architecture

### Pipeline (current)
1. **Voice cloning:** ElevenLabs API clones subject voice from audio sample
2. **TTS:** ElevenLabs generates speech audio from text using cloned voice
3. **Lip-sync video:** Atlas API (North Model Labs) takes TTS audio + face image -> lip-synced MP4
4. **Scene video:** Sora (browser automation) or Kling (FAL API) for background/scene generation
5. **Orchestration:** OBS WebSocket for scene control, tkinter GUI controller

### Atlas Integration (lip-sync)
- Atlas replaces the old viseme-based lip sync pipeline (MediaPipe + InsightFace compositing)
- Offline API: audio + image -> submit job -> poll -> download MP4 (~40-50s)
- Realtime API: WebRTC via LiveKit for live interactive avatars
- API docs: `docs/atlas_api.md`
- Client: `src/video/atlas_client.py`

### Key Config
- `config/config.yaml` — all service settings
- `.env` — API keys (ATLAS_API_KEY, ELEVENLABS_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)

## Dev Notes
- Python 3.12
- The viseme pipeline (`src/viseme/`) is legacy; Atlas is the preferred lip-sync path
- Cross-platform: macOS primary, Windows paths still supported
