# Cloner Architecture

Real-time human clone system: captures a subject's voice + face, then generates talking-head video responses on demand.

## System Overview

```
User Input (text/question)
        |
        v
   [LLM Response]  (Anthropic Claude / OpenAI GPT)
        |
        v
   [Voice Clone TTS]  (ElevenLabs API)
        |
        v
   [Lip-Sync Video]  (Atlas API)  <-- audio + face image -> MP4
        |
        v
   [Playback / OBS Scene]
```

## Components

### Voice Pipeline
- **ElevenLabs Client** (`src/voice/elevenlabs_client.py`) — voice cloning + TTS
- **Voice Cloning Service** (`src/voice/voice_cloning_service.py`) — higher-level voice management
- **Audio Processor** (`src/voice/audio_processor.py`) — audio preprocessing

### Lip-Sync Video (Atlas — primary)
- **Atlas Client** (`src/video/atlas_client.py`) — North Model Labs Atlas API
  - Offline: audio + face image -> submit job -> poll -> download MP4 (~40-50s)
  - Realtime: WebRTC via LiveKit (sub-second latency)
  - $4/hr of output video
- API docs: `docs/atlas_api.md`

### Lip-Sync Video (Viseme — legacy)
- **Viseme Library** (`src/viseme/viseme_library.py`) — builds mouth shape library from video
- **TTS Viseme** (`src/viseme/tts_viseme.py`) — TTS with viseme timing data
- **Compositor** (`src/viseme/realtime_compositor.py`, `enhanced_compositor.py`) — frame compositing

### Scene Video Generation
- **Sora Client** (`src/video/sora_client.py`) — browser automation for Sora
- **Kling Client** (`src/video/kling_client.py`) — FAL API for Kling models
- **Video Backend Factory** (`src/video/video_backend.py`) — factory for all backends

### Streaming Pipeline (`src/streaming/`)
- **StreamingPipeline** (`pipeline.py`) — orchestrator: question -> video frames in ~2.7s
- **SentenceSplitter** (`sentence_splitter.py`) — splits LLM token stream on sentence boundaries
- **LLMStreamer** (`llm_streamer.py`) — streaming Anthropic/OpenAI with sentence splitting
- **TTSStreamer** (`tts_streamer.py`) — ElevenLabs streaming TTS per sentence
- **AtlasRealtimeSession** (`atlas_realtime.py`) — LiveKit WebRTC session for real-time lip-sync

### Orchestration (legacy)
- **Clone Pipeline** (`src/clone_pipeline.py`) — offline: text -> talking video (Atlas or viseme)
- **Realtime Clone** (`src/realtime_clone.py`) — real-time response system (offline Atlas)
- **Clone Interview** (`src/clone_interview.py`) — meeting capture + auto-response
- **Clone Controller** (`scripts/clone_controller.py`) — tkinter GUI + OBS control

### Infrastructure
- **Config Loader** (`src/utils/config_loader.py`) — YAML config with env var expansion
- **Config** (`config/config.yaml`) — all service settings
- **Meeting Capture** (`src/capture/`) — browser-based meeting join + recording

## Data Flow: Text -> Talking Video

### Streaming Pipeline (default, ~2.7s to first frame)
```
User speaks -> Whisper transcription (~1s)
  -> LLM streaming tokens (~500ms TTFB)
  -> SentenceSplitter (accumulate until sentence boundary)
  -> ElevenLabs TTS per sentence (~500ms TTFB)
  -> Publish audio to Atlas Realtime WebRTC session
  -> Receive lip-synced video frames (~100ms)
  -> Display / OBS
```

Thread architecture: 4 parallel workers connected by queues:
- LLM Streamer -> tts_queue -> TTS Worker -> audio_queue -> Audio Publisher -> LiveKit -> Video Frames

### Atlas Offline Path (fallback)
```
1. Text -> ElevenLabs TTS -> audio.mp3
2. audio.mp3 + face.png -> Atlas API -> lip-synced.mp4
```

### Viseme Path (legacy)
```
1. Text -> ElevenLabs TTS w/ viseme timing -> audio.mp3 + viseme_events[]
2. viseme_events + viseme_library -> compositor -> frames[]
3. frames + audio -> ffmpeg -> video.mp4
```

## Configuration

All config in `config/config.yaml`. API keys in `.env`:
- `ATLAS_API_KEY` — lip-sync video (Atlas)
- `ELEVENLABS_API_KEY` — voice cloning + TTS
- `OPENAI_API_KEY` — LLM responses + Whisper transcription
- `ANTHROPIC_API_KEY` — Claude LLM responses

## Key Design Decisions

1. **Atlas over viseme compositing**: Atlas produces higher quality lip-sync with zero local compute. Trade-off is ~40-50s latency per clip and API cost ($4/hr), but quality is significantly better than local viseme blending.

2. **ElevenLabs for voice**: Best-in-class voice cloning quality. Audio is generated first, then fed to Atlas — the two services are independent.

3. **Backend factory pattern**: `get_video_backend()` abstracts scene video providers. `lipsync_backend` config controls Atlas vs viseme. Both pipelines coexist for fallback.

4. **Cross-platform**: Originally Windows, now macOS primary. Path handling uses `sys.platform` checks.
