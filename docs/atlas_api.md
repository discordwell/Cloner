# Atlas API v8.0 Reference

**Provider:** North Model Labs
**Base URL:** `https://api.atlasv1.com`
**Docs:** https://www.northmodellabs.com/api

## Authentication

All endpoints except `GET /` and `/v1/health` require Bearer token:

```
Authorization: Bearer <api_key>
```

Env var: `ATLAS_API_KEY`

## Pricing

- **Offline generation:** $4/hour of output video ($0.067/min, $0.0011/sec)
- **Realtime sessions:** $4/hour prorated per second
- **Billing:** Pay-as-you-go; enterprise plans available

## Limits & Constraints

| Constraint | Value |
|------------|-------|
| Max upload size | 50 MB combined |
| Server processing timeout (video) | 300s (5 min) |
| Max retries on failure | 3 |
| Rate limit (default) | 30 RPM |
| Job result availability | 24 hours after completion |
| Video output format | MP4 |
| Typical video generation | 40-50s |
| Realtime session max duration | 1 hour (configurable) |
| Face image max size (PATCH) | 10 MB |
| face_url max length | 2048 chars |

## Supported Formats

- **Audio input:** WAV, MP3, MPEG, OGG, WebM
- **Image input:** PNG, JPEG, WebP
- **Video output:** MP4

---

## Offline (Async) Video Generation

### 3-Step Flow

1. **Submit:** `POST /v1/generate` with audio + image -> 202 `{job_id}`
2. **Poll:** `GET /v1/jobs/{id}` until status = `completed` or `failed`
3. **Download:** `GET /v1/jobs/{id}/result` -> presigned URL valid 24 hours

### POST /v1/generate

Submit lip-sync avatar video generation job.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Details |
|-------|------|----------|---------|
| audio | file | yes | WAV, MP3, MPEG, OGG, WebM |
| image | file | yes | PNG, JPEG, WebP |
| X-Callback-URL | header | no | Webhook URL for async notification |

**Response 202 Accepted:**
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "pending",
  "message": "Job accepted. Poll GET /v1/jobs/a1b2c3d4e5f6 for status."
}
```

**cURL Example:**
```bash
curl -X POST "https://api.atlasv1.com/v1/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "X-Callback-URL: https://yourapp.com/webhook/atlas" \
  -F "audio=@speech.mp3" \
  -F "image=@face.jpg"
```

**Python Example:**
```python
import requests, time

API_KEY = "YOUR_API_KEY"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Submit
job = requests.post(
    "https://api.atlasv1.com/v1/generate",
    headers=headers,
    files={
        "audio": ("speech.mp3", open("speech.mp3", "rb"), "audio/mp3"),
        "image": ("face.jpg", open("face.jpg", "rb"), "image/jpeg"),
    },
).json()

job_id = job["job_id"]

# Poll until complete
while True:
    status = requests.get(
        f"https://api.atlasv1.com/v1/jobs/{job_id}",
        headers=headers,
    ).json()
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        raise Exception(f"Job failed: {status['error']}")
    time.sleep(2)

# Download
result = requests.get(
    f"https://api.atlasv1.com/v1/jobs/{job_id}/result",
    headers=headers,
).json()

video = requests.get(result["url"])
with open("output.mp4", "wb") as f:
    f.write(video.content)
```

---

## Job Queue Management

### GET /v1/jobs

List recent jobs, newest first. Paginated.

| Param | Type | Default | Max |
|-------|------|---------|-----|
| limit | int | 20 | 100 |
| offset | int | 0 | -- |

**Response 200:**
```json
{
  "jobs": [
    {
      "job_id": "a1b2c3d4e5f6",
      "type": "video",
      "status": "completed",
      "created_at": "2026-03-25T16:47:07Z",
      "completed_at": "2026-03-25T16:47:52Z",
      "latency_ms": 43000,
      "output_duration": null,
      "input_chars": null,
      "error_code": null
    }
  ],
  "count": 42,
  "limit": 20,
  "offset": 0
}
```

### GET /v1/jobs/{id}

Poll status of specific job.

**Job Statuses:**

| Status | Meaning |
|--------|---------|
| pending | Queued, waiting to process |
| processing | Actively being processed |
| completed | Output ready to download |
| failed | Check error field for details |

**Response 200 (completed):**
```json
{
  "job_id": "a1b2c3d4e5f6",
  "type": "video",
  "status": "completed",
  "queue_position": 0,
  "input": { "audio_size": 105000, "image_size": 52000 },
  "output": { "duration": null, "size_bytes": 4634000, "has_result": true },
  "error": null,
  "error_code": null,
  "timing": {
    "created_at": "2026-03-25T16:47:07Z",
    "started_at": "2026-03-25T16:47:09Z",
    "completed_at": "2026-03-25T16:47:52Z",
    "latency_ms": 43000
  },
  "url": "https://storage.example.com/jobs/.../output.mp4?...",
  "expires_in": 86400,
  "result_url": "/v1/jobs/a1b2c3d4e5f6/result"
}
```

### GET /v1/jobs/{id}/result

Get presigned download URL. Valid 24 hours. Requires job status = `completed`.

**Response 200:**
```json
{
  "url": "https://storage.example.com/jobs/.../output.mp4?...",
  "content_type": "video/mp4",
  "expires_in": 86400
}
```

**Error 409 (not ready):**
```json
{
  "error": "not_ready",
  "message": "Job is still processing. Poll GET /v1/jobs/{id} until status is completed."
}
```

---

## Account & Status

### GET /v1/me

```json
{
  "authenticated": true,
  "key_prefix": "sk_live_A7xQ",
  "name": "My API Key",
  "tier": "starter",
  "requests_used": 14,
  "rate_limit": {
    "requests_per_minute": 30,
    "remaining": 28,
    "resets_in": "42s"
  },
  "billing": "pay_as_you_go"
}
```

### GET /v1/status

```json
{
  "status": "operational",
  "services": {
    "avatar_generation": "available",
    "voice_synthesis": "available"
  }
}
```

Service values: `available` or `busy`

### GET /v1/health (no auth)

```json
{
  "status": "all systems go",
  "services": { "avatar_generation": "online" }
}
```

---

## Realtime Avatar API (WebRTC)

Live interactive avatars with sub-second latency over WebRTC via LiveKit.
**Mode:** passthrough (you provide audio, Atlas renders lip-synced video).

### POST /v1/realtime/session

Create session. Returns LiveKit room token and connection URL.

**Content-Type:** `application/json` or `multipart/form-data`

| Field | Type | Required | Details |
|-------|------|----------|---------|
| face_url | string | no | HTTPS URL of face image (max 2048 chars) |
| face | file | no | Face image file (PNG/JPEG/WebP, max 10 MB) |
| mode | string | no | `"passthrough"` (default) |

**Response 200:**
```json
{
  "session_id": "ses_x9y8z7w6v5u4...",
  "livekit_url": "wss://your-livekit-instance.livekit.cloud",
  "token": "<livekit_jwt_token>",
  "room": "atlas-rt-ses_x9y8z7w6v5u4...",
  "mode": "passthrough",
  "max_duration_seconds": 3600,
  "pricing": "$4/hour, prorated per second"
}
```

### POST /v1/realtime/session/{session_id}/viewer

Issue view-only LiveKit token for existing session. Zero GPU cost.

**Response 200:**
```json
{
  "session_id": "ses_a1b2c3d4e5f6...",
  "livekit_url": "wss://livekit.example.com",
  "token": "eyJ...",
  "room": "atlas-rt-ses_a1b2c3d4e5f6...",
  "viewer_id": "viewer-abc123def456",
  "role": "viewer",
  "permissions": {
    "can_publish": false,
    "can_subscribe": true,
    "can_publish_data": false
  }
}
```

### GET /v1/realtime/session/{session_id}

Get session status. Only accessible by API key that created it.

**Response 200 (active):**
```json
{
  "session_id": "ses_a1b2c3d4e5f6...",
  "status": "active",
  "room": "atlas-rt-ses_a1b2c3d4e5f6...",
  "started_at": "2026-04-01T01:30:00Z",
  "ended_at": null,
  "duration_seconds": 142.5,
  "max_duration_seconds": 3600
}
```

### PATCH /v1/realtime/session/{session_id}

Hot-swap avatar face during active session. File upload only (no URLs).

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Details |
|-------|------|----------|---------|
| face | file | yes | PNG/JPEG/WebP, max 10 MB |

**Response 200:**
```json
{
  "session_id": "ses_a1b2c3d4e5f6...",
  "face_updated": true,
  "metadata_pushed": true,
  "message": "Face image updated. The avatar will switch within seconds."
}
```

### DELETE /v1/realtime/session/{session_id}

End session. Records billing duration.

**Response 200:**
```json
{
  "session_id": "ses_a1b2c3d4e5f6...",
  "status": "ended",
  "duration_seconds": 322.0,
  "estimated_cost": "$0.8944",
  "credits_deducted_cents": 89
}
```

---

## Plugin: External LiveKit Room

### POST /v1/avatar/session

Use when you already have a LiveKit room and want Atlas to join.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Details |
|-------|------|----------|---------|
| livekit_url | string | yes | LiveKit server URL |
| livekit_token | string | yes | LiveKit auth token |
| room_name | string | yes | Room name to join |
| avatar_image | file | no | Avatar image file |

**Response 200:**
```json
{ "session_id": "ses_...", "status": "ok" }
```

---

## Webhooks

Pass `X-Callback-URL` header on `POST /v1/generate` to skip polling.

| Rule | Detail |
|------|--------|
| Protocol | HTTPS only (HTTP and localhost rejected) |
| Retries | 3 attempts with backoff (5s, 30s, 120s) |
| Timeout | 10s per attempt |
| Success | Any 2xx counts as delivered |

**Completed payload:**
```json
{
  "event": "job.completed",
  "job_id": "a1b2c3d4e5f6",
  "type": "video",
  "status": "completed",
  "url": "https://storage.example.com/jobs/.../output.mp4?...",
  "expires_in": 86400,
  "result_url": "https://api.atlasv1.com/v1/jobs/a1b2c3d4e5f6/result",
  "created_at": "2026-03-31T16:47:07+00:00",
  "completed_at": "2026-03-31T16:47:52+00:00"
}
```

**Failed payload:**
```json
{
  "event": "job.failed",
  "job_id": "a1b2c3d4e5f6",
  "type": "video",
  "status": "failed",
  "error_code": "generation_failed",
  "created_at": "2026-03-31T16:47:07+00:00",
  "completed_at": "2026-03-31T16:47:40+00:00"
}
```

### Verifying Webhook Signatures

Headers: `X-Atlas-Signature`, `X-Atlas-Timestamp`

```python
import hmac, hashlib

def verify_atlas_webhook(body: bytes, signature: str, timestamp: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        f"{timestamp}.{body.decode()}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

---

## Error Codes

All errors return: `{"error": "error_code", "message": "Human-readable description"}`

| Code | Error | Description |
|------|-------|-------------|
| 400 | invalid_input | Empty or invalid audio/image file |
| 400 | invalid_mode | mode must be "passthrough" |
| 400 | invalid_face_url | HTTPS required or exceeds 2048 chars |
| 401 | unauthorized | Missing or malformed Authorization header |
| 403 | forbidden | Invalid API key |
| 404 | not_found | Endpoint or job not found |
| 404 | no_output | No stored output available |
| 409 | not_ready | Downloading result before job completes |
| 409 | already_ended | Realtime session already ended |
| 409 | session_not_active | Session ended or not active |
| 413 | payload_too_large | Upload exceeds 50 MB |
| 415 | unsupported_media_type | Format not supported |
| 422 | validation_error | Missing required fields |
| 429 | rate_limit_exceeded | Rate limit hit (includes retry_after_seconds) |
| 429 | monthly_cap_exceeded | Monthly request cap reached |
| 500 | internal_error | Unexpected server error |
| 502 | generation_failed | Generation failed after retries |
| 503 | queue_unavailable | Job queue temporarily unavailable |
| 503 | storage_unavailable | Output storage temporarily unavailable |
| 503 | no_capacity | All GPU pods busy (retry after 30s) |

---

## React SDK

**Package:** `@northmodellabs/atlas-react`
**Install:** `npm install @northmodellabs/atlas-react livekit-client`
**Hook:** `useAtlasSession()`

See full React/LiveKit integration examples at:
- https://github.com/NorthModelLabs/atlas-offline-example
- https://github.com/NorthModelLabs/atlas-realtime-example
