# M5StickS3 Voice Terminal — Status Quo

Date: 2026-05-13

## Architecture

```
[M5StickS3]
  BtnA held
  → record 16 kHz mono PCM until release (up to 10s; minimum ~300ms)
  → wrap as WAV
  → POST /voice-command
        │
        ▼
[FastAPI server :8010]
  → mlx-whisper STT (mlx-community/whisper-tiny)
  → job queued to agent_jobs.json
        │
        ▼
[agent_worker.py]
  → pi minimax/MiniMax-M2.7-highspeed (thinking=off)
  → tools: read,bash,grep,find,ls (internet capable)
  → POST /agent/jobs/{id}/result
        │
        ▼
[M5StickS3]
  polls GET /agent/jobs/{id} every 1.5s
  → status=done → display result_text/image on LCD and play audio_url WAV when present
```

## Current Runtime State

| Process | Port | PID | Status |
|---|---|---|---|
| `uvicorn server.app:app` | 8010 | 81140 | running |
| `agent_worker.py --base-url http://127.0.0.1:8010` | — | 91467 | running |

**Endpoints:**

```
GET  /health                       → always 200
GET  /models                       → stt/tts backend info
POST /command                      → text command, returns job_id
POST /voice-command                → audio upload, STT, returns job_id
GET  /agent/jobs                   → all jobs (optional ?status=queued)
GET  /agent/jobs/next?worker=pi   → claim oldest queued job
GET  /agent/jobs/{id}             → job detail including result_text, sentiment, audio_url
POST /agent/jobs/{id}/result      → post result from agent; server generates Supertonic WAV
GET  /audio/{job_id}              → download generated WAV response audio; audio id equals job id
GET  /image/{job_id}              → download generated RGB565 sentiment image; image id equals job id
```

Completed job response shape now includes:

```json
{
  "status": "done",
  "result_text": "Audio response JSON ready",
  "sentiment": "happy",
  "image_url": "/image/<job_id>",
  "audio_url": "/audio/<job_id>"
}
```

## How to Run / Restart

```sh
# Server
cd m5-voice-terminal
env STT_BACKEND=mlx-whisper MLX_WHISPER_MODEL=mlx-community/whisper-tiny \
  uvicorn server.app:app --host 0.0.0.0 --port 8010

# Agent worker (separate terminal)
env PI_WORKER_MODEL=minimax/MiniMax-M2.7-highspeed \
    PI_WORKER_THINKING=off \
    PI_WORKER_TOOLS=read,bash,grep,find,ls \
    python3 scripts/agent_worker.py --base-url http://127.0.0.1:8010

# Logs
tail -f /tmp/m5-agent-worker.log
tail -f /tmp/m5-voice-terminal.log
```

## Firmware Status

**Active firmware:** `firmware/m5sticks3-arduino/M5VoiceTerminal/M5VoiceTerminal.ino`

- Board: M5StickS3 (`m5stack:esp32:m5stack_sticks3`)
- Libraries: M5Unified 0.2.14, M5GFX 0.2.20, ArduinoJson 7.4.3
- Upload: `./scripts/arduino_upload.sh /dev/cu.usbmodem213301`
- Requires download mode if esptool fails: hold side reset until green LED flashes

**Legacy:** `stick/main.py` (MicroPython) is retained as fallback reference.

## Model Choices

| Component | Default | Env var |
|---|---|---|
| STT | mlx-whisper / `mlx-community/whisper-tiny` | `STT_BACKEND`, `MLX_WHISPER_MODEL` |
| STT fallback | openai-whisper / `base` | `WHISPER_MODEL` |
| Pi model | minimax/MiniMax-M2.7-highspeed | `PI_WORKER_MODEL` |
| Pi thinking | off | `PI_WORKER_THINKING` |

## Safety Posture

- `edit` and `write` tools **NOT** enabled in voice worker.
- `read,bash,grep,find,ls` tools enabled — worker can inspect and run safe shell commands.
- No destructive commands (delete, clean, reset-hard, push, deploy) auto-executed; worker asks confirmation.
- Wi-Fi password and API keys are gitignored in `stick/config.py` and `firmware/**/config.h`.

## Timing Metrics (server + worker)

Each completed job records in `job.metrics`:

```json
{
  "audio_bytes": 64782,
  "server_upload_read_s": 0.0,
  "server_stt_s": 1.992,
  "server_voice_command_s": 1.993,
  "agent_worker_s": 20.104,
  "agent_worker_mode": "pi"
}
```

Current bottleneck: agent response generation (~6–25s). STT is fast (<2s warm).

## Known Issues

- Server default port changed to **8010** (8000 already in use by another service); older helper defaults may still need explicit `--base-url` if not updated.
- First mlx-whisper run is slow due to model download; subsequent runs are fast.
- Audio playback speed issue was traced to writing Supertonic output as 24 kHz. Server now writes PCM_16 WAV at Supertonic's native sample rate, currently 44.1 kHz.
- The server deletes old generated WAV/image files whenever a new response artifact is generated; the Stick downloads artifacts into PSRAM and frees them after playback/display.

## Next Steps

1. **VAD / silence filter** — reject audio that is too quiet or contains no speech to reduce hallucination.
2. **Real mic benchmark** — capture and test actual StickS3 microphone recordings vs synthetic samples.
3. **Hold-to-talk UX** — tune maximum duration, minimum duration, and progress indication after real-device testing.
4. **Reduce agent latency** — compare MiniMax-M2.7-highspeed vs other model aliases; test with tools disabled for comparison.
5. **Runtime runbook** — replace volatile PID/status rows with a repeatable process supervisor or check script.

## Project Structure

```
m5-voice-terminal/
├── server/app.py              FastAPI server with STT + queue + TTS
├── scripts/
│   ├── agent_worker.py        Background Pi worker (active)
│   ├── agent_inbox.py         Manual CLI for job inspection/completion
│   ├── arduino_upload.sh      Firmware compile + upload helper
│   ├── benchmark_stt.py      STT backend benchmark script
│   ├── upload_voice_sample.py  Test audio upload to server
│   └── make_stt_samples.sh     Generate synthetic test WAV files
├── firmware/m5sticks3-arduino/M5VoiceTerminal/
│   ├── M5VoiceTerminal.ino   Active Arduino/M5Unified firmware
│   └── config.h.example       Template (Wi-Fi/secrets not committed)
├── stick/                     MicroPython fallback client
│   ├── main.py, st7789.py, m5pm1.py
│   ├── config.py               Local secrets (gitignored)
│   └── config.example.py      Shareable template
├── docs/
│   ├── stt-review.md          STT model benchmark analysis
│   └── status-quo.md          This file
├── samples/                   Synthetic WAV test samples
└── data/agent_jobs.json       Job queue (runtime artifact)
```