# M5StickS3 Voice Terminal MVP

A voice-command terminal for M5StickS3. The active firmware records a hold-to-talk microphone clip, uploads it to a local FastAPI server, the server transcribes it, queues it for a Pi/agent worker, generates Supertonic WAV audio after the worker replies, and the Stick displays either a bundled sentiment face or a generated image while playing the result.

## Current flow

```text
M5StickS3 BtnA
  -> hold button to record up to 10s of 16 kHz mono PCM
  -> wrap as WAV
  -> POST /voice-command
  -> server STT (empty transcript short-circuits to "No howl heard")
  -> queued agent job
  -> scripts/agent_worker.py handles prompt, with recent same-device history
  -> POST /agent/jobs/{id}/result
  -> server optionally generates RGB565 image, then Supertonic WAV
  -> Stick polls job, displays generated image or bundled face/result, downloads audio, plays WAV
  -> optional BtnB/BtnA menu sends follow-up /command choices
```

The legacy MicroPython client in `stick/` is retained as a fallback text-command reference. The active device firmware is the Arduino/M5Unified sketch in `firmware/m5sticks3-arduino/M5VoiceTerminal/`.

## Server

```sh
cd m5-voice-terminal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Default if STT_BACKEND is unset is whisper/base.
# Recommended current runtime on Apple Silicon:
env STT_BACKEND=mlx-whisper MLX_WHISPER_MODEL=mlx-community/whisper-tiny \
  uvicorn server.app:app --host 0.0.0.0 --port 8010
```

Find the Mac LAN IP:

```sh
ipconfig getifaddr en0
```

Then copy the firmware config template and set the server URL:

```sh
cp firmware/m5sticks3-arduino/M5VoiceTerminal/config.h.example \
  firmware/m5sticks3-arduino/M5VoiceTerminal/config.h
```

```c
#define VOICE_URL "http://<mac-lan-ip>:8010/voice-command"
```

`config.h` contains local Wi-Fi secrets and is ignored by git.

## Agent worker

Voice/text commands are queued for a local Python worker. By default the worker uses `VOICE_WORKER_BACKEND=pi-sdk-full`, invoking Pi programmatically from Python with extensions/skills/prompts/context files loaded. Set `VOICE_WORKER_BACKEND=pi-sdk` for a faster minimal SDK path, `VOICE_WORKER_BACKEND=minimax-direct` for the direct MiniMax shortcut, or `VOICE_WORKER_BACKEND=pi` to use the slower `pi -p` CLI path.

In `pi-sdk-full` mode, if `PI_WORKER_TOOLS` is unset, the worker currently enables `web_fetch,web_search,read,bash,grep,find,ls`.

```sh
env VOICE_WORKER_BACKEND=pi-sdk-full \
    PI_WORKER_MODEL=minimax/MiniMax-M2.7-highspeed \
    PI_WORKER_THINKING=off \
    PI_WORKER_TOOLS=web_fetch,web_search,read,bash,grep,find,ls \
    python3 scripts/agent_worker.py --base-url http://127.0.0.1:8010
```

Manual polling helpers:

```sh
./scripts/agent_inbox.py --base-url http://127.0.0.1:8010 next
./scripts/agent_inbox.py --base-url http://127.0.0.1:8010 done <job_id> "Result text to show/play back"
./scripts/agent_inbox.py --base-url http://127.0.0.1:8010 list
```

HTTP endpoints:

```text
GET  /health
GET  /models
POST /command
POST /voice-command
GET  /agent/jobs
GET  /agent/jobs/next?worker=pi
GET  /agent/jobs/{job_id}
POST /agent/jobs/{job_id}/result
GET  /image/{job_id}
GET  /audio/{job_id}
```

Completed job response shape includes:

```json
{
  "status": "done",
  "result_text": "Answer text",
  "sentiment": "happy",
  "image_url": "/image/<job_id>",
  "audio_url": "/audio/<job_id>",
  "options": ["Again", "Details", "New request"],
  "metrics": {}
}
```

If sentiment is omitted, the server infers a simple `happy`, `neutral`, or `sad` fallback from the result text/status. If an agent result includes `image_prompt`, the server generates a 135×135 little-endian RGB565 image at `/image/<job_id>` and returns it as `image_url`; the Stick displays that image instead of the bundled sentiment face. The server keeps up to three worker-provided options and appends `New request` as the fourth option.

## Stick firmware

Active firmware:

```text
firmware/m5sticks3-arduino/M5VoiceTerminal/M5VoiceTerminal.ino
firmware/m5sticks3-arduino/M5VoiceTerminal/config.h.example
```

Arduino CLI setup on this Mac:

- `arduino-cli` installed via Homebrew.
- M5Stack board index installed.
- `m5stack:esp32@3.3.7` installed.
- Libraries installed: `M5Unified`, `M5GFX`, `ArduinoJson`.
- Sketch compiles for `m5stack:esp32:m5stack_sticks3`.

Upload to the connected StickS3:

```sh
./scripts/arduino_upload.sh /dev/cu.usbmodem213301
```

If upload says `Failed to connect to ESP32-S3: No serial data received`, put the StickS3 into bootloader/download mode first: connect USB-C, hold the side reset button until the internal green LED flashes, then rerun the upload command.

## STT backend selection

The server has a configurable speech-to-text backend. If no environment variables are set, code defaults to `STT_BACKEND=whisper` and `WHISPER_MODEL=base`. Current recommended Apple Silicon runtime:

```sh
STT_BACKEND=mlx-whisper MLX_WHISPER_MODEL=mlx-community/whisper-tiny uvicorn server.app:app --host 0.0.0.0 --port 8010
```

Reliable PyTorch Whisper baseline:

```sh
STT_BACKEND=whisper WHISPER_MODEL=base uvicorn server.app:app --host 0.0.0.0 --port 8010
```

Benchmark real or synthetic recordings with:

```sh
./scripts/benchmark_stt.py --backend whisper --model base sample.wav
./scripts/benchmark_stt.py --backend mlx-whisper --model mlx-community/whisper-tiny sample.wav
```

See `docs/stt-review.md` for benchmark notes and current STT recommendation.

## Testing

Fast Pi SDK startup smoke test:

```sh
make test
# optional real model call after SDK startup:
node scripts/test_pi_programmatic_start.mjs --prompt
```

`make test` invokes Pi programmatically through the SDK directly from Node and through the same Python→Node SDK helper used by the worker, with discovery disabled and an in-memory session. It fails if startup exceeds `PI_STARTUP_THRESHOLD_MS` (default 5000 ms).

Server-side sample upload:

```sh
./scripts/upload_voice_sample.py samples/check_status.wav --base-url http://127.0.0.1:8010
```

Latency reporting:

```sh
./scripts/latency_report.py --base-url http://127.0.0.1:8010
```

## Safety posture

The default voice worker should run with inspection-oriented tools only. The current `pi-sdk-full` default includes web tools plus `read,bash,grep,find,ls`; do not add edit/write/git tools for voice commands. For stricter operation, set `PI_WORKER_TOOLS` explicitly to a smaller non-empty list, or use `VOICE_WORKER_BACKEND=pi-sdk` with `PI_WORKER_TOOLS` unset for no tools.

## Remaining work

1. Improve VAD/silence filtering beyond the current Whisper empty-transcript short-circuit.
2. Benchmark actual StickS3 microphone recordings against synthetic samples.
3. Reduce agent latency by comparing model aliases and tool configurations.
4. Validate generated RGB565 image display and Supertonic playback on-device after recent changes.
5. Add stronger runtime process management/runbook coverage.
