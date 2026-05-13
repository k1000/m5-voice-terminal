# M5StickS3 Voice Terminal MVP

A voice-command terminal for M5StickS3. The active firmware records a hold-to-talk microphone clip, uploads it to a local FastAPI server, the server transcribes it, queues it for a Pi/agent worker, generates Supertonic WAV audio, and the Stick displays a bundled sentiment face while playing the result.

## Current flow

```text
M5StickS3 BtnA
  -> hold button to record up to 10s of 16 kHz mono PCM
  -> wrap as WAV
  -> POST /voice-command
  -> server STT
  -> queued agent job
  -> scripts/agent_worker.py handles prompt
  -> POST /agent/jobs/{id}/result
  -> server generates Supertonic WAV
  -> Stick polls job, displays bundled sentiment face/result, downloads audio, plays WAV
```

The legacy MicroPython client in `stick/` is retained as a fallback text-command reference. The active device firmware is the Arduino/M5Unified sketch in `firmware/m5sticks3-arduino/M5VoiceTerminal/`.

## Server

```sh
cd m5-voice-terminal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Recommended current runtime on Apple Silicon
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

Voice/text commands are queued for a local worker. By default the worker calls MiniMax directly for lower latency; set `VOICE_WORKER_BACKEND=pi` to route through `pi -p`.

```sh
env PI_WORKER_MODEL=minimax/MiniMax-M2.7-highspeed \
    PI_WORKER_THINKING=off \
    PI_WORKER_TOOLS=read,bash,grep,find,ls \
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
  "image_url": null,
  "audio_url": "/audio/<job_id>"
}
```

If sentiment is omitted, the server infers a simple `happy`, `neutral`, or `sad` fallback from the result text/status. `image_url` remains available for future custom RGB565 images; standard faces are bundled in firmware.

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

The server has a configurable speech-to-text backend. Current Apple Silicon runtime:

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

Server-side sample upload:

```sh
./scripts/upload_voice_sample.py samples/check_status.wav --base-url http://127.0.0.1:8010
```

Latency reporting:

```sh
./scripts/latency_report.py --base-url http://127.0.0.1:8010
```

## Safety posture

The default voice worker should run with inspection-oriented tools only, for example `read,bash,grep,find,ls`. It should not auto-enable destructive edit/write/git operations for voice commands.

## Remaining work

1. Add VAD/silence filtering so quiet clips do not hallucinate commands.
2. Benchmark actual StickS3 microphone recordings against synthetic samples.
3. Reduce agent latency by comparing model aliases and tool configurations.
4. Tune hold-to-talk recording limits and UX after real-device testing.
5. Add stronger runtime process management/runbook coverage.
