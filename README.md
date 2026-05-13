# M5StickS3 Voice Terminal MVP

Iteration 1: the StickS3 connects to Wi-Fi, waits for a button press, sends a JSON command to a server, and displays the returned text on the LCD.

No microphone, STT, agent bridge, TTS, or speaker playback yet.

## Server

```sh
cd m5-voice-terminal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Find the Mac LAN IP:

```sh
ipconfig getifaddr en0
```

Then edit `stick/config.py` and set:

```python
SERVER_URL = "http://<mac-lan-ip>:8000/command"
```

## Stick firmware

`stick/config.py` contains the local Wi-Fi password and is ignored by git. `stick/config.example.py` is the shareable template.

Upload to the connected StickS3:

```sh
./scripts/upload.sh /dev/cu.usbmodem213301
```

The device should show Wi-Fi status. Press the front button on GPIO11 to call the server. The server returns a job id; the Stick polls `/agent/jobs/<job_id>` until the Pi/agent posts a result, then displays the final `result_text`.

## Planned next iterations

1. Replace fixed text with recorded microphone audio upload.
2. Server performs speech-to-text and forwards transcript to the agent.
3. Return response text plus optional audio URL.
4. Stick downloads and plays generated audio through the speaker.

## Pi agent queue bridge

Voice/text commands are now queued for the local agent worker instead of answered immediately. By default the worker calls MiniMax directly for lower latency; set `VOICE_WORKER_BACKEND=pi` to route through `pi -p`.

Flow:

```text
StickS3 /command or /voice-command
  -> server transcribes audio if needed
  -> server stores queued job
  -> Pi/agent polls /agent/jobs/next
  -> Pi/agent posts result to /agent/jobs/<id>/result
```

Manual polling helpers:

```sh
# Claim next queued voice prompt
./scripts/agent_inbox.py next

# Mark it done after the agent handles it
./scripts/agent_inbox.py done <job_id> "Result text to show/play back"

# Inspect all jobs
./scripts/agent_inbox.py list
```

HTTP endpoints:

```text
GET  /agent/jobs
GET  /agent/jobs/next?worker=pi
GET  /agent/jobs/{job_id}
POST /agent/jobs/{job_id}/result
```

### Stick result polling

The Stick uses simple HTTP polling for the response path:

```text
POST /command or /voice-command -> receives meta.job_id
GET  /agent/jobs/<job_id> every 1.5s
status=done -> display result_text plus happy/neutral/sad face from sentiment
status=failed -> display error plus sad face
```

Agent results support `sentiment: "happy" | "neutral" | "sad"`. The Stick draws the matching Wolfenstein-inspired status face when the response arrives. If omitted, the server infers a simple fallback sentiment from the result text/status.

This is intentionally simpler and more reliable on ESP32/MicroPython than SSE, WebSocket, or WebRTC.

## STT backend selection

The server has a configurable speech-to-text backend. Default is the reliable baseline:

```sh
STT_BACKEND=whisper WHISPER_MODEL=base uvicorn server.app:app --host 0.0.0.0 --port 8010
```

Apple Silicon MLX candidate for lower latency:

```sh
STT_BACKEND=mlx-whisper MLX_WHISPER_MODEL=mlx-community/whisper-tiny uvicorn server.app:app --host 0.0.0.0 --port 8010
```

Higher-accuracy MLX candidate:

```sh
STT_BACKEND=mlx-whisper MLX_WHISPER_MODEL=mlx-community/whisper-base-mlx uvicorn server.app:app --host 0.0.0.0 --port 8010
# or: MLX_WHISPER_MODEL=mlx-community/whisper-large-v3-turbo
```

Benchmark real Stick recordings with:

```sh
./scripts/benchmark_stt.py --backend whisper --model base sample.wav
./scripts/benchmark_stt.py --backend mlx-whisper --model mlx-community/whisper-tiny sample.wav
```

See `docs/stt-review.md` for benchmark notes and current STT recommendation.

## Real StickS3 microphone path

MicroPython is still used for the current button/text polling client. For real microphone capture, the recommended path is Arduino/M5Unified because StickS3 audio uses the ES8311 codec and M5Unified already handles codec setup.

Scaffolded firmware:

```text
firmware/m5sticks3-arduino/M5VoiceTerminal/M5VoiceTerminal.ino
firmware/m5sticks3-arduino/M5VoiceTerminal/config.h.example
```

Flow in that sketch:

```text
BtnA -> record 3s 16 kHz mono PCM -> wrap as WAV -> POST /voice-command
```

Arduino CLI setup has been completed on this Mac:

- `arduino-cli` installed via Homebrew.
- M5Stack board index installed.
- `m5stack:esp32@3.3.7` installed.
- Libraries installed: `M5Unified`, `M5GFX`, `ArduinoJson`.
- Sketch compiles for `m5stack:esp32:m5stack_sticks3`.

Upload command:

```sh
./scripts/arduino_upload.sh /dev/cu.usbmodem213301
```

If upload says `Failed to connect to ESP32-S3: No serial data received`, put the StickS3 into bootloader/download mode first: connect USB-C, hold the side reset button until the internal green LED flashes, then rerun the upload command.

Server-side sample upload is already tested with:

```sh
./scripts/upload_voice_sample.py samples/check_status.wav --base-url http://127.0.0.1:8010
```
