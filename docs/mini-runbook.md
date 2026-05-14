# Running the voice terminal on `mini`

Target: run the FastAPI server and Pi worker on the Mac mini, then point StickS3 `VOICE_URL` at the mini LAN IP.

## One-time setup on mini

```sh
ssh mini
mkdir -p ~/Development/_MY
cd ~/Development/_MY
git clone git@github.com:k1000/m5-voice-terminal.git
cd m5-voice-terminal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mini.txt
```

`requirements-mini.txt` intentionally omits PyTorch/OpenAI Whisper to save disk and uses the MLX Whisper path. Use full `requirements.txt` only if mini has enough free disk and you need the PyTorch Whisper fallback.

Pi worker requirements:

```sh
PATH=/opt/homebrew/bin:$PATH ffmpeg -version | head -1
PATH=/opt/homebrew/bin:$PATH node --version
PATH=/opt/homebrew/bin:$PATH bun --version  # optional but preferred by run_worker.sh when present
PATH=/opt/homebrew/bin:$PATH pi --version
PATH=/opt/homebrew/bin:$PATH make test
PATH=/opt/homebrew/bin:$PATH make test JS_RUNTIME=bun  # optional Bun SDK smoke test
```

If `node` or `pi` is missing, install/configure Pi on mini before using `VOICE_WORKER_BACKEND=pi-sdk-full`. Bun is optional; when installed, `scripts/run_worker.sh` uses it for the Pi SDK helper via `PI_JS_RUNTIME=bun`. The server can still run without the worker for `/health`, `/models`, and upload/STT testing.

## Start server

```sh
cd ~/Development/_MY/m5-voice-terminal
./scripts/run_server.sh
```

Defaults:

- `HOST=0.0.0.0`
- `PORT=8010`
- `STT_BACKEND=mlx-whisper`
- `MLX_WHISPER_MODEL=mlx-community/whisper-tiny`

Override with environment variables or a local `.env` file.

## Start worker

In a second shell:

```sh
cd ~/Development/_MY/m5-voice-terminal
./scripts/run_worker.sh
```

Defaults:

- `BASE_URL=http://127.0.0.1:8010`
- `VOICE_WORKER_BACKEND=pi-sdk-full`
- `PI_WORKER_MODEL=minimax/MiniMax-M2.7-highspeed`
- `PI_WORKER_THINKING=off`
- `PI_JS_RUNTIME=bun` if `bun` is on PATH, otherwise Node

## Point StickS3 at mini

On mini:

```sh
ipconfig getifaddr en0
```

Then set firmware config on the machine used to flash the StickS3:

```c
#define VOICE_URL "http://<mini-lan-ip>:8010/voice-command"
```

Upload firmware with:

```sh
./scripts/arduino_upload.sh /dev/cu.usbmodem213301
```

## Health checks

```sh
curl http://127.0.0.1:8010/health
curl http://<mini-lan-ip>:8010/health
./scripts/upload_voice_sample.py samples/check_status.wav --base-url http://127.0.0.1:8010
./scripts/agent_inbox.py --base-url http://127.0.0.1:8010 list
```

macOS may prompt to allow incoming connections to Python/uvicorn. Allow it, or configure the firewall for port `8010`.
