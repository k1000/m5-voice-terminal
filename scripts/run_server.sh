#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ ! -d .venv ]; then
  echo "Missing .venv. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

# Prefer the fast Apple Silicon STT path for mini; callers can override these.
export STT_BACKEND="${STT_BACKEND:-mlx-whisper}"
export MLX_WHISPER_MODEL="${MLX_WHISPER_MODEL:-mlx-community/whisper-tiny}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8010}"

exec .venv/bin/python -m uvicorn server.app:app --host "$HOST" --port "$PORT"
