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

export VOICE_WORKER_BACKEND="${VOICE_WORKER_BACKEND:-pi-sdk-full}"
export PI_WORKER_MODEL="${PI_WORKER_MODEL:-minimax/MiniMax-M2.7-highspeed}"
export PI_WORKER_THINKING="${PI_WORKER_THINKING:-off}"
export BASE_URL="${BASE_URL:-http://127.0.0.1:8010}"

exec .venv/bin/python scripts/agent_worker.py --base-url "$BASE_URL"
