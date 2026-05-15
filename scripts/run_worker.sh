#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Non-interactive ssh/launchd shells on mini may not include Homebrew.
if [ -d /opt/homebrew/bin ]; then
  export PATH="/opt/homebrew/bin:$PATH"
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -z "${PI_JS_RUNTIME:-}" ] && command -v bun >/dev/null 2>&1; then
  export PI_JS_RUNTIME=bun
fi

export VOICE_WORKER_BACKEND="${VOICE_WORKER_BACKEND:-pi-sdk-full}"
export PI_WORKER_MODEL="${PI_WORKER_MODEL:-minimax/MiniMax-M2.7-highspeed}"
export PI_WORKER_THINKING="${PI_WORKER_THINKING:-off}"
export BASE_URL="${BASE_URL:-http://127.0.0.1:8010}"

# Auto-restart loop: worker dies → reconnect loop.  Ctrl-C exits cleanly.
echo "agent_worker auto-restart loop PID $$ "$(date)"" >&2
while true; do
  python3 scripts/agent_worker.py --base-url "$BASE_URL" 2>&1 \
    | while IFS= read -r line; do echo "$(date +%H:%M:%S) $line"; done \
    >> data/agent_worker.log
  EXIT=${PIPESTATUS[0]}
  if [ $EXIT -eq 0 ] || [ $EXIT -eq 130 ]; then
    echo "agent_worker exited cleanly (code $EXIT)" >&2
    break
  fi
  echo "agent_worker crashed (code $EXIT), restarting in 2s..." >&2
  sleep 2
done
