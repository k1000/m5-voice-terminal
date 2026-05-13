#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-}"
if [[ -z "$PORT" ]]; then
  echo "Usage: $0 /dev/cu.usbmodemXXXX" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mpremote connect "$PORT" fs cp "$ROOT/stick/m5pm1.py" :m5pm1.py
mpremote connect "$PORT" fs cp "$ROOT/stick/st7789.py" :st7789.py
mpremote connect "$PORT" fs cp "$ROOT/stick/config.py" :config.py
mpremote connect "$PORT" fs cp "$ROOT/stick/main.py" :main.py
mpremote connect "$PORT" reset
