#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-/dev/cu.usbmodem213301}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKETCH="$ROOT/firmware/m5sticks3-arduino/M5VoiceTerminal"
FQBN="m5stack:esp32:m5stack_sticks3"

arduino-cli compile --fqbn "$FQBN" "$SKETCH"
arduino-cli upload -p "$PORT" --fqbn "$FQBN" "$SKETCH"
