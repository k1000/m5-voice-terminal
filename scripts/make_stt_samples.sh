#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/samples"
mkdir -p "$OUT"

make_sample() {
  local name="$1"
  local text="$2"
  local aiff="$OUT/$name.aiff"
  local wav="$OUT/$name.wav"
  say -v Daniel -o "$aiff" "$text"
  ffmpeg -y -hide_banner -loglevel error -i "$aiff" -ac 1 -ar 16000 -sample_fmt s16 "$wav"
  rm -f "$aiff"
  echo "$wav"
}

make_sample check_status "Check the current project status."
make_sample list_files "List the files in the current project."
make_sample run_tests "Run the tests and summarize the result."
make_sample create_note "Create a short note that says the voice loop is working."
