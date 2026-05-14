#!/usr/bin/env python3
"""Smoke test that Python can start Pi programmatically through the SDK helper."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_MS = float(os.environ.get("PI_STARTUP_THRESHOLD_MS", "5000"))

payload = {
    "dryRun": True,
    "systemPrompt": "You are a concise voice-terminal assistant.",
    "tools": [],
    "thinking": "off",
    "fullResources": os.environ.get("PI_SDK_FULL", "").lower() in {"1", "true", "yes"},
}

result = subprocess.run(
    ["node", str(ROOT / "scripts" / "pi_sdk_once.mjs")],
    input=json.dumps(payload),
    text=True,
    capture_output=True,
    timeout=10,
    env={**os.environ, "PI_OFFLINE": os.environ.get("PI_OFFLINE", "1")},
)
if result.returncode != 0:
    print(result.stderr or result.stdout, file=sys.stderr)
    raise SystemExit(result.returncode)

metrics = json.loads(result.stdout)
metrics["threshold_ms"] = THRESHOLD_MS
print(json.dumps(metrics, indent=2))
if float(metrics["startup_ms"]) > THRESHOLD_MS:
    print(f"Pi SDK startup exceeded {THRESHOLD_MS} ms", file=sys.stderr)
    raise SystemExit(1)
