#!/usr/bin/env python3
"""Call the configured MiniMax Anthropic-compatible API directly, bypassing `pi`.

This is for latency testing and for the voice worker fast path. The API key is read
from ~/.pi/agent/models.json and is never printed.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"

SYSTEM_PROMPT = """You are the server-side voice responder for a tiny M5StickS3 screen.
Interpret the transcript generously. Return ONLY compact JSON with exactly these keys:
{"text":"concise response for the user","sentiment":"happy|neutral|sad"}
Use happy for success/positive confirmation, neutral for normal info/questions, sad for errors/bad news.
"""


def load_minimax_config() -> dict[str, Any]:
    cfg = json.loads(MODELS_JSON.read_text())
    return cfg["providers"]["minimax"]


def extract_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in data.get("content", []):
        if isinstance(block, dict) and "text" in block:
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def call_minimax(
    prompt: str,
    model: str = "MiniMax-M2.7-highspeed",
    timeout: int = 60,
    system_prompt: str = SYSTEM_PROMPT,
    raw_user_content: bool = False,
) -> tuple[str, float, dict[str, Any]]:
    cfg = load_minimax_config()
    url = cfg["baseUrl"].rstrip("/") + "/v1/messages"
    user_content = prompt if raw_user_content else f"Voice transcript to interpret:\n{prompt}"
    payload = {
        "model": model,
        "max_tokens": 256,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }
    headers = {
        "authorization": "Bearer " + cfg["apiKey"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    started = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode())
    elapsed = time.perf_counter() - started
    return extract_text(data), elapsed, data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Say latency test is ready")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    for i in range(args.runs):
        text, elapsed, data = call_minimax(args.prompt)
        content_types = [block.get("type") for block in data.get("content", []) if isinstance(block, dict)]
        print(json.dumps({
            "run": i + 1,
            "seconds": round(elapsed, 3),
            "content_types": content_types,
            "text": text,
        }))


if __name__ == "__main__":
    main()
