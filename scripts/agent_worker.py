#!/usr/bin/env python3
"""Auto-worker that turns queued StickS3 voice prompts into agent results.

Default mode calls MiniMax directly through its Anthropic-compatible API for low
latency. Set VOICE_WORKER_BACKEND=pi to use `pi -p --no-tools --no-session` instead.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from http_json import request_json
from minimax_direct import call_minimax

DEFAULT_BASE_URL = "http://127.0.0.1:8010"
SYSTEM_PROMPT = """You are the server-side Pi agent for a handheld M5StickS3 voice terminal.

The user prompt you receive was spoken aloud, recorded by a small microphone, and
transcribed by speech-to-text. It may contain transcription errors, repeated words,
wrong homophones, missing punctuation, or partial phrases.

Your job is to infer the user's intended command in the context of the current project:
an M5StickS3 voice terminal connected to a FastAPI server, STT, an agent job queue,
and eventual TTS/audio output. Interpret likely voice mistakes generously, but do not
pretend certainty when a command is ambiguous.

Behavior rules:
- First mentally normalize the transcript into the most likely intended command.
- If the command is clear, answer or act according to that intended command.
- If the transcript is ambiguous, ask one concise clarifying question.
- If the command could be unsafe/destructive, ask for confirmation.
- Respond concisely for a tiny screen. Keep textual answers under 400 characters unless needed.
- Do not mention transcription uncertainty unless it affects the answer.
- Return ONLY compact JSON with exactly these keys:
  {"text":"response for the user","sentiment":"happy|neutral|sad","options":["option"]}
- Use sentiment=happy for success/positive confirmation, neutral for normal info/questions,
  sad for errors, blocked operations, or bad news.
- If useful, include up to 3 options, each max 2 words. The server adds a 4th "New request" option automatically.
"""


def complete(base_url: str, job_id: str, text: str, status: str = "done", error: str | None = None, metrics: dict[str, Any] | None = None, sentiment: str | None = None, options: list[str] | None = None) -> None:
    payload: dict[str, Any] = {"status": status, "metrics": metrics or {}, "options": options or []}
    if sentiment is not None:
        payload["sentiment"] = sentiment
    if status == "done":
        payload["text"] = text
    else:
        payload["error"] = error or text
    request_json("POST", f"{base_url}/agent/jobs/{job_id}/result", payload)


def normalize_options(options: Any) -> list[str]:
    if not isinstance(options, list):
        return []
    out = []
    for opt in options[:3]:
        text = str(opt).strip()
        if text and len(text.split()) <= 2:
            out.append(text[:24])
    return out


def response_tuple(obj: dict[str, Any], raw: str) -> tuple[str, str, list[str]]:
    text = str(obj.get("text") or obj.get("response") or raw).strip()
    sentiment = str(obj.get("sentiment") or "neutral").strip().lower()
    if sentiment not in {"happy", "neutral", "sad"}:
        sentiment = "neutral"
    return text, sentiment, normalize_options(obj.get("options"))


def parse_agent_json(raw: str) -> tuple[str, str, list[str]]:
    """Parse Pi's requested JSON response, with fallback for non-JSON output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start:end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return response_tuple(obj, raw)

    fallback = raw.strip()
    if len(fallback) > 500:
        fallback = fallback[:497].rstrip() + "..."
    return fallback, "neutral", []


def current_time_context() -> str:
    local_now = datetime.now().astimezone()
    utc_now = datetime.now(timezone.utc)
    return (
        "\n\nCurrent date/time for this voice request:\n"
        f"- Local: {local_now.isoformat(timespec='seconds')}\n"
        f"- UTC: {utc_now.isoformat(timespec='seconds')}\n"
    )


def build_voice_request(prompt: str) -> tuple[str, str]:
    system_prompt = SYSTEM_PROMPT + current_time_context()
    voice_request = (
        "Voice transcript to interpret:\n"
        f"{prompt}\n\n"
        "Return only this JSON object, no markdown/code fences:\n"
        "{\"text\":\"concise response for tiny screen\",\"sentiment\":\"happy|neutral|sad\",\"options\":[\"max two words\"]}\n"
        "Options: include at most 3; server adds New request as option 4."
    )
    return system_prompt, voice_request


def run_pi(prompt: str, timeout: int) -> str:
    model = os.environ.get("PI_WORKER_MODEL", "minimax/MiniMax-M2.7-highspeed")
    thinking = os.environ.get("PI_WORKER_THINKING", "off")
    system_prompt, voice_request = build_voice_request(prompt)
    cmd = [
        "pi", "-p", "--no-session", "--no-tools",
        "--model", model, "--thinking", thinking,
        "--append-system-prompt", system_prompt,
        prompt,
    ]
    extra_tools = os.environ.get("PI_WORKER_TOOLS", "").strip()
    if extra_tools:
        # Explicit opt-in, e.g. PI_WORKER_TOOLS=read,bash. Voice-triggered tools are risky.
        cmd = [
            "pi", "-p", "--no-session", "--tools", extra_tools,
            "--model", model, "--thinking", thinking,
            "--append-system-prompt", system_prompt,
            prompt,
        ]
    cmd[-1] = voice_request
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or f"pi exited {result.returncode}").strip())
    return result.stdout.strip() or "[agent returned no text]"


def next_queued_job(base_url: str) -> dict[str, Any] | None:
    data = request_json("GET", f"{base_url}/agent/jobs?status=queued")
    jobs = data.get("jobs", []) if isinstance(data, dict) else []
    if not jobs:
        return None
    # Process the newest job first so the currently waiting Stick gets a fast response.
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--echo", action="store_true", help="debug mode: do not call pi, just echo transcript")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    print(f"agent_worker listening on {base_url}", flush=True)
    while True:
        try:
            job = next_queued_job(base_url)
            if job is None:
                if args.once:
                    return
                time.sleep(args.interval)
                continue

            job_id = job["id"]
            prompt = job.get("prompt") or ""
            print(f"processing {job_id}: {prompt}", flush=True)
            try:
                queue_wait_s = None
                created_at = job.get("created_at")
                if created_at:
                    try:
                        queue_wait_s = (datetime.now(timezone.utc) - datetime.fromisoformat(created_at)).total_seconds()
                    except ValueError:
                        queue_wait_s = None
                start = time.perf_counter()
                if args.echo:
                    text = f"I heard: {prompt}"
                    sentiment = "neutral"
                    options = []
                    agent_s = time.perf_counter() - start
                    mode = "echo"
                else:
                    worker_backend = os.environ.get("VOICE_WORKER_BACKEND", "minimax-direct").strip().lower()
                    if worker_backend == "minimax-direct":
                        system_prompt, voice_request = build_voice_request(prompt)
                        raw_text, direct_s, _ = call_minimax(
                            voice_request,
                            timeout=args.timeout,
                            system_prompt=system_prompt,
                            raw_user_content=True,
                        )
                        text, sentiment, options = parse_agent_json(raw_text)
                        agent_s = time.perf_counter() - start
                        mode = "minimax-direct"
                    else:
                        raw_text = run_pi(prompt, args.timeout)
                        text, sentiment, options = parse_agent_json(raw_text)
                        agent_s = time.perf_counter() - start
                        mode = "pi"
                metrics = {"agent_worker_s": round(agent_s, 3), "agent_worker_mode": mode}
                if queue_wait_s is not None:
                    metrics["agent_queue_wait_s"] = round(queue_wait_s, 3)
                complete(base_url, job_id, text, metrics=metrics, sentiment=sentiment, options=options)
                print(f"done {job_id} agent_s={agent_s:.3f}", flush=True)
            except Exception as exc:
                complete(base_url, job_id, "", status="failed", error=repr(exc))
                print(f"failed {job_id}: {exc!r}", file=sys.stderr, flush=True)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"server unavailable: {exc!r}", file=sys.stderr, flush=True)
            time.sleep(args.interval)

        if args.once:
            return


if __name__ == "__main__":
    main()
