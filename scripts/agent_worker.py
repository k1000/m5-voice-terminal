#!/usr/bin/env python3
"""Auto-worker that turns queued StickS3 voice prompts into agent results.

Default mode invokes Pi programmatically through its SDK with full resource
loading. Set VOICE_WORKER_BACKEND=pi-sdk for a faster minimal SDK path, or
VOICE_WORKER_BACKEND=pi to use `pi -p --no-tools --no-session` instead.
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
- For current weather, news, dates, or other live facts, you MUST use available tools/internet first (for example bash with curl or python). Do not answer from memory, and do not say live data is unavailable unless a tool/web fetch actually fails.
- For weather questions, fetch a current forecast from the web, then summarize the relevant city/date concisely.
- Respond concisely for a tiny screen. Keep textual answers under 400 characters unless needed.
- Do not mention transcription uncertainty unless it affects the answer.
- Return ONLY compact JSON with these keys:
  {"text":"response for the user","sentiment":"happy|neutral|sad","options":["option"],"image_prompt":"optional visual prompt"}
- Use sentiment=happy for success/positive confirmation, neutral for normal info/questions,
  sad for errors, blocked operations, or bad news.
- If the user asks to draw, show, create, or visualize something, include image_prompt: a concise square image prompt.
- If useful, include up to 3 options, each max 2 words. The server adds a 4th "New request" option automatically.
"""


def complete(base_url: str, job_id: str, text: str, status: str = "done", error: str | None = None, metrics: dict[str, Any] | None = None, sentiment: str | None = None, options: list[str] | None = None, image_prompt: str | None = None) -> None:
    payload: dict[str, Any] = {"status": status, "metrics": metrics or {}, "options": options or []}
    if sentiment is not None:
        payload["sentiment"] = sentiment
    if image_prompt:
        payload["image_prompt"] = image_prompt
    if status == "done":
        payload["text"] = text
    else:
        payload["error"] = error or text
    # Result POST can block while the server generates Supertonic audio and/or an image.
    request_json("POST", f"{base_url}/agent/jobs/{job_id}/result", payload, timeout=180)


def normalize_options(options: Any) -> list[str]:
    if not isinstance(options, list):
        return []
    out = []
    for opt in options[:3]:
        text = str(opt).strip()
        if text and len(text.split()) <= 2:
            out.append(text[:24])
    return out


def response_tuple(obj: dict[str, Any]) -> tuple[str, str, list[str], str | None] | None:
    text = str(obj.get("text") or obj.get("response") or "").strip()
    if not text:
        return None
    sentiment = str(obj.get("sentiment") or "neutral").strip().lower()
    if sentiment not in {"happy", "neutral", "sad"}:
        sentiment = "neutral"
    image_prompt = str(obj.get("image_prompt") or obj.get("image") or "").strip()[:500] or None
    return text[:400], sentiment, normalize_options(obj.get("options")), image_prompt


def parse_agent_json(raw: str) -> tuple[str, str, list[str], str | None]:
    """Parse Pi's requested JSON response, with fallback for malformed/empty output."""
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
        if isinstance(obj, dict) and (parsed := response_tuple(obj)) is not None:
            return parsed

    fallback = raw.strip()
    if not fallback or fallback.startswith("{") or fallback.startswith("["):
        return "Wolf got confused. Try again.", "sad", ["Try again"], None
    if len(fallback) > 400:
        fallback = fallback[:397].rstrip() + "..."
    return fallback, "neutral", [], None


def current_time_context() -> str:
    local_now = datetime.now().astimezone()
    utc_now = datetime.now(timezone.utc)
    return (
        "\n\nCurrent date/time for this voice request:\n"
        f"- Local: {local_now.isoformat(timespec='seconds')}\n"
        f"- UTC: {utc_now.isoformat(timespec='seconds')}\n"
    )


def recent_history(base_url: str, current_job: dict[str, Any], limit: int = 5) -> list[dict[str, str]]:
    """Return recent same-device turns so option clicks like 'Berlin' keep context."""
    try:
        data = request_json("GET", f"{base_url}/agent/jobs")
    except Exception:
        return []
    jobs = data.get("jobs", []) if isinstance(data, dict) else []
    current_id = current_job.get("id")
    device = current_job.get("device")
    created_at = current_job.get("created_at", "")
    turns = []
    for job in jobs:
        if job.get("id") == current_id or job.get("device") != device or job.get("status") != "done":
            continue
        if created_at and job.get("created_at", "") >= created_at:
            continue
        prompt = str(job.get("prompt") or "").strip()
        reply = str(job.get("result_text") or job.get("error") or "").strip()
        if prompt and reply:
            turns.append({"prompt": prompt[:160], "reply": reply[:220]})
    return turns[-limit:]


def format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    lines = ["Recent same-device conversation, oldest to newest:"]
    for turn in history:
        lines.append(f"User: {turn['prompt']}")
        lines.append(f"Assistant: {turn['reply']}")
    return "\n".join(lines) + "\n\n"


def build_voice_request(prompt: str, history: list[dict[str, str]] | None = None) -> tuple[str, str]:
    system_prompt = SYSTEM_PROMPT + current_time_context()
    voice_request = (
        format_history(history or [])
        + "Voice transcript to interpret now:\n"
        f"{prompt}\n\n"
        "Use recent conversation only to resolve short follow-ups and option clicks. "
        "Do not claim you fetched news, weather, music, or external data unless that result is explicitly present above or you fetched it with a tool in this turn.\n"
        "If this asks for weather, forecast, news, or other live data, use bash/curl/python internet access before answering. Do not say you lack live access until the tool call fails.\n"
        "Return only this JSON object, no markdown/code fences:\n"
        "{\"text\":\"concise response for tiny screen\",\"sentiment\":\"happy|neutral|sad\",\"options\":[\"max two words\"],\"image_prompt\":\"omit unless user asked for image\"}\n"
        "Options: include at most 3; server adds New request as option 4. "
        "Only include image_prompt when the user asks to draw, show, create, or visualize something."
    )
    return system_prompt, voice_request


def worker_tools() -> list[str]:
    extra_tools = os.environ.get("PI_WORKER_TOOLS", "").strip()
    if not extra_tools and os.environ.get("VOICE_WORKER_BACKEND", "pi-sdk-full").strip().lower() == "pi-sdk-full":
        extra_tools = "web_fetch,web_search,read,bash,grep,find,ls"
    if not extra_tools:
        return []
    # Explicit opt-in, e.g. PI_WORKER_TOOLS=read,bash. Voice-triggered tools are risky.
    return [tool.strip() for tool in extra_tools.split(",") if tool.strip()]


def js_runtime() -> str:
    """Return JS runtime for the Pi SDK helper; set PI_JS_RUNTIME=bun to use Bun."""
    return os.environ.get("PI_JS_RUNTIME") or os.environ.get("JS_RUNTIME") or "node"


def run_pi_sdk(prompt: str, timeout: int, history: list[dict[str, str]] | None = None, full_resources: bool = False) -> str:
    """Invoke Pi programmatically through its SDK from this Python worker."""
    model = os.environ.get("PI_WORKER_MODEL", "minimax/MiniMax-M2.7-highspeed")
    thinking = os.environ.get("PI_WORKER_THINKING", "off")
    system_prompt, voice_request = build_voice_request(prompt, history)
    payload = {
        "systemPrompt": system_prompt,
        "prompt": voice_request,
        "model": model,
        "thinking": thinking,
        "tools": worker_tools(),
        "fullResources": full_resources,
    }
    script = os.path.join(os.path.dirname(__file__), "pi_sdk_once.mjs")
    result = subprocess.run(
        [js_runtime(), script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout,
        env={**os.environ, "PI_OFFLINE": os.environ.get("PI_OFFLINE", "1")},
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or f"pi SDK exited {result.returncode}").strip())
    return result.stdout.strip() or "[agent returned no text]"


def run_pi(prompt: str, timeout: int, history: list[dict[str, str]] | None = None) -> str:
    model = os.environ.get("PI_WORKER_MODEL", "minimax/MiniMax-M2.7-highspeed")
    thinking = os.environ.get("PI_WORKER_THINKING", "off")
    system_prompt, voice_request = build_voice_request(prompt, history)
    cmd = [
        "pi", "-p", "--no-session", "--no-tools",
        "--model", model, "--thinking", thinking,
        "--append-system-prompt", system_prompt,
        prompt,
    ]
    extra_tools = os.environ.get("PI_WORKER_TOOLS", "").strip()
    if extra_tools:
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
            history = recent_history(base_url, job)
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
                    image_prompt = None
                    agent_s = time.perf_counter() - start
                    mode = "echo"
                else:
                    worker_backend = os.environ.get("VOICE_WORKER_BACKEND", "pi-sdk-full").strip().lower()
                    if worker_backend == "minimax-direct":
                        system_prompt, voice_request = build_voice_request(prompt, history)
                        raw_text, direct_s, _ = call_minimax(
                            voice_request,
                            timeout=args.timeout,
                            system_prompt=system_prompt,
                            raw_user_content=True,
                        )
                        text, sentiment, options, image_prompt = parse_agent_json(raw_text)
                        agent_s = time.perf_counter() - start
                        mode = "minimax-direct"
                    elif worker_backend in {"pi-sdk", "pi-sdk-full"}:
                        raw_text = run_pi_sdk(prompt, args.timeout, history, full_resources=worker_backend == "pi-sdk-full")
                        text, sentiment, options, image_prompt = parse_agent_json(raw_text)
                        agent_s = time.perf_counter() - start
                        mode = worker_backend
                    else:
                        raw_text = run_pi(prompt, args.timeout, history)
                        text, sentiment, options, image_prompt = parse_agent_json(raw_text)
                        agent_s = time.perf_counter() - start
                        mode = "pi"
                metrics = {"agent_worker_s": round(agent_s, 3), "agent_worker_mode": mode}
                if queue_wait_s is not None:
                    metrics["agent_queue_wait_s"] = round(queue_wait_s, 3)
                complete(base_url, job_id, text, metrics=metrics, sentiment=sentiment, options=options, image_prompt=image_prompt)
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
