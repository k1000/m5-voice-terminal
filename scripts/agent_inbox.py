#!/usr/bin/env python3
"""Tiny CLI for Pi/agent-side polling of StickS3 voice prompts.

Examples:
  python scripts/agent_inbox.py next
  python scripts/agent_inbox.py done <job_id> "Done, I checked it."
  python scripts/agent_inbox.py list
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error

from http_json import request_json

DEFAULT_BASE_URL = "http://127.0.0.1:8010"


def request(method: str, url: str, payload: dict | None = None) -> dict | None:
    try:
        return request_json(method, url, payload)
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8"), file=sys.stderr)
        raise SystemExit(exc.code) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")
    sub.add_parser("next")

    done = sub.add_parser("done")
    done.add_argument("job_id")
    done.add_argument("text")
    done.add_argument("--sentiment", choices=["happy", "neutral", "sad"], help="override the face shown on the Stick")

    fail = sub.add_parser("fail")
    fail.add_argument("job_id")
    fail.add_argument("error")

    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    if args.command == "list":
        print(json.dumps(request("GET", f"{base}/agent/jobs"), indent=2))
    elif args.command == "next":
        job = request("GET", f"{base}/agent/jobs/next?worker=pi")
        if job is None:
            print("No queued jobs.")
            return
        print(f"JOB_ID={job['id']}")
        print(f"DEVICE={job['device']}")
        print(f"PROMPT={job['prompt']}")
    elif args.command == "done":
        payload = {"status": "done", "text": args.text}
        if args.sentiment:
            payload["sentiment"] = args.sentiment
        print(json.dumps(request("POST", f"{base}/agent/jobs/{args.job_id}/result", payload), indent=2))
    elif args.command == "fail":
        print(json.dumps(request("POST", f"{base}/agent/jobs/{args.job_id}/result", {"status": "failed", "error": args.error}), indent=2))


if __name__ == "__main__":
    main()
