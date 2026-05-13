#!/usr/bin/env python3
"""Summarize where StickS3 voice-terminal response time is spent."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_JOBS_FILE = Path(__file__).resolve().parents[1] / "data" / "agent_jobs.json"

STAGES = [
    ("server_stt_s", "speech-to-text"),
    ("agent_queue_wait_s", "queue wait before Pi worker"),
    ("agent_worker_s", "Pi agent/model response"),
    ("server_image_s", "face image generation"),
    ("server_tts_s", "TTS audio generation"),
    ("server_result_post_s", "server result save/post processing"),
]


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def total_seconds(job: dict[str, Any]) -> float | None:
    metrics = job.get("metrics") or {}
    if "server_total_until_done_s" in metrics:
        return float(metrics["server_total_until_done_s"])
    created = parse_time(job.get("created_at"))
    updated = parse_time(job.get("updated_at"))
    if created and updated:
        return (updated - created).total_seconds()
    return None


def summarize(values: list[float]) -> str:
    if not values:
        return "-"
    return f"avg={mean(values):.2f}s med={median(values):.2f}s max={max(values):.2f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-file", type=Path, default=DEFAULT_JOBS_FILE)
    parser.add_argument("--last", type=int, default=20, help="number of completed/failed jobs to inspect")
    args = parser.parse_args()

    jobs = json.loads(args.jobs_file.read_text())
    jobs = [j for j in jobs if j.get("status") in {"done", "failed"}]
    jobs = jobs[-args.last :]
    if not jobs:
        print("No completed jobs found.")
        return

    print(f"Latency report for last {len(jobs)} completed jobs")
    totals = [t for j in jobs if (t := total_seconds(j)) is not None]
    print(f"total created->done: {summarize(totals)}")
    print()

    stage_values: dict[str, list[float]] = {key: [] for key, _ in STAGES}
    for job in jobs:
        metrics = job.get("metrics") or {}
        for key, _ in STAGES:
            if key in metrics:
                stage_values[key].append(float(metrics[key]))

    for key, label in STAGES:
        print(f"{label:32} {summarize(stage_values[key])}")

    latest = jobs[-1]
    print("\nLatest job:")
    print(f"  id: {latest.get('id')} status={latest.get('status')} prompt={latest.get('prompt')!r}")
    print(f"  total: {total_seconds(latest)}s")
    metrics = latest.get("metrics") or {}
    ranked = [(label, float(metrics[key])) for key, label in STAGES if key in metrics]
    for label, value in sorted(ranked, key=lambda item: item[1], reverse=True):
        print(f"  {label:32} {value:.3f}s")


if __name__ == "__main__":
    main()
