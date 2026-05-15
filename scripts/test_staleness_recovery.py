#!/usr/bin/env python3
"""Verify _fail_stale_in_progress_jobs marks stuck jobs failed correctly."""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Point the server at a throwaway data dir BEFORE importing.
_tmp = tempfile.mkdtemp(prefix="m5_staleness_test_")
os.environ["M5_VOICE_DATA_DIR"] = _tmp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.app import (  # noqa: E402
    AgentJob,
    STALENESS_THRESHOLD_S,
    _fail_stale_in_progress_jobs,
    _load_jobs_unlocked,
    _save_jobs_unlocked,
)


def _make_job(job_id: str, status: str, age_seconds: float) -> AgentJob:
    updated = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    iso = updated.isoformat(timespec="seconds")
    return AgentJob(
        id=job_id,
        device="test",
        event="test",
        prompt="test",
        status=status,  # type: ignore[arg-type]
        created_at=iso,
        updated_at=iso,
    )


def test_stale_in_progress_is_failed() -> None:
    _save_jobs_unlocked([_make_job("j1", "in_progress", STALENESS_THRESHOLD_S + 5)])
    recovered = _fail_stale_in_progress_jobs()
    assert recovered == 1, f"expected 1 recovered, got {recovered}"
    jobs = _load_jobs_unlocked()
    assert jobs[0].status == "failed"
    assert jobs[0].error == "agent worker was unreachable"


def test_fresh_in_progress_is_left_alone() -> None:
    _save_jobs_unlocked([_make_job("j2", "in_progress", 1)])
    recovered = _fail_stale_in_progress_jobs()
    assert recovered == 0, f"expected 0 recovered, got {recovered}"
    assert _load_jobs_unlocked()[0].status == "in_progress"


def test_non_in_progress_jobs_are_ignored() -> None:
    _save_jobs_unlocked([
        _make_job("j3", "queued", STALENESS_THRESHOLD_S + 60),
        _make_job("j4", "done", STALENESS_THRESHOLD_S + 60),
        _make_job("j5", "failed", STALENESS_THRESHOLD_S + 60),
    ])
    recovered = _fail_stale_in_progress_jobs()
    assert recovered == 0
    statuses = {j.id: j.status for j in _load_jobs_unlocked()}
    assert statuses == {"j3": "queued", "j4": "done", "j5": "failed"}


def test_malformed_updated_at_treated_as_stale() -> None:
    """A job with an unparseable timestamp should be treated as stale and failed —
    safer than leaving it pinned in_progress forever."""
    job = _make_job("j6", "in_progress", 0)
    job.updated_at = "not-a-real-iso-date"
    _save_jobs_unlocked([job])
    recovered = _fail_stale_in_progress_jobs()
    assert recovered == 1
    assert _load_jobs_unlocked()[0].status == "failed"


def test_mixed_batch_only_stale_recovered() -> None:
    _save_jobs_unlocked([
        _make_job("fresh", "in_progress", 1),
        _make_job("stale1", "in_progress", STALENESS_THRESHOLD_S + 10),
        _make_job("stale2", "in_progress", STALENESS_THRESHOLD_S + 99),
        _make_job("done_old", "done", STALENESS_THRESHOLD_S + 99),
    ])
    recovered = _fail_stale_in_progress_jobs()
    assert recovered == 2
    statuses = {j.id: j.status for j in _load_jobs_unlocked()}
    assert statuses == {
        "fresh": "in_progress",
        "stale1": "failed",
        "stale2": "failed",
        "done_old": "done",
    }


def main() -> None:
    tests = [
        test_stale_in_progress_is_failed,
        test_fresh_in_progress_is_left_alone,
        test_non_in_progress_jobs_are_ignored,
        test_malformed_updated_at_treated_as_stale,
        test_mixed_batch_only_stale_recovered,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} staleness-recovery tests passed.")


if __name__ == "__main__":
    main()
