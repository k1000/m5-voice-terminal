#!/usr/bin/env python3
"""Benchmark local STT backends on the same audio files.

Usage:
  ./scripts/benchmark_stt.py samples/*.wav
  ./scripts/benchmark_stt.py --backend mlx-whisper --model mlx-community/whisper-base sample.wav
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def transcribe_whisper(path: Path, model_name: str) -> tuple[str, dict[str, Any]]:
    import whisper

    t0 = time.perf_counter()
    model = whisper.load_model(model_name)
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    result = model.transcribe(str(path), fp16=False, condition_on_previous_text=False, no_speech_threshold=0.6)
    infer_s = time.perf_counter() - t1
    return str(result.get("text", "")).strip(), {
        "backend": "whisper",
        "model": model_name,
        "load_s": load_s,
        "infer_s": infer_s,
        "language": result.get("language"),
    }


def transcribe_mlx_whisper(path: Path, model_name: str) -> tuple[str, dict[str, Any]]:
    import mlx_whisper

    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=model_name,
        verbose=False,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    total_s = time.perf_counter() - t0
    return str(result.get("text", "")).strip(), {
        "backend": "mlx-whisper",
        "model": model_name,
        "total_s": total_s,
        "language": result.get("language"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--backend", choices=["whisper", "mlx-whisper"], default="whisper")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.model is None:
        args.model = "base" if args.backend == "whisper" else "mlx-community/whisper-tiny"

    for audio in args.audio:
        if args.backend == "whisper":
            text, meta = transcribe_whisper(audio, args.model)
        else:
            text, meta = transcribe_mlx_whisper(audio, args.model)
        print(json.dumps({"audio": str(audio), "text": text, **meta}, indent=2))


if __name__ == "__main__":
    main()
