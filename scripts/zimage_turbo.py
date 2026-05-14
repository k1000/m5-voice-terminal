#!/usr/bin/env -S uv run --with mflux --quiet python3
"""
Z-Image-Turbo image generation helper for m5-voice-terminal.

Usage (via server):
    python scripts/zimage_turbo.py "a happy robot" --output /tmp/img.png --square 135

Standalone usage:
    python scripts/zimage_turbo.py "a happy robot" -o /tmp/img.png --steps 4

Uses the mflux pip package which wraps Z-Image-Turbo.
For --square N: rounds down to nearest multiple of 16 (e.g. 135→128).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from mflux.models.z_image import ZImageTurbo
except ImportError:
    print("ERROR: mflux not installed. Run: uv pip install mflux", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Z-Image-Turbo image generation")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--output", "-o", required=True, help="Output PNG path")
    parser.add_argument("--square", type=int, default=128,
                        help="Square dimension; rounds down to nearest multiple of 16")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    # Compatibility args accepted but ignored.
    parser.add_argument("--model", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    # Server always passes --square N (135); round down to multiple of 16 for mflux.
    size = (args.square // 16) * 16
    if size < 16:
        size = 128

    model = ZImageTurbo(quantize=8)
    image = model.generate_image(
        prompt=args.prompt,
        seed=args.seed,
        num_inference_steps=args.steps,
        width=size,
        height=size,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(out_path))
    print(f"Saved {size}×{size} image to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
