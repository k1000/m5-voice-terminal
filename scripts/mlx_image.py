#!/usr/bin/env python3
"""Generate one image locally using MLX-optimized Stable Diffusion via PyTorch MPS.

This helper generates images entirely offline on Apple Silicon using PyTorch's MPS
(Metal Performance Shaders) backend.  It mirrors the interface of the MiniMax
helper so the server can switch backends without code changes.

Usage:
    python3 mlx_image.py "a happy robot face" --square 135 --output /tmp/robot.jpg
    python3 mlx_image.py "a wolf" --size 270x270 --output /tmp/wolf.jpg

Backends tried in order:
    1. PyTorch MPS  (Apple Silicon GPU, installed with torch)
    2. CPU fallback (slow; not recommended)

Models tried in order:
    1. stabilityai/sdxl-turbo  — 2 steps, good quality, ~7GB VRAM
    2. stabilityai/stable-diffusion-2-1  — more steps, larger model
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Argument parsing (mirrors minimax_image.py interface)
# ---------------------------------------------------------------------------

def parse_size(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise SystemExit(f"Invalid size {s!r}; use WxH, e.g. 135x135")
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise SystemExit(f"Invalid size {s!r}: {exc}") from exc
    if w <= 0 or h <= 0:
        raise SystemExit(f"Size must be positive: {w}x{h}")
    return w, h


def infer_aspect_ratio(args: argparse.Namespace) -> str:
    if hasattr(args, "square") and args.square is not None:
        return "1:1"
    if hasattr(args, "size") and args.size:
        w, h = parse_size(args.size)
        g = math.gcd(w, h)
        return f"{w // g}:{h // g}"
    return "1:1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image locally with PyTorch MPS")
    parser.add_argument("prompt", help="Image prompt")
    parser.add_argument("--output", "-o", default="mlx_output.jpg", help="Output image path")
    parser.add_argument(
        "--size",
        help="Final output size as WIDTHxHEIGHT, e.g. 135x135",
    )
    parser.add_argument(
        "--square",
        type=int,
        help="Shortcut: --size NxN and aspect ratio 1:1",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of diffusion steps (default: 4; sdxl-turbo is good at 2-4)",
    )
    parser.add_argument(
        "--model",
        default="stabilityai/sdxl-turbo",
        help="HuggingFace model id (default: stabilityai/sdxl-turbo)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale (default: 0.0; sdxl-turbo ignores it)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    if args.square is not None and args.square > 0:
        args.size = f"{args.square}x{args.square}"
    if not args.size:
        args.size = "256x256"
    return args


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

_pipe = None
_pipe_model_id = None


def get_pipe(model_id: str, device: str):
    """Lazily load (and cache) the diffusers pipeline."""
    global _pipe, _pipe_model_id
    if _pipe is not None and _pipe_model_id == model_id:
        return _pipe

    print(f"Loading model {model_id!r} on {device} ...", file=sys.stderr)
    t0 = time.perf_counter()

    import torch

    torch_device = torch.device(device)

    # Use the specific pipeline class to avoid auto_pipeline importing broken
    # glm_image pipeline (diffusers 0.38 + transformers 5 incompatibility).
    if "turbo" in model_id.lower() or "sdxl" in model_id.lower():
        from diffusers import StableDiffusionXLPipeline
        pipe_cls = StableDiffusionXLPipeline
    else:
        from diffusers import StableDiffusionPipeline
        pipe_cls = StableDiffusionPipeline

    pipe = pipe_cls.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        variant="fp16" if device == "mps" else None,
    )
    pipe = pipe.to(torch_device)

    # Warm-up on first load (slow)
    try:
        pipe("", num_inference_steps=1, guidance_scale=0.0)
        print(f"Model loaded in {time.perf_counter() - t0:.1f}s (warm-up done)", file=sys.stderr)
    except Exception as warmup_err:
        print(f"Warning: warm-up failed ({warmup_err}); continuing", file=sys.stderr)

    _pipe = pipe
    _pipe_model_id = model_id
    print(f"Model ready in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return pipe


def generate(prompt: str, args: argparse.Namespace) -> Image.Image:
    device = get_device()
    if device == "cpu":
        print("Warning: MPS not available; using CPU (slow)", file=sys.stderr)

    pipe = get_pipe(args.model, device)

    generator: dict | None = None
    if args.seed is not None:
        import torch
        generator = {"generator": torch.Generator(device=device).manual_seed(args.seed)}

    w, h = parse_size(args.size)

    # SDXL latent dimensions must be divisible by 8; round up and resize afterward.
    latent_w = ((w + 7) // 8) * 8
    latent_h = ((h + 7) // 8) * 8

    image = pipe(
        prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=latent_h,
        width=latent_w,
        **({} if generator is None else generator),
    ).images[0]

    # Resize to exact requested dimensions.
    if image.width != w or image.height != h:
        image = image.resize((w, h), Image.Resampling.LANCZOS)
    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    prompt = args.prompt.strip()
    if not prompt:
        raise SystemExit("Error: prompt cannot be empty")

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    image = generate(prompt, args)
    print(f"Generated in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    image = image.convert("RGB")
    # Always save as PNG to preserve exact pixel values.  The server converts
    # to RGB565 and RLE-compresses; lossy JPEG would destroy RLE runs.
    png_output = output.with_suffix(".png")
    image.save(png_output, format="PNG", compress_level=0)
    # Also write the requested path if it differs (so callers expecting .jpg work).
    if png_output != output:
        import shutil
        shutil.copy(png_output, output)
    print(f"Saved: {png_output} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
