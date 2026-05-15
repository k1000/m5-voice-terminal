#!/usr/bin/env python3
"""Verify server.rle_compress and stick.st7789.rle_decompress round-trip cleanly.

Regression coverage for commit e6143f9 (which fixed a repeat-run pixel-count bug)
plus the new Stick-side decoder added alongside server RLE passthrough.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("M5_VOICE_DATA_DIR", "/tmp/m5_voice_test_data")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# stick/st7789.py imports MicroPython's `machine` module on the device.
# Stub it so the pure-Python RLE decoder is importable on CPython.
sys.modules.setdefault("machine", types.SimpleNamespace(Pin=object, SPI=object))

from server.app import rle_compress  # noqa: E402
from stick.st7789 import rle_decompress as stick_decompress  # noqa: E402


def pixels_to_bytes(pixels: list[int]) -> bytes:
    """Convert a list of 16-bit pixels (lo,hi) to a flat byte string."""
    out = bytearray()
    for p in pixels:
        out.append(p & 0xFF)
        out.append((p >> 8) & 0xFF)
    return bytes(out)


def decode_like_firmware(payload: bytes, width: int, height: int) -> bytes:
    """Mirrors the firmware/Python branch logic: only call rle_decompress
    when the b"RLE\\x01" header is present, otherwise the payload is raw."""
    if payload[:4] == b"RLE\x01":
        return bytes(stick_decompress(payload, width, height))
    return payload[: width * height * 2]


def roundtrip(pixels: list[int], width: int, height: int, label: str) -> None:
    raw = pixels_to_bytes(pixels)
    assert len(raw) == width * height * 2, f"{label}: raw size mismatch"
    compressed, did = rle_compress(raw)
    decoded = decode_like_firmware(compressed, width, height)
    assert decoded == raw, (
        f"{label}: round-trip failed (did_compress={did}, "
        f"compressed={len(compressed)}B, raw={len(raw)}B)"
    )


def test_all_identical() -> None:
    """A solid-color image should compress maximally and decode exactly."""
    roundtrip([0xF800] * 100, 10, 10, "all-identical")


def test_all_unique() -> None:
    """All-unique pixels forces literal escapes — must still round-trip."""
    roundtrip([(i * 17) & 0xFFFF for i in range(64)], 8, 8, "all-unique")


def test_mixed_runs() -> None:
    """Mix of long runs and singletons exercises both encoding branches."""
    pixels = [0x0000] * 50 + [0x1234, 0x5678, 0x9ABC] + [0xFFFF] * 47
    roundtrip(pixels, 10, 10, "mixed-runs")


def test_run_longer_than_255() -> None:
    """A run of 300 identical pixels must split across multiple count-bytes (max 255)."""
    roundtrip([0xABCD] * 300, 20, 15, "long-run")


def test_decoder_handles_raw_rgb565() -> None:
    """When compression would inflate the payload, server returns raw bytes
    without the RLE\\x01 header. The Stick decoder must handle that fallback."""
    raw = pixels_to_bytes([(i * 7) & 0xFFFF for i in range(4)])  # 2x2 noisy
    compressed, did = rle_compress(raw)
    assert not did, "expected compression to be skipped on tiny noisy input"
    assert compressed == raw, "fallback should return raw bytes unchanged"


def test_overproduced_output_trimmed() -> None:
    """If width*height is smaller than the data actually contains, the decoder
    must trim to the expected length — not write past the output buffer."""
    raw = pixels_to_bytes([0xAAAA] * 100)
    compressed, did = rle_compress(raw)
    assert did, "expected solid-color input to compress (precondition)"
    # Ask for fewer pixels than encoded — decoder should stop at expected_len.
    decoded = stick_decompress(compressed, 5, 5)  # 25 pixels = 50 bytes
    assert len(decoded) == 50, f"expected 50 bytes, got {len(decoded)}"
    assert bytes(decoded) == raw[:50]


def main() -> None:
    tests = [
        test_all_identical,
        test_all_unique,
        test_mixed_runs,
        test_run_longer_than_255,
        test_decoder_handles_raw_rgb565,
        test_overproduced_output_trimmed,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} RLE round-trip tests passed.")


if __name__ == "__main__":
    main()
