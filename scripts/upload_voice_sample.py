#!/usr/bin/env python3
"""Upload a WAV/MP3/M4A file to /voice-command for end-to-end STT queue testing."""

from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.request
from pathlib import Path


def encode_multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = "----m5voiceboundary7MA4YWxkTrZu0gW"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(
        (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{file_field}\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--device", default="sample-uploader")
    args = parser.parse_args()

    body, content_type = encode_multipart(
        {"device": args.device, "event": "sample_upload"},
        "audio",
        args.audio,
    )
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/voice-command",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        print(json.dumps(json.loads(response.read().decode()), indent=2))


if __name__ == "__main__":
    main()
