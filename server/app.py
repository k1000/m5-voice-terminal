from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock, Thread
from typing import Any, Literal
from uuid import uuid4
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
import unicodedata

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket, WebSocketDisconnect

app = FastAPI(title="M5StickS3 Voice Terminal MVP")

# STT_BACKEND choices:
# - whisper: OpenAI Whisper via PyTorch. Reliable baseline.
# - mlx-whisper: Apple Silicon/MLX Whisper. Usually faster on this Mac; supports HF repos.
STT_BACKEND = os.environ.get("STT_BACKEND", "whisper")
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "base")
MLX_WHISPER_MODEL = os.environ.get("MLX_WHISPER_MODEL", "mlx-community/whisper-tiny")
DATA_DIR = Path(os.environ.get("M5_VOICE_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
JOBS_FILE = DATA_DIR / "agent_jobs.json"
AUDIO_DIR = DATA_DIR / "audio"
IMAGE_DIR = DATA_DIR / "images"
IMAGE_CACHE_DIR = IMAGE_DIR / "cache"
TTS_VOICE = os.environ.get("TTS_VOICE", "M1")
TTS_LANG = os.environ.get("TTS_LANG", "en")
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "44100"))

_whisper_model: Any | None = None
_tts_engine: Any | None = None
_jobs_lock = Lock()
_job_subscribers: dict[str, list[WebSocket]] = {}
_subscribers_lock = asyncio.Lock()


class CommandRequest(BaseModel):
    device: str
    event: str
    text: str | None = None


class CommandResponse(BaseModel):
    text: str
    transcript: str | None = None
    meta: dict[str, Any]


class AgentJob(BaseModel):
    id: str
    device: str
    event: str
    prompt: str
    status: Literal["queued", "in_progress", "done", "failed"] = "queued"
    created_at: str
    updated_at: str
    claimed_by: str | None = None
    result_text: str | None = None
    error: str | None = None
    sentiment: Literal["happy", "neutral", "sad"] = "neutral"
    audio_url: str | None = None
    image_url: str | None = None
    options: list[str] = Field(default_factory=list, max_length=4)
    metrics: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    text: str | None = None
    status: Literal["done", "failed"] = "done"
    error: str | None = None
    # If omitted, the server infers a simple happy/neutral/sad face from the result text/status.
    sentiment: Literal["happy", "neutral", "sad"] | None = None
    audio_url: str | None = None
    image_url: str | None = None
    # Optional prompt that asks the server to generate a 135x135 RGB565 response image.
    image_prompt: str | None = None
    options: list[str] = Field(default_factory=list, max_length=4)
    metrics: dict[str, Any] = Field(default_factory=dict)


class AgentJobList(BaseModel):
    jobs: list[AgentJob]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_jobs_unlocked() -> list[AgentJob]:
    _ensure_data_dir()
    if not JOBS_FILE.exists():
        return []
    raw = json.loads(JOBS_FILE.read_text())
    return [AgentJob.model_validate(item) for item in raw]


def _save_jobs_unlocked(jobs: list[AgentJob]) -> None:
    _ensure_data_dir()
    tmp = JOBS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps([job.model_dump() for job in jobs], indent=2))
    tmp.replace(JOBS_FILE)


def infer_sentiment(text: str | None, status: str = "done") -> Literal["happy", "neutral", "sad"]:
    """Small fallback classifier for selecting the StickS3 response face."""
    if status == "failed":
        return "sad"
    lowered = (text or "").lower()
    sad_terms = (
        "fail", "failed", "error", "sorry", "can't", "cannot", "blocked", "timeout",
        "problem", "issue", "bad", "not found", "unable", "denied", "unsafe",
    )
    happy_terms = (
        "done", "success", "successful", "ok", "great", "good", "ready", "fixed",
        "working", "complete", "completed", "created", "saved", "uploaded", "yes",
    )
    if any(term in lowered for term in sad_terms):
        return "sad"
    if any(term in lowered for term in happy_terms):
        return "happy"
    return "neutral"


def enqueue_agent_job(device: str, event: str, prompt: str, metrics: dict[str, Any] | None = None) -> AgentJob:
    now = utc_now()
    job = AgentJob(
        id=uuid4().hex[:12],
        device=device,
        event=event,
        prompt=prompt,
        created_at=now,
        updated_at=now,
        metrics=metrics or {},
    )
    with _jobs_lock:
        jobs = _load_jobs_unlocked()
        jobs.append(job)
        _save_jobs_unlocked(jobs)
    return job


def get_whisper_model() -> Any:
    """Lazy-load local OpenAI Whisper only when an audio request arrives."""
    global _whisper_model
    if _whisper_model is None:
        import whisper

        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    return _whisper_model


def get_tts_engine() -> Any:
    """Lazy-load Supertonic TTS when an agent result needs audio."""
    global _tts_engine
    if _tts_engine is None:
        from supertonic import TTS

        _tts_engine = TTS(auto_download=True)
    return _tts_engine


def cleanup_old_audio(keep_job_id: str | None = None) -> int:
    """Delete prior generated audio files so responses do not accumulate."""
    _ensure_data_dir()
    removed = 0
    keep_name = f"{keep_job_id}.wav" if keep_job_id else None
    for path in AUDIO_DIR.glob("*.wav"):
        if keep_name and path.name == keep_name:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def image_cache_key(prompt: str) -> str:
    """Stable cache key for generated Stick-sized images."""
    normalized = " ".join(prompt.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def generate_response_image(prompt: str, job_id: str) -> tuple[str | None, dict[str, Any]]:
    """Generate or reuse a 135x135 RGB565 image and return its download URL if successful."""
    _ensure_data_dir()
    started = time.perf_counter()
    cache_key = image_cache_key(prompt)
    metrics: dict[str, Any] = {"server_image_prompt": prompt[:180], "server_image_cache_key": cache_key}
    helper = Path(os.environ.get("MINIMAX_IMAGE_HELPER", "/Users/kamil/.pi/agent/skills/minimax-image/scripts/minimax_image.py"))
    jpg_path = IMAGE_DIR / f"{job_id}.jpg"
    rgb565_path = IMAGE_DIR / f"{job_id}.rgb565"
    cache_rgb565_path = IMAGE_CACHE_DIR / f"{cache_key}.rgb565"
    cache_jpg_path = IMAGE_CACHE_DIR / f"{cache_key}.jpg"
    try:
        if cache_rgb565_path.exists():
            shutil.copyfile(cache_rgb565_path, rgb565_path)
            metrics.update({
                "server_image_s": round(time.perf_counter() - started, 3),
                "server_image_size": "135x135",
                "server_image_format": "rgb565-le",
                "server_image_cached": True,
            })
            return f"/image/{job_id}", metrics

        if not helper.exists():
            raise RuntimeError(f"MiniMax image helper not found: {helper}")
        import subprocess
        subprocess.run(
            [sys.executable, str(helper), prompt, "--square", "135", "--output", str(jpg_path)],
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        from PIL import Image

        with Image.open(jpg_path) as image:
            image = image.convert("RGB").resize((135, 135), Image.Resampling.LANCZOS)
            image.save(cache_jpg_path, quality=95)
            raw = bytearray()
            for r, g, b in image.getdata():
                value = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
                raw.extend(value.to_bytes(2, "little"))
        cache_rgb565_path.write_bytes(raw)
        shutil.copyfile(cache_rgb565_path, rgb565_path)
        metrics.update({
            "server_image_s": round(time.perf_counter() - started, 3),
            "server_image_size": "135x135",
            "server_image_format": "rgb565-le",
            "server_image_cached": False,
        })
        return f"/image/{job_id}", metrics
    except Exception as exc:
        metrics.update({
            "server_image_s": round(time.perf_counter() - started, 3),
            "server_image_error": repr(exc),
        })
        return None, metrics


def schedule_response_image(prompt: str, job_id: str) -> None:
    """Generate response image in the background so voice answers are not delayed."""
    def worker() -> None:
        image_url, image_metrics = generate_response_image(prompt, job_id)
        with _jobs_lock:
            jobs = _load_jobs_unlocked()
            for index, job in enumerate(jobs):
                if job.id == job_id:
                    if image_url:
                        job.image_url = image_url
                    job.metrics.update(image_metrics)
                    job.updated_at = utc_now()
                    jobs[index] = job
                    _save_jobs_unlocked(jobs)
                    return

    Thread(target=worker, name=f"image-{job_id}", daemon=True).start()


def sanitize_tts_text(text: str) -> str:
    """Remove invisible/emoji variation chars that Supertonic rejects."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in normalized
        if unicodedata.category(ch) not in {"Mn", "Me", "Cf"}
    ).strip()


def generate_tts_audio(text: str, job_id: str) -> tuple[str, dict[str, Any]]:
    """Generate a WAV response with Supertonic and return its download URL."""
    started = time.perf_counter()
    removed = cleanup_old_audio()
    tts_text = sanitize_tts_text(text)
    if not tts_text:
        return "", {"server_tts_s": 0, "server_tts_skipped": True, "tts_removed_old_files": removed}

    import soundfile as sf

    engine = get_tts_engine()
    style = engine.get_voice_style(voice_name=TTS_VOICE)
    try:
        wav, duration = engine.synthesize(tts_text, voice_style=style)
    except ValueError as exc:
        ascii_text = tts_text.encode("ascii", "ignore").decode().strip()
        if not ascii_text or ascii_text == tts_text:
            return "", {"server_tts_s": round(time.perf_counter() - started, 3), "server_tts_error": repr(exc), "tts_removed_old_files": removed}
        wav, duration = engine.synthesize(ascii_text, voice_style=style)
        tts_text = ascii_text
    path = AUDIO_DIR / f"{job_id}.wav"
    sample_rate = int(getattr(engine, "sample_rate", TTS_SAMPLE_RATE) or TTS_SAMPLE_RATE)
    # Write PCM_16 because M5Unified Speaker.playWav supports PCM WAV up to 16-bit.
    # Supertonic currently outputs 44.1 kHz; using 24 kHz made Stick playback too slow.
    sf.write(path, wav.squeeze().astype("float32"), sample_rate, subtype="PCM_16")
    # Public URL uses the exact job id; the server maps it to <job_id>.wav internally.
    return f"/audio/{job_id}", {
        "server_tts_s": round(time.perf_counter() - started, 3),
        "tts_engine": "supertonic",
        "tts_voice": TTS_VOICE,
        "tts_lang": TTS_LANG,
        "tts_sample_rate": sample_rate,
        "tts_removed_old_files": removed,
        "tts_duration_s": round(float(duration[0]) if hasattr(duration, "__len__") else float(duration), 3),
        "tts_text_sanitized": tts_text != text.strip(),
    }


def transcribe_audio(path: str) -> tuple[str, dict[str, Any]]:
    """Transcribe uploaded audio using the configured local STT backend."""
    if STT_BACKEND == "whisper":
        model = get_whisper_model()
        result = model.transcribe(
            path,
            fp16=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        transcript = str(result.get("text", "")).strip()
        segments = result.get("segments") or []
        no_speech_prob = None
        if segments:
            probs = [s.get("no_speech_prob") for s in segments if s.get("no_speech_prob") is not None]
            if probs:
                no_speech_prob = max(probs)
        return transcript, {
            "stt_backend": "whisper",
            "stt_model": WHISPER_MODEL_NAME,
            "language": result.get("language"),
            "no_speech_prob": no_speech_prob,
        }

    if STT_BACKEND == "mlx-whisper":
        import mlx_whisper

        result = mlx_whisper.transcribe(
            path,
            path_or_hf_repo=MLX_WHISPER_MODEL,
            verbose=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        transcript = str(result.get("text", "")).strip()
        return transcript, {
            "stt_backend": "mlx-whisper",
            "stt_model": MLX_WHISPER_MODEL,
            "language": result.get("language"),
        }

    raise HTTPException(status_code=500, detail=f"unsupported STT_BACKEND={STT_BACKEND!r}")


def normalize_options(options: list[str]) -> list[str]:
    """Keep only concise button labels; the Stick has room for four short choices."""
    normalized = [opt.strip()[:24] for opt in options[:3] if opt and len(opt.strip().split()) <= 2]
    return [*normalized, "New request"]


def queue_response(device: str, prompt: str, event: str, meta_extra: dict[str, Any] | None = None) -> CommandResponse:
    job = enqueue_agent_job(device=device, event=event, prompt=prompt, metrics=meta_extra)
    return CommandResponse(
        text=f"Queued for Pi agent. Job {job.id}: {prompt}",
        transcript=prompt,
        meta={
            "job_id": job.id,
            "status": job.status,
            "agent_next_url": "/agent/jobs/next",
            "agent_result_url": f"/agent/jobs/{job.id}/result",
            **(meta_extra or {}),
        },
    )


def completed_response(
    device: str,
    event: str,
    prompt: str,
    text: str,
    sentiment: Literal["happy", "neutral", "sad"] = "neutral",
    meta_extra: dict[str, Any] | None = None,
    options: list[str] | None = None,
    generate_audio: bool = False,
) -> CommandResponse:
    """Create an already-done job for deterministic server-side replies."""
    now = utc_now()
    job = AgentJob(
        id=uuid4().hex[:12],
        device=device,
        event=event,
        prompt=prompt,
        status="done",
        created_at=now,
        updated_at=now,
        result_text=text,
        sentiment=sentiment,
        options=normalize_options(options or []),
        metrics=meta_extra or {},
    )
    if generate_audio:
        audio_url, audio_metrics = generate_tts_audio(text, job.id)
        job.audio_url = audio_url or None
        job.metrics.update(audio_metrics)
    job.metrics["server_short_circuit"] = True
    job.metrics["server_total_until_done_s"] = 0
    with _jobs_lock:
        jobs = _load_jobs_unlocked()
        jobs.append(job)
        _save_jobs_unlocked(jobs)
    return CommandResponse(
        text=text,
        transcript=prompt,
        meta={"job_id": job.id, "status": job.status, **(meta_extra or {})},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/image/{image_id}")
def download_image(image_id: str) -> FileResponse:
    """Download generated RGB565 sentiment image by job id."""
    safe_id = Path(image_id).name
    path = IMAGE_DIR / f"{safe_id}.rgb565"
    if not path.exists() and safe_id.endswith(".rgb565"):
        path = IMAGE_DIR / safe_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.get("/audio/{audio_id}")
def download_audio(audio_id: str) -> FileResponse:
    """Download generated Supertonic response audio by job id.

    The public audio id is the same as the job id. Internally the file is stored
    as <job_id>.wav.
    """
    safe_id = Path(audio_id).name
    path = AUDIO_DIR / f"{safe_id}.wav"
    if not path.exists() and safe_id.endswith(".wav"):
        # Backward compatibility for older /audio/<job_id>.wav URLs.
        path = AUDIO_DIR / safe_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/models")
def models() -> dict[str, Any]:
    """Report locally available server-side audio/STT/TTS options discovered for this MVP."""
    import importlib.util

    return {
        "selected_stt": {
            "backend": STT_BACKEND,
            "whisper_model": WHISPER_MODEL_NAME,
            "mlx_whisper_model": MLX_WHISPER_MODEL,
        },
        "audio_tools": {
            "openai_whisper": importlib.util.find_spec("whisper") is not None,
            "mlx_whisper": importlib.util.find_spec("mlx_whisper") is not None,
            "supertonic_tts": importlib.util.find_spec("supertonic") is not None,
            "mlx": importlib.util.find_spec("mlx") is not None,
            "transformers": importlib.util.find_spec("transformers") is not None,
        },
        "recommended_stt_path": [
            "MVP/default: STT_BACKEND=whisper WHISPER_MODEL=base",
            "Apple Silicon speed test: STT_BACKEND=mlx-whisper MLX_WHISPER_MODEL=mlx-community/whisper-tiny",
            "Accuracy test: STT_BACKEND=mlx-whisper MLX_WHISPER_MODEL=mlx-community/whisper-base-mlx or mlx-community/whisper-large-v3-turbo",
            "Future low-latency test: Moonshine or Parakeet if we add those dependencies.",
        ],
        "notes": [
            "Whisper is the reliable baseline for speech-to-text.",
            "mlx-whisper is installed and is the next candidate for faster Apple Silicon inference.",
            "Supertonic is for text-to-speech output, not STT.",
            "Voice prompts are queued for Pi/agent processing via /agent/jobs/next and /agent/jobs/{id}/result.",
        ],
    }


@app.post("/command", response_model=CommandResponse)
def command(payload: CommandRequest) -> CommandResponse:
    prompt = payload.text or "button press"
    return queue_response(payload.device, prompt, payload.event)


@app.post("/voice-command", response_model=CommandResponse)
async def voice_command(
    device: str = Form("m5sticks3-01"),
    event: str = Form("audio_upload"),
    audio: UploadFile = File(...),
) -> CommandResponse:
    """Accept audio from StickS3, transcribe on the server, queue prompt for Pi."""
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    total_start = time.perf_counter()
    bytes_written = 0
    with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        write_start = time.perf_counter()
        while chunk := await audio.read(1024 * 1024):
            bytes_written += len(chunk)
            tmp.write(chunk)
        tmp.flush()
        upload_read_s = time.perf_counter() - write_start

        stt_start = time.perf_counter()
        transcript, stt_meta = transcribe_audio(tmp.name)
        stt_s = time.perf_counter() - stt_start

    stt_meta.update({
        "audio_bytes": bytes_written,
        "server_upload_read_s": round(upload_read_s, 3),
        "server_stt_s": round(stt_s, 3),
        "server_voice_command_s": round(time.perf_counter() - total_start, 3),
    })
    if not transcript:
        return completed_response(
            device=device,
            event=event,
            prompt="[no speech detected]",
            text="No howl heard. Try again.",
            sentiment="neutral",
            meta_extra={**stt_meta, "server_no_speech_short_circuit": True},
            options=["Try again"],
        )
    return queue_response(device, transcript, event, meta_extra=stt_meta)


@app.get("/agent/jobs", response_model=AgentJobList)
def list_agent_jobs(status: str | None = None) -> AgentJobList:
    """List queued/in-progress/done jobs for debugging and manual operation."""
    with _jobs_lock:
        jobs = _load_jobs_unlocked()
    if status is not None:
        jobs = [job for job in jobs if job.status == status]
    return AgentJobList(jobs=jobs)


@app.get("/agent/jobs/next", response_model=AgentJob | None)
def claim_next_agent_job(worker: str = "pi") -> AgentJob | None:
    """Claim the oldest queued voice prompt so Pi/agent can process it."""
    with _jobs_lock:
        jobs = _load_jobs_unlocked()
        for index, job in enumerate(jobs):
            if job.status == "queued":
                job.status = "in_progress"
                job.claimed_by = worker
                job.updated_at = utc_now()
                jobs[index] = job
                _save_jobs_unlocked(jobs)
                return job
    return None


def _get_job_unlocked(jobs: list[AgentJob], job_id: str) -> AgentJob | None:
    """Look up a job by id within an already-loaded list; returns None if not found."""
    for job in jobs:
        if job.id == job_id:
            return job
    return None


async def _notify_subscribers(job_id: str) -> None:
    """Push the final job result to every WebSocket subscriber for *job_id*.

    Sends the payload followed by a _ws_close marker so the handler exits its
    receive loop and closes the connection.  We do *not* call ws.close() here
    because the handler's finally-block also closes — we make both sides
    idempotent with try/except.
    """
    async with _subscribers_lock:
        sockets = _job_subscribers.pop(job_id, [])
    if not sockets:
        return
    with _jobs_lock:
        jobs = _load_jobs_unlocked()
    job = _get_job_unlocked(jobs, job_id)
    if job is None:
        return
    payload = job.model_dump()
    for ws in sockets:
        try:
            await ws.send_json(payload)
            await ws.send_json({"_ws_close": True})
        except Exception:
            # Socket already closed or errored — handler's finally will clean up.
            pass


@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_status(websocket: WebSocket, job_id: str) -> None:
    """Push real-time job status to the Stick via WebSocket.

    The client connects immediately after receiving a job_id from /voice-command
    and receives push messages rather than polling."""
    await websocket.accept()

    # Register as subscriber so set_agent_job_result can notify us.
    async with _subscribers_lock:
        _job_subscribers.setdefault(job_id, []).append(websocket)

    try:
        # Send a snapshot of the current state right away.
        with _jobs_lock:
            jobs = _load_jobs_unlocked()
        job = _get_job_unlocked(jobs, job_id)
        if job:
            await websocket.send_json(job.model_dump())
            if job.status in ("done", "failed"):
                async with _subscribers_lock:
                    sockets = _job_subscribers.get(job_id, [])
                    if websocket in sockets:
                        sockets.remove(websocket)
                await websocket.close()
                return

        # Wait for the job to complete.  _notify_subscribers sends the final
        # payload followed by {"_ws_close": true}; we break and close when we see it.
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            elif '"_ws_close"' in data:
                break
    except WebSocketDisconnect:
        pass
    finally:
        async with _subscribers_lock:
            sockets = _job_subscribers.get(job_id, [])
            if websocket in sockets:
                sockets.remove(websocket)
        try:
            await websocket.close()
        except Exception:
            pass  # idempotent: already closed by early-return or _notify_subscribers caller.


@app.get("/agent/jobs/{job_id}", response_model=AgentJob)
def get_agent_job(job_id: str) -> AgentJob:
    with _jobs_lock:
        jobs = _load_jobs_unlocked()
    job = _get_job_unlocked(jobs, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/agent/jobs/{job_id}/result", response_model=AgentJob)
def set_agent_job_result(job_id: str, result: AgentResult, background_tasks: BackgroundTasks) -> AgentJob:
    """Store Pi/agent output so the device or UI can retrieve it."""
    result_post_start = time.perf_counter()
    image_prompt_to_schedule: str | None = None
    with _jobs_lock:
        jobs = _load_jobs_unlocked()
        for index, job in enumerate(jobs):
            if job.id == job_id:
                job.status = result.status
                job.result_text = result.text
                job.error = result.error
                job.sentiment = result.sentiment or infer_sentiment(result.text or result.error, result.status)
                job.audio_url = result.audio_url
                job.image_url = result.image_url
                job.options = normalize_options(result.options)
                job.metrics.update(result.metrics)
                if result.status == "done" and result.image_prompt and not job.image_url:
                    image_prompt_to_schedule = result.image_prompt
                    job.metrics.update({
                        "server_image_prompt": result.image_prompt[:180],
                        "server_image_queued": True,
                    })
                if result.status == "done" and result.text and not job.audio_url:
                    audio_url, tts_metrics = generate_tts_audio(result.text, job.id)
                    job.audio_url = audio_url or None
                    job.metrics.update(tts_metrics)
                now = datetime.now(timezone.utc)
                try:
                    created_at = datetime.fromisoformat(job.created_at)
                    job.metrics["server_total_until_done_s"] = round((now - created_at).total_seconds(), 3)
                except ValueError:
                    pass
                job.metrics["server_result_post_s"] = round(time.perf_counter() - result_post_start, 3)
                job.updated_at = now.isoformat(timespec="seconds")
                jobs[index] = job
                _save_jobs_unlocked(jobs)
                if image_prompt_to_schedule:
                    schedule_response_image(image_prompt_to_schedule, job.id)
                background_tasks.add_task(_notify_subscribers, job.id)
                return job
    raise HTTPException(status_code=404, detail="job not found")
