# STT Backend Review

Date: 2026-05-13

Goal: choose a speech-to-text backend for short StickS3 voice commands.

## Current candidates

| Backend | Model | Status | Notes |
|---|---|---:|---|
| OpenAI Whisper | `tiny` | installed/tested | Fast and accurate on synthetic command samples. |
| OpenAI Whisper | `base` | installed/tested | Reliable baseline, slightly slower than tiny on short commands. |
| MLX Whisper | `mlx-community/whisper-tiny` | installed/tested | Fastest after first model download on Apple Silicon. |
| MLX Whisper | `mlx-community/whisper-base-mlx` | installed/tested | Slower than MLX tiny and similar/slower than PyTorch base for very short files in this synthetic test. |
| MLX Whisper | `mlx-community/whisper-large-v3-turbo` | candidate | Expected higher accuracy; not benchmarked yet. |
| Moonshine | candidate | Promising low-latency voice-agent model; not installed in this project yet. |
| Parakeet/Canary | candidate | Potentially higher accuracy; heavier integration. |

## Synthetic command benchmark

Samples generated with macOS `say`, converted to 16 kHz mono WAV:

- `Check the current project status.`
- `List the files in the current project.`
- `Run the tests and summarize the result.`
- `Create a short note that says the voice loop is working.`

All tested models transcribed all four phrases correctly.

Approximate warm inference time per short command after initial download/load:

| Backend/model | Warm inference |
|---|---:|
| `mlx-whisper` / `mlx-community/whisper-tiny` | ~0.10–0.12s |
| `whisper` / `tiny` | ~0.19–0.25s |
| `whisper` / `base` | ~0.28–0.34s |
| `mlx-whisper` / `mlx-community/whisper-base-mlx` | ~0.35–0.43s |

First run includes model download/load and is much slower; ignore for steady-state UX.

## Recommendation

For the next StickS3 voice iteration:

```sh
STT_BACKEND=mlx-whisper MLX_WHISPER_MODEL=mlx-community/whisper-tiny
```

Reason: for short command-style prompts, MLX tiny was accurate on the current sample set and fastest on Apple Silicon.

Keep fallback:

```sh
STT_BACKEND=whisper WHISPER_MODEL=base
```

Reason: reliable baseline if real StickS3 microphone audio is noisy or if MLX tiny misses commands.

## Next required test

Synthetic samples are not enough. Capture real StickS3 microphone audio and benchmark the same command phrases. The final choice should be based on:

1. exact transcript accuracy,
2. latency after model warmup,
3. hallucination behavior on silence/noise,
4. robustness to distance from microphone.
