# Xybrid CLI

Command-line interface for running on-device ML inference — LLMs, TTS, ASR, and pipelines.

## Installation

See the full [Installation Guide](../../docs/INSTALLATION.md) for all options.

**Quick install (no Rust required):**

```bash
# macOS / Linux
curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.ps1 | iex
```

**From source:**

```bash
cargo install --git https://github.com/xybrid-ai/xybrid xybrid-cli --features platform-macos

# Vision-language models add the VLM feature to the platform preset:
cargo install --git https://github.com/xybrid-ai/xybrid xybrid-cli --features platform-macos,llm-llamacpp-vision
```

`platform-macos` includes MLX registry/metadata support but not the linked MLX
runtime. To run MLX SafeTensors models on Apple Silicon, build with
`--features platform-macos,llm-mlx-runtime` from a local clone after running
`./tools/scripts/fetch-mlx-xcframework.sh` when the pinned download is
available or `./tools/scripts/build-local-mlx-xcframework.sh` for source-build
validation. In external build environments, set `MLX_XCFRAMEWORK_PATH` to a
prebuilt `mlx.xcframework`.

## Quick Start

```bash
# List available models
xybrid models list

# Text-to-speech
xybrid run --model kokoro-82m --input-text "Hello world" --output hello.wav

# Speech-to-text
xybrid run --model whisper-tiny-ggml --input-audio recording.wav

# Chat with an LLM (interactive)
xybrid repl --model smollm2-360m --stream

# Vision-language prompt (requires a vision-capable build and model)
xybrid run --model lfm2-vl-450m --input-text "Describe this image" --input-image photo.jpg

# Chat through a specific local backend when available
xybrid repl --model qwen3-4b --backend mlx --stream

# Run any GGUF from HuggingFace
xybrid run --huggingface "unsloth/SmolLM2-360M-Instruct-GGUF:Q4_K_M" --input-text "Hello!"
```

## Documentation

- [Installation & CLI Reference](../../docs/INSTALLATION.md) — install methods, all commands, flags, and options
- [Main README](../../README.md) — project overview and SDK quickstarts
