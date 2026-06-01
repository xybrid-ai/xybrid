# Installation

## Quick Install (Recommended)

The install scripts detect your OS and architecture, download the latest release binary, and add it to your PATH.

**macOS / Linux:**

```bash
curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.ps1 | iex
```

## Download Binary

Pre-built binaries are available on the [Releases](https://github.com/xybrid-ai/xybrid/releases) page for:

| Platform | Architecture | Binary |
|----------|-------------|--------|
| macOS | Apple Silicon (M1+) | `xybrid-v*-macos-arm64` |
| Linux | x86_64 | `xybrid-v*-linux-x86_64` |
| Windows | x86_64 | `xybrid-v*-windows-x86_64.exe` |

Download the binary for your platform, make it executable, and move it to your PATH:

```bash
# Example: macOS Apple Silicon
chmod +x xybrid-v*-macos-arm64
sudo mv xybrid-v*-macos-arm64 /usr/local/bin/xybrid
```

## Install from Source

Requires the [Rust toolchain](https://rustup.rs/) (1.75+).

```bash
cargo install --git https://github.com/xybrid-ai/xybrid xybrid-cli
```

### Platform Features

By default, `cargo install` builds without hardware acceleration. For optimal performance, add platform features:

```bash
# macOS — Metal GPU + Apple Neural Engine + llama.cpp
cargo install --git https://github.com/xybrid-ai/xybrid xybrid-cli --features platform-macos

# Linux / Windows — ONNX + llama.cpp
cargo install --git https://github.com/xybrid-ai/xybrid xybrid-cli --features platform-desktop
```

The prebuilt binaries from the install script already include the correct features for each platform.

<details>
<summary>All available feature flags</summary>

| Feature | Description |
|---------|-------------|
| **Platform presets** | |
| `platform-macos` | ONNX download + CoreML + Metal + text-only llama.cpp |
| `platform-ios` | ONNX download + CoreML + Metal + text-only llama.cpp |
| `platform-android` | ONNX dynamic + text-only llama.cpp |
| `platform-desktop` | ONNX download + text-only llama.cpp |
| **Individual flags** | |
| `ort-download` | Download prebuilt ONNX Runtime binaries |
| `ort-dynamic` | Load ONNX Runtime .so at runtime |
| `ort-coreml` | Apple Neural Engine acceleration |
| `candle` | Candle ML backend (Whisper models) |
| `candle-metal` | Metal GPU acceleration for Candle |
| `candle-cuda` | CUDA GPU acceleration for Candle |
| `llm-llamacpp` | llama.cpp LLM runtime backend (recommended) |
| `vision` | Image envelope and preprocessing support |
| `llm-llamacpp-vision` | llama.cpp vision-language support (`mmproj` / `mtmd`) |
| `llm-mistral` | mistral.rs LLM backend (alternative) |

</details>

### Build from a Local Clone

```bash
git clone https://github.com/xybrid-ai/xybrid.git
cd xybrid
cargo build --release -p xybrid-cli --features platform-macos
# Binary at target/release/xybrid
```

## Verify Installation

```bash
xybrid --help
```

Expected output:

```
Xybrid CLI - Run hybrid cloud-edge AI inference pipelines

Usage: xybrid [OPTIONS] <COMMAND>

Commands:
  init     Generate model_metadata.json by inspecting model files
  models   Manage models from the registry
  run      Run a pipeline, model, or GGUF file
  repl     Interactive REPL mode
  fetch    Pre-download models from the registry
  cache    Manage the local model cache
  ...
```

Check available models:

```bash
xybrid models list
```

## Getting Started

### Text-to-Speech

```bash
xybrid run --model kokoro-82m --input-text "Hello world" --output hello.wav
```

### Speech-to-Text

```bash
xybrid run --model whisper-tiny --input-audio recording.wav
```

### Chat with an LLM

```bash
# Interactive chat (keeps model loaded between messages)
xybrid repl --model smollm2-360m --stream

# Single inference
xybrid run --model smollm2-360m --input-text "What is the capital of France?"
```

### Vision-Language Input

Vision-language models require a build with `vision` or `llm-llamacpp-vision`
enabled, plus a model bundle that includes the vision encoder artifact.
Platform presets alone are text-only; compose a preset with
`llm-llamacpp-vision` for local VLM generation.

```bash
cargo build --release -p xybrid-cli --features platform-macos,llm-llamacpp-vision
```

```bash
# Single vision turn
xybrid run --model lfm2-vl-450m \
  --input-text "Describe this image" \
  --input-image photo.jpg

# Interactive vision chat
xybrid repl --model lfm2-vl-450m --stream
/image photo.jpg
What is in this image?
```

### Run Any GGUF from HuggingFace

No registry entry needed — point directly at a HuggingFace repo:

```bash
xybrid run --huggingface "unsloth/SmolLM2-360M-Instruct-GGUF:Q4_K_M" \
  --input-text "Hello!"

# Interactive chat
xybrid repl --huggingface "unsloth/SmolLM2-360M-Instruct-GGUF:Q4_K_M" --stream
```

### Run a Local GGUF File

```bash
xybrid run --model-file ./my-model.gguf --input-text "Hello!"
```

### Pipelines

Chain models together with a YAML file:

```yaml
# voice-assistant.yaml
name: voice-assistant
stages:
  - model: whisper-tiny
  - model: smollm2-360m
  - model: kokoro-82m
```

```bash
xybrid run --config voice-assistant.yaml --input-audio question.wav --output response.wav
```

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `xybrid run` | Run inference on a model, pipeline, bundle, or GGUF file |
| `xybrid repl` | Interactive REPL — keeps models loaded for fast repeated inference |
| `xybrid init` | Generate `model_metadata.json` by inspecting a model directory |
| `xybrid models list` | List all models in the registry |
| `xybrid models search <query>` | Search models by name, task, or description |
| `xybrid models info <id>` | Show details about a specific model |
| `xybrid models voices <id>` | List available voices for a TTS model |
| `xybrid fetch --model <id>` | Pre-download a model from the registry |
| `xybrid fetch --huggingface <repo>` | Pre-download a model from HuggingFace |
| `xybrid cache list` | List cached models |
| `xybrid cache status` | Show cache size and statistics |
| `xybrid cache clear [id]` | Clear cached models (all or specific) |
| `xybrid prepare <file>` | Validate a pipeline YAML |
| `xybrid plan <file>` | Show execution plan for a pipeline |
| `xybrid bundle <model>` | Fetch a model and create a `.xyb` bundle |
| `xybrid pack <name>` | Package local model artifacts into a `.xyb` bundle |
| `xybrid trace` | View and analyze telemetry from past sessions |

### Global Flags

| Flag | Description |
|------|-------------|
| `-v`, `-vv` | Increase verbosity |
| `-q`, `--quiet` | Suppress output, show errors only |
| `--api-key` | Platform API key for telemetry (or `XYBRID_API_KEY` env) |

### `run` Input Sources

The `run` command accepts multiple input sources (mutually exclusive):

| Flag | Description | Example |
|------|-------------|---------|
| `--model <id>` | Registry model | `--model kokoro-82m` |
| `--config <file>` | Pipeline YAML | `--config pipeline.yaml` |
| `--pipeline <name>` | Built-in pipeline | `--pipeline hiiipe` |
| `--bundle <file>` | `.xyb` bundle | `--bundle model.xyb` |
| `--directory <dir>` | Local model dir | `--directory ./my-model/` |
| `--huggingface <repo>` | HuggingFace repo | `--huggingface "org/model:Q4_K_M"` |
| `--model-file <path>` | Local GGUF file | `--model-file model.gguf` |

### `run` Options

| Flag | Description |
|------|-------------|
| `--input-text <text>` | Text input (for TTS, LLM) |
| `--input-audio <file>` | Audio input WAV file (for ASR) |
| `--input-image <file>` | Image input for vision-language models; repeatable |
| `--voice <id>` | TTS voice ID (e.g., `af_bella`) |
| `--output <file>` | Output file (.wav for audio, .txt for text) |
| `--target <format>` | Target format (onnx, coreml, tflite) |
| `--dry-run` | Validate without executing |
| `--trace` | Enable execution tracing |
| `--trace-export <file>` | Export trace to JSON (Chrome trace format) |

### `repl` Options

| Flag | Description |
|------|-------------|
| `--model <id>` | Registry model to load |
| `--huggingface <repo>` | HuggingFace model to load |
| `--model-file <path>` | Local GGUF file to load |
| `--stream` | Stream tokens as generated (LLM) |
| `--system <prompt>` | System prompt for the conversation |
| `--voice <id>` | TTS voice ID |

## Model Cache

Downloaded models are cached at `~/.xybrid/cache/`. Manage with:

```bash
xybrid cache status    # Show cache size
xybrid cache list      # List cached models
xybrid cache clear     # Clear all
xybrid cache clear kokoro-82m  # Clear specific model
```

## Uninstall

**Installed via script:**

```bash
rm $(which xybrid)
# Optionally remove cache
rm -rf ~/.xybrid
```

**Installed via cargo:**

```bash
cargo uninstall xybrid-cli
rm -rf ~/.xybrid
```

**Windows:**

```powershell
Remove-Item "$env:USERPROFILE\.xybrid" -Recurse -Force
```
