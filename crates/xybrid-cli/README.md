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

# Run any GGUF from HuggingFace
xybrid run --huggingface "unsloth/SmolLM2-360M-Instruct-GGUF:Q4_K_M" --input-text "Hello!"
```

## Observability

Cloud authentication (`--api-key` or `XYBRID_API_KEY`) does not enable export.
Add `--observability` to send traces to `https://ingest.xybrid.dev/v1/traces`.
Use `--traces-endpoint` with that flag for a custom full trace URL; the cloud
gateway selected by `--platform-url` remains independent.

For an OpenTelemetry Collector, set `OTEL_EXPORTER_OTLP_ENDPOINT` or the
per-signal `OTEL_EXPORTER_OTLP_*_ENDPOINT` variables instead. Collector export
does not require a Xybrid API key. `OTEL_SDK_DISABLED=true` disables export in
either mode. The CLI flushes and shuts down its exporter when a command ends.
Enabled observability captures prompts and outputs by default. Set
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT=no_content` for metadata only.

## Documentation

- [Installation & CLI Reference](../../docs/INSTALLATION.md) — install methods, all commands, flags, and options
- [Main README](../../README.md) — project overview and SDK quickstarts
