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

Cloud authentication (`--api-key`, `XYBRID_API_KEY`, or the key `xybrid connect`
saved for this project) does not enable export. Add `--observability` to send
traces to `https://ingest.xybrid.dev/v1/traces` (with `--platform-url
https://staging.api.xybrid.dev`, to staging's
`https://staging.ingest.xybrid.dev/v1/traces`); it fails with a clear error when
no key resolves. Use `--traces-endpoint` with that flag for a custom full trace
URL; the cloud gateway selected by `--platform-url` remains independent. A saved
key is only ever sent to its own platform's ingest endpoint.

For an OpenTelemetry Collector, set `OTEL_EXPORTER_OTLP_ENDPOINT` or the
per-signal `OTEL_EXPORTER_OTLP_*_ENDPOINT` variables instead. Collector export
does not require a Xybrid API key. `OTEL_SDK_DISABLED=true` disables export in
either mode. The CLI flushes and shuts down its exporter when a command ends.
Enabled observability captures prompts and outputs by default. Set
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT=no_content` for metadata only.

## Connect a project (coding agents)

`xybrid connect` gives a project a provisional, claimable Xybrid workspace
without a browser or an account. It is meant for coding agents setting up an
app for a user:

```bash
xybrid connect --agent claude-code --write-env .env --json
```

- It skips itself in CI (`--allow-ci` overrides only that) and when
  `XYBRID_TELEMETRY_OPTOUT`, `DO_NOT_TRACK` or `OTEL_SDK_DISABLED=true` is set.
- An existing key (flag, `XYBRID_API_KEY`, a value in the file) is never
  replaced. A file tracked by Git is refused, and so is a file in another
  repository than the one `connect` runs in (run it from that repository).
  Otherwise the key is written to the `--write-env` file (dotenv,
  `local.properties` as `xybrid.apiKey`, or an `.xcconfig`; `--env-name
  EXPO_PUBLIC_XYBRID_API_KEY` for Expo), which is added to `.gitignore` and
  checked with `git check-ignore`.
- The key, claim link and deadlines are kept in `~/.xybrid/credentials.json`
  (`0600`, per platform host and repository; `XYBRID_CREDENTIALS_FILE` moves
  it, and a directory inside a repository is gitignored whole). The store is
  refused, with the `chmod` that fixes it, when another account could replace
  it: a directory writable by other accounts (a group-writable one is fine
  when the group is your own private group, as with a `002` umask), or a file
  that another account owns or can access. Output never contains the raw key.
- It does not enable trace export: wire the app's observability explicitly,
  with metadata-only content.

| Exit | Status |
|---|---|
| 0 | `created`, `existing` |
| 3 | `signup_closed` |
| 4 | `offline` |
| 5 | `ci_skipped`, `opted_out` |
| 6 | `env_file_tracked` |
| 7 | `workspace_expired` (it can still be claimed) |
| 1 | `error` (see `error.code`; `claim_expired` once the claim period ended) |

`xybrid auth status [--json] [--show-claim-url]` reports the key's source, the
workspace lifecycle and whether traces from this project have been stored.
`xybrid auth claim` shows (and on a terminal opens) the private claim link,
after asking the platform whether the workspace was already claimed.
`xybrid auth logout` forgets the saved credentials; it does not revoke the key.
Other commands print a daily stderr reminder while the saved workspace is
unclaimed (as of the last `connect` or `auth` check); `--quiet`, JSON output,
CI and `XYBRID_QUIET=1` silence it.
`--platform-url` selects production (default) or staging; any other host needs
`--allow-custom-host` and HTTPS unless it is localhost.

## Documentation

- [Installation & CLI Reference](../../docs/INSTALLATION.md) — install methods, all commands, flags, and options
- [Main README](../../README.md) — project overview and SDK quickstarts
