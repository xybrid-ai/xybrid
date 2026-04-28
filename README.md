<div align="center">
<p align="center">
  <a href="./README.md">English</a> · <a href="./README.zh-CN.md">简体中文</a> · <a href="./README.ja-JP.md">日本語</a>
</p>


<p align="center">
  <img src="./docs/logo.jpg" alt="Xybrid Logo" width="180"/>
</p>

<h1 align="center">Xybrid</h1>

<p align="center">
  <strong>Run LLMs, ASR, and TTS natively in apps and games.</strong><br/>
  Rust core · iOS · Android · Flutter · Unity<br/>
  Private, offline, no cloud required.
</p>

<p align="center">
  <a href="https://docs.xybrid.dev">Documentation</a> ·
  <a href="#sdks">SDKs</a> ·
  <a href="https://www.xybrid.ai/models">Models</a> ·
  <a href="https://discord.gg/YhFHHkhbad">Join Discord</a> ·
  <a href="https://x.com/xybrid_ai">Follow on X</a> ·
  <a href="https://github.com/xybrid-ai/xybrid/issues">Issues</a>
</p>

<p align="center">

[![Website][website-shield]][website-url]
[![Docs][docs-shield]][docs-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
<br>
[![License][license-shield]][license-url]
[![Build][build-shield]][build-url]
[![OpenSSF Scorecard][scorecard-shield]][scorecard-url]
[![Stars][stars-shield]][stars-url]
[![Release][release-shield]][release-url]
[![Release Date][release-date-shield]][release-url]
<br>
[![pub.dev][pubdev-shield]][pubdev-url]
[![Maven Central][maven-shield]][maven-url]
[![Swift Package Manager][spm-shield]][spm-url]
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=xybrid-ai.xybrid)](https://github.com/xybrid-ai/xybrid)

</p>

[website-shield]: https://img.shields.io/badge/xybrid.ai-4285F4?style=flat
[website-url]: https://www.xybrid.ai/
[docs-shield]: https://img.shields.io/badge/docs-xybrid.dev-1F6FEB?style=flat&logo=readthedocs&logoColor=white
[docs-url]: https://docs.xybrid.dev/
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FYhFHHkhbad%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=5865F2&suffix=%20members
[discord-url]: https://discord.gg/YhFHHkhbad
[twitter-shield]: https://img.shields.io/twitter/follow/xybrid_ai
[twitter-url]: https://x.com/xybrid_ai
[license-shield]: https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat
[license-url]: https://opensource.org/licenses/Apache-2.0
[build-shield]: https://img.shields.io/github/actions/workflow/status/xybrid-ai/xybrid/ci.yml?branch=master&style=flat
[build-url]: https://github.com/xybrid-ai/xybrid/actions
[scorecard-shield]: https://api.scorecard.dev/projects/github.com/xybrid-ai/xybrid/badge
[scorecard-url]: https://scorecard.dev/viewer/?uri=github.com/xybrid-ai/xybrid
[stars-shield]: https://img.shields.io/github/stars/xybrid-ai/xybrid?style=flat
[stars-url]: https://github.com/xybrid-ai/xybrid/stargazers
[release-shield]: https://img.shields.io/github/v/release/xybrid-ai/xybrid?style=flat&sort=semver
[release-url]: https://github.com/xybrid-ai/xybrid/releases
[release-date-shield]: https://img.shields.io/github/release-date/xybrid-ai/xybrid?style=flat
[pubdev-shield]: https://img.shields.io/pub/v/xybrid_flutter?style=flat&label=pub.dev
[pubdev-url]: https://pub.dev/packages/xybrid_flutter
[maven-shield]: https://img.shields.io/maven-central/v/ai.xybrid/xybrid-kotlin?style=flat&label=Maven%20Central
[maven-url]: https://central.sonatype.com/artifact/ai.xybrid/xybrid-kotlin
[spm-shield]: https://img.shields.io/badge/Swift_Package_Manager-compatible-F05138?style=flat&logo=swift&logoColor=white
[spm-url]: https://github.com/xybrid-ai/xybrid
</div>

<p align="center">
  <img src="docs/demo-desktop.gif" alt="Desktop demo" width="540">&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/demo-android.gif" alt="Android demo" width="150">
</p>



## Start Here

| Goal | Path |
|------|------|
| Fastest demo (2 min) | [Install CLI →](#quick-start) |
| Build a mobile or desktop app | [Flutter SDK →](bindings/flutter/) |
| Add AI NPCs to your game | [Unity SDK →](bindings/unity/) and try the [3D tavern demo](https://github.com/xybrid-ai/xybrid-unity-tavern) |
| Android native | [Kotlin SDK →](bindings/kotlin/) |
| Rust / embedded | [Core crate →](crates/) |
---

<p align="center">
  <img src="docs/game-demo.gif" alt="Game demo" width="540">
</p>

## SDKs

Xybrid is a **Rust-powered runtime** with native bindings for every major platform.

| SDK | Platforms | Install | Status | Sample |
|-----|-----------|---------|--------|--------|
| **[Flutter](bindings/flutter/)** | iOS, Android, macOS, Linux, Windows | [pub.dev](https://pub.dev/packages/xybrid_flutter) | Available | [README](examples/flutter/README.md) |
| **[Unity](bindings/unity/)** | macOS, Windows, Linux, iOS, Android | [See below](#quick-start) | Available | [Unity 3D AI tavern](https://github.com/xybrid-ai/xybrid-unity-tavern) |
| **[Swift](bindings/apple/)** | iOS, macOS | Swift Package Manager | Coming Soon | [README](examples/ios/README.md) |
| **[Kotlin](bindings/kotlin/)** | Android | Maven Central | Available | [README](examples/android/README.md) |
| **[CLI](https://github.com/xybrid-ai/xybrid/releases)** | macOS, Linux, Windows | `curl -sSL .../install.sh \| sh` | Available | — |
| **[Rust](crates/)** | All | `xybrid-core` / `xybrid-sdk` | Available | — |

Every SDK wraps the same Rust core — identical model support and behavior across all platforms.

---

## Quick Start

Install and run a model in your language of choice. Each section includes the install snippet and a minimal example.

See the full [Installation Guide](https://docs.xybrid.dev/en/docs/quickstart) for all options.

<details>
<summary><b>Flutter</summary>

**Install** in `pubspec.yaml`:

```yaml
dependencies:
  xybrid_flutter: ^0.1.0
```

**Run a model:**

```dart
final model = await Xybrid.model('kokoro-82m').load();
final result = await model.run(XybridEnvelope.text('Hello world'));
// result → 24kHz WAV audio
```

</details>

<details>
<summary><b>Kotlin</summary>

**Install** in `build.gradle.kts`:

```gradle
dependencies {
    implementation("ai.xybrid:xybrid-kotlin:0.1.0-beta12")
}
```

**Run a model:**

```kotlin
val model = XybridModelLoader.fromRegistry("kokoro-82m").load()
val result = model.run(Envelope.text("Hello world"))
// result → 24kHz WAV audio
```

</details>

<details>
<summary><b>Swift</summary>

**Install** in `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/xybrid-ai/xybrid.git", from: "0.1.0")
]
```

**Run a model:**

```swift
let model = try ModelLoader.fromRegistry(modelId: "kokoro-82m").load()
let result = try model.run(envelope: Envelope.text("Hello world"))
// result → 24kHz WAV audio
```

</details>

<details>
<summary><b>Unity/C#</summary>

**Install** via Unity Package Manager:

```sh
https://github.com/xybrid-ai/xybrid.git#upm
```

**Run a model:**

```csharp
var model = XybridClient.LoadModel("kokoro-82m");
var result = model.Run(Envelope.Text("Hello world"));
// result → 24kHz WAV audio
```

</details>

<details>
<summary>Rust</summary>

**Install** in `Cargo.toml`:

```toml
[dependencies]
xybrid-sdk = "0.1"
```

**Run a model:**

```rust
let model = Xybrid::model("kokoro-82m").load()?;
let result = model.run(&Envelope::text("Hello world"))?;
// result → 24kHz WAV audio
```

</details>

<details>
<summary><b>CLI</b> — macOS, Linux, Windows</summary>

**Install:**

```bash
# macOS / Linux
curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.ps1 | iex
```

**Run a model:**

```sh
xybrid run --model kokoro-82m --input-text "Hello world" -o output.wav
```

</details>

For platform-specific setup, see each SDK's README: [Flutter](bindings/flutter/) · [Unity](bindings/unity/) · [Swift](bindings/apple/) · [Kotlin](bindings/kotlin/) · [Rust](crates/).

<details>
<summary><h3>Pipelines (Experimental)</h3></summary>

Chain models together — build a voice assistant in 3 lines of YAML:

```yaml
# voice-assistant.yaml
name: voice-assistant
stages:
  - model: whisper-tiny    # Speech → text
  - model: qwen2.5-0.5b    # Process with LLM
  - model: kokoro-82m      # Text → speech
```

**CLI:**
```sh
xybrid run --config voice-assistant.yaml --input-audio question.wav -o response.wav
```

**Flutter:**
```dart
final pipeline = Xybrid.pipeline(yaml: yamlString);
final result = await pipeline.run(XybridEnvelope.audio(bytes: audioBytes, sampleRate: 16000));
```

**Kotlin:**
```kotlin
// Pipeline support coming soon — use single model loading for now
```

**Swift:**
```swift
// Pipeline support coming soon — use single model loading for now
```

**Unity (C#):**
```csharp
// Pipeline support coming soon — use single model loading for now
```

**Rust:**
```rust
let pipeline = Xybrid::pipeline(&yaml_string).load()?;
pipeline.load_models()?;
let result = pipeline.run(&Envelope::audio(audio_bytes))?;
```

</details>

---

## Supported Models

All models run entirely on-device. No cloud, no API keys required. Browse the full registry with `xybrid models list`.

### Start with these

| Model | Type | Params | Why start here |
|-------|------|--------|----------------|
| **SmolLM2 360M** | LLM | 360M | Best quality-to-size ratio for any device |
| **Kokoro 82M** | TTS | 82M | High-quality speech, 24 voices, fast |
| **Whisper Tiny** | ASR | 39M | Accurate multilingual transcription |

### Speech-to-Text

| Model | Params | Format | Description |
|-------|--------|--------|-------------|
| Whisper Tiny | 39M | SafeTensors | Multilingual transcription (Candle runtime) |
| Wav2Vec2 Base | 95M | ONNX | English ASR with CTC decoding |

### Text-to-Speech

| Model | Params | Format | Description |
|-------|--------|--------|-------------|
| Kokoro 82M | 82M | ONNX | High-quality, 24 natural voices |
| KittenTTS Nano | 15M | ONNX | Ultra-lightweight, 8 voices |

### Language Models

| Model | Params | Format | Description |
|-------|--------|--------|-------------|
| Gemma 3 1B | 1B | GGUF Q4_K_M | Google's mobile-optimized LLM |
| LFM2.5 350M | 354M | GGUF Q4_K_M | Liquid AI's hybrid conv+attention, 9 languages, tool calling |
| Llama 3.2 1B | 1B | GGUF Q4_K_M | Meta's general purpose, 128K context |
| Qwen 2.5 0.5B | 500M | GGUF Q4_K_M | Compact on-device chat |
| Qwen 3.5 0.8B | 800M | GGUF Q4_K_M | Latest Qwen with reasoning (thinking mode) |
| Qwen 3.5 2B | 2B | GGUF Q4_K_M | Larger Qwen 3.5 with extended reasoning |
| SmolLM2 360M | 360M | GGUF Q4_K_M | Best tiny LLM, excellent quality/size ratio |

### Coming Soon

| Model | Type | Params | Priority | Status |
|-------|------|--------|----------|--------|
| Phi-4 Mini | LLM | 3.8B | P2 | Spec Ready (first multi-quant: Q4, Q8, FP16) |
| Qwen3 0.6B | LLM | 600M | P2 | Planned |
| Trinity Nano | LLM (MoE) | 6B (1B active) | P2 | Planned |
| LFM2-VL 700M | Vision+LLM | 700M | P2 | Planned |
| Nomic Embed Text v1.5 | Embeddings | 137M | P1 | Blocked (needs Tokenize/MeanPool steps) |
| LFM2-VL 450M | Vision | 450M | P2 | Planned |
| Whisper Tiny CoreML | ASR | 39M | P2 | Planned |
| Qwen3-TTS 0.6B | TTS | 600M | P2 | Blocked (needs custom SafeTensors runtime) |
| Chatterbox Turbo | TTS | 350M | P3 | Blocked (needs ModelGraph template) |

<details>
<summary><h3>Bring Your Own Model (Experimental)</h3></summary>

> **Note**: BYM support is experimental. The `model_metadata.json` schema is stable, but the AI-assisted tooling (`/xybrid-init`) is under active development and may not handle all model types yet.

Xybrid works with **any** ONNX, GGUF, or SafeTensors model. You just need a `model_metadata.json` that tells xybrid how to run it.

**With an AI assistant** (Claude Code, Codex, etc.):

```sh
# Install xybrid skills into your project
curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/tools/scripts/install-skills.sh | sh

# Generate model_metadata.json from a HuggingFace model
claude /xybrid-init hexgrad/Kokoro-82M-v1.0-ONNX
```

Skills are agent-agnostic and live in [`agents/skills/`](agents/skills/). The installer symlinks them for Claude Code (`.claude/skills`) and Codex (`.codex/skills`).

**Manually** — create `model_metadata.json` in your model directory:

```json
{
  "model_id": "my-model",
  "version": "1.0",
  "execution_template": { "type": "Onnx", "model_file": "model.onnx" },
  "preprocessing": [],
  "postprocessing": [],
  "files": ["model.onnx"],
  "metadata": { "task": "text-generation" }
}
```

See the [model metadata docs](docs/sdk/API_REFERENCE.md) for the full schema, or look at existing examples in [`integration-tests/fixtures/models/`](integration-tests/fixtures/models/).

</details>

---

## Features

| Capability | iOS | Android | macOS | Linux | Windows |
|------------|-----|---------|-------|-------|---------|
| Speech-to-Text | ✅ | ✅ | ✅ | ✅ | ✅ |
| Text-to-Speech | ✅ | ✅ | ✅ | ✅ | ✅ |
| Language Models | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vision Models | 🔜 | 🔜 | 🔜 | 🔜 | 🔜 |
| Embeddings | 🔜 | 🔜 | 🔜 | 🔜 | 🔜 |
| Pipeline Orchestration | ✅ | ✅ | ✅ | ✅ | ✅ |
| Model Download & Caching | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hardware Acceleration | Metal, ANE | CPU | Metal, ANE | CUDA | CUDA |

**SDK pipeline support:** Flutter ✅ · Rust ✅ · Kotlin 🔜 · Swift 🔜 · Unity 🔜

---

## Why Xybrid?

- **Privacy first** — All inference runs on-device. Your data never leaves the device. The SDK attaches a small fleet-attribution header on registry metadata calls — see [registry telemetry](docs/telemetry/registry.md).
- **Offline capable** — No internet required after initial model download.
- **Cross-platform** — One API across iOS, Android, macOS, Linux, and Windows.
- **Pipeline orchestration** — Chain models together (ASR → LLM → TTS) in a single call.
- **Automatic optimization** — Hardware acceleration on Apple Neural Engine, Metal, and CUDA.

### How it compares

| | Xybrid | Ollama | llama.cpp | ONNX Runtime |
|---|---|---|---|---|
| Mobile (iOS/Android) | ✅ | ❌ | ❌ | ✅ |
| Game engine (Unity) | ✅ | ❌ | ❌ | ❌ |
| Multi-stage pipelines | ✅ | ❌ | ❌ | ❌ |
| ASR + TTS + LLM in one SDK | ✅ | ❌ | ❌ | ❌ |
| Runs in-process (no server) | ✅ | ❌ | ✅ | ✅ |
| No cloud required | ✅ | ✅ | ✅ | ✅ |

---

## Community

- [Documentation](https://docs.xybrid.dev)
- [Discord](https://discord.gg/YhFHHkhbad)
- [X (Twitter)](https://x.com/xybrid_ai)
- [GitHub Issues](https://github.com/xybrid-ai/xybrid/issues)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on setting up your development environment, submitting pull requests, and adding new models.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=xybrid-ai/xybrid&type=date&legend=bottom-right)](https://www.star-history.com/#xybrid-ai/xybrid&type=date&legend=bottom-right)

## License

Apache License 2.0 — see [LICENSE](./LICENSE) for details.
