<p align="center">
  <img src="./docs/logo.jpg" alt="Xybrid Logo" width="180"/>
</p>

<h1 align="center">Xybrid</h1>

<p align="center">
  <strong>On-device AI for mobile, desktop, and edge.</strong><br/>
  Run speech, language, and vision models locally — private, offline, fast.
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://github.com/xybrid-ai/xybrid/actions"><img src="https://img.shields.io/github/actions/workflow/status/xybrid-ai/xybrid/ci.yml?branch=main&style=flat-square" alt="Build Status"></a>
  <a href="https://github.com/xybrid-ai/xybrid/stargazers"><img src="https://img.shields.io/github/stars/xybrid-ai/xybrid?style=flat-square" alt="Stars"></a>
  <a href="https://discord.gg/cgd3tbFPWx"><img src="https://img.shields.io/discord/0?label=Discord&style=flat-square&color=5865F2" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://docs.xybrid.dev">Documentation</a> ·
  <a href="#sdks">SDKs</a> ·
  <a href="#supported-models">Models</a> ·
  <a href="https://discord.gg/cgd3tbFPWx">Discord</a> ·
  <a href="https://github.com/xybrid-ai/xybrid/issues">Issues</a>
</p>

---

## SDKs

Xybrid is a **Rust-powered runtime** with native bindings for every major platform. Pick your SDK:

| SDK | Platforms | Install | Status |
|-----|-----------|---------|--------|
| **[Flutter](bindings/flutter/)** | iOS, Android, macOS, Linux, Windows | [See below](#install) | Available |
| **[Unity](bindings/unity/)** | macOS, Windows, Linux | [See below](#install) | Available |
| **[Swift](bindings/apple/)** | iOS, macOS | Swift Package Manager | Coming Soon |
| **[Kotlin](bindings/kotlin/)** | Android | Maven Central | Available |
| **[CLI](https://github.com/xybrid-ai/xybrid/releases)** | macOS, Linux, Windows | [Download binary](https://github.com/xybrid-ai/xybrid/releases) | Available |
| **[Rust](crates/)** | All | `xybrid-core` / `xybrid-sdk` | Available |

Every SDK wraps the same Rust core — identical model support and behavior across all platforms.

### Install

**Flutter** — add to your `pubspec.yaml`:

```yaml
dependencies:
  xybrid_flutter:
    git:
      url: https://github.com/xybrid-ai/xybrid.git
      ref: main
      path: bindings/flutter
```

**Kotlin (Android)** — add to your `build.gradle.kts`:

```kotlin
dependencies {
    implementation("ai.xybrid:xybrid-kotlin:0.1.0-alpha3")
}
```

**Unity** — Package Manager → Add from git URL:

```
https://github.com/xybrid-ai/xybrid.git?path=bindings/unity
```

---

## Quick Start

Run a single model:

```dart
import 'package:xybrid_flutter/xybrid_flutter.dart';

await Xybrid.init();
final model = await Xybrid.model(modelId: 'kokoro-82m').load();
final result = await model.run(envelope: Envelope.text(text: 'Hello world'));
// → 24kHz WAV audio output
```

Or chain models into a pipeline — build a voice assistant in 3 lines of YAML:

```yaml
# voice-assistant.yaml
name: voice-assistant
stages:
  - model: whisper-tiny    # Speech → text
  - model: qwen2.5-0.5b    # Process with LLM
  - model: kokoro-82m      # Text → speech
```

```dart
// Run the pipeline from Flutter
final pipeline = await Xybrid.pipeline(yamlContent: yamlString).load();
await pipeline.loadModels();
final result = await pipeline.run(envelope: Envelope.audio(bytes: audioBytes));
```

```bash
# Or from the CLI
xybrid run voice-assistant.yaml --input question.wav -o response.wav
```

See each SDK's README for platform-specific setup: [Flutter](bindings/flutter/) · [Unity](bindings/unity/) · [Swift](bindings/apple/) · [Kotlin](bindings/kotlin/) · [Rust](crates/)

---

## Supported Models

All models run entirely on-device. No cloud, no API keys required. Browse the full registry with `xybrid models list`.

### Speech-to-Text

| Model | Size | Description |
|-------|------|-------------|
| Whisper Tiny | ~89 MB | Real-time English transcription |
| Wav2Vec2 Base | ~231 MB | English ASR |

### Text-to-Speech

| Model | Size | Description |
|-------|------|-------------|
| Kokoro 82M | ~183 MB | 24 natural voices |
| KittenTTS Nano | ~19 MB | Lightweight, fast |

### Language Models

| Model | Size | Description |
|-------|------|-------------|
| Gemma 3 1B | ~786 MB | Google's compact LLM |
| Llama 3.2 1B | ~791 MB | Meta's general purpose |
| Mistral 7B | ~4.3 GB | High-quality reasoning |
| Phi-4 Mini | ~8.7 GB | Microsoft's compact reasoning LLM |
| Qwen 2.5 0.5B | ~477 MB | Smallest on-device chat |
| SmolLM2 360M | ~267 MB | Ultra-lightweight |

### Coming Soon

| Model | Type | Status |
|-------|------|--------|
| Qwen3 0.6B | LLM | Planned |
| Whisper Tiny CoreML | ASR | Planned |
| Nomic Embed Text v1.5 | Embeddings | Blocked |
| Chatterbox Turbo | TTS | Blocked |

---

## Features

| Capability | iOS | Android | macOS | Linux | Windows |
|------------|-----|---------|-------|-------|---------|
| Speech-to-Text | ✅ | ✅ | ✅ | ✅ | ✅ |
| Text-to-Speech | ✅ | ✅ | ✅ | ✅ | ✅ |
| Language Models | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vision Models | ✅ | ✅ | ✅ | ✅ | ✅ |
| Embeddings | ✅ | ✅ | ✅ | ✅ | ✅ |
| Pipeline Orchestration | ✅ | ✅ | ✅ | ✅ | ✅ |
| Model Download & Caching | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hardware Acceleration | Metal, ANE | CPU | Metal, ANE | CUDA | CUDA |

---

## Why Xybrid?

- **Privacy first** — All inference runs on-device. Your data never leaves the device.
- **Offline capable** — No internet required after initial model download.
- **Cross-platform** — One API across iOS, Android, macOS, Linux, and Windows.
- **Pipeline orchestration** — Chain models together (ASR → LLM → TTS) in a single call.
- **Automatic optimization** — Hardware acceleration on Apple Neural Engine, Metal, and CUDA.

---

## Community

- [Documentation](https://docs.xybrid.dev)
- [Discord](https://discord.gg/cgd3tbFPWx)
- [GitHub Issues](https://github.com/xybrid-ai/xybrid/issues)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on setting up your development environment, submitting pull requests, and adding new models.

## License

Apache License 2.0 — see [LICENSE](./LICENSE) for details.
