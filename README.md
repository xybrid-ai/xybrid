<p align="center">
  <img src="./docs/logo.jpg" alt="Xybrid Logo" width="180"/>
</p>

<h1 align="center">Xybrid</h1>

<p align="center">
  <strong>On-device AI for mobile, desktop, and edge.</strong><br/>
  Run ASR, TTS, LLMs, and vision models locally—private, offline, fast.
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://github.com/xybrid-ai/xybrid/actions"><img src="https://img.shields.io/github/actions/workflow/status/xybrid-ai/xybrid/ci.yml?branch=main" alt="Build Status"></a>
  <a href="https://pub.dev/packages/xybrid_flutter"><img src="https://img.shields.io/pub/v/xybrid_flutter.svg" alt="Pub.dev"></a>
  <a href="https://discord.gg/xybrid"><img src="https://img.shields.io/discord/1234567890?label=Discord&logo=discord" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://docs.xybrid.dev">Documentation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-models">Models</a> •
  <a href="https://discord.gg/xybrid">Discord</a>
</p>

---

## Why Xybrid?

One SDK for all your on-device AI needs:

| Capability | Models | Use Case |
|------------|--------|----------|
| **Speech-to-Text** | Whisper, Wav2Vec2 | Offline transcription, voice commands |
| **Text-to-Speech** | Kokoro, Piper | Natural voice synthesis |
| **LLMs** | Qwen, Llama, Mistral | On-device chat, summarization |
| **Vision** | MobileNet, ResNet | Image classification, object detection |
| **Embeddings** | MiniLM | Semantic search, RAG |

**Plus:**
- **Pipeline orchestration** — Chain models together (ASR → LLM → TTS)
- **Intelligent routing** — Automatic edge/cloud decisions based on device state
- **Hardware acceleration** — CoreML ANE, Metal, CUDA (6.8x speedup on vision)
- **Cross-platform** — iOS, Android, macOS, Linux, Windows + CLI

---

## Quick Start

### Flutter

```yaml
# pubspec.yaml
dependencies:
  xybrid_flutter: ^0.1.0
```

```dart
import 'package:xybrid_flutter/xybrid_flutter.dart';

// Initialize once
await Xybrid.init();

// Load a model
final loader = Xybrid.model(modelId: 'whisper-tiny');
final model = await loader.load();

// Run inference
final result = await model.run(
  envelope: Envelope.audio(bytes: audioBytes),
);
print('Transcription: ${result.unwrapText()}');
```

### CLI

```bash
# Install
git clone https://github.com/xybrid-ai/xybrid.git
cargo install --path xybrid/cli

# List available models
xybrid models list

# Download and run
xybrid fetch whisper-tiny
xybrid run whisper-tiny --input recording.wav
```

---

## Supported Models

| Model | Type | Size | Platforms | Notes |
|-------|------|------|-----------|-------|
| **whisper-tiny** | ASR | ~75 MB | All | English, real-time capable |
| **wav2vec2-base-960h** | ASR | ~360 MB | All | English, CTC decoding |
| **kokoro-82m** | TTS | ~330 MB | All | Natural voice, 24kHz |
| **kitten-tts-nano** | TTS | ~50 MB | All | Lightweight, fast |
| **qwen2.5-0.5b** | LLM | ~500 MB | macOS, Linux | GGUF, 4096 context |
| **all-MiniLM-L6-v2** | Embeddings | ~90 MB | All | 384-dim vectors |
| **mobilenet-v2** | Vision | ~14 MB | All | ImageNet classification |

> Models are downloaded on-demand and cached locally (~/.xybrid/cache/).

---

## Platform Support

| Platform | Status | Hardware Acceleration |
|----------|--------|----------------------|
| **macOS (ARM)** | ✅ Stable | CoreML ANE, Metal GPU |
| **macOS (x86)** | ✅ Stable | CoreML GPU |
| **iOS** | ✅ Stable | CoreML ANE, Metal GPU |
| **Android** | ✅ Stable | NNAPI (planned) |
| **Linux** | ✅ Stable | CUDA |
| **Windows** | ✅ Stable | CUDA, DirectML |

---

## Benchmarks

Tested on Apple M3 MacBook Pro:

| Model | CPU | CoreML ANE | Speedup |
|-------|-----|------------|---------|
| MobileNetV2 (224×224) | 8.1 ms | **1.2 ms** | **6.8×** |
| Whisper Tiny (30s audio) | 2.1 s | 0.8 s | 2.6× |
| Kokoro TTS (100 chars) | 303 ms | 356 ms | CPU faster* |

*Dynamic-shape models like TTS run faster on CPU due to CoreML dispatch overhead.

### When to Use CoreML ANE

| Model Type | Expected Speedup | Recommendation |
|------------|------------------|----------------|
| Vision (CNNs) | **5-10×** | Always use ANE |
| Embeddings | **2-4×** | Use ANE |
| Whisper encoder | **2-5×** | Use ANE |
| TTS / autoregressive | ~1× | Use CPU |

Xybrid automatically selects the optimal execution provider based on model characteristics.

---

## Pipeline Orchestration

Chain multiple models into a single workflow:

```yaml
# voice-assistant.yaml
name: voice-assistant
stages:
  - id: transcribe
    model: whisper-tiny

  - id: think
    model: qwen2.5-0.5b
    target: cloud  # Optional: force cloud execution

  - id: speak
    model: kokoro-82m
```

---

## Hybrid Cloud Fallback

Xybrid automatically routes to cloud APIs when:
- Device resources are constrained (low battery, memory pressure)
- Model isn't available locally
- You explicitly request cloud execution

```yaml
stages:
  - model: whisper-tiny
    fallback: openai/whisper-1  # Use OpenAI if local fails
```

```dart
// Set API keys for cloud providers
Xybrid.setProviderApiKey('openai', 'sk-...');
Xybrid.setProviderApiKey('anthropic', 'sk-ant-...');
```

---

## SDKs

| Package | Platform | Link |
|---------|----------|------|
| **xybrid_flutter** | iOS, Android, macOS | [![](https://img.shields.io/pub/v/xybrid_flutter.svg)](https://pub.dev/packages/xybrid_flutter) |
| **xybrid-cli** | macOS, Linux, Windows | [GitHub](https://github.com/xybrid-ai/xybrid) |

---

## Comparison

| Feature | Xybrid | RunAnywhere | Cactus | TensorFlow Lite |
|---------|--------|-------------|--------|-----------------|
| **ASR** | ✅ Whisper, Wav2Vec2 | ✅ Whisper | ❌ | ✅ (manual) |
| **TTS** | ✅ Kokoro, Piper | ✅ Piper | ❌ | ❌ |
| **LLMs** | ✅ GGUF | ✅ GGUF | ✅ Proprietary | ❌ |
| **Vision** | ✅ | ⏳ Coming | ❌ | ✅ |
| **Pipeline chaining** | ✅ | ❌ | ❌ | ❌ |
| **Flutter SDK** | ✅ | ✅ | ❌ | ❌ |
| **CLI** | ✅ | ❌ | ❌ | ❌ |
| **CoreML ANE** | ✅ Auto-select | ? | ❌ | ❌ |
| **Cloud fallback** | ✅ | ✅ | ✅ | ❌ |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## Community

- [Documentation](https://docs.xybrid.dev)
- [Discord](https://discord.gg/xybrid)
- [GitHub Issues](https://github.com/xybrid-ai/xybrid/issues)
- [Roadmap](./ROADMAP.md)

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE) for details.

The core runtime, SDK, CLI, and Flutter bindings are fully open source.
