<p align="center">
  <img src="./docs/logo.jpg" alt="Xybrid Logo" width="180"/>
</p>

<h1 align="center">Xybrid</h1>

<p align="center">
  <strong>On-device AI for mobile, desktop, and edge.</strong><br/>
  Run speech, language, and vision models locally—private, offline, fast.
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://github.com/xybrid-ai/xybrid/actions"><img src="https://img.shields.io/github/actions/workflow/status/xybrid-ai/xybrid/ci.yml?branch=main" alt="Build Status"></a>
  <a href="https://pub.dev/packages/xybrid_flutter"><img src="https://img.shields.io/pub/v/xybrid_flutter.svg" alt="Pub.dev"></a>
</p>

<p align="center">
  <a href="https://docs.xybrid.dev">Documentation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-models">Models</a> •
  <a href="https://discord.gg/xybrid">Discord</a>
</p>

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

## Supported Models

**Speech Recognition:**
- Whisper Tiny (~75 MB) - English, real-time
- Wav2Vec2 Base (~360 MB) - English

**Voice Synthesis:**
- Kokoro 82M (~330 MB) - 24 natural voices
- KittenTTS Nano (~50 MB) - Lightweight, fast

**Language Models:**
- Qwen 2.5 0.5B (~500 MB) - On-device chat
- Llama 3.2 1B (~1 GB) - General purpose

**Vision & Embeddings:**
- MobileNetV2 (~14 MB) - Image classification
- MiniLM L6 (~90 MB) - Text embeddings

All models run entirely on-device. No cloud, no API keys required.

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

// Load and run a model
final model = await Xybrid.model(modelId: 'whisper-tiny').load();
final result = await model.run(envelope: Envelope.audio(bytes: audioBytes));
print('Transcription: ${result.unwrapText()}');
```

### Unity

Add to your `Packages/manifest.json`:

```json
{
  "dependencies": {
    "ai.xybrid.sdk": "https://github.com/xybrid-ai/xybrid.git?path=bindings/unity"
  }
}
```

```csharp
using Xybrid.Native;
using UnityEngine;

public class XybridExample : MonoBehaviour
{
    void Start()
    {
        NativeMethods.xybrid_init();
        var loader = NativeMethods.xybrid_model_loader_new();
        NativeMethods.xybrid_model_loader_load(loader, "kokoro-82m", IntPtr.Zero, out var model);

        var envelope = NativeMethods.xybrid_envelope_new_text("Hello world");
        NativeMethods.xybrid_model_run(model, envelope, out var output);
        Debug.Log(Marshal.PtrToStringUTF8(NativeMethods.xybrid_result_get_text(output)));
    }
}
```

### CLI

```bash
# Install
cargo install --git https://github.com/xybrid-ai/xybrid.git xybrid-cli

# Or download from releases:
# https://github.com/xybrid-ai/xybrid/releases

# List models
xybrid models list

# Text-to-speech
xybrid run --model kokoro-82m --input-text "Hello world" -o speech.wav

# Speech-to-text
xybrid run --model whisper-tiny --input-audio recording.wav -o transcript.txt
```

---

## SDKs

| Package | Platform | Status |
|---------|----------|--------|
| **xybrid_flutter** | iOS, Android, macOS | [![](https://img.shields.io/pub/v/xybrid_flutter.svg)](https://pub.dev/packages/xybrid_flutter) |
| **xybrid-unity** | Windows, macOS, Linux | [UPM Package](bindings/unity/) |
| **xybrid-cli** | macOS, Linux, Windows | [Releases](https://github.com/xybrid-ai/xybrid/releases) |
| **xybrid-swift** | iOS, macOS | Coming Soon |
| **xybrid-kotlin** | Android | Coming Soon |

---

## Pipeline Orchestration

Chain multiple models into a single workflow:

```yaml
# voice-assistant.yaml
name: voice-assistant
stages:
  - model: whisper-tiny    # Speech to text
  - model: qwen2.5-0.5b    # Process with LLM
  - model: kokoro-82m      # Text to speech
```

```bash
xybrid run voice-assistant.yaml --input question.wav -o response.wav
```

Build voice assistants, transcription pipelines, or any multi-model workflow—all running locally on-device.

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
- [Discord](https://discord.gg/xybrid)
- [GitHub Issues](https://github.com/xybrid-ai/xybrid/issues)

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE) for details.
