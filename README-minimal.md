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
</p>

---

```dart
import 'package:xybrid_flutter/xybrid_flutter.dart';

await Xybrid.init();
final model = await Xybrid.model(modelId: 'kokoro-82m').load();
final result = await model.run(envelope: Envelope.text(text: 'Hello world'));
```

---

## SDKs

| SDK | Platforms | Docs |
|-----|-----------|------|
| **Flutter** | iOS, Android, macOS, Linux, Windows | [Get Started →](bindings/flutter/) |
| **Swift** | iOS, macOS | [Get Started →](bindings/apple/) |
| **Kotlin** | Android | [Get Started →](bindings/kotlin/) |
| **Unity** | macOS, Windows, Linux | [Get Started →](bindings/unity/) |
| **CLI** | macOS, Linux, Windows | [Download →](https://github.com/xybrid-ai/xybrid/releases) |
| **Rust** | All | [Crates →](crates/) |

---

## Models

Whisper Tiny · Wav2Vec2 · Kokoro 82M · KittenTTS · Qwen 2.5 0.5B · Llama 3.2 1B · MobileNetV2 · MiniLM L6

All models run entirely on-device. No cloud, no API keys.

---

<p align="center">
  <a href="https://docs.xybrid.dev"><strong>Documentation</strong></a> ·
  <a href="https://discord.gg/xybrid">Discord</a> ·
  <a href="./CONTRIBUTING.md">Contributing</a> ·
  <a href="./LICENSE">Apache 2.0</a>
</p>
