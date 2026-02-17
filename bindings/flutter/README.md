# Xybrid Flutter SDK

Run ML models on-device or in the cloud with intelligent hybrid routing and streaming inference.

[![pub package](https://img.shields.io/pub/v/xybrid_flutter.svg)](https://pub.dev/packages/xybrid_flutter)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Installation

```bash
flutter pub add xybrid_flutter
```

Or add to your `pubspec.yaml`:

```yaml
dependencies:
  xybrid_flutter: ^0.1.0
```

<details>
<summary>Alternative installation (git / local path)</summary>

**From git** (unreleased changes):

```yaml
dependencies:
  xybrid_flutter:
    git:
      url: https://github.com/xybrid-ai/xybrid.git
      ref: main
      path: bindings/flutter
```

</details>

## Quick Start

```dart
import 'package:xybrid_flutter/xybrid_flutter.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Xybrid.init();

  // Load a TTS model from the registry
  final model = await XybridModelLoader.fromRegistry('kokoro-82m').load();

  // Run text-to-speech
  final result = await model.run(XybridEnvelope.text('Hello from Xybrid!'));
  print('Audio: ${result.audioBytes?.length} bytes');
}
```

## Features

### Model Loading

Load models from the Xybrid registry or local bundles:

```dart
// From registry (downloads + caches automatically)
final model = await XybridModelLoader.fromRegistry('kokoro-82m').load();

// From local bundle
final model = await XybridModelLoader.fromBundle('path/to/model.xyb').load();

// Check if already cached
if (Xybrid.isModelCached('kokoro-82m')) {
  print('Model ready, no download needed');
}
```

### Download Progress

Track model downloads with progress events:

```dart
final loader = XybridModelLoader.fromRegistry('kokoro-82m');

await for (final event in loader.loadWithProgress()) {
  switch (event) {
    case LoadProgress(:final progress):
      print('Downloading: ${(progress * 100).toInt()}%');
    case LoadComplete():
      print('Model ready!');
    case LoadError(:final message):
      print('Error: $message');
  }
}
```

### Input Envelopes

Type-safe inputs for different model types:

```dart
// Text (for TTS or LLM)
final textInput = XybridEnvelope.text('Hello world');

// Text with TTS voice selection
final ttsInput = XybridEnvelope.text('Hello', voiceId: 'af_heart', speed: 1.0);

// Audio (for ASR / speech-to-text)
final audioInput = XybridEnvelope.audio(
  bytes: wavBytes,
  sampleRate: 16000,
  channels: 1,
);

// Embedding vector
final embeddingInput = XybridEnvelope.embedding([0.1, 0.2, 0.3]);
```

### Inference Results

```dart
final result = await model.run(envelope);

if (result.success) {
  // Text output (ASR transcription or LLM response)
  print(result.text);

  // Audio output (TTS) — get as WAV for playback
  final wav = result.audioAsWav(sampleRate: 24000, channels: 1);

  // Embedding output
  print(result.embedding);

  // Inference timing
  print('Latency: ${result.latencyMs}ms');
}
```

### LLM Streaming

Stream tokens in real-time as the LLM generates:

```dart
final model = await XybridModelLoader.fromRegistry('qwen-2.5-0.5b').load();

await for (final token in model.runStreaming(XybridEnvelope.text('What is ML?'))) {
  stdout.write(token.token);

  if (token.isFinal) {
    print('\n--- Done (${token.finishReason}) ---');
  }
}
```

### Conversation Memory

Multi-turn LLM conversations with automatic history management:

```dart
final model = await XybridModelLoader.fromRegistry('qwen-2.5-0.5b').load();
final context = ConversationContext();
context.setSystem('You are a helpful assistant.');

// Turn 1
context.pushText('What is Rust?', MessageRole.user);
final result = await model.runWithContext(
  XybridEnvelope.text('What is Rust?'),
  context,
);
context.pushText(result.text!, MessageRole.assistant);

// Turn 2 — the model remembers Turn 1
context.pushText('How does it compare to Go?', MessageRole.user);
final result2 = await model.runWithContext(
  XybridEnvelope.text('How does it compare to Go?'),
  context,
);
```

Streaming with context also supported:

```dart
await for (final token in model.runStreamingWithContext(envelope, context)) {
  stdout.write(token.token);
}
```

### Pipeline Execution

Run multi-stage ML pipelines from YAML:

```dart
final pipeline = XybridPipeline.fromYaml('''
name: "Speech-to-Text"
stages:
  - "whisper-tiny@1.0"
''');

print('Pipeline: ${pipeline.name}, ${pipeline.stageCount} stages');

final result = await pipeline.run(XybridEnvelope.audio(
  bytes: audioBytes,
  sampleRate: 16000,
));
print('Transcription: ${result.text}');
```

## Platform Support

| Platform | ONNX Runtime | Candle | LLM (llama.cpp) | Notes |
|----------|:---:|:---:|:---:|-------|
| **macOS** | ✅ | ✅ Metal | ✅ | Apple Silicon only (M1+) |
| **iOS** | ✅ CoreML | ✅ Metal | ✅ | arm64, downloads ORT from HuggingFace |
| **Android** | ✅ | — | ✅ | arm64-v8a, x86_64; ORT from Maven Central |
| **Linux** | ✅ | ✅ CPU | ✅ | x86_64 |
| **Windows** | ✅ | ✅ CPU | ✅ | x86_64 |

### Model Support

| Model | Type | All Platforms |
|-------|------|:---:|
| Kokoro-82M | TTS | ✅ |
| KittenTTS Nano | TTS | ✅ |
| Whisper Tiny (Candle) | ASR | ✅ |
| Wav2Vec2 (ONNX) | ASR | ✅ |
| Qwen 2.5 0.5B | LLM | ✅ |

### Platform Requirements

- **macOS**: 13.3+, Xcode 15+, Apple Silicon
- **iOS**: 13.0+, Xcode 15+
- **Android**: minSdk 21, NDK r25+, 64-bit only
- **Linux/Windows**: x86_64

## Native Libraries

Native ML runtimes are resolved automatically at build time:

- **Android**: ONNX Runtime pulled from Maven Central (`com.microsoft.onnxruntime:onnxruntime-android`)
- **iOS**: ONNX Runtime xcframework downloaded from HuggingFace and cached at `~/.xybrid/cache/ort-ios/`
- **macOS/Linux/Windows**: ONNX Runtime downloaded by the `ort` Rust crate at compile time

Precompiled Rust binaries are available for all platforms via [cargokit](https://github.com/nicklocking/cargokit) — no Rust toolchain required for most users.

## Example App

A full example app with 8 demo screens (TTS, ASR, LLM chat, pipelines, device info) is available:

```
https://github.com/xybrid-ai/xybrid/tree/main/examples/flutter
```

## API Reference

| Class | Purpose |
|-------|---------|
| `Xybrid` | SDK initialization, cache checking, factory methods |
| `XybridModelLoader` | Load models from registry or local bundle |
| `XybridModel` | Run inference (batch, streaming, with context) |
| `XybridEnvelope` | Type-safe inputs: audio, text, embedding |
| `XybridResult` | Inference output: text, audio, embedding, latency |
| `StreamToken` | Individual LLM token during streaming |
| `ConversationContext` | Multi-turn conversation history with FIFO pruning |
| `XybridPipeline` | Multi-stage pipeline execution from YAML |
| `MessageRole` | Enum: `system`, `user`, `assistant` |
| `LoadEvent` | Download progress events: `LoadProgress`, `LoadComplete`, `LoadError` |

Full API documentation: [pub.dev/documentation/xybrid_flutter](https://pub.dev/documentation/xybrid_flutter/latest/)

## License

Apache 2.0 — see [LICENSE](LICENSE)
