# Xybrid Flutter Example

A comprehensive Flutter example app demonstrating all major features of the Xybrid SDK for on-device ML inference.

## Features

This example includes 7 interactive demo screens:

| Demo | Description |
|------|-------------|
| **SDK Initialization** | Async `Xybrid.init()` with loading, ready, and error states |
| **Model Loading** | Load models from registry with progress indication |
| **Text-to-Speech** | Generate speech from text with audio playback |
| **Speech-to-Text** | Record audio and transcribe with Whisper |
| **Pipelines** | Multi-stage inference workflows from YAML |
| **Error Handling** | Exception patterns, retry logic, and error display |
| **Demo Hub** | Navigation between all demo screens |

## Prerequisites

| Tool | Required Version | Installation |
|------|------------------|--------------|
| Flutter | 3.19.0+ | [flutter.dev/docs/get-started/install](https://flutter.dev/docs/get-started/install) |
| Dart | 3.3.0+ | Included with Flutter |
| Xcode | 14.0+ | Mac App Store (for iOS/macOS) |
| Android Studio | Hedgehog+ | [developer.android.com/studio](https://developer.android.com/studio) |

### Platform-Specific Requirements

**iOS/macOS:**
- Valid Apple Developer account (for device testing)
- Xcode command line tools: `xcode-select --install`

**Android:**
- Android SDK API 28+ (Android 9.0)
- Android NDK (for native library builds)

## Quick Start

```bash
# Navigate to the example
cd examples/flutter

# Install dependencies
flutter pub get

# Run on default device
flutter run

# Run on specific platform
flutter run -d macos
flutter run -d ios
flutter run -d android
```

## Project Structure

```
flutter/
├── lib/
│   └── main.dart              # All demo screens and navigation
├── pubspec.yaml               # Dependencies
├── ios/                       # iOS platform configuration
│   └── Runner/Info.plist      # Microphone permission
├── android/                   # Android platform configuration
│   └── app/src/main/
│       └── AndroidManifest.xml  # Microphone permission
├── macos/                     # macOS platform configuration
│   └── Runner/
│       ├── Info.plist           # Microphone permission
│       └── *.entitlements       # Audio input entitlement
└── test/                      # Widget tests
```

## SDK Usage Examples

### Initialize SDK

```dart
import 'package:xybrid_flutter/xybrid.dart';

try {
  await Xybrid.init();
  print('SDK ready!');
} catch (e) {
  print('Init failed: $e');
}
```

### Load Model from Registry

```dart
try {
  final loader = XybridModelLoader.fromRegistry(modelId: 'whisper-tiny');
  final model = await loader.load();
  print('Loaded: ${model.modelId}');
} catch (e) {
  print('Load failed: $e');
}
```

### Text-to-Speech (TTS)

```dart
// Create text envelope
final envelope = XybridEnvelope.text(
  text: 'Hello, world!',
  voiceId: 'af_bella',  // Optional
  speed: 1.0,           // Optional
);

// Run inference
final result = await model.run(envelope: envelope);

// Play audio
if (result.audioBytes != null) {
  print('Generated audio in ${result.latencyMs}ms');
  // Use just_audio or audioplayers to play result.audioBytes
}
```

### Speech-to-Text (ASR)

```dart
// Record audio (16kHz mono WAV for Whisper)
final audioBytes = await recordAudio();

// Create audio envelope
final envelope = XybridEnvelope.audio(
  audioBytes: audioBytes,
  sampleRate: 16000,
  channels: 1,
);

// Run inference
final result = await model.run(envelope: envelope);
print('Transcription: ${result.text}');
```

### Multi-Stage Pipelines

```dart
// Define pipeline in YAML
const yaml = '''
name: voice-assistant
stages:
  - model: whisper-tiny
  - model: llama-3-8b
  - model: kokoro-82m
''';

// Load and run pipeline
final pipeline = await XybridPipeline.fromYaml(yaml: yaml);
final result = await pipeline.run(envelope: audioEnvelope);
print('Pipeline output: ${result.text}');
```

### Error Handling

```dart
try {
  final loader = XybridModelLoader.fromRegistry(modelId: 'unknown-model');
  await loader.load();
} on XybridException catch (e) {
  // Handle SDK-specific errors
  print('Xybrid error: ${e.message}');
} catch (e) {
  // Handle other errors
  print('Error: $e');
}
```

### Retry Logic

```dart
Future<XybridModel> loadWithRetry(String modelId, {int maxRetries = 3}) async {
  for (var attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      final loader = XybridModelLoader.fromRegistry(modelId: modelId);
      return await loader.load();
    } catch (e) {
      if (attempt == maxRetries) rethrow;
      await Future.delayed(Duration(milliseconds: 200 * attempt));
    }
  }
  throw Exception('Failed after $maxRetries retries');
}
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `xybrid` | local | Xybrid SDK (Flutter binding) |
| `just_audio` | ^0.9.43 | Audio playback for TTS |
| `record` | ^5.0.0 | Microphone recording for ASR |
| `permission_handler` | ^11.0.0 | Runtime permission requests |
| `path_provider` | ^2.1.0 | Temporary file storage |

## Platform Permissions

### iOS (ios/Runner/Info.plist)

```xml
<key>NSMicrophoneUsageDescription</key>
<string>Microphone access is required for speech recognition.</string>
```

### Android (android/app/src/main/AndroidManifest.xml)

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
```

### macOS (macos/Runner/*.entitlements)

```xml
<key>com.apple.security.device.audio-input</key>
<true/>
```

And in `macos/Runner/Info.plist`:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>Microphone access is required for speech recognition.</string>
```

## Troubleshooting

### "Package xybrid not found"

Ensure the SDK path is correct in `pubspec.yaml`:

```yaml
dependencies:
  xybrid:
    path: ../../bindings/flutter
```

Then run:
```bash
flutter pub get
```

### "MissingPluginException"

The native library isn't built. Build it from the repo root:

```bash
cargo xtask build-flutter
```

### Widget tests fail

Widget tests require the native library which isn't available in the test environment. This is expected for flutter_rust_bridge apps.

Run typecheck instead:
```bash
flutter analyze
```

### Microphone permission denied

1. Check permission is declared in platform files (see above)
2. On macOS, add `com.apple.security.device.audio-input` entitlement
3. Ensure app has been granted permission in system settings

### Audio doesn't play

1. Verify `just_audio` is properly configured for the platform
2. Check that audio bytes are not empty
3. Ensure audio format matches expected output (PCM)

### Model download fails

1. Check internet connectivity
2. Verify model ID: `xybrid models list`
3. Try downloading manually: `xybrid models pull <model-id>`

## Running Tests

```bash
# Type checking (primary)
flutter analyze

# Unit tests (will fail due to FFI)
flutter test

# Integration tests (requires device)
flutter test integration_test/
```

## API Reference

- **[SDK API Reference](../../docs/sdk/API_REFERENCE.md)** - Complete API specification
- **[Flutter SDK Docs](../../docs/sdk/flutter.md)** - Flutter-specific documentation
- **[Online Documentation](https://docs.xybrid.dev/sdk/flutter)** - Full guides and tutorials

## Related

- [Main Examples README](../README.md) - All platform examples
- [Xybrid Repository](https://github.com/xybrid-ai/xybrid)
- [Flutter Binding Source](../../bindings/flutter)

## License

MIT License - See [LICENSE](../../LICENSE) for details.
