# Xybrid Flutter SDK

Flutter SDK for the Xybrid universal AI inference runtime. Execute AI pipelines on-device with intelligent local/cloud routing.

## Features

- **Pipeline Execution**: Run AI inference pipelines defined in YAML
- **Audio Processing**: Record audio and run speech-to-text models
- **Real-time Streaming**: Real-time ASR with `XybridStreamer`
- **Real-time Telemetry**: Stream pipeline execution events for monitoring
- **Envelope System**: Type-safe data passing (Audio, Text, Embedding)
- **Device Intelligence**: Hardware capabilities detection (GPU, NPU, memory, thermal)
- **Cross-platform**: Supports iOS, macOS, Android, Linux, and Windows

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  xybrid_flutter:
    git:
      url: https://github.com/xybrid-ai/bindings/flutter.git
      ref: main
```

Or for local development:

```yaml
dependencies:
  xybrid_flutter:
     path: ../../bindings/flutter
```

### Platform Requirements

#### macOS
- macOS 13.3+ (for ONNX Runtime CoreML)
- Xcode 15+
- **Apple Silicon only** (M1/M2/M3/M4) - Intel Macs are not supported
- Supports: ONNX Runtime + Candle (Metal acceleration)

#### iOS
- iOS 13.0+
- Xcode 15+
- Supports: Candle only (Metal acceleration)
- Note: ONNX Runtime requires manual xcframework bundling (see [PLAN-IOS-ONNX.md](../../PLAN-IOS-ONNX.md))

#### Android
- NDK r25+ (r27 recommended)
- minSdkVersion 21
- 64-bit only: arm64-v8a, x86_64 (32-bit excluded for APK size)
- Supports: ONNX Runtime (bundled .so from Microsoft AAR)

#### Linux / Windows
- Supports: ONNX Runtime (download-binaries)

### Rust Toolchain

```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install required targets
# macOS (Apple Silicon only)
rustup target add aarch64-apple-darwin

# iOS
rustup target add aarch64-apple-ios aarch64-apple-ios-sim

# Android
rustup target add aarch64-linux-android x86_64-linux-android

# Linux (if cross-compiling)
rustup target add x86_64-unknown-linux-gnu

# Windows (if cross-compiling)
rustup target add x86_64-pc-windows-msvc
```

## Quick Start

### Initialize the SDK

```dart
import 'package:xybrid_flutter/xybrid_flutter.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();

  // Initialize Rust logging - logs will appear in:
  // - Android: Logcat with tag "xybrid"
  // - iOS: Xcode console
  // - Desktop: stderr (terminal)
  initLogging();

  runApp(MyApp());
}
```

### Run a Pipeline

```dart
// Load pipeline YAML from assets
final yamlContent = await rootBundle.loadString('assets/pipelines/speech-to-text.yaml');

// Run without input
final result = runPipelineFromYaml(yamlContent: yamlContent, envelope: null);

// Parse result (JSON)
final data = jsonDecode(result);
print('Pipeline: ${data['name']}');
print('Total latency: ${data['total_latency_ms']}ms');
```

### Run with Audio Input

```dart
// Create audio envelope from recorded bytes
final audioBytes = await readAudioFile(recordingPath);
final envelope = Envelope.audio(
  audioBytes: audioBytes,
  sampleRate: 16000,
  channels: 1,
);

// Run pipeline with audio input
final result = runPipelineFromYaml(
  yamlContent: yamlContent,
  envelope: envelope,
);
```

### AudioEnvelope API (v0.0.7)

The `AudioEnvelope` class provides a high-level API for audio processing with automatic format conversion, resampling, and WAV parsing.

#### From WAV File

```dart
import 'dart:io';
import 'package:xybrid_flutter/xybrid_flutter.dart';

// Load WAV file
final wavBytes = await File('recording.wav').readAsBytes();
final audio = AudioEnvelope.fromWav(wavBytes: wavBytes);

// Check audio properties
print('Sample rate: ${audio.sampleRate}Hz');
print('Channels: ${audio.channels}');
print('Duration: ${audio.durationMs}ms');

// Convert to pipeline Envelope and run inference
final envelope = audio.toEnvelope();
final result = runInferenceFromBundle(
  bundlePath: 'models/wav2vec2.xyb',
  envelope: envelope,
);
```

#### From Raw Recording (PCM16)

```dart
// From mobile audio recorder (e.g., record package)
final audio = AudioEnvelope.fromRecording(
  pcmBytes: recorder.getBytes(),  // Raw PCM16 bytes
  sampleRate: 44100,              // Your recording's sample rate
  channels: 2,                    // Stereo recording
);

// Prepare for ASR (converts to 16kHz mono float32)
final prepared = audio.prepareForAsr();

// Run inference
final result = runInferenceFromBundle(
  bundlePath: 'models/wav2vec2.xyb',
  envelope: prepared.toEnvelope(),
);
```

#### From Float32 Samples

```dart
// If you already have normalized float samples (-1.0 to 1.0)
final audio = AudioEnvelope.fromSamples(
  samples: floatSamples,
  sampleRate: 16000,
  channels: 1,
);
```

#### Audio Transformations

```dart
// Convert stereo to mono
final mono = audio.toMono();

// Resample to different rate
final resampled = audio.resample(targetRate: 16000);

// Prepare for ASR (convenience: mono + 16kHz)
final prepared = audio.prepareForAsr();

// Get raw samples for custom processing
final Float32List samples = audio.samples();

// Get format info
final AudioFormat format = audio.format();
print('Format: ${format.formatType}');  // "float32", "pcm16", etc.
```

#### Full Example: Voice Recording to Transcription

```dart
import 'package:xybrid_flutter/xybrid_flutter.dart';

Future<String> transcribeRecording(Uint8List pcmBytes) async {
  // 1. Create AudioEnvelope from recording
  final audio = AudioEnvelope.fromRecording(
    pcmBytes: pcmBytes,
    sampleRate: 44100,  // Common mobile recording rate
    channels: 2,        // Stereo
  );

  // 2. Prepare for ASR (auto-converts to 16kHz mono)
  final prepared = audio.prepareForAsr();

  // 3. Convert to pipeline Envelope
  final envelope = prepared.toEnvelope();

  // 4. Run inference
  final xybrid = Xybrid.simple();
  final result = xybrid.runInferenceFromBundle(
    'assets/models/wav2vec2.xyb',
    envelope: envelope,
  );

  if (result.success) {
    return result.text ?? '';
  } else {
    throw XybridException(result.error ?? 'Inference failed');
  }
}
```

### Stream Telemetry Events

```dart
// Subscribe to real-time pipeline events
final stream = subscribeTelemetryEvents();
stream.listen((event) {
  print('${event.eventType}: ${event.stageName ?? "pipeline"}');
  if (event.latencyMs != null) {
    print('  Latency: ${event.latencyMs}ms');
  }
});
```

## API Reference

### Pipeline Execution

```dart
/// Run a pipeline from YAML content
String runPipelineFromYaml({
  required String yamlContent,
  Envelope? envelope,
});

/// Run a pipeline from file path
String runPipeline(String configPath);

/// Get model cache status
String getCacheStatus();
```

### Envelope Types

```dart
/// Audio envelope (PCM bytes)
Envelope.audio(
  audioBytes: Uint8List,
  sampleRate: int?,  // default: 16000
  channels: int?,    // default: 1
);

/// Text envelope
Envelope.text(String text);

/// Embedding envelope
Envelope.embedding(List<double> values);
```

### AudioEnvelope (v0.0.7)

```dart
/// Create from WAV file bytes
AudioEnvelope.fromWav({required List<int> wavBytes});

/// Create from raw PCM16 recording
AudioEnvelope.fromRecording({
  required List<int> pcmBytes,
  required int sampleRate,
  required int channels,
});

/// Create from float32 samples
AudioEnvelope.fromSamples({
  required List<double> samples,
  required int sampleRate,
  required int channels,
});

/// Properties
int get sampleRate;          // Sample rate in Hz
int get channels;            // Number of channels
BigInt get sampleCount;      // Number of samples
double get durationMs;       // Duration in milliseconds
AudioFormat format();        // Format info

/// Transformations
AudioEnvelope toMono();                        // Convert to mono
AudioEnvelope resample({required int targetRate}); // Resample
AudioEnvelope prepareForAsr();                 // Convert to 16kHz mono

/// Conversion
Envelope toEnvelope();       // Convert to pipeline Envelope
Float32List samples();       // Get raw float32 samples
String toJson();             // Export as JSON (debugging)
```

### AudioFormat

```dart
/// Standard audio format specification
AudioFormat.pcm16({required int sampleRate, required int channels});
AudioFormat.float32({required int sampleRate, required int channels});
AudioFormat.asrDefault();  // 16kHz mono float32
```

### Device Capabilities (v0.0.6)

```dart
/// Get current device hardware capabilities
HardwareCapabilities getDeviceCapabilities();

/// Check if a model can be loaded (memory check)
bool canLoadModel(int modelSizeMb);
```

The `HardwareCapabilities` class provides:

```dart
class HardwareCapabilities {
  bool hasGpu;              // GPU available
  String gpuType;           // metal, vulkan, directx, none
  bool hasMetal;            // iOS/macOS Metal
  bool hasNnapi;            // Android NNAPI
  bool hasNpu;              // Neural Processing Unit
  String npuType;           // coreml, nnapi, directml, none
  int memoryAvailableMb;    // Available RAM
  int memoryTotalMb;        // Total RAM
  int batteryLevel;         // 0-100
  String thermalState;      // normal, warm, hot, critical
  String platform;          // macos, ios, android, linux, windows
  bool shouldThrottle;      // true if battery low or thermal critical

  // v0.0.7 additions
  double cpuUsagePercent;   // Current CPU usage (0-100)
  int cpuCores;             // Number of CPU cores
  String memoryConfidence;  // Detection confidence: high, medium, low
  String gpuConfidence;     // Detection confidence: high, medium, low
  String npuConfidence;     // Detection confidence: high, medium, low
}
```

Example usage:

```dart
final caps = getDeviceCapabilities();
print('Platform: ${caps.platform}');
print('GPU: ${caps.hasGpu ? caps.gpuType : "none"}');
print('NPU: ${caps.hasNpu ? caps.npuType : "none"}');
print('Memory: ${caps.memoryAvailableMb}/${caps.memoryTotalMb} MB');

if (caps.shouldThrottle) {
  // Use cloud inference or smaller models
  print('Device is throttled - battery: ${caps.batteryLevel}%, thermal: ${caps.thermalState}');
}

// v0.0.7: Check detection confidence
print('Memory confidence: ${caps.memoryConfidence}'); // high, medium, or low
print('CPU: ${caps.cpuCores} cores, ${caps.cpuUsagePercent}% usage');
```

### Session Metrics (v0.0.6)

```dart
/// Get current session metrics
SessionMetrics getSessionMetrics();

/// Export telemetry as JSON for backend ingestion
String exportTelemetryJson();

/// Reset session and start fresh
void resetSession();

/// Record an inference call (usually automatic)
void recordInference(String modelId, String version, int latencyMs);

/// Record an error (usually automatic)
void recordError(String? modelId, String error);

/// End session and get final export
String endSession();
```

The `SessionMetrics` class provides:

```dart
class SessionMetrics {
  String sessionId;         // Unique session ID
  String deviceId;          // Device identifier
  int startedAt;            // Session start (ms since epoch)
  int endedAt;              // Session end (0 if active)
  String sdkVersion;        // SDK version
  int totalInferences;      // Total inference calls
  int totalLatencyMs;       // Cumulative latency
  int avgLatencyMs;         // Average latency
  List<String> modelsUsed;  // Models used in session
  int errorCount;           // Total errors
  int durationMs;           // Session duration
  bool isActive;            // Is session active
}
```

Example usage:

```dart
// Get current metrics
final metrics = getSessionMetrics();
print('Inferences: ${metrics.totalInferences}');
print('Avg Latency: ${metrics.avgLatencyMs}ms');
print('Models: ${metrics.modelsUsed.join(", ")}');

// Export for analytics
final json = exportTelemetryJson();
await http.post(analyticsEndpoint, body: json);

// On app exit
final finalReport = endSession();
await sendToAnalytics(finalReport);
```

### Telemetry Events

```dart
class TelemetryEvent {
  String eventType;        // PipelineStart, StageComplete, etc.
  String? stageName;       // Stage name if applicable
  String? target;          // local, cloud, or fallback
  int? latencyMs;          // Execution time
  String? error;           // Error message if failed
  String? data;            // Additional JSON data
  int timestampMs;         // Unix timestamp
}
```

Event types:
- `PipelineStart` / `PipelineComplete`
- `StageStart` / `StageComplete`
- `RoutingDecided`
- `StageError` / `ExecutionFailed`

## Example App

The example app demonstrates all SDK features:

```bash
cd xybrid_flutter/example
flutter run -d macos   # or -d ios, -d android
```

Features:
- Pipeline selection (Speech-to-Text, Hiiipe)
- Audio recording with mic input
- Real-time telemetry viewer
- Device capabilities display (GPU, NPU, memory, thermal)
- Cache status monitoring

## Pipeline YAML Format

```yaml
name: "Speech-to-Text Demo"

registry:
  url: "http://0.0.0.0:8080"

stages:
  - "whisper-tiny@1.2"

input:
  kind: "AudioRaw"

metrics:
  network_rtt: 50
  battery: 80
  temperature: 22.0

availability:
  "whisper-tiny@1.2": true
```

## Architecture

```
xybrid_flutter/
├── rust/                    # Rust FFI layer
│   └── src/api/
│       ├── xybrid.rs        # Pipeline execution API
│       ├── envelope.rs      # Input data types
│       ├── logging.rs       # Rust-to-Flutter logging bridge
│       └── telemetry.rs     # Event streaming
├── lib/                     # Generated Dart bindings
│   └── src/rust/            # flutter_rust_bridge output
└── example/                 # Demo application
    ├── lib/main.dart        # Full-featured demo
    ├── macos/Runner/*.entitlements  # macOS sandbox permissions
    └── assets/pipelines/    # Sample YAML configs
```

## Building

The SDK uses [flutter_rust_bridge](https://cjycode.com/flutter_rust_bridge/) v2.11.1 for Rust-Dart interop.

```bash
# Regenerate bindings (if Rust API changes)
cd xybrid_flutter
flutter_rust_bridge_codegen generate

# Build for development
cd example
flutter build macos --debug
```

## Logging

The SDK includes a Rust-to-Flutter logging bridge. Initialize it at app startup:

```dart
// Initialize after RustLib.init()
initLogging();

// Test logging (optional)
testLogging();

// Log from Dart to Rust logger
logInfo(message: "App started");
logDebug(message: "Debug info");
logWarn(message: "Warning");
logError(message: "Error occurred");
```

Logs appear in:
- **Android**: Logcat with tag `xybrid`
- **iOS**: Xcode console via oslog
- **Desktop**: stderr (visible in terminal)

## macOS Entitlements

The example app includes required entitlements for macOS sandbox:

```xml
<!-- DebugProfile.entitlements and Release.entitlements -->
<key>com.apple.security.network.client</key>
<true/>  <!-- Required for HTTP requests to registry -->
<key>com.apple.security.network.server</key>
<true/>  <!-- Required for local server connections -->
<key>com.apple.security.device.audio-input</key>
<true/>  <!-- Required for microphone recording -->
```

## Model Support by Platform

| Model | Type | macOS | iOS | Android | Linux/Windows |
|-------|------|-------|-----|---------|---------------|
| Whisper (Candle) | ASR | ✅ Metal | ✅ Metal | ❌ | ✅ CPU |
| Wav2Vec2 (ONNX) | ASR | ✅ | ❌ | ✅ | ✅ |
| Kokoro-82M (ONNX) | TTS | ✅ | ❌ | ✅ | ✅ |
| KittenTTS (ONNX) | TTS | ✅ | ❌ | ✅ | ✅ |

**Notes**:
- iOS currently supports Candle models only (no ONNX Runtime prebuilt binaries)
- Android uses bundled ONNX Runtime .so from Microsoft AAR
- macOS supports both ONNX and Candle runtimes

## Notes

- **ASR Models**: Whisper (via Candle) and Wav2Vec2 (via ONNX) are both supported. Whisper ONNX has decoder issues (see docs/archive/).
- **TTS Models**: Kokoro-82M and KittenTTS work on all ONNX-supported platforms.
- **iOS Limitation**: ONNX models require bundling xcframework manually. See [PLAN-IOS-ONNX.md](../../PLAN-IOS-ONNX.md) for options.
- **macOS Intel**: Intel Macs (x86_64) are not supported. ONNX Runtime does not provide prebuilt binaries for x86_64-apple-darwin.
- **macOS Version Warnings**: Linker warnings about macOS version mismatches are expected due to ONNX Runtime build configuration.
- **Registry URL**: Use `http://localhost:8080` for local development. The app needs `network.client` entitlement to connect.
