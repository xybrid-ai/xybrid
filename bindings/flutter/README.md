# Xybrid Flutter SDK

Flutter/Dart bindings for [Xybrid](https://github.com/xybrid-ai/xybrid), the hybrid cloud-edge ML inference orchestrator.

This SDK provides a clean Dart API for running ML models on-device or in the cloud, with intelligent routing based on device capabilities.

## Installation

Add the dependency to your `pubspec.yaml`:

```yaml
dependencies:
  xybrid:
    git:
      url: https://github.com/xybrid-ai/xybrid.git
      path: repos/xybrid/bindings/flutter
```

Then run:

```bash
flutter pub get
```

## Quick Start

### Initialization

**Important:** You must initialize the Xybrid SDK before using any other functionality.

```dart
import 'package:xybrid/xybrid.dart';

void main() async {
  // Initialize the Xybrid runtime
  await Xybrid.init();

  runApp(MyApp());
}
```

Initialization is idempotent - calling `Xybrid.init()` multiple times is safe and subsequent calls are no-ops. The SDK handles concurrent initialization attempts gracefully.

You can check if the SDK is initialized using:

```dart
if (Xybrid.isInitialized) {
  // SDK is ready
}
```

### Basic Usage

Load a model and run inference:

```dart
import 'package:xybrid/xybrid.dart';

Future<void> textToSpeech() async {
  // Initialize once at app startup
  await Xybrid.init();

  // Load a model from the registry
  final loader = XybridModelLoader.fromRegistry('kokoro-82m');
  final model = await loader.load();

  // Create input envelope
  final envelope = XybridEnvelope.text('Hello, world!');

  // Run inference
  final result = await model.run(envelope);

  // Access output
  if (result.success) {
    final audioBytes = result.audioBytes; // Uint8List for TTS output
    print('Generated ${audioBytes?.length} bytes of audio');
    print('Inference took ${result.latencyMs}ms');
  }
}
```

### Speech-to-Text (ASR)

```dart
Future<void> speechToText() async {
  await Xybrid.init();

  final loader = XybridModelLoader.fromRegistry('whisper-small');
  final model = await loader.load();

  // Create audio envelope from raw bytes
  final envelope = XybridEnvelope.audio(
    bytes: audioData,       // List<int> of audio bytes
    sampleRate: 16000,      // Sample rate in Hz
    channels: 1,            // Mono audio
  );

  final result = await model.run(envelope);
  print('Transcription: ${result.text}');
}
```

### Text-to-Speech (TTS)

```dart
Future<void> textToSpeechWithOptions() async {
  await Xybrid.init();

  final loader = XybridModelLoader.fromRegistry('kokoro-82m');
  final model = await loader.load();

  // Create text envelope with optional parameters
  final envelope = XybridEnvelope.text(
    'Hello, world!',
    voiceId: 'af_heart',   // Optional voice selection
    speed: 1.0,            // Optional speed multiplier
  );

  final result = await model.run(envelope);
  final audioBytes = result.audioBytes; // Uint8List of synthesized audio
}
```

### Loading from Local Bundle

```dart
Future<void> loadLocalModel() async {
  await Xybrid.init();

  // Load from a local bundle directory
  final loader = XybridModelLoader.fromBundle('/path/to/model-bundle');
  final model = await loader.load();

  final envelope = XybridEnvelope.text('Test');
  final result = await model.run(envelope);
}
```

### Using Pipelines

Pipelines chain multiple models together for complex workflows:

```dart
Future<void> runPipeline() async {
  await Xybrid.init();

  // Load pipeline from YAML string
  final pipeline = XybridPipeline.fromYaml('''
name: speech-to-text
stages:
  - model: whisper-small
    input: audio
''');

  print('Pipeline: ${pipeline.name}');
  print('Stages: ${pipeline.stageNames}');

  // Run the pipeline
  final envelope = XybridEnvelope.audio(
    bytes: audioData,
    sampleRate: 16000,
  );
  final result = await pipeline.run(envelope);
  print('Transcription: ${result.text}');
}
```

**Loading pipelines from files:**

```dart
// From a YAML file path
final pipeline = XybridPipeline.fromFile('/path/to/pipeline.yaml');

// From a bundle directory
final pipeline = XybridPipeline.fromBundle('/path/to/pipeline-bundle');
```

### Error Handling

All Xybrid operations throw `XybridException` on failure:

```dart
try {
  final loader = XybridModelLoader.fromRegistry('nonexistent-model');
  final model = await loader.load();
} on XybridException catch (e) {
  print('Failed to load model: $e');
}
```

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `Xybrid` | Static class for SDK initialization |
| `XybridModelLoader` | Prepares and loads models from registry or bundles |
| `XybridModel` | A loaded model ready for inference |
| `XybridEnvelope` | Input data wrapper for different modalities |
| `XybridResult` | Inference output containing text, audio, or embeddings |
| `XybridPipeline` | Multi-stage inference pipeline |
| `XybridException` | Exception thrown on operation failure |

### XybridEnvelope Factory Constructors

| Constructor | Use Case |
|-------------|----------|
| `XybridEnvelope.audio(bytes, sampleRate, channels)` | Speech recognition (ASR) |
| `XybridEnvelope.text(text, voiceId?, speed?)` | Text-to-speech (TTS) |
| `XybridEnvelope.embedding(data)` | Embedding models |

### XybridResult Properties

| Property | Type | Description |
|----------|------|-------------|
| `success` | `bool` | Whether inference completed successfully |
| `text` | `String?` | Text output (ASR models) |
| `audioBytes` | `Uint8List?` | Audio output (TTS models) |
| `embedding` | `List<double>?` | Embedding vector output |
| `latencyMs` | `int` | Inference latency in milliseconds |

## Prerequisites

### Required Tools

| Tool | Required Version | Purpose |
|------|------------------|---------|
| Rust | 1.70+ | Compile native libraries |
| Flutter | 3.10+ | Flutter SDK |
| flutter_rust_bridge_codegen | 2.0+ | Generate Dart bindings |
| LLVM/Clang | 15+ | FFI parsing (used by FRB) |

### Installing Flutter Rust Bridge (FRB)

This project uses FRB v2.x for generating Dart bindings from Rust code.

**Required Version:** `flutter_rust_bridge ^2.0.0`

**Option 1: Using Cargo (Recommended)**

```bash
cargo install flutter_rust_bridge_codegen
```

**Option 2: Using Dart**

```bash
dart pub global activate flutter_rust_bridge
```

**Verify installation:**

```bash
flutter_rust_bridge_codegen --version
# Should print: flutter_rust_bridge_codegen 2.x.x
```

### Installing LLVM/Clang

FRB requires LLVM/Clang for parsing Rust FFI types.

**macOS:**
```bash
brew install llvm

# Add to PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
```

**Ubuntu/Debian:**
```bash
sudo apt install llvm-dev libclang-dev clang
```

**Windows:**
```powershell
# Using Chocolatey
choco install llvm

# Or download from: https://releases.llvm.org/
```

**Verify installation:**
```bash
clang --version
# Should print: clang version 15.x.x or higher
```

### Installing Rust Targets

For cross-platform builds, install the required targets:

```bash
# From the xybrid repo root
cargo xtask setup-targets

# Or manually for specific platforms:
rustup target add aarch64-apple-ios aarch64-apple-ios-sim  # iOS
rustup target add aarch64-apple-darwin x86_64-apple-darwin # macOS
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android  # Android
rustup target add x86_64-pc-windows-msvc     # Windows
rustup target add x86_64-unknown-linux-gnu   # Linux
```

### Platform-Specific Requirements

| Platform | Additional Requirements |
|----------|-------------------------|
| iOS/macOS | Xcode 14+, Apple targets installed |
| Android | Android NDK r26+, `ANDROID_NDK_HOME` set, cargo-ndk |
| Windows | Visual Studio Build Tools 2019+ |
| Linux | `build-essential`, `pkg-config` |

## Building

Use the xtask command to build Flutter native libraries:

```bash
# Build for your current platform
cargo xtask build-flutter --platform macos

# Build for specific platforms
cargo xtask build-flutter --platform ios
cargo xtask build-flutter --platform android
cargo xtask build-flutter --platform linux
cargo xtask build-flutter --platform windows

# Debug build (unoptimized, with symbols)
cargo xtask build-flutter --platform macos --debug
```

This will:
1. Run `flutter_rust_bridge_codegen generate` to regenerate Dart bindings
2. Build the native Rust library for the specified platform
3. Output libraries to `rust/target/{target}/{profile}/`

### Build Output

| Platform | Output Path | Library Name |
|----------|-------------|--------------|
| iOS | `rust/target/aarch64-apple-ios/release/` | `libxybrid_flutter_ffi.a` |
| macOS | `rust/target/aarch64-apple-darwin/release/` | `libxybrid_flutter_ffi.a` |
| Android | `rust/target/{abi}/release/` | `libxybrid_flutter_ffi.so` |
| Linux | `rust/target/x86_64-unknown-linux-gnu/release/` | `libxybrid_flutter_ffi.so` |
| Windows | `rust/target/x86_64-pc-windows-msvc/release/` | `xybrid_flutter_ffi.dll` |

### FRB Codegen

The FRB codegen step runs automatically during `cargo xtask build-flutter`. To run it manually:

```bash
# From the bindings/flutter directory
cd bindings/flutter
flutter_rust_bridge_codegen generate
```

**Generated Files:**

```
lib/src/rust/              # All files in this directory are generated
├── frb_generated.dart     # Main generated bindings
├── frb_generated.io.dart  # Platform-specific (mobile)
└── frb_generated.web.dart # Web platform
```

These files are gitignored and regenerated on every build.

### Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `ANDROID_NDK_HOME` | Android NDK path | Android builds |
| `ANDROID_HOME` | Android SDK path | Android builds |
| `PATH` | Include LLVM/Clang | All platforms |
| `LIBCLANG_PATH` | LLVM lib path | macOS (if Homebrew LLVM) |

## Configuration

FRB configuration is defined in `flutter_rust_bridge.yaml`:

- **Input**: `rust/src/api/` - Rust types with `#[frb]` attributes
- **Output**: `lib/src/rust/` - Generated Dart bindings (gitignored)

## Project Structure

```
flutter/
├── pubspec.yaml              # Flutter plugin manifest
├── flutter_rust_bridge.yaml  # FRB configuration
├── lib/                      # Dart API wrappers
│   ├── xybrid.dart           # Main entry point
│   └── src/
│       ├── envelope.dart     # Envelope types
│       ├── model_loader.dart # Model loading API
│       ├── result.dart       # Result types
│       └── rust/             # Generated by FRB (gitignored)
├── rust/                     # FRB bridge (Rust code)
│   ├── Cargo.toml            # depends on xybrid-sdk
│   └── src/api/              # Thin #[frb] wrappers
├── ios/
├── android/
├── macos/
├── windows/
└── linux/
```

## Development Workflow

1. Modify Rust types in `rust/src/api/`
2. Run `cargo xtask build-flutter --platform <your-platform>`
3. FRB regenerates Dart bindings automatically
4. Update manual Dart wrappers in `lib/src/` if needed

## Troubleshooting

### "flutter_rust_bridge_codegen: command not found"

**Cause**: FRB codegen not installed or not in PATH.

**Fix**:
```bash
# Install with Cargo
cargo install flutter_rust_bridge_codegen

# Or add Dart pub cache to PATH
export PATH="$PATH:$HOME/.pub-cache/bin"
```

### "fatal error: 'clang-c/Index.h' file not found"

**Cause**: LLVM/Clang not installed or not found.

**Fix (macOS)**:
```bash
brew install llvm
export LIBCLANG_PATH="/opt/homebrew/opt/llvm/lib"
```

**Fix (Linux)**:
```bash
sudo apt install llvm-dev libclang-dev clang
```

### "error: failed to run custom build command for 'flutter_rust_bridge_codegen'"

**Cause**: Missing build dependencies.

**Fix**: Install LLVM development files:
```bash
# macOS
brew install llvm

# Ubuntu/Debian
sudo apt install llvm-dev libclang-dev
```

### "dart analyze" fails with type errors

**Cause**: Generated Dart code out of sync with Rust changes.

**Fix**: Regenerate bindings:
```bash
cd bindings/flutter
flutter_rust_bridge_codegen generate
flutter pub get
```

### "flutter pub get" fails with dependency conflicts

**Cause**: Mismatched flutter_rust_bridge versions.

**Fix**: Ensure pubspec.yaml uses the same major version as the codegen tool:
```yaml
dependencies:
  flutter_rust_bridge: ^2.0.0  # Match codegen version
```

### Android build fails with "linker not found"

**Cause**: Android NDK not configured.

**Fix**: See [Android Build Requirements](#android-build-requirements) section below.

## Android Build Requirements

Android builds require the following setup:

### 1. Install Android NDK

**Option A: Using Android Studio (Recommended)**

1. Open Android Studio → Tools → SDK Manager
2. Go to SDK Tools tab
3. Check "NDK (Side by side)" and install
4. Set environment variable:
   ```bash
   export ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk/{version}"
   ```

**Option B: Manual Download**

Download from https://developer.android.com/ndk/downloads and set `ANDROID_NDK_HOME`.

### 2. Install cargo-ndk

```bash
cargo install cargo-ndk
```

### 3. Install Rust Targets

```bash
rustup target add aarch64-linux-android x86_64-linux-android
```

### 4. Build

```bash
cargo xtask build-flutter --platform android
```

### Supported Android ABIs

| ABI | Rust Target | Description |
|-----|-------------|-------------|
| arm64-v8a | `aarch64-linux-android` | 64-bit ARM (most modern devices) |
| x86_64 | `x86_64-linux-android` | 64-bit x86 (emulators, some Chromebooks) |

**Note**: armeabi-v7a (32-bit ARM) is not currently supported due to build constraints.

### API Level

The build uses **API level 28** (Android 9.0 Pie) for compilation headers. This is required because:
- `POSIX_MADV_*` constants (used by llama.cpp) require API 23+
- `aws-lc-sys` (used for TLS) requires API 28+ for `getentropy()`

This does **not** affect your app's `minSdkVersion` - it only determines which NDK headers are used during native library compilation. Your Flutter app can still target older Android versions.

### Build Output

After a successful build, `.so` files are located at:
```
target/aarch64-linux-android/release/libxybrid_flutter_ffi.so  # arm64-v8a
target/x86_64-linux-android/release/libxybrid_flutter_ffi.so   # x86_64
```

These are ELF shared libraries that Flutter will automatically bundle into your APK/AAB.

### iOS build fails with "target not installed"

**Cause**: Missing iOS Rust targets.

**Fix**:
```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
```

### Windows build fails with "link.exe not found"

**Cause**: Visual Studio Build Tools not installed.

**Fix**: Install Visual Studio Build Tools 2019+ with "Desktop development with C++" workload.

### "FRB codegen warning: skipping..." messages

**Cause**: Some Rust types not compatible with FRB or missing `#[frb]` attributes.

**Fix**: Check the skipped types and add appropriate annotations or exclude them from the API.

## Migration from xybrid-flutter

If you were previously using the standalone [xybrid-flutter](https://github.com/xybrid-ai/xybrid-flutter) package, here are the key changes:

### Breaking Changes

1. **Package location**: The Flutter SDK now lives in the monorepo at `repos/xybrid/bindings/flutter/` instead of a separate repository.

2. **Initialization required**: You must now call `await Xybrid.init()` before using any SDK functionality:
   ```dart
   // Before: No initialization needed
   // After:
   await Xybrid.init();
   ```

3. **Async inference**: `model.run()` and `pipeline.run()` now return `Future<XybridResult>`:
   ```dart
   // Before: final result = model.run(envelope);
   // After:
   final result = await model.run(envelope);
   ```

### Dependency Update

Update your `pubspec.yaml` from:

```yaml
dependencies:
  xybrid_flutter: ^0.x.x
```

To:

```yaml
dependencies:
  xybrid:
    git:
      url: https://github.com/xybrid-ai/xybrid.git
      path: repos/xybrid/bindings/flutter
```

### Import Changes

```dart
// Before:
import 'package:xybrid_flutter/xybrid_flutter.dart';

// After:
import 'package:xybrid/xybrid.dart';
```

### API Compatibility

The core API remains similar:
- `XybridModelLoader.fromRegistry(modelId)` - unchanged
- `XybridModelLoader.fromBundle(path)` - unchanged
- `XybridEnvelope.text(text)` - unchanged
- `XybridEnvelope.audio(bytes, sampleRate, channels)` - unchanged
- `XybridResult` properties - unchanged

### New Features

This version adds:
- **Pipeline support**: Chain multiple models with `XybridPipeline`
- **Better error handling**: All errors wrapped in `XybridException`
- **Initialization state**: Check `Xybrid.isInitialized` before operations

## Full Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the complete restructuring plan.
