# Xybrid Kotlin Binding (Android)

> **Status**: Active - UniFFI bindings generated

This directory contains the Android library for Xybrid, providing native Kotlin/Java support via UniFFI-generated bindings.

## Installation

### Gradle (Maven Central) - Coming Soon

```kotlin
// In your app's build.gradle.kts
dependencies {
    implementation("ai.xybrid:xybrid:0.1.0")
}
```

### Local Development

For local development, add this module as a project dependency:

```kotlin
// In your settings.gradle.kts
include(":xybrid")
project(":xybrid").projectDir = file("/path/to/xybrid/bindings/kotlin")

// In your app's build.gradle.kts
dependencies {
    implementation(project(":xybrid"))
}
```

## Usage

### Loading a Model from Registry

```kotlin
import ai.xybrid.XybridModelLoader
import ai.xybrid.XybridEnvelope
import ai.xybrid.XybridException

// Load a model from the registry
val loader = XybridModelLoader.fromRegistry("kokoro-82m")
val model = loader.load()

// Run text-to-speech
val envelope = XybridEnvelope.Text(
    text = "Hello, world!",
    voiceId = "af_bella",
    speed = 1.0
)
val result = model.run(envelope)

if (result.success) {
    val audioBytes = result.audioBytes
    // Play audio...
} else {
    println("Error: ${result.error}")
}
```

### Loading a Model from Bundle

```kotlin
import ai.xybrid.XybridModelLoader

// Load from a local bundle path
val loader = XybridModelLoader.fromBundle("/path/to/model/bundle")
val model = loader.load()
```

### Speech Recognition (ASR)

```kotlin
import ai.xybrid.XybridEnvelope

// Create audio envelope
val audioBytes: ByteArray = loadAudioFile()  // Your audio loading code
val envelope = XybridEnvelope.Audio(
    bytes = audioBytes,
    sampleRate = 16000u,  // 16kHz for most ASR models
    channels = 1u         // Mono audio
)

val result = model.run(envelope)
if (result.success) {
    println("Transcription: ${result.text}")
}
```

### Embeddings

```kotlin
import ai.xybrid.XybridEnvelope

// Create embedding envelope
val embedding = XybridEnvelope.Embedding(
    data = listOf(0.1f, 0.2f, 0.3f)  // Input vector
)

val result = model.run(embedding)
if (result.success && result.embedding != null) {
    val outputVector = result.embedding!!
    // Use embedding...
}
```

### Error Handling

```kotlin
import ai.xybrid.XybridException

try {
    val loader = XybridModelLoader.fromRegistry("unknown-model")
    val model = loader.load()
} catch (e: XybridException.ModelNotFound) {
    println("Model not found: ${e.modelId}")
} catch (e: XybridException.InferenceFailed) {
    println("Inference failed: ${e.message}")
} catch (e: XybridException.InvalidInput) {
    println("Invalid input: ${e.message}")
} catch (e: XybridException.IoException) {
    println("I/O error: ${e.message}")
}
```

## API Reference

### Types

| Type | Description |
|------|-------------|
| `XybridModelLoader` | Factory for loading models from registry or bundle |
| `XybridModel` | Loaded model ready for inference |
| `XybridEnvelope` | Input data (Audio, Text, or Embedding) |
| `XybridResult` | Inference output with success/error and result data |
| `XybridException` | Error types (ModelNotFound, InferenceFailed, etc.) |

### XybridModelLoader

| Method | Description |
|--------|-------------|
| `fromRegistry(modelId: String)` | Load model from Xybrid registry |
| `fromBundle(path: String)` | Load model from local bundle path |
| `load(): XybridModel` | Fetch and load the model |

### XybridEnvelope

| Variant | Fields |
|---------|--------|
| `Audio` | `bytes: ByteArray`, `sampleRate: UInt`, `channels: UInt` |
| `Text` | `text: String`, `voiceId: String?`, `speed: Double?` |
| `Embedding` | `data: List<Float>` |

### XybridResult

| Field | Type | Description |
|-------|------|-------------|
| `success` | `Boolean` | Whether inference succeeded |
| `error` | `String?` | Error message if failed |
| `outputType` | `String` | Type of output ("text", "audio", "embedding") |
| `text` | `String?` | Text output (for ASR) |
| `audioBytes` | `ByteArray?` | Audio output (for TTS) |
| `embedding` | `List<Float>?` | Embedding output |
| `latencyMs` | `UInt` | Inference latency in milliseconds |

## Directory Structure

```
kotlin/
├── build.gradle.kts                     # Gradle build configuration
├── README.md                            # This file
├── libs/                                # Prebuilt native libraries
│   ├── armeabi-v7a/
│   │   └── libxybrid_uniffi.so
│   ├── arm64-v8a/
│   │   └── libxybrid_uniffi.so
│   └── x86_64/
│       └── libxybrid_uniffi.so
└── src/main/kotlin/ai/xybrid/
    └── xybrid_uniffi.kt                 # UniFFI-generated bindings
```

## FFI Strategy

The Kotlin bindings are generated from `crates/xybrid-uniffi/` using UniFFI:
- Single Rust source generates both Swift and Kotlin bindings
- Memory-safe wrappers with proper resource cleanup
- Uses JNA for native library loading

## Building Native Libraries

Native `.so` files must be built for each target architecture before the library can be used.

### Prerequisites

| Tool | Required Version | Installation |
|------|------------------|--------------|
| Rust | 1.70+ | [rustup.rs](https://rustup.rs) |
| Android NDK | r26+ (recommended: r26b) | Android Studio or sdkmanager |
| cargo-ndk | Latest | `cargo install cargo-ndk` |

### Installing Android NDK

**Option 1: Android Studio (Recommended)**

1. Open Android Studio
2. Go to **Tools > SDK Manager**
3. Select **SDK Tools** tab
4. Check **NDK (Side by side)** and click Apply
5. Note the installation path (e.g., `$ANDROID_HOME/ndk/26.1.10909125`)

**Option 2: Command Line (sdkmanager)**

```bash
# Install NDK via sdkmanager
sdkmanager --install "ndk;26.1.10909125"

# Find your SDK location
echo $ANDROID_HOME
# Typically: ~/Library/Android/sdk (macOS) or ~/Android/Sdk (Linux)
```

### Environment Variables

Set these environment variables before building:

| Variable | Description | Example |
|----------|-------------|---------|
| `ANDROID_HOME` | Android SDK root directory | `~/Library/Android/sdk` |
| `ANDROID_NDK_HOME` | NDK installation directory | `$ANDROID_HOME/ndk/26.1.10909125` |

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export ANDROID_HOME="$HOME/Library/Android/sdk"  # macOS
# export ANDROID_HOME="$HOME/Android/Sdk"        # Linux
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/26.1.10909125"
export PATH="$PATH:$ANDROID_HOME/cmdline-tools/latest/bin"
```

### Installing Rust Targets

```bash
# From the xybrid repo root
cargo xtask setup-targets

# Or manually:
rustup target add aarch64-linux-android      # arm64-v8a
rustup target add armv7-linux-androideabi    # armeabi-v7a
rustup target add x86_64-linux-android       # x86_64
```

### Building

**Using xtask (Recommended)**

```bash
# Build all ABIs
cargo xtask build-android

# Build specific ABI only
cargo xtask build-android --abi arm64-v8a

# Debug build (with symbols, unoptimized)
cargo xtask build-android --debug

# With explicit version
cargo xtask build-android --version 0.2.0
```

**Manual Build (without cargo-ndk)**

```bash
# Set up linker for each target (API 21+)
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android21-clang"
export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/armv7a-linux-androideabi21-clang"
export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/x86_64-linux-android21-clang"

# Build each target
cargo build -p xybrid-uniffi --lib --release --target aarch64-linux-android
cargo build -p xybrid-uniffi --lib --release --target armv7-linux-androideabi
cargo build -p xybrid-uniffi --lib --release --target x86_64-linux-android
```

### Build Output

After a successful build:

```
bindings/kotlin/libs/
├── arm64-v8a/
│   └── libxybrid_uniffi.so
├── armeabi-v7a/
│   └── libxybrid_uniffi.so
├── x86_64/
│   └── libxybrid_uniffi.so
└── {version}/                    # Versioned copy
    ├── arm64-v8a/
    │   └── libxybrid_uniffi.so
    ├── armeabi-v7a/
    │   └── libxybrid_uniffi.so
    └── x86_64/
        └── libxybrid_uniffi.so
```

### Troubleshooting

#### "error: linker 'aarch64-linux-android21-clang' not found"

**Cause**: NDK not found or `ANDROID_NDK_HOME` not set.

**Fix**:
```bash
# Verify NDK is installed
ls $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/

# Set environment variable
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/26.1.10909125"
```

#### "error: target 'aarch64-linux-android' not installed"

**Cause**: Missing Rust target.

**Fix**: Run `cargo xtask setup-targets` or `rustup target add aarch64-linux-android`

#### "error: could not find 'cargo-ndk'"

**Cause**: cargo-ndk not installed.

**Fix**: `cargo install cargo-ndk`

#### "ld: error: undefined symbol" (at link time)

**Cause**: Wrong NDK version or missing system libraries.

**Fix**: Use NDK r26+ and ensure API level 21+ is targeted.

#### ".so file not loading in Android app"

**Cause**: ABI mismatch between built library and device.

**Fix**:
1. Verify you built for the correct ABI (check `adb shell getprop ro.product.cpu.abi`)
2. Ensure the .so file is in the correct `jniLibs/{abi}/` directory

#### "java.lang.UnsatisfiedLinkError: dlopen failed"

**Cause**: Missing native library or corrupted .so file.

**Fix**:
1. Verify the .so file is valid: `file libs/arm64-v8a/libxybrid_uniffi.so`
2. Should show: `ELF 64-bit LSB shared object, ARM aarch64`
3. Rebuild with `cargo xtask build-android`

### Platform Notes

| Platform | NDK Prebuilt Path |
|----------|-------------------|
| macOS (Intel) | `darwin-x86_64` |
| macOS (Apple Silicon) | `darwin-x86_64` (Rosetta 2) |
| Linux | `linux-x86_64` |
| Windows | `windows-x86_64`

## Supported Android Versions

| Android API | Version Name |
|-------------|--------------|
| API 24+ | Android 7.0 (Nougat) |

## NDK ABIs

| Architecture | ABI | Device Examples |
|--------------|-----|-----------------|
| ARMv7 | armeabi-v7a | Older Android phones |
| ARM64 | arm64-v8a | Most modern Android phones |
| x86_64 | x86_64 | Android emulator on Intel/AMD |

## Full Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the complete restructuring plan.
