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

Native `.so` files must be built for each target architecture:

```bash
# Build for all Android targets
cargo xtask build-android

# Or use cross-compilation manually
cargo build --lib --release --target aarch64-linux-android
cargo build --lib --release --target armv7-linux-androideabi
cargo build --lib --release --target x86_64-linux-android
```

Copy the resulting `libxybrid_uniffi.so` files to the appropriate `libs/` subdirectories.

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
