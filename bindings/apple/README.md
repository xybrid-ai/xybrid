# Xybrid Apple Binding (Swift)

> **Status**: In Development - UniFFI bindings generated

This directory contains the Swift package for Xybrid, providing native iOS and macOS support via UniFFI-generated bindings.

## Installation

### Swift Package Manager (Recommended)

Add the Xybrid package to your Xcode project:

1. In Xcode, select **File > Add Package Dependencies...**
2. Enter the repository URL: `https://github.com/xybrid-ai/xybrid`
3. Select the package product **Xybrid**

Or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/xybrid-ai/xybrid", from: "0.1.0")
]
```

Then add the dependency to your target:

```swift
.target(
    name: "YourApp",
    dependencies: ["Xybrid"]
)
```

## Usage

```swift
import Xybrid

// Load a model from the registry
let loader = XybridModelLoader.fromRegistry(modelId: "kokoro-82m")
let model = try loader.load()

// Create an envelope for TTS
let envelope = XybridEnvelope.text(
    text: "Hello, world!",
    voiceId: "af",
    speed: 1.0
)

// Run inference
let result = try model.run(envelope: envelope)

// Access the result
if result.success {
    if let audioBytes = result.audioBytes {
        // Play or save the audio
    }
}
```

### Available Types

| Type | Description |
|------|-------------|
| `XybridModelLoader` | Loads models from registry or local bundles |
| `XybridModel` | Represents a loaded model ready for inference |
| `XybridEnvelope` | Input data container (audio, text, or embedding) |
| `XybridResult` | Inference result with success status and output data |
| `XybridError` | Error enum for error handling |

### Creating Envelopes

```swift
// Text-to-Speech input
let ttsEnvelope = XybridEnvelope.text(
    text: "Convert this to speech",
    voiceId: "af",     // Optional voice ID
    speed: 1.0         // Optional speed multiplier
)

// Speech-to-Text input
let asrEnvelope = XybridEnvelope.audio(
    bytes: audioData,
    sampleRate: 16000,
    channels: 1
)

// Embedding input
let embeddingEnvelope = XybridEnvelope.embedding(
    data: [0.1, 0.2, 0.3, ...]
)
```

## Structure

```
apple/
├── Package.swift                    # Swift Package manifest
├── Sources/
│   ├── Xybrid/                      # Swift source
│   │   └── xybrid_uniffi.swift      # UniFFI-generated Swift bindings
│   └── XybridFFI/                   # C FFI layer
│       ├── include/
│       │   └── xybrid_uniffiFFI.h   # C header for FFI
│       └── xybrid_uniffiFFI.modulemap
└── XCFrameworks/                    # Prebuilt binaries (future)
    └── XybridFFI.xcframework
```

## Supported Platforms

| Platform | Minimum Version |
|----------|-----------------|
| iOS | 13.0 |
| macOS | 10.15 (Catalina) |

## Building the XCFramework

The XCFramework containing the compiled Rust library must be built separately:

```bash
# From the xybrid repo root
cargo xtask build-xcframework
```

This will produce `XCFrameworks/XybridFFI.xcframework` containing:
- iOS device (arm64)
- iOS simulator (arm64, x86_64)
- macOS (arm64, x86_64)

## FFI Strategy

The Swift bindings are generated from `crates/xybrid-uniffi/` using [UniFFI](https://mozilla.github.io/uniffi-rs/):
- Single Rust source generates both Swift and Kotlin
- Native async/await support
- Memory-safe wrappers
- Automatic error handling

## Full Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the complete restructuring plan.
