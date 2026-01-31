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

The XCFramework containing the compiled Rust library must be built before using the Swift package.

### Prerequisites

| Tool | Required Version | Installation |
|------|------------------|--------------|
| Xcode | 14.0+ | Mac App Store |
| Rust | 1.70+ | [rustup.rs](https://rustup.rs) |
| Xcode Command Line Tools | Latest | `xcode-select --install` |

### Installing Rust Targets

Install the required cross-compilation targets:

```bash
# From the xybrid repo root
cargo xtask setup-targets

# Or manually:
rustup target add aarch64-apple-ios        # iOS device (arm64)
rustup target add aarch64-apple-ios-sim    # iOS simulator (arm64)
rustup target add x86_64-apple-ios         # iOS simulator (x86_64)
rustup target add aarch64-apple-darwin     # macOS (arm64)
rustup target add x86_64-apple-darwin      # macOS (x86_64)
```

### Building

```bash
# From the xybrid repo root
cargo xtask build-xcframework

# With debug symbols (slower, larger binaries)
cargo xtask build-xcframework --debug

# With explicit version
cargo xtask build-xcframework --version 0.2.0
```

This produces `XCFrameworks/XybridFFI.xcframework` containing:
- iOS device (arm64)
- iOS simulator (arm64, x86_64 universal)
- macOS (arm64, x86_64 universal)

### Build Output

After a successful build:

```
bindings/apple/XCFrameworks/
├── XybridFFI.xcframework/
│   ├── ios-arm64/
│   │   └── libxybrid_uniffi.a
│   ├── ios-arm64_x86_64-simulator/
│   │   └── libxybrid_uniffi.a
│   └── macos-arm64_x86_64/
│       └── libxybrid_uniffi.a
└── XybridFFI-{version}.xcframework/    # Versioned copy
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEVELOPER_DIR` | Path to Xcode.app | Auto-detected |

### Troubleshooting

#### "error: linker 'cc' not found"

**Cause**: Xcode Command Line Tools not installed.

**Fix**: Run `xcode-select --install`

#### "error: target 'aarch64-apple-ios' not installed"

**Cause**: Missing Rust target.

**Fix**: Run `cargo xtask setup-targets` or `rustup target add aarch64-apple-ios`

#### "xcodebuild: error: cannot be used together with -create-xcframework"

**Cause**: Conflicting xcodebuild options or incompatible library format.

**Fix**: Ensure you're using static libraries (.a files), not dynamic (.dylib).

#### Build works but Swift can't find the module

**Cause**: XCFramework not in expected location or not linked.

**Fix**: Ensure `XCFrameworks/XybridFFI.xcframework` exists and is listed in your Xcode project's "Frameworks, Libraries, and Embedded Content".

#### "Undefined symbols for architecture arm64"

**Cause**: XCFramework built for different architecture than target.

**Fix**: Rebuild with `cargo xtask build-xcframework` ensuring all targets are installed.

### Non-macOS Developers

XCFramework builds require macOS with Xcode. If you're developing on Linux or Windows:

1. **Use prebuilt XCFrameworks**: Download from [GitHub Releases](https://github.com/xybrid-ai/xybrid/releases)
2. **Use CI**: Push your changes and let GitHub Actions build the XCFramework
3. **Use a macOS VM or CI service**: If you need local builds

## FFI Strategy

The Swift bindings are generated from `crates/xybrid-uniffi/` using [UniFFI](https://mozilla.github.io/uniffi-rs/):
- Single Rust source generates both Swift and Kotlin
- Native async/await support
- Memory-safe wrappers
- Automatic error handling

## Full Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the complete restructuring plan.
