# Xybrid Swift SDK (iOS/macOS)

> **Status**: Coming Soon — Swift bindings are in development. Use [Flutter](../flutter/) or [Kotlin](../kotlin/) for production use today.

Native iOS and macOS SDK for [Xybrid](https://github.com/xybrid-ai/xybrid), providing on-device ML inference via BoltFFI-generated Swift bindings.

## Installation

### Swift Package Manager (Recommended)

Add Xybrid to your Xcode project:

1. In Xcode, select **File > Add Package Dependencies...**
2. Enter: `https://github.com/xybrid-ai/xybrid`
3. Set **Dependency Rule** to **Up to Next Major Version** → `0.6.0`
4. Select the **Xybrid** library product

Or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/xybrid-ai/xybrid", from: "0.6.0")
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

// Describing the source is cheap; load() is the explicit async boundary.
let model = try await Xybrid.model("kokoro-82m").load()

// Create an envelope for TTS
let envelope = XybridEnvelope.text(
    text: "Hello, world!",
    voiceId: "af",
    speed: 1.0
)

// Run inference without blocking the caller's executor.
let result = try await model.runAsync(envelope: envelope)

// Access the result
if result.success {
    if let audioBytes = result.audioBytes {
        // Play or save the audio
    }
}
```

### Reasoning (thinking models)

Reasoning models (metadata `reasoning: true`, e.g. `lfm2.5-1.2b-thinking`)
produce a chain-of-thought before their answer. Xybrid keeps it out of the
answer text and surfaces it on `reasoningContent` — `nil` for non-thinking
models. Nothing to enable; just read it if you want it.

```swift
let model = try await Xybrid.model("lfm2.5-1.2b-thinking").load()
let result = try await model.runAsync(envelope: XybridEnvelope.text(
    "Is 97 a prime number? Reason, then answer."))

if let answer = result.text { print("Answer:", answer) }
if let reasoning = result.reasoningContent { print("Reasoning:", reasoning) }
```

### Available Types

| Type | Description |
|------|-------------|
| `ModelSource` | Registry, bundle, directory, or Hugging Face model location |
| `ModelLoader` | Cheap model reference; call `load()` to perform I/O |
| `XybridModel` | A loaded model ready for inference |
| `XybridEnvelope` | Input data container (audio, text, embedding, image, or multi-part user message) |
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

// Vision-language input
let image = try XybridEnvelope.image(imageData, format: "jpeg")
let prompt = try XybridEnvelope.userMessage(
    "Describe this image",
    images: [image]
)
```

For larger VLM variants, validate on iPhone 15 Pro or newer. Smaller devices should use
capability checks before loading a vision bundle so unsupported paths fail with a clear
runtime error instead of an opaque memory pressure failure.

## Structure

```
apple/
├── Package.swift                    # Swift Package manifest (local dev: path-based binaryTarget)
├── Sources/
│   └── Xybrid/                      # Swift source
│       ├── Xybrid.swift             # Public API, extensions, type aliases
│       └── xybrid_bolt.swift        # BoltFFI-generated Swift bindings (DO NOT EDIT)
└── XCFrameworks/                    # Local-dev unzip target for the Bazel-built xcframework
    └── XybridFFI.xcframework/       # (gitignored; unzipped from bazel-bin)
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
| Bazel (via bazelisk) | per `.bazelversion` | `brew install bazelisk` |
| Xcode Command Line Tools | Latest | `xcode-select --install` |

### Building

The xcframework builds with Bazel (which brings its own Rust toolchain — no
rustup targets needed). Install Bazel via
[bazelisk](https://github.com/bazelbuild/bazelisk) (`brew install bazelisk`),
then:

```bash
# From the xybrid repo root
bazel build --config=ios //bindings/apple:XybridFFI

# Unzip it where the Swift package's local-natives mode looks
unzip -o bazel-bin/bindings/apple/XybridFFI.xcframework.zip -d bindings/apple/XCFrameworks
```

This produces `XCFrameworks/XybridFFI.xcframework` containing:
- iOS device (arm64)
- iOS simulator (arm64)

### Build Output

After a successful build:

```
bindings/apple/XCFrameworks/
└── XybridFFI.xcframework/
    ├── Info.plist
    ├── ios-arm64/
    │   └── XybridFFI.framework/        # static framework (binary, Headers/, Modules/)
    └── ios-arm64-simulator/
        └── XybridFFI.framework/
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

**Fix**: Build via Bazel (`bazel build --config=ios //bindings/apple:XybridFFI`) — it provides its own Rust toolchain and targets

#### "xcodebuild: error: cannot be used together with -create-xcframework"

**Cause**: Conflicting xcodebuild options or incompatible library format.

**Fix**: Ensure you're using static libraries (.a files), not dynamic (.dylib).

#### Build works but Swift can't find the module

**Cause**: XCFramework not in expected location or not linked.

**Fix**: Ensure `XCFrameworks/XybridFFI.xcframework` exists and is listed in your Xcode project's "Frameworks, Libraries, and Embedded Content".

#### "Undefined symbols for architecture arm64"

**Cause**: XCFramework built for different architecture than target.

**Fix**: Rebuild with `bazel build --config=ios //bindings/apple:XybridFFI`.

### Non-macOS Developers

XCFramework builds require macOS with Xcode. If you're developing on Linux or Windows:

1. **Use prebuilt XCFrameworks**: Download from [GitHub Releases](https://github.com/xybrid-ai/xybrid/releases)
2. **Use CI**: Push your changes and let GitHub Actions build the XCFramework
3. **Use a macOS VM or CI service**: If you need local builds

## FFI Strategy

The Swift bindings are generated from `crates/xybrid-bolt/` using [BoltFFI](https://crates.io/crates/boltffi):
- Single Rust source generates Swift, Kotlin, Java, C#, WASM, and a C header
- Native async/await support
- Memory-safe wrappers
- Automatic error handling

## Telemetry

The Apple binding reports `binding=swift` in a small `X-Xybrid-Client` header attached to registry metadata calls. See [docs/telemetry/registry.md](../../docs/telemetry/registry.md) for the exact wire format and the opt-out switch (`XYBRID_TELEMETRY_OPTOUT=1`).

## Full Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the complete restructuring plan.
