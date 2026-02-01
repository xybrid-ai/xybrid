# Xybrid iOS Example

A native iOS example app demonstrating Xybrid SDK integration using SwiftUI.

## Features

- **SDK Initialization**: Shows proper async SDK startup flow
- **Model Loading**: Text field for model ID, load button with progress indicator
- **Text-to-Speech Inference**: Run TTS inference with text input
- **Result Display**: Shows inference result, latency metrics, and play button
- **Error Handling**: User-friendly error messages with retry capability
- **Swift Concurrency**: Proper async/await usage throughout

## Prerequisites

| Tool | Required Version | Installation |
|------|------------------|--------------|
| Xcode | 14.0+ | Mac App Store |
| iOS | 15.0+ | Deployment target |
| Rust | 1.70+ | [rustup.rs](https://rustup.rs) (for building XCFramework) |

## Quick Start

### 1. Open in Xcode

```bash
open XybridExample.xcodeproj
```

Or double-click `XybridExample.xcodeproj` in Finder.

### 2. Build and Run

1. Select an iOS Simulator or device
2. Press **Cmd+R** to build and run

The example app will launch with:
1. Welcome screen with "Initialize SDK" button
2. After initialization, an inference demo screen with:
   - Model ID input (default: `kokoro-82m`)
   - "Load Model" button
   - Text input for TTS
   - "Run Inference" button
   - Result display with latency and play button

## Building with XCFramework (Full SDK Functionality)

To enable actual SDK functionality (model loading, inference), you need to build the XCFramework:

### 1. Install Rust Targets

```bash
# From the xybrid repo root
cd ../../..  # Navigate to xybrid repo root
cargo xtask setup-targets

# Or manually:
rustup target add aarch64-apple-ios        # iOS device (arm64)
rustup target add aarch64-apple-ios-sim    # iOS simulator (arm64)
rustup target add x86_64-apple-ios         # iOS simulator (x86_64)
```

### 2. Build XCFramework

```bash
# From xybrid repo root
cargo xtask build-xcframework
```

### 3. Enable Xybrid Import

Edit `XybridExample/ContentView.swift`:

```swift
// Change this:
// import Xybrid

// To this:
import Xybrid
```

### 4. Add Xybrid Package Dependency

In Xcode:

1. Select the project in the navigator
2. Go to **Package Dependencies**
3. Click **+** and add local package: `../../bindings/apple`
4. Select **Xybrid** library

### 5. Rebuild

Press **Cmd+R** to build and run with full SDK functionality.

## Project Structure

```
ios/
├── XybridExample.xcodeproj/     # Xcode project
├── XybridExample/
│   ├── XybridExampleApp.swift   # App entry point
│   ├── ContentView.swift        # Main UI with inference demo
│   ├── Assets.xcassets/         # App icons and colors
│   └── Info.plist               # App configuration
└── README.md                    # This file
```

## SDK Usage Examples

### Initialize SDK

```swift
import Xybrid

// Using Swift async/await
Task {
    do {
        try await Xybrid.initialize()
        print("SDK ready!")
    } catch {
        print("Init failed: \(error.localizedDescription)")
    }
}
```

### Load Model from Registry

```swift
// Load model asynchronously
Task {
    do {
        let loader = XybridModelLoader.fromRegistry(modelId: "kokoro-82m")
        let model = try await loader.load()
        print("Loaded: \(model.modelId())")
    } catch {
        print("Load failed: \(error.localizedDescription)")
    }
}
```

### Run Inference (TTS)

```swift
// Create text envelope for TTS
let envelope = XybridEnvelope.text(
    text: "Hello, world!",
    voiceId: "af",
    speed: 1.0
)

// Run inference with timing
let startTime = CFAbsoluteTimeGetCurrent()

Task {
    do {
        let result = try await model.run(envelope: envelope)
        let latencyMs = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000)

        if result.success, let audio = result.audioBytes {
            print("Generated audio in \(latencyMs)ms")
            // Play or save audio
        }
    } catch {
        print("Inference failed: \(error.localizedDescription)")
    }
}
```

### Run Inference (ASR)

```swift
// Create audio envelope for ASR
let envelope = XybridEnvelope.audio(
    bytes: audioData,
    sampleRate: 16000,
    channels: 1
)

Task {
    do {
        let result = try await model.run(envelope: envelope)

        if let text = result.text {
            print("Transcription: \(text)")
        }
    } catch {
        print("Inference failed: \(error.localizedDescription)")
    }
}
```

## Troubleshooting

### "Module 'Xybrid' not found"

The XCFramework hasn't been built. Follow the "Building with XCFramework" section above.

### Linker errors (undefined symbols)

Same as above - the Rust library needs to be compiled and linked via XCFramework.

### "Target 'aarch64-apple-ios' not installed"

Install the required Rust target:

```bash
rustup target add aarch64-apple-ios
```

### Build fails on Intel Mac

Ensure you have all simulator targets:

```bash
rustup target add x86_64-apple-ios
```

### App crashes on launch

Check that:
1. XCFramework is built for the correct architecture
2. Xybrid Swift Package is properly linked
3. All required frameworks are embedded

## Architecture

This example uses:

- **SwiftUI** for declarative UI
- **Swift Concurrency** (async/await) for asynchronous operations
- **Swift Package Manager** for Xybrid SDK dependency
- **UniFFI** for Rust-Swift bridging (via XCFramework)

The SDK is linked via a local Swift Package reference to `../../bindings/apple`.

## API Reference

- **[SDK API Reference](../../docs/sdk/API_REFERENCE.md)** - Complete API specification
- **[Swift SDK](../../docs/sdk/swift.md)** - Swift-specific documentation (planned)
- **[Online Documentation](https://docs.xybrid.dev/sdk/swift)** - Full guides and tutorials

## Related

- [Main Examples README](../README.md) - All platform examples
- [Xybrid Swift SDK](../../bindings/apple/README.md)
- [Building XCFramework](../../bindings/apple/README.md#building-the-xcframework)
