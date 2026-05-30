# Xybrid iOS Example

A native iOS example app demonstrating Xybrid SDK integration using SwiftUI.

## Features

- **SDK Initialization**: Cache directory setup plus `Xybrid.initialize()` (pass an `apiKey` to enable dashboard telemetry)
- **Model Loading**: Downloads models from the xybrid registry with async/await
- **Text-to-Speech**: Run TTS inference with voice selection
- **Audio Playback**: Play generated audio via AVAudioPlayer
- **Error Handling**: User-friendly error messages with retry capability

## Prerequisites

| Tool | Required Version | Installation |
|------|------------------|--------------|
| Xcode | 14.0+ | Mac App Store |
| iOS | 15.0+ | Deployment target |
| Rust | 1.70+ | [rustup.rs](https://rustup.rs) (for building XCFramework) |

## Quick Start

### 1. Build the XCFramework

```bash
# From the xybrid repo root
rustup target add aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin
cargo xtask build-xcframework --release
```

### 2. Open in Xcode

```bash
open examples/ios/XybridExample.xcodeproj
```

### 3. Add Xybrid Package Dependency

In Xcode:

1. Select the project in the navigator
2. Go to **Package Dependencies**
3. Click **+** and add local package: `../../bindings/apple`
4. Select **Xybrid** library

### 4. Build and Run

1. Select an iOS device (arm64) — simulator requires separate ORT build
2. Press **Cmd+R** to build and run

The app will:
1. Show a welcome screen with "Initialize SDK" button
2. After init, show inference demo with:
   - Model ID input (default: `kokoro-82m`)
   - Voice picker (populated after model loads)
   - Text input for TTS
   - "Run Inference" button
   - Audio playback of generated speech

## Project Structure

```
ios/
├── XybridExample.xcodeproj/     # Xcode project
├── XybridExample/
│   ├── XybridExampleApp.swift   # App entry point
│   ├── ContentView.swift        # Main UI with real SDK integration
│   ├── Assets.xcassets/         # App icons and colors
│   └── Info.plist               # App configuration
└── README.md                    # This file
```

## SDK Usage Patterns

### Initialize

```swift
import Xybrid

let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
    .first!.appendingPathComponent("xybrid").path
initSdkCacheDir(cacheDir: cacheDir)

// Runs locally with no key. Pass an apiKey to light up the dashboard:
//   Xybrid.initialize(apiKey: ProcessInfo.processInfo.environment["XYBRID_API_KEY"])
Xybrid.initialize()
```

### Load Model

```swift
let loader = XybridModelLoader.fromRegistry(modelId: "kokoro-82m")
let model = try await loader.load()
let voices = model.voices()  // [XybridVoiceInfo]?
```

### Run Inference

```swift
let envelope = XybridEnvelope.text(text: "Hello!", voiceId: "af", speed: 1.0)
let result = try await model.run(envelope: envelope, config: nil)

if result.success, let audio = result.audioBytes {
    // Play audio (wrap in WAV header for AVAudioPlayer)
}
```

### Generation Config (LLM)

```swift
let config = XybridGenerationConfig(
    maxTokens: 512,
    temperature: 0.7,
    topP: 0.9,
    minP: nil, topK: nil,
    repetitionPenalty: nil,
    stopSequences: nil
)
let result = try await model.run(envelope: envelope, config: config)
```

## Troubleshooting

### "Module 'Xybrid' not found"

The XCFramework hasn't been built or the package dependency hasn't been added.

1. Run `cargo xtask build-xcframework --release` from the xybrid repo root
2. Add the local package dependency in Xcode (see Quick Start step 3)

### Linker errors

Ensure the XCFramework includes headers. Rebuild with `cargo xtask build-xcframework --release`.

### App crashes on launch

Check that:
1. XCFramework is built for the correct architecture (arm64 for device)
2. `initSdkCacheDir()` is called before any model operations

## Related

- [Xybrid Swift SDK](../../bindings/apple/README.md)
- [SDK API Reference](../../docs/sdk/API_REFERENCE.md)
- [Main Examples README](../README.md)
