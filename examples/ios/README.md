# Xybrid iOS Example

A native iOS example app demonstrating Xybrid SDK integration using SwiftUI.

## Features

- **SDK Initialization**: Cache directory setup plus `Xybrid.initialize()` (pass an `apiKey` to enable dashboard telemetry)
- **Model Loading**: Downloads models from the xybrid registry with async/await
- **Text-to-Speech** (Speech tab): Run TTS inference with voice selection + audio playback
- **Live camera vision** (Vision tab): point the camera at a scene and get an on-device VLM caption that refreshes as the scene changes — AVFoundation capture + a cheap luma change-gate firing batch multimodal turns
- **Error Handling**: User-friendly error messages with retry capability

The two demos live behind a `TabView` once the SDK is initialized.

### Live vision: responsiveness model

The Vision tab is **drop-if-busy**, not cancel-and-replace: while one frame is
being answered, newly gated frames are dropped rather than preempting the
in-flight run. That's because the Swift/UniFFI bindings currently expose only the
batch `model.run(...)`; streaming tokens, cancel-and-replace, and raw-frame
(`imageRaw`) envelopes — the realtime-vision primitives already in the Flutter
SDK — are not yet bound for Swift. When that surface lands the loop can move to
latest-frame-wins. See `LiveVision.swift` for the gate + batch-VLM wiring.

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

### 4. Set your API key (optional)

The app reads `XYBRID_API_KEY` (and optionally `XYBRID_PLATFORM_URL`, the
telemetry ingest endpoint) from the scheme's environment — the Xcode-native way
to inject values at run time without committing them. This is the iOS analog of
Flutter's `flutter run --dart-define=...`.

1. **Product → Scheme → Edit Scheme…** (or `Cmd+<`)
2. Select **Run** → **Arguments** tab
3. Under **Environment Variables**, add:
   - `XYBRID_API_KEY` = `sk_test_...`
   - `XYBRID_PLATFORM_URL` = `https://your-platform.example.com` *(optional)*

The values are saved in your *user* scheme under `xcuserdata/`, which is
gitignored — so they never land in the repo. Leave **Shared** unchecked
(the default) to keep it that way. Both are optional: without a key the app
runs anonymously (on-device inference, telemetry disabled); without a platform
URL it uses the default Xybrid endpoint (`XYBRID_PLATFORM_URL` is handy for
pointing a debug build at a self-hosted or tunneled platform). Get a free key
at [dashboard.xybrid.dev](https://dashboard.xybrid.dev).

To inject them from the command line on a Simulator instead, prefix the
launch environment:

```bash
SIMCTL_CHILD_XYBRID_API_KEY=sk_test_... \
SIMCTL_CHILD_XYBRID_PLATFORM_URL=https://your-platform.example.com \
  xcrun simctl launch --console booted ai.xybrid.example
```

### 5. Build and Run

1. Select an iOS device (arm64) — simulator requires separate ORT build
2. Press **Cmd+R** to build and run

The app will:
1. Show a welcome screen with "Initialize SDK" button
2. After init, show a two-tab demo:
   - **Speech** — Model ID input (default: `kokoro-82m`), voice picker, text input,
     "Run Inference", and audio playback of generated speech.
   - **Vision** — a vision-model ID input (default: `lfm2-vl-450m`, editable — use
     any registry-resolvable VLM with an mmproj artifact), a question field,
     "Start live", and a viewfinder with a replace-in-place caption + history.
     Requires camera access (the app declares `NSCameraUsageDescription`).

## Project Structure

```
ios/
├── XybridExample.xcodeproj/     # Xcode project
├── XybridExample/
│   ├── XybridExampleApp.swift   # App entry point
│   ├── ContentView.swift        # SDK init + TabView; the Speech (TTS) demo
│   ├── LiveVision.swift         # Vision tab: camera service + luma gate + batch VLM
│   ├── Assets.xcassets/         # App icons and colors
│   └── Info.plist               # App configuration (incl. NSCameraUsageDescription)
└── README.md                    # This file
```

## SDK Usage Patterns

### Initialize

```swift
import Xybrid

let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
    .first!.appendingPathComponent("xybrid").path
initSdkCacheDir(cacheDir: cacheDir)

// Reads the key and platform URL from scheme environment variables (see
// "Set your API key" above). Empty/unset → anonymous, local-only init.
let env = ProcessInfo.processInfo.environment
let apiKey = env["XYBRID_API_KEY"]
let platformUrl = env["XYBRID_PLATFORM_URL"]
Xybrid.initialize(
    apiKey: (apiKey ?? "").isEmpty ? nil : apiKey,
    ingestUrl: (platformUrl ?? "").isEmpty ? nil : platformUrl
)
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

### Vision (image → VLM, batch)

```swift
// Encode a camera frame (or any image) to JPEG, then send it as a multimodal
// user turn. Batch only on Swift today — see "responsiveness model" above.
let image = XybridEnvelope.image(bytes: jpegData, format: "jpeg")
let envelope = XybridEnvelope.userMessage(text: "What do you see?", images: [image])
let result = try await model.run(envelope: envelope, config: config)
let caption = result.text
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
