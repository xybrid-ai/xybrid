# Xybrid iOS Example

A native iOS example app demonstrating Xybrid SDK integration using SwiftUI.

## Features

- **SDK Initialization**: Cache directory setup plus `Xybrid.initialize()`; with an `apiKey`, also starts dashboard trace export via `Xybrid.Observability`
- **Model Loading**: Downloads models from the xybrid registry with async/await
- **Text-to-Speech** (Speech tab): Run TTS inference with voice selection + audio playback
- **Live camera vision** (Vision tab): point the camera at a scene and get an on-device VLM caption that refreshes as the scene changes — AVFoundation capture + a cheap luma change-gate firing batch multimodal turns
- **Structured extraction** (Extract tab): LFM2.5-230M turns messy text (receipts, email signatures) into schema-valid JSON via GBNF constrained decoding — a JSON Schema is converted with `jsonSchemaToGbnf` and attached as `XybridGenerationConfig.grammar`, so the model *cannot* emit output that violates the schema. Toggle the constraint off to watch the raw model drift into markdown fences and made-up keys.
- **Error Handling**: User-friendly error messages with retry capability

The three demos live behind a `TabView` once the SDK is initialized.

### Live vision: responsiveness model

The Vision tab is **drop-if-busy**, not cancel-and-replace: while one frame is
being answered, newly gated frames are dropped rather than preempting the
in-flight run, and each frame is answered by one batch `model.run(...)` call.

That is this demo's choice, not a missing binding. Swift streams tokens with
`model.streamTokens(envelope:options:)` (see "Stream Tokens" below), and
cancelling the task that consumes the stream stops generation at the next
token. Two realtime-vision primitives in the Flutter SDK are not bound for
Swift yet: the `preempt` flag, which cancels the previous in-flight run for you
(cancel-and-replace), and raw-frame (`imageRaw`) envelopes, so camera frames go
in as encoded JPEG. See `LiveVision.swift` for the gate + batch-VLM wiring.

## Prerequisites

| Tool | Required Version | Installation |
|------|------------------|--------------|
| Xcode | 15.3+ | Mac App Store |
| iOS | 16.0+ | Deployment target |
| Bazel (via bazelisk) | per `.bazelversion` | `brew install bazelisk` (builds the XCFramework with its own Rust toolchain) |

## Quick Start

### 1. Build the XCFramework

```bash
# From the xybrid repo root (Bazel brings its own Rust toolchain)
bazel build --config=ios //bindings/apple:XybridFFI

# Unzip it where the Swift package's local-natives mode looks
unzip -o bazel-bin/bindings/apple/XybridFFI.xcframework.zip -d bindings/apple/XCFrameworks

# Point the Swift package at the local build. Without this, SPM downloads the
# published release xcframework, whose headers may predate symbols the checked-
# out wrapper calls — the failure mode is a Swift error like
# "Cannot find 'boltffi_json_schema_to_gbnf' in scope".
./bindings/apple/scripts/set-natives-mode.sh --set-local
```

> `useLocalNatives = true` is a local-dev setting — run
> `./bindings/apple/scripts/set-natives-mode.sh --set-remote` before committing
> anything that touches `Package.swift`.

### 2. Open in Xcode

```bash
open examples/ios/XybridExample.xcodeproj
```

### 3. Add Xybrid Package Dependency

In Xcode:

1. Select the project in the navigator
2. Go to **Package Dependencies**
3. The project already references the Xybrid package at the repo root (`../..`).
   If the reference is missing, click **+** and add the local package at the
   repo root, the folder that holds `Package.swift`
4. Select **Xybrid** library

### 4. Set your API key (optional)

The app reads `XYBRID_API_KEY` (and optionally `XYBRID_TRACES_ENDPOINT`, a full
OTLP trace URL) from the scheme's environment — the Xcode-native way
to inject values at run time without committing them. This is the iOS analog of
Flutter's `flutter run --dart-define=...`.

1. **Product → Scheme → Edit Scheme…** (or `Cmd+<`)
2. Select **Run** → **Arguments** tab
3. Under **Environment Variables**, add:
   - `XYBRID_API_KEY` = `sk_test_...`
   - `XYBRID_TRACES_ENDPOINT` = `https://your-platform.example.com/v1/traces` *(optional)*

The values are saved in your *user* scheme under `xcuserdata/`, which is
gitignored — so they never land in the repo. Leave **Shared** unchecked
(the default) to keep it that way. Both are optional: without a key the app
runs anonymously (on-device inference, no trace export). With a key, the app
passes it to `Xybrid.initialize` and also starts trace export with
`Xybrid.Observability.xybrid(apiKey:)` — the SDK never exports on the key
alone. `XYBRID_TRACES_ENDPOINT` points those traces at a self-hosted or
tunneled endpoint instead of Xybrid. Get a key at
[console.xybrid.dev](https://console.xybrid.dev).

To inject them from the command line on a Simulator instead, prefix the
launch environment:

```bash
SIMCTL_CHILD_XYBRID_API_KEY=sk_test_... \
SIMCTL_CHILD_XYBRID_TRACES_ENDPOINT=https://your-platform.example.com/v1/traces \
  xcrun simctl launch --console booted dev.xybrid.example
```

### 5. Build and Run

1. Select an iOS device, or an iOS Simulator on an Apple Silicon Mac (the
   xcframework's simulator slice is arm64 only). The Vision tab needs a camera,
   which the Simulator does not have.
2. Press **Cmd+R** to build and run

The app will:
1. Show a welcome screen with "Initialize SDK" button
2. After init, show a three-tab demo:
   - **Speech** — Model ID input (default: `kokoro-82m`), voice picker, text input,
     "Run Inference", and audio playback of generated speech.
   - **Vision** — a vision-model ID input (default: `lfm2-vl-450m`, editable — use
     any registry-resolvable VLM with an mmproj artifact), a question field,
     "Start live", and a viewfinder with a replace-in-place caption + history.
     Requires camera access (the app declares `NSCameraUsageDescription`).
   - **Extract** — LFM2.5-230M turns pasted text into schema-valid JSON, with a
     toggle that turns the grammar constraint off.

## Project Structure

```
ios/
├── XybridExample.xcodeproj/     # Xcode project
├── XybridExample/
│   ├── XybridExampleApp.swift   # App entry point
│   ├── ContentView.swift        # SDK init + TabView; the Speech (TTS) demo
│   ├── LiveVision.swift         # Vision tab: camera service + luma gate + batch VLM
│   ├── ExtractionView.swift     # Extract tab: JSON Schema → GBNF constrained decoding
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

// Reads the key from the scheme environment (see "Set your API key" above).
// Empty/unset → anonymous, local-only init.
let apiKey = ProcessInfo.processInfo.environment["XYBRID_API_KEY"]
    .flatMap { $0.isEmpty ? nil : $0 }
Xybrid.initialize(apiKey: apiKey)

// The key alone exports nothing; start dashboard traces explicitly.
if let apiKey {
    try Xybrid.Observability.xybrid(apiKey: apiKey)
        .service("XybridExample")
        .start()
}
```

### Load Model

```swift
// Describing the model is cheap; `load()` resolves, downloads and initializes
// it off the caller's executor, so the UI stays responsive.
let model = try await Xybrid.model("kokoro-82m").load()
let voices = model.voices()  // [XybridVoiceInfo]
```

### Run Inference

```swift
let envelope = XybridEnvelope.text(text: "Hello!", voiceId: "af", speed: 1.0)
let result = try model.run(envelope: envelope)

if let audio = result.audioBytes {
    // Play audio (wrap in WAV header for AVAudioPlayer)
}
```

### Generation Config (LLM)

```swift
let config = XybridGenerationConfig.make(
    maxTokens: 512,
    temperature: 0.7,
    topP: 0.9
)
let result = try model.run(
    envelope: envelope,
    options: .make(generationConfig: config)
)
```

### Stream Tokens (LLM)

```swift
// Tokens arrive as an AsyncSequence. Cancelling the consuming task, or breaking
// out of the loop, stops generation at the next token.
for try await token in model.streamTokens(
    envelope: XybridEnvelope.text("Tell me a joke"),
    options: .make(generationConfig: config)
) {
    print(token.token, terminator: "")
}
```

### Vision (image → VLM, batch)

```swift
// Encode a camera frame (or any image) to JPEG, then send it as a multimodal
// user turn. The example answers each frame with one batch run — see
// "responsiveness model" above.
let image = try XybridEnvelope.image(jpegData, format: "jpeg")
let envelope = try XybridEnvelope.userMessage("What do you see?", images: [image])
let result = try model.run(
    envelope: envelope,
    options: .make(generationConfig: config)
)
let caption = result.text
```

## Troubleshooting

### "Module 'Xybrid' not found"

The XCFramework hasn't been built or the package dependency hasn't been added.

1. Run `bazel build --config=ios //bindings/apple:XybridFFI` from the xybrid repo root
2. Add the local package dependency in Xcode (see Quick Start step 3)

### Linker errors

Ensure the XCFramework includes headers. Rebuild with `bazel build --config=ios //bindings/apple:XybridFFI`.

### App crashes on launch

Check that:
1. XCFramework is built for the correct architecture (arm64 for device)
2. `initSdkCacheDir()` is called before any model operations

## Related

- [Xybrid Swift SDK](../../bindings/apple/README.md)
- [SDK API Reference](../../docs/sdk/API_REFERENCE.md)
- [Main Examples README](../README.md)
