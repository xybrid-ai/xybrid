# Xybrid Example Apps

Platform-specific reference implementations demonstrating how to integrate the Xybrid SDK for on-device ML inference.

## Quick Start

| Platform | Directory | Command | Sample |
|----------|-----------|---------|--------|
| [Flutter](#flutter) | `flutter/` | `cd flutter && flutter run` | [README](flutter/README.md) |
| [iOS (Swift)](#ios-swift) | `ios/` | `open ios/XybridExample.xcodeproj` | [README](ios/README.md) |
| [Android (Kotlin)](#android-kotlin) | `android/` | `cd android && ./gradlew assembleDebug` | [README](android/README.md) |
| [Unity](#unity) | `unity/` | Open in Unity Hub | [xybrid-unity-tavern](https://github.com/xybrid-ai/xybrid-unity-tavern) |

---

## Flutter

Cross-platform example app for iOS, Android, and macOS using the Flutter SDK binding.

### Prerequisites

| Tool | Required Version | Installation |
|------|------------------|--------------|
| Flutter | 3.19.0+ | [flutter.dev/docs/get-started/install](https://flutter.dev/docs/get-started/install) |
| Dart | 3.3.0+ | Included with Flutter |
| Xcode | 14.0+ | Mac App Store (for iOS/macOS) |
| Android Studio | Hedgehog+ | [developer.android.com/studio](https://developer.android.com/studio) (for Android) |

### Getting Started

```bash
cd flutter
flutter pub get
flutter run
```

### Features Demonstrated

- **SDK Initialization**: Async `Xybrid.init()` with loading/error states
- **Model Loading**: `XybridModelLoader.fromRegistry()` with progress indication
- **Text-to-Speech (TTS)**: `XybridEnvelope.text()` inference with audio playback
- **Speech-to-Text (ASR)**: Microphone recording with `XybridEnvelope.audio()`
- **Pipelines**: Multi-stage workflows via `XybridPipeline.fromYaml()`
- **Error Handling**: `XybridException` patterns with retry logic

See [flutter/README.md](flutter/README.md) for detailed documentation.

---

## iOS (Swift)

Native iOS example using SwiftUI and the Xybrid Swift SDK.

### Prerequisites

| Tool | Required Version | Installation |
|------|------------------|--------------|
| Xcode | 14.0+ | Mac App Store |
| iOS Deployment Target | 15.0+ | - |
| Rust | 1.70+ | [rustup.rs](https://rustup.rs) (for XCFramework build) |

### Getting Started

```bash
# Open in Xcode
open ios/XybridExample.xcodeproj

# Or build from command line
cd ios
xcodebuild -project XybridExample.xcodeproj \
  -scheme XybridExample \
  -destination 'generic/platform=iOS Simulator' \
  CODE_SIGNING_ALLOWED=NO
```

**Note**: To enable full SDK functionality, build the XCFramework first:

```bash
# From xybrid repo root
cargo xtask build-xcframework
```

### Features Demonstrated

- **SDK Initialization**: Async startup with Swift concurrency
- **Model Loading**: Text field for model ID, load button with progress
- **TTS Inference**: Text input, inference execution, audio playback
- **Result Display**: Status, latency metrics, Play Audio button
- **Error Handling**: User-friendly error messages with retry

See [ios/README.md](ios/README.md) for detailed documentation.

---

## Android (Kotlin)

Native Android example using Jetpack Compose and the Xybrid Kotlin SDK.

### Prerequisites

| Tool | Required Version | Installation |
|------|------------------|--------------|
| Android Studio | Hedgehog (2023.1.1)+ | [developer.android.com/studio](https://developer.android.com/studio) |
| JDK | 11+ | Included with Android Studio |
| Android SDK | API 28+ | Via SDK Manager |
| Rust | 1.70+ | [rustup.rs](https://rustup.rs) (for native lib build) |

### Getting Started

```bash
cd android

# Set Android SDK location (if not using Android Studio)
export ANDROID_HOME="$HOME/Library/Android/sdk"  # macOS
# export ANDROID_HOME="$HOME/Android/Sdk"        # Linux

# Build
./gradlew assembleDebug

# Install to connected device
adb install app/build/outputs/apk/debug/app-debug.apk
```

**Note**: To enable full SDK functionality, build native libraries first:

```bash
# From xybrid repo root
cargo xtask build-android
```

### Features Demonstrated

- **SDK Initialization**: Proper startup flow with loading states
- **Model Loading**: Registry-based model download and caching
- **TTS Inference**: Text input, inference execution with result display
- **Coroutines**: Kotlin coroutine usage for async operations
- **Material 3**: Modern Jetpack Compose UI

See [android/README.md](android/README.md) for detailed documentation.

---

## Unity

Unity example project for on-device ML inference in games.

### Prerequisites

| Tool | Required Version | Installation |
|------|------------------|--------------|
| Unity | 2022.3 LTS | [unity.com/download](https://unity.com/download) |
| Xcode | 14.0+ | Mac App Store (for iOS builds) |
| Android Studio | Hedgehog+ | [developer.android.com/studio](https://developer.android.com/studio) (for Android builds) |
| Rust | 1.70+ | [rustup.rs](https://rustup.rs) (for native lib build) |

### Getting Started

1. Open Unity Hub
2. Click "Open" and select the `examples/unity` directory
3. Wait for Unity to import the project
4. Open `Assets/Scenes/SampleScene`
5. Press Play to test in the editor

**Note**: To enable full SDK functionality, build native libraries first:

```bash
# From xybrid repo root
# For iOS/macOS
cargo xtask build-xcframework

# For Android
cargo xtask build-android
```

Then copy the built libraries to `Assets/Plugins/` (see [unity/README.md](unity/README.md) for paths).

### Features Demonstrated

- **SDK Initialization**: Automatic startup on scene load
- **Model Loading**: Registry-based model download
- **TTS Inference**: UI Canvas with text input and results display
- **Latency Metrics**: Real-time timing measurement
- **Cross-Platform**: Builds for iOS, Android, macOS

See [unity/README.md](unity/README.md) for detailed documentation.

---

## Common Issues

### "Module/Library not found"

Native libraries need to be built from Rust before the SDK works:

| Platform | Build Command |
|----------|---------------|
| iOS/macOS | `cargo xtask build-xcframework` |
| Android | `cargo xtask build-android` |

### "Model not found"

1. Check internet connectivity
2. Verify the model ID exists in the registry: `xybrid models list`
3. For offline usage, download models first: `xybrid models pull <model-id>`

### Build failures

| Issue | Solution |
|-------|----------|
| "SDK location not found" (Android) | Set `ANDROID_HOME` environment variable |
| "Target 'aarch64-apple-ios' not installed" | Run `rustup target add aarch64-apple-ios` |
| "minSdk is too high" (Android) | Use Android 9.0+ device/emulator (API 28) |
| Xcode signing errors | Configure development team in Xcode |

### Microphone permission denied

Ensure permission declarations are configured:

| Platform | File | Setting |
|----------|------|---------|
| iOS | `Info.plist` | `NSMicrophoneUsageDescription` |
| Android | `AndroidManifest.xml` | `android.permission.RECORD_AUDIO` |
| macOS | `Info.plist` + entitlements | `NSMicrophoneUsageDescription` + `com.apple.security.device.audio-input` |

---

## SDK API Reference

The Xybrid SDK follows a consistent pattern across all platforms:

```
1. Initialize SDK    →  Xybrid.init()
2. Load Model        →  loader.load()
3. Run Inference     →  model.run(envelope)
```

### Core API Documentation

- **[API Reference](../docs/sdk/API_REFERENCE.md)** - Complete SDK API specification
- **[Flutter SDK](../docs/sdk/flutter.md)** - Flutter-specific documentation
- **[Kotlin SDK](../docs/sdk/kotlin.md)** - Kotlin-specific documentation
- **[Swift SDK](../docs/sdk/swift.md)** - Swift-specific documentation (planned)

### Online Documentation

- **[docs.xybrid.dev](https://docs.xybrid.dev)** - Full documentation site
- **[docs.xybrid.dev/sdk](https://docs.xybrid.dev/sdk)** - SDK guides and tutorials

---

## Project Structure

```
examples/
├── README.md           # This file
├── flutter/            # Flutter cross-platform example
│   ├── lib/main.dart   # Demo screens (7 demos)
│   ├── pubspec.yaml    # Dependencies
│   └── README.md       # Flutter-specific docs
├── ios/                # Native iOS (Swift) example
│   ├── XybridExample/  # SwiftUI source
│   └── README.md       # iOS-specific docs
├── android/            # Native Android (Kotlin) example
│   ├── app/            # Jetpack Compose source
│   └── README.md       # Android-specific docs
└── unity/              # Unity game example
    ├── Assets/         # Scenes, scripts, plugins
    └── README.md       # Unity-specific docs
```

---

## Contributing Examples

When adding new examples:

1. **Create the example** in the appropriate platform directory
2. **Include a README.md** with prerequisites, setup, and run instructions
3. **Keep dependencies minimal** and well-documented
4. **Test on actual devices** when possible
5. **Follow existing patterns** and code style
6. **Link to API documentation** where appropriate

---

## License

MIT License - See [LICENSE](../LICENSE) for details.

## Related

- [Main Xybrid Repository](https://github.com/xybrid-ai/xybrid)
- [Xybrid Model Registry](https://huggingface.co/xybrid-ai)
- [Xybrid Documentation](https://docs.xybrid.dev)
