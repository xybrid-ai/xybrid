# Xybrid Android Example

This is a native Android example app demonstrating how to integrate the Xybrid SDK using Kotlin and Jetpack Compose.

## Features

- **SDK Initialization**: Shows proper SDK startup flow with loading and error states
- **Model Loading**: Demonstrates loading models from the Xybrid registry
- **TTS Inference**: Shows text-to-speech inference with result display and latency metrics
- **Jetpack Compose UI**: Modern declarative UI with Material 3 design
- **Proper Async Handling**: Uses Kotlin coroutines for async operations

## Prerequisites

| Tool | Required Version | Notes |
|------|------------------|-------|
| Android Studio | Hedgehog (2023.1.1) or later | [Download](https://developer.android.com/studio) |
| JDK | 11+ | Included with Android Studio |
| Android SDK | API 28+ (Android 9.0) | Via SDK Manager |

## Quick Start

### 1. Clone and Open

```bash
# Clone the repository
git clone https://github.com/xybrid-ai/xybrid.git
cd xybrid/examples/android

# Open in Android Studio
open -a "Android Studio" .
```

### 2. Configure SDK Path

When opening in Android Studio, it will automatically configure `local.properties` with your SDK path. For command-line builds, set `ANDROID_HOME`:

```bash
export ANDROID_HOME="$HOME/Library/Android/sdk"  # macOS
# export ANDROID_HOME="$HOME/Android/Sdk"        # Linux
```

### 3. Build and Run

**Android Studio:**
1. Wait for Gradle sync to complete
2. Select a device or emulator (API 28+)
3. Click **Run** (Shift+F10)

**Command Line:**
```bash
./gradlew assembleDebug
# APK will be at: app/build/outputs/apk/debug/app-debug.apk

# Install to connected device
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Enabling Xybrid SDK

The example app includes simulated SDK calls by default. To use the real SDK:

### 1. Build Native Libraries

The Xybrid SDK requires native `.so` libraries for Android. Build them with:

```bash
# From the xybrid repo root
cargo xtask build-android
```

This requires:
- Rust toolchain with Android targets
- Android NDK (r26+)
- `cargo-ndk` tool

See [bindings/kotlin/README.md](../../bindings/kotlin/README.md) for detailed build instructions.

### 2. Enable SDK Dependency

**In `settings.gradle.kts`**, uncomment:
```kotlin
include(":xybrid")
project(":xybrid").projectDir = file("../../bindings/kotlin")
```

**In `app/build.gradle.kts`**, uncomment:
```kotlin
implementation(project(":xybrid"))
```

### 3. Update Application Code

In `XybridExampleApp.kt`, uncomment the SDK imports and replace the simulated calls:

```kotlin
// Replace this:
withContext(Dispatchers.IO) {
    kotlinx.coroutines.delay(500)
}

// With real SDK calls:
import ai.xybrid.XybridModelLoader
import ai.xybrid.Envelope

val loader = XybridModelLoader.fromRegistry(modelId)
val model = loader.load()
val envelope = Envelope.text(inputText)
val result = model.run(envelope)
```

## SDK Usage Patterns

### Model Loading

```kotlin
import ai.xybrid.XybridModelLoader
import ai.xybrid.XybridException

try {
    val loader = XybridModelLoader.fromRegistry("kokoro-82m")
    val model = loader.load()
    // Model ready for inference
} catch (e: XybridException.ModelNotFound) {
    println("Model not found: ${e.modelId}")
} catch (e: XybridException.IoException) {
    println("Network error: ${e.message}")
}
```

### Text-to-Speech (TTS)

```kotlin
import ai.xybrid.Envelope

val envelope = Envelope.text(
    text = "Hello, world!",
    voiceId = "af_bella",  // Optional voice ID
    speed = 1.0            // Optional speed multiplier
)

val result = model.run(envelope)
if (result.success) {
    val audioBytes = result.audioBytes
    val latencyMs = result.latencyMs
    // Play audio with MediaPlayer or ExoPlayer
}
```

### Speech Recognition (ASR)

```kotlin
import ai.xybrid.Envelope

val audioBytes: ByteArray = recordAudio()  // Your audio recording code
val envelope = Envelope.audio(
    bytes = audioBytes,
    sampleRate = 16000u,  // 16kHz for Whisper
    channels = 1u          // Mono audio
)

val result = model.run(envelope)
if (result.success) {
    val transcription = result.text
    println("Transcribed: $transcription")
}
```

### Error Handling with Coroutines

```kotlin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

suspend fun loadModel(modelId: String): XybridModel? {
    return withContext(Dispatchers.IO) {
        try {
            val loader = XybridModelLoader.fromRegistry(modelId)
            loader.load()
        } catch (e: XybridException) {
            withContext(Dispatchers.Main) {
                showError(e.displayMessage)
            }
            null
        }
    }
}
```

## Project Structure

```
android/
├── app/
│   ├── src/main/
│   │   ├── java/ai/xybrid/example/
│   │   │   ├── MainActivity.kt         # Activity entry point
│   │   │   ├── XybridExampleApp.kt      # Main Compose UI
│   │   │   └── ui/theme/                # Material 3 theme
│   │   ├── res/                         # Android resources
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts                 # App module build config
├── build.gradle.kts                     # Root build config
├── settings.gradle.kts                  # Module includes
├── gradle.properties                    # Gradle settings
└── README.md                            # This file
```

## Troubleshooting

### Build Fails: "SDK location not found"

Set `ANDROID_HOME` environment variable or let Android Studio configure it automatically:

```bash
export ANDROID_HOME="$HOME/Library/Android/sdk"
```

### Build Fails: "minSdk is too high"

Ensure your emulator or device runs Android 9.0 (API 28) or later.

### UnsatisfiedLinkError at Runtime

Native libraries aren't built or included. Follow "Enabling Xybrid SDK" steps above to build them.

### Gradle Sync Failed

1. File → Invalidate Caches / Restart
2. Delete `.gradle` and `build` directories
3. Sync Project with Gradle Files

## Requirements

| Requirement | Value |
|-------------|-------|
| Minimum SDK | API 28 (Android 9.0) |
| Target SDK | API 34 (Android 14) |
| Kotlin | 1.8.22 |
| Compose BOM | 2023.06.01 |
| AGP | 7.4.2 |

## API Reference

- **[SDK API Reference](../../docs/sdk/API_REFERENCE.md)** - Complete API specification
- **[Kotlin SDK Docs](../../docs/sdk/kotlin.md)** - Kotlin-specific documentation
- **[Online Documentation](https://docs.xybrid.dev/sdk/kotlin)** - Full guides and tutorials

## Related

- [Main Examples README](../README.md) - All platform examples
- [Xybrid Kotlin SDK](../../bindings/kotlin/README.md)
- [Xybrid Repository](https://github.com/xybrid-ai/xybrid)

## License

MIT License - See [LICENSE](../../LICENSE) for details.
