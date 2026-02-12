# Xybrid Android Example

This is a native Android example app demonstrating how to integrate the Xybrid SDK using Kotlin and Jetpack Compose. It loads a model from a local bundle directory and runs real TTS inference via `TemplateExecutor`.

## Features

- **SDK Initialization**: Shows proper SDK startup flow with loading and error states
- **Model Loading**: Loads models from a local bundle path using `XybridModelLoader.fromBundle()`
- **TTS Inference**: Runs text-to-speech inference with result display and latency metrics from `XybridResult.latencyMs`
- **Error Handling**: Displays `XybridException` messages (ModelNotFound, InferenceFailed, InvalidInput, IoException)
- **Jetpack Compose UI**: Modern declarative UI with Material 3 design
- **Proper Async Handling**: Uses Kotlin coroutines for all SDK operations on `Dispatchers.IO`

## Prerequisites

| Tool | Required Version | Notes |
|------|------------------|-------|
| Android Studio | Hedgehog (2023.1.1) or later | [Download](https://developer.android.com/studio) |
| JDK | 11+ | Included with Android Studio |
| Android SDK | API 28+ (Android 9.0) | Via SDK Manager |
| Rust toolchain | With Android targets | For building native libraries |
| Android NDK | r26+ | For cross-compilation |

## Quick Start

### 1. Build Native Libraries

The Xybrid SDK requires native `.so` libraries for Android:

```bash
# From the xybrid repo root
cargo xtask build-android
```

This builds `libxybrid_uniffi.so` and bundles `libonnxruntime.so` + `libc++_shared.so` from `vendor/ort-android/` for each ABI (arm64-v8a, x86_64).

See [bindings/kotlin/README.md](../../bindings/kotlin/README.md) for detailed build instructions.

### 2. Open and Build

```bash
cd xybrid/examples/android
open -a "Android Studio" .
```

Or build from the command line:

```bash
./gradlew assembleDebug
```

### 3. Obtain Model Files

The app loads models from the device filesystem. You need to place model files on the device before running inference.

**Option A: Push from fixtures (recommended for development)**

If you have model files in the repo's `fixtures/models/` directory:

```bash
# Push a model directory to the device/emulator
adb push ../../integration-tests/fixtures/models/kokoro-82m /data/local/tmp/models/kokoro-82m
```

Then update the model path in the app to `/data/local/tmp/models/kokoro-82m`.

**Option B: Push to app-private storage**

```bash
# Get the app's files directory path (after installing the app)
adb shell run-as ai.xybrid.example mkdir -p files/models/kokoro-82m

# Push model files
adb push model_metadata.json /sdcard/tmp_model/
adb push model.onnx /sdcard/tmp_model/
adb push tokens.txt /sdcard/tmp_model/
adb push voices.bin /sdcard/tmp_model/

# Copy into app storage
adb shell run-as ai.xybrid.example cp /sdcard/tmp_model/* files/models/kokoro-82m/
```

The app defaults to `<app-files-dir>/models/kokoro-82m` — you can edit the path in the UI.

**Required model files** (for kokoro-82m TTS):
- `model_metadata.json` — Execution configuration
- `model.onnx` — ONNX model file
- `tokens.txt` — Phoneme token vocabulary
- `voices.bin` — Voice embeddings

### 4. Run the App

1. Select a device or emulator (API 28+)
2. Click **Run** (Shift+F10)
3. Tap **Initialize SDK**
4. Enter the model bundle path (or use the default)
5. Tap **Load Model**
6. Enter text and tap **Run Inference**

## SDK Usage Patterns

### Model Loading (from bundle)

```kotlin
import ai.xybrid.XybridModelLoader
import ai.xybrid.XybridException
import ai.xybrid.displayMessage

try {
    val loader = XybridModelLoader.fromBundle("/path/to/model/directory")
    val model = loader.load()
    // Model ready for inference
} catch (e: XybridException.ModelNotFound) {
    println("Model not found: ${e.displayMessage}")
} catch (e: XybridException.IoException) {
    println("I/O error: ${e.displayMessage}")
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
    val audioBytes = result.audioBytes   // PCM audio data
    val latencyMs = result.latencyMs     // Inference latency
}
```

### Error Handling with Coroutines

```kotlin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import ai.xybrid.displayMessage

suspend fun loadAndRun(path: String, text: String) {
    withContext(Dispatchers.IO) {
        try {
            val loader = XybridModelLoader.fromBundle(path)
            val model = loader.load()
            val result = model.run(Envelope.text(text))
            // Handle result...
        } catch (e: XybridException) {
            println("Error: ${e.displayMessage}")
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
│   │   │   ├── XybridExampleApp.kt      # Main Compose UI with real SDK calls
│   │   │   └── ui/theme/                # Material 3 theme
│   │   ├── res/                         # Android resources
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts                 # App module build config
├── build.gradle.kts                     # Root build config
├── settings.gradle.kts                  # Module includes (Xybrid SDK enabled)
├── gradle.properties                    # Gradle settings
└── README.md                            # This file
```

## Troubleshooting

### Build Fails: "SDK location not found"

Set `ANDROID_HOME` environment variable or let Android Studio configure it automatically:

```bash
export ANDROID_HOME="$HOME/Library/Android/sdk"
```

### UnsatisfiedLinkError at Runtime

Native libraries aren't built or included. Build them with:

```bash
cargo xtask build-android
```

### Model Not Found Error

Ensure the model directory exists on the device and contains `model_metadata.json`. Check with:

```bash
adb shell ls -la /data/local/tmp/models/kokoro-82m/
```

### Gradle Sync Failed

1. File > Invalidate Caches / Restart
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
