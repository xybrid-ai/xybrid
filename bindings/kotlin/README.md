# Xybrid Kotlin Binding (Android)

> **Status**: Active - Real inference via TemplateExecutor

This directory contains the Android library for Xybrid, providing native Kotlin/Java support via BoltFFI-generated bindings. The SDK supports real ML inference (TTS, ASR, embeddings) on-device via ONNX Runtime.

## Installation

### Maven Central (Recommended)

Add to your `build.gradle.kts`:

```gradle
dependencies {
    implementation("ai.xybrid:xybrid-kotlin:0.16.0")
}
```

Releases 0.16.0 and later require Android 10 (API 29): set `minSdk = 29` or
higher in your app's `defaultConfig`, or the build fails at the manifest merge.
Releases before 0.16.0 require API 24.

Downloading models from the registry needs network access, and the SDK
declares no permissions. Add this to your app's `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

### Local Development

Add this module as a project dependency:

```kotlin
// In your settings.gradle.kts
include(":xybrid")
project(":xybrid").projectDir = file("/path/to/xybrid/bindings/kotlin")

// In your app's build.gradle.kts
dependencies {
    implementation(project(":xybrid"))
}
```

## Usage

### Loading a Model from Registry

```kotlin
import ai.xybrid.Envelope
import ai.xybrid.Xybrid
import ai.xybrid.audioBytes
import ai.xybrid.runAsync

// Once, in Application.onCreate(): sets the model cache folder, which
// Android needs before any load.
Xybrid.init(context)

// Describing the source is cheap; load() is the explicit suspend boundary.
val model = Xybrid.model("kokoro-82m").load()

// Run text-to-speech
val envelope = Envelope.text("Hello, world!", voiceId = "af_bella", speed = 1.0)
val result = model.runAsync(envelope)

if (result.success) {
    val audioBytes = result.audioBytes
    // Play audio...
} else {
    println("Error: ${result.error}")
}
```

### Loading a Model from Bundle

```kotlin
import ai.xybrid.ModelSource
import ai.xybrid.Xybrid

// Load from a local bundle path
val model = Xybrid.model(ModelSource.bundle("/path/to/model/bundle")).load()
```

### Speech (Phonon / Kitten TTS 2)

The arm64-v8a library links the zzz speech engine statically, so arm64
devices run Phonon and Kitten TTS 2 on-device (Android 10 or later, the SDK's
floor). The library is chosen by ABI, so arm64-v8a emulator images, such as
those on Apple Silicon Macs, include speech too. The armeabi-v7a and x86_64
libraries, including the one x86_64 emulator images load, are built without
it: there the registry does not serve `phonon-31m-en` and a Kitten bundle
fails to run. Speech therefore needs the app to load the arm64-v8a library, so
keep that ABI in any `abiFilters` or App Bundle split.

Both models return raw PCM16 little-endian audio, 24 kHz mono:

```kotlin
import ai.xybrid.Envelope
import ai.xybrid.ModelSource
import ai.xybrid.Xybrid
import ai.xybrid.audioBytes
import ai.xybrid.runAsync
import java.io.File

// Phonon: phonon-31m-en from the registry; voiceId picks one of its voices.
val phonon = Xybrid.model("phonon-31m-en").load()
val hello = phonon.runAsync(Envelope.text("Hello from Phonon.", voiceId = "easton"))
val pcm = hello.audioBytes

// Kitten TTS 2: a local directory holding the four bundle files beside a
// model_metadata.json with a ZzzEmbed template.
val kitten = Xybrid.model(ModelSource.directory(File(context.filesDir, "kitten-tts-2").path)).load()
val speech = kitten.runAsync(Envelope.text("Hello from Kitten."))
```

[Phonon speech](../../docs/sdk/PHONON.md) lists Phonon's voices and options.
The Kitten bundle's files, checksums and `model_metadata.json` are in the
[Unity Kitten sample](../unity/Samples~/KittenTts/README.md); keep
`accelerate` false on Android and measure `threads` on the target device. To
play audio while it is synthesized, `runStream` + `streamNext` deliver
`XybridStreamEventKind.AUDIO` events whose `XybridTtsAudioChunk` carries the
PCM, sample rate and channel count.

The engine's runtime terms and upstream licenses (fourteen files) ship in the
AAR under `assets/xybrid/zzz/`, for example
`assets/xybrid/zzz/RUNTIME_TERMS.md` and
`assets/xybrid/zzz/LICENSES/Zig-MIT.txt`. Android merges them into your APK,
so every app that ships the SDK carries them; do not exclude them from
packaging. Read them with `context.assets.open("xybrid/zzz/RUNTIME_TERMS.md")`,
for example to show them beside your other open-source notices. The
Apache-2.0 license in the published POM covers xybrid's own sources; the
engine linked into arm64-v8a comes under those runtime terms and licenses
instead.

### Speech Recognition (ASR)

```kotlin
import ai.xybrid.XybridEnvelope

// Create audio envelope
val audioBytes: ByteArray = loadAudioFile()  // Your audio loading code
val envelope = XybridEnvelope.Audio(
    bytes = audioBytes,
    sampleRate = 16000u,  // 16kHz for most ASR models
    channels = 1u         // Mono audio
)

val result = model.run(envelope)
if (result.success) {
    println("Transcription: ${result.text}")
}
```

### Embeddings

```kotlin
import ai.xybrid.XybridEnvelope

// Create embedding envelope
val embedding = XybridEnvelope.Embedding(
    data = listOf(0.1f, 0.2f, 0.3f)  // Input vector
)

val result = model.run(embedding)
if (result.success && result.embedding != null) {
    val outputVector = result.embedding!!
    // Use embedding...
}
```

### Vision-Language Input

```kotlin
import ai.xybrid.Envelope

val imageBytes: ByteArray = loadImageBytes()
val image = Envelope.image(imageBytes, format = "jpeg")
val prompt = Envelope.userMessage(
    text = "Describe this image",
    images = listOf(image)
)

val result = model.run(prompt)
if (result.success) {
    println(result.text)
}
```

### Reasoning (thinking models)

Reasoning models (metadata `reasoning: true`, e.g. `lfm2.5-1.2b-thinking`)
produce a chain-of-thought before their answer. Xybrid keeps it out of the
answer text and surfaces it on `reasoningContent` — `null` for non-thinking
models. Nothing to enable; just read it if you want it.

```kotlin
val result = model.run(Envelope.text("Is 97 a prime number? Reason, then answer."))
result.text?.let { println("Answer: $it") }
result.reasoningContent?.let { println("Reasoning: $it") }
```

### Error Handling

```kotlin
import ai.xybrid.XybridException

try {
    val model = Xybrid.model("unknown-model").load()
} catch (e: XybridException.ModelNotFound) {
    println("Model not found: ${e.id}")
} catch (e: XybridException.LoadError) {
    println("Load error: ${e.message}")
} catch (e: XybridException.InferenceError) {
    println("Inference failed: ${e.message}")
} catch (e: XybridException.IoError) {
    println("I/O error: ${e.message}")
}
```

## API Reference

### Types

| Type | Description |
|------|-------------|
| `ModelSource` | Registry, bundle, directory, or Hugging Face model location |
| `ModelLoader` / `XybridModelLoader` | Cheap model reference; call `load()` to perform I/O |
| `XybridModel` | Loaded model ready for inference |
| `Envelope` | Factory for `XybridEnvelope` inputs (`text`, `audio`, `embedding`, `image`, `userMessage`) |
| `XybridEnvelope` | Input data container |
| `XybridResult` | Inference output with success/error and result data |
| `XybridException` | Error types (ModelNotFound, InferenceError, etc.) |

### ModelLoader

| Call | Description |
|--------|-------------|
| `Xybrid.model(id: String)` | Describe a registry model without performing I/O |
| `Xybrid.model(source: ModelSource)` | Describe an explicitly typed model source |
| `ModelLoader.fromBundle(path: String)` | Describe a local `.xyb` bundle |
| `ModelLoader.fromDirectory(path: String)` | Describe an extracted directory |
| `ModelLoader.fromHuggingFace(repo: String)` | Describe a Hugging Face repository |
| `loader.load()` | Resolve, download if needed, and load without blocking the caller's thread |
| `loader.loadBlocking()` | Explicit synchronous load for an existing worker thread |

Creating a source or loader is always cheap. Only `load()` and
`loadBlocking()` may access the network, disk, or native inference runtime.

### XybridEnvelope

| Variant | Fields |
|---------|--------|
| `Audio` | `bytes: ByteArray`, `sampleRate: UInt`, `channels: UInt` |
| `Text` | `text: String`, `voiceId: String?`, `speed: Double?` |
| `Embedding` | `data: List<Float>` |
| `Image` | `bytes: ByteArray`, `format: String` |
| `UserMessage` | `text: String`, `images: List<XybridEnvelope>` |

### XybridResult

| Field | Type | Description |
|-------|------|-------------|
| `success` | `Boolean` | Whether inference succeeded |
| `error` | `String?` | Error message if failed |
| `outputType` | `String` | Type of output ("text", "audio", "embedding") |
| `text` | `String?` | Text output (for ASR) |
| `audioBytes` | `ByteArray?` | Audio output (for TTS) |
| `embedding` | `List<Float>?` | Embedding output |
| `latencyMs` | `UInt` | Inference latency in milliseconds |

## Directory Structure

```
kotlin/
├── build.gradle.kts                     # Gradle build configuration
├── README.md                            # This file
├── libs/                                # Native libraries (libxybrid_bolt.so built locally/CI, not committed)
│   ├── armeabi-v7a/
│   │   ├── libxybrid_bolt.so
│   │   └── libc++_shared.so             # C++ runtime (from the NDK sysroot)
│   ├── arm64-v8a/
│   │   ├── libxybrid_bolt.so
│   │   ├── libonnxruntime.so            # ORT shared library (from vendor/)
│   │   └── libc++_shared.so             # C++ runtime (from the NDK sysroot)
│   └── x86_64/
│       ├── libxybrid_bolt.so
│       ├── libonnxruntime.so            # ORT shared library (from vendor/)
│       └── libc++_shared.so             # C++ runtime (from the NDK sysroot)
└── src/main/kotlin/ai/xybrid/
    ├── Xybrid.kt                         # Public convenience API
    └── XybridBolt.kt                     # BoltFFI-generated bindings
```

## Native Dependencies

The SDK bundles ONNX Runtime (`libonnxruntime.so`) and the C++ shared library (`libc++_shared.so`) alongside `libxybrid_bolt.so`. These are included automatically in the AAR — no manual setup required.

> **Note:** `libxybrid_bolt.so` is a build output and is **not** committed to the repository. The AAR published to Maven Central includes it (built in CI). For a **local** build, build the Bazel AAR and stage its jniLibs (see [Building Native Libraries](#building-native-libraries) below) so `libs/<abi>/` is populated before running `./gradlew`.

| Library | Purpose | Source |
|---------|---------|--------|
| `libxybrid_bolt.so` | Xybrid Rust SDK via BoltFFI | Built from `crates/xybrid-bolt/` |
| `libonnxruntime.so` | ONNX Runtime inference engine | Vendored at `vendor/ort-android/` |
| `libc++_shared.so` | C++ standard library runtime | The NDK sysroot (`@androidndk`), matching the NDK that links `libxybrid_bolt.so` |

The Bazel AAR bundles `libonnxruntime.so` and `libc++_shared.so` into `jni/<abi>/` automatically, so staging its jniLibs populates everything Gradle needs.

## FFI Strategy

The Kotlin bindings are generated from `crates/xybrid-bolt/` using [BoltFFI](https://crates.io/crates/boltffi):
- Single Rust source generates Swift, Kotlin, Java, C#, WASM, and a C header
- Memory-safe wrappers with proper resource cleanup

Regenerate the Kotlin wrapper and its Bazel JNI inputs with the script, never
by hand:

```bash
python3 tools/scripts/gen_kotlin_bolt.py            # regenerate + write
python3 tools/scripts/gen_kotlin_bolt.py --check    # fail on drift
```

It runs `boltffi generate kotlin`, commits the generated `jni_glue.c` and C
header used by the AAR's Bazel link, and applies the Kotlin post-processes the
output needs. The `--check` form compares all three outputs so JVM declarations
cannot drift away from their native implementations.

## Building Native Libraries

Native `.so` files must be built for each target architecture before the library can be used.

### Prerequisites

The recommended Bazel path below needs **only Bazel** (install via
[bazelisk](https://github.com/bazelbuild/bazelisk)) — it downloads its own
Rust toolchain, Android targets, and NDK. The Android SDK (`ANDROID_HOME`)
is required only for the Gradle steps (assembling / publishing the AAR):

```bash
export ANDROID_HOME="$HOME/Library/Android/sdk"  # macOS
# export ANDROID_HOME="$HOME/Android/Sdk"        # Linux
```

The manual cargo build (below) additionally needs the Android NDK (r26+,
`ANDROID_NDK_HOME`) and the rustup Android targets.

### Building

**Using Bazel (Recommended)**

Builds every ABI (the AAR always ships the full set) and needs no local NDK or
rustup targets — Bazel downloads its own pinned toolchains. From the repo root:

```bash
bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar
rm -rf bindings/kotlin/libs && mkdir -p bindings/kotlin/libs /tmp/aar
unzip -o -q bazel-bin/bindings/kotlin/xybrid-kotlin.aar 'jni/*' -d /tmp/aar
cp -r /tmp/aar/jni/* bindings/kotlin/libs/
```

This local AAR has no speech engine. The released one is built with
`--//:tts_zzz=true` and a privately staged `aarch64-linux-android` engine
slice (see [zzz-sys](../../crates/zzz-sys/README.md#android-sdk)), which adds
the engine to arm64-v8a and its notices under `assets/`.

**Manual Build (without cargo-ndk)**

```bash
# Set up a linker for each target, at the AAR's API levels: arm64-v8a is
# built for Android 10 (API 29), the other ABIs for API 28
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android29-clang"
export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/armv7a-linux-androideabi28-clang"
export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/x86_64-linux-android28-clang"

# Build each target
cargo build -p xybrid-bolt --lib --release --target aarch64-linux-android
cargo build -p xybrid-bolt --lib --release --target armv7-linux-androideabi
cargo build -p xybrid-bolt --lib --release --target x86_64-linux-android
```

### Build Output

After a successful build:

```
bindings/kotlin/libs/
├── arm64-v8a/
│   ├── libxybrid_bolt.so
│   ├── libonnxruntime.so         # Bundled from vendor/ort-android/
│   └── libc++_shared.so          # Bundled from the NDK sysroot
├── armeabi-v7a/
│   ├── libxybrid_bolt.so
│   └── libc++_shared.so          # Bundled from the NDK sysroot
├── x86_64/
│   ├── libxybrid_bolt.so
│   ├── libonnxruntime.so         # Bundled from vendor/ort-android/
│   └── libc++_shared.so          # Bundled from the NDK sysroot
└── {version}/                    # Versioned copy
    ├── arm64-v8a/
    │   ├── libxybrid_bolt.so
    │   ├── libonnxruntime.so
    │   └── libc++_shared.so
    ├── armeabi-v7a/
    │   ├── libxybrid_bolt.so
    │   └── libc++_shared.so
    └── x86_64/
        ├── libxybrid_bolt.so
        ├── libonnxruntime.so
        └── libc++_shared.so
```

### Troubleshooting

#### "error: linker 'aarch64-linux-android29-clang' not found"

**Cause**: NDK not found or `ANDROID_NDK_HOME` not set.

**Fix**:
```bash
# Verify NDK is installed
ls $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/

# Set environment variable
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/26.1.10909125"
```

#### "error: target 'aarch64-linux-android' not installed"

**Cause**: Missing Rust target.

**Fix**: Build via Bazel (see Building above) — it provides its own Rust toolchain and Android targets

#### "error: could not find 'cargo-ndk'"

**Cause**: cargo-ndk not installed.

**Fix**: `cargo install cargo-ndk`

#### "ld: error: undefined symbol" (at link time)

**Cause**: Wrong NDK version or missing system libraries.

**Fix**: Use NDK r26+ and the linkers above: API 29 for arm64-v8a, API 28
for armeabi-v7a and x86_64.

#### ".so file not loading in Android app"

**Cause**: ABI mismatch between built library and device.

**Fix**:
1. Verify you built for the correct ABI (check `adb shell getprop ro.product.cpu.abi`)
2. Ensure the .so file is in the correct `jniLibs/{abi}/` directory

#### "java.lang.UnsatisfiedLinkError: dlopen failed"

**Cause**: Missing native library or corrupted .so file.

**Fix**:
1. Verify the .so file is valid: `file libs/arm64-v8a/libxybrid_bolt.so`
2. Should show: `ELF 64-bit LSB shared object, ARM aarch64`
3. Rebuild the Bazel AAR and restage its jniLibs (see Building Native Libraries above)

### Platform Notes

| Platform | NDK Prebuilt Path |
|----------|-------------------|
| macOS (Intel) | `darwin-x86_64` |
| macOS (Apple Silicon) | `darwin-x86_64` (Rosetta 2) |
| Linux | `linux-x86_64` |
| Windows | `windows-x86_64`

## Supported Android Versions

| Android API | Version Name | SDK releases |
|-------------|--------------|--------------|
| API 29+ | Android 10 | 0.16.0 and later |
| API 24+ | Android 7.0 (Nougat) | before 0.16.0 |

## NDK ABIs

| Architecture | ABI | Device Examples |
|--------------|-----|-----------------|
| ARMv7 | armeabi-v7a | Older Android phones |
| ARM64 | arm64-v8a | Most modern Android phones |
| x86_64 | x86_64 | Android emulator on Intel/AMD |

## Telemetry

The Android binding reports `binding=kotlin` in a small `X-Xybrid-Client` header attached to registry metadata calls. See [docs/telemetry/registry.md](../../docs/telemetry/registry.md) for the exact wire format and the opt-out switch (`XYBRID_TELEMETRY_OPTOUT=1`).

## Full Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the complete restructuring plan.
