# Xybrid Kotlin Binding (Android)

> **Status**: Active - Real inference via TemplateExecutor

This directory contains the Android library for Xybrid, providing native Kotlin/Java support via BoltFFI-generated bindings. The SDK supports real ML inference (TTS, ASR, embeddings) on-device via ONNX Runtime.

## Installation

### Maven Central (Recommended)

Add to your `build.gradle.kts`:

```gradle
dependencies {
    implementation("ai.xybrid:xybrid-kotlin:0.5.0")
}
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
├── libs/                                # Native libraries (libxybrid-bolt.so built locally/CI, not committed)
│   ├── armeabi-v7a/
│   │   └── libxybrid-bolt.so
│   ├── arm64-v8a/
│   │   ├── libxybrid-bolt.so
│   │   ├── libonnxruntime.so            # ORT shared library (symlink → vendor/)
│   │   └── libc++_shared.so             # C++ runtime (symlink → vendor/)
│   └── x86_64/
│       ├── libxybrid-bolt.so
│       ├── libonnxruntime.so            # ORT shared library (symlink → vendor/)
│       └── libc++_shared.so             # C++ runtime (symlink → vendor/)
└── src/main/kotlin/ai/xybrid/
    ├── Xybrid.kt                         # Public convenience API
    └── XybridBolt.kt                     # BoltFFI-generated bindings
```

## Native Dependencies

The SDK bundles ONNX Runtime (`libonnxruntime.so`) and the C++ shared library (`libc++_shared.so`) alongside `libxybrid-bolt.so`. These are included automatically in the AAR — no manual setup required.

> **Note:** `libxybrid-bolt.so` is a build output and is **not** committed to the repository. The AAR published to Maven Central includes it (built in CI). For a **local** build, build the Bazel AAR and stage its jniLibs (see [Building Native Libraries](#building-native-libraries) below) so `libs/<abi>/` is populated before running `./gradlew`.

| Library | Purpose | Source |
|---------|---------|--------|
| `libxybrid-bolt.so` | Xybrid Rust SDK via BoltFFI | Built from `crates/xybrid-bolt/` |
| `libonnxruntime.so` | ONNX Runtime inference engine | Vendored at `vendor/ort-android/` |
| `libc++_shared.so` | C++ standard library runtime | Vendored at `vendor/ort-android/` |

The Bazel AAR bundles the ORT libraries into `jni/<abi>/` automatically, so staging its jniLibs populates everything Gradle needs.

## FFI Strategy

The Kotlin bindings are generated from `crates/xybrid-bolt/` using [BoltFFI](https://crates.io/crates/boltffi):
- Single Rust source generates Swift, Kotlin, Java, C#, WASM, and a C header
- Memory-safe wrappers with proper resource cleanup

Regenerate `XybridBolt.kt` with the script, never by hand:

```bash
python3 tools/scripts/gen_kotlin_bolt.py            # regenerate + write
python3 tools/scripts/gen_kotlin_bolt.py --check    # fail on drift
```

It runs `boltffi generate kotlin` and applies the one post-process the output
needs: boltffi 0.29 emits each `XybridError` payload field verbatim, so the
fourteen variants carrying a `message` collide with `Throwable.message` and the
binding does not compile without an `override` modifier. A plain copy of the
generator output silently reintroduces that break.

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

**Manual Build (without cargo-ndk)**

```bash
# Set up linker for each target (API 21+)
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android21-clang"
export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/armv7a-linux-androideabi21-clang"
export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/x86_64-linux-android21-clang"

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
│   ├── libxybrid-bolt.so
│   ├── libonnxruntime.so         # Bundled from vendor/ort-android/
│   └── libc++_shared.so          # Bundled from vendor/ort-android/
├── armeabi-v7a/
│   └── libxybrid-bolt.so
├── x86_64/
│   ├── libxybrid-bolt.so
│   ├── libonnxruntime.so         # Bundled from vendor/ort-android/
│   └── libc++_shared.so          # Bundled from vendor/ort-android/
└── {version}/                    # Versioned copy
    ├── arm64-v8a/
    │   ├── libxybrid-bolt.so
    │   ├── libonnxruntime.so
    │   └── libc++_shared.so
    ├── armeabi-v7a/
    │   └── libxybrid-bolt.so
    └── x86_64/
        ├── libxybrid-bolt.so
        ├── libonnxruntime.so
        └── libc++_shared.so
```

### Troubleshooting

#### "error: linker 'aarch64-linux-android21-clang' not found"

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

**Fix**: Use NDK r26+ and ensure API level 21+ is targeted.

#### ".so file not loading in Android app"

**Cause**: ABI mismatch between built library and device.

**Fix**:
1. Verify you built for the correct ABI (check `adb shell getprop ro.product.cpu.abi`)
2. Ensure the .so file is in the correct `jniLibs/{abi}/` directory

#### "java.lang.UnsatisfiedLinkError: dlopen failed"

**Cause**: Missing native library or corrupted .so file.

**Fix**:
1. Verify the .so file is valid: `file libs/arm64-v8a/libxybrid-bolt.so`
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

| Android API | Version Name |
|-------------|--------------|
| API 24+ | Android 7.0 (Nougat) |

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
