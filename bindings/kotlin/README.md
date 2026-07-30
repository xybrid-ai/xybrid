# Xybrid Kotlin Binding (Android)

> **Status**: Active - Real inference via TemplateExecutor

This directory contains the Android library for Xybrid, providing native Kotlin/Java support via BoltFFI-generated bindings. The SDK supports real ML inference (TTS, ASR, embeddings) on-device via ONNX Runtime.

## Installation

### Maven Central (Recommended)

Add to your `build.gradle.kts`:

```gradle
dependencies {
    implementation("ai.xybrid:xybrid-kotlin:0.3.0")
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
import ai.xybrid.XybridModel
import ai.xybrid.Envelope

// Load a model from the registry (the constructor resolves + loads it)
val model = XybridModel("kokoro-82m")

// Run text-to-speech
val envelope = Envelope.text("Hello, world!", voiceId = "af_bella", speed = 1.0)
val result = model.run(envelope)

if (result.success) {
    val audioBytes = result.audioBytes
    // Play audio...
} else {
    println("Error: ${result.error}")
}
```

### Loading a Model from Bundle

```kotlin
import ai.xybrid.XybridModel

// Load from a local bundle path
val model = XybridModel.fromBundle("/path/to/model/bundle")
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
import ai.xybrid.reasoningContent

val result = model.run(Envelope.text("Is 97 a prime number? Reason, then answer."))
result.text?.let { println("Answer: $it") }
result.reasoningContent?.let { println("Reasoning: $it") }
```

### Error Handling

```kotlin
import ai.xybrid.XybridException

try {
    val model = XybridModel("unknown-model")
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
| `XybridModel` | Loaded model ready for inference (construct/factory to load) |
| `Envelope` | Factory for `XybridEnvelope` inputs (`text`, `audio`, `embedding`, `image`, `userMessage`) |
| `XybridEnvelope` | Input data container |
| `XybridResult` | Inference output with success/error and result data |
| `XybridException` | Error types (ModelNotFound, InferenceError, etc.) |

### XybridModel (loading)

| Call | Description |
|--------|-------------|
| `XybridModel(id: String)` | Resolve and load a model from the Xybrid registry |
| `XybridModel.fromBundle(path: String)` | Load a model from a local `.xyb` bundle |
| `XybridModel.fromDirectory(path: String)` | Load a model from an extracted directory |
| `XybridModel.fromHuggingface(repo: String)` | Resolve and load a HuggingFace repo |

Each loads synchronously; use the `…Async` suspend variants
(`XybridModel.fromRegistryAsync(id)`, etc.) to load off the calling thread.

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

## Building Native Libraries

Native `.so` files must be built for each target architecture before the library can be used.

### Prerequisites

| Tool | Required Version | Installation |
|------|------------------|--------------|
| Rust | 1.70+ | [rustup.rs](https://rustup.rs) |
| Android NDK | r26+ (recommended: r26b) | Android Studio or sdkmanager |
| cargo-ndk | Latest | `cargo install cargo-ndk` |

### Installing Android NDK

**Option 1: Android Studio (Recommended)**

1. Open Android Studio
2. Go to **Tools > SDK Manager**
3. Select **SDK Tools** tab
4. Check **NDK (Side by side)** and click Apply
5. Note the installation path (e.g., `$ANDROID_HOME/ndk/26.1.10909125`)

**Option 2: Command Line (sdkmanager)**

```bash
# Install NDK via sdkmanager
sdkmanager --install "ndk;26.1.10909125"

# Find your SDK location
echo $ANDROID_HOME
# Typically: ~/Library/Android/sdk (macOS) or ~/Android/Sdk (Linux)
```

### Environment Variables

Set these environment variables before building:

| Variable | Description | Example |
|----------|-------------|---------|
| `ANDROID_HOME` | Android SDK root directory | `~/Library/Android/sdk` |
| `ANDROID_NDK_HOME` | NDK installation directory | `$ANDROID_HOME/ndk/26.1.10909125` |

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export ANDROID_HOME="$HOME/Library/Android/sdk"  # macOS
# export ANDROID_HOME="$HOME/Android/Sdk"        # Linux
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/26.1.10909125"
export PATH="$PATH:$ANDROID_HOME/cmdline-tools/latest/bin"
```

### Installing Rust Targets

```bash
# From the xybrid repo root
cargo xtask setup-targets

# Or manually:
rustup target add aarch64-linux-android      # arm64-v8a
rustup target add armv7-linux-androideabi    # armeabi-v7a
rustup target add x86_64-linux-android       # x86_64
```

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

**Fix**: Run `cargo xtask setup-targets` or `rustup target add aarch64-linux-android`

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
