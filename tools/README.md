# Tools

Build automation and scripts for the xybrid project.

## Directory Structure

```
tools/
├── scripts/        # Shell scripts for platform builds
│   ├── build-xcframework.sh  # Build XCFramework for Apple platforms
│   └── build-android.sh      # Build AAR for Android
└── README.md       # This file

xtask/              # Cargo xtask for build automation (at repo root)
├── Cargo.toml
└── src/
    ├── main.rs
    └── setup_env.rs
```

## xtask Pattern

The `xtask` crate (located at the **repo root**, not under `tools/`) follows the [cargo-xtask](https://github.com/matklad/cargo-xtask) pattern for build automation. This provides:

- **Cross-platform builds**: Run the same automation on macOS, Linux, and Windows
- **Type-safe scripting**: Use Rust instead of shell scripts for complex logic
- **IDE integration**: Get autocomplete and error checking in your editor
- **Dependency management**: Use crates.io ecosystem for common tasks

## xtask vs build.rs

The build system has two layers:

| Layer | Role | Examples |
|-------|------|----------|
| **xtask** | Orchestration | Multi-target builds, packaging, NDK detection for cargo-ndk |
| **build.rs** | Compilation | llama.cpp CMake, native linking, compile-time feature detection |

**xtask** handles high-level orchestration: selecting targets, invoking cargo with the right flags, creating universal binaries with `lipo`, packaging artifacts.

**build.rs** (in xybrid-core) handles native dependency compilation: building llama.cpp via CMake, setting up linker flags, detecting NDK paths for CMake toolchain.

Note: NDK detection happens in both places intentionally - xtask detects it for cargo-ndk, build.rs detects it for CMake toolchain configuration.

## Available Commands

### `setup-test-env` - Setup Integration Test Environment

Downloads models and sets up fixtures for integration tests.

```bash
cargo xtask setup-test-env
cargo xtask setup-test-env --registry <custom-registry-url>
```

**Options:**
- `--registry <url>` - Custom registry URL for model downloads (default: api.xybrid.dev)

### `build-ffi` - Build the Unity native library

Builds the `xybrid-bolt` native library (the BoltFFI native the Unity SDK
loads). Pass `--deploy-unity` to copy it into the Unity plugins tree.

```bash
cargo xtask build-ffi
cargo xtask build-ffi --target x86_64-unknown-linux-gnu --release --deploy-unity
```

**Options:**
- `--target <triple>` - Target triple
- `--release` - Build in release mode
- `--deploy-unity` - Copy the built library into `bindings/unity/Runtime/Plugins/<Platform>/`

**Outputs:**
- Dynamic library: `target/<target>/<profile>/libxybrid_bolt.{dylib,so,dll}`
- Static library: `target/<target>/<profile>/libxybrid_bolt.a`

### `build-xcframework` - Build Apple XCFramework (macOS only)

Builds the Apple XCFramework via `boltffi pack apple` — compiles
`xybrid-bolt` for every Apple slice in `crates/xybrid-bolt/boltffi.toml`,
generates the Swift wrapper (`bindings/apple/Sources/Xybrid/xybrid_bolt.swift`),
and packs the slices into an xcframework.

```bash
cargo xtask build-xcframework --release
cargo xtask build-xcframework --debug --version 1.0.0
```

**Options:**
- `--release` - Build in release mode (default: true)
- `--debug` - Build in debug mode (overrides --release)
- `--version <ver>` - Override version (default: from Cargo.toml or git tag)

**Requirements:**
- macOS host
- `boltffi` CLI (`cargo install boltffi_cli`)
- iOS ORT is fetched at build time via `xybrid-core/ort-download` (the
  `platform-ios` feature) — no manual ORT vendoring needed.

**Slices** (driven by `boltffi.toml`): iOS arm64 (device) + iOS
Simulator arm64. macOS is excluded by config.

**Output:**
- `bindings/apple/XCFrameworks/XybridFFI.xcframework` (unversioned)
- `bindings/apple/XCFrameworks/XybridFFI-<version>.xcframework` (versioned)
- `bindings/apple/Sources/Xybrid/xybrid_bolt.swift` (generated Swift wrapper)

### Android .so files — direct Bazel (no xtask command)

`build-android` was removed: it was a thin proxy over Bazel. Build the
feature-complete 3-ABI AAR (text + candle voice + mtmd vision) directly and,
if you need loose `.so` files for Gradle, stage its jniLibs the way CI does
(`release-prep.yml`, `test-ci.yml`, `build-react-native.yml`):

```bash
bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar
rm -rf bindings/kotlin/libs && mkdir -p bindings/kotlin/libs /tmp/aar
unzip -o -q bazel-bin/bindings/kotlin/xybrid-kotlin.aar 'jni/*' -d /tmp/aar
cp -r /tmp/aar/jni/* bindings/kotlin/libs/
```

The NDK is a pinned Bazel download — no machine setup. Each
`libxybrid-bolt.so` is a clean one-link output (16 KB-aligned,
`libc++_shared` in DT_NEEDED, no patchelf) with the ORT runtime bundled.

**Requirements:**
- Android NDK r27 (`ANDROID_NDK_HOME`, or installed under `$ANDROID_HOME/ndk/`)

**Output:**
- `bindings/kotlin/libs/<abi>/libxybrid-bolt.so` (native library; 16 KB-aligned,
  `libc++_shared` linked in — a clean linker output that survives a consumer's
  AGP strip, no patchelf)
- `bindings/kotlin/libs/<abi>/{libonnxruntime.so,libc++_shared.so}` (bundled runtime)

### `build-flutter` - Build Flutter Native Libraries

Builds native libraries for Flutter plugin on a specific platform.

```bash
cargo xtask build-flutter --platform macos --release
cargo xtask build-flutter --platform android
cargo xtask build-flutter --platform linux --debug
```

**Options:**
- `--platform <plat>` - Target platform: `ios`, `android`, `macos`, `windows`, `linux` (required)
- `--release` - Build in release mode (default: true)
- `--debug` - Build in debug mode (overrides --release)
- `--version <ver>` - Override version

**Platform requirements:**
- iOS/macOS: macOS host
- Windows: Windows host
- Linux: Linux host
- Android: Any host with NDK

**Note:** Runs `flutter_rust_bridge_codegen` to generate Dart bindings before building.

### `setup-targets` - Install Cross-Compilation Targets

Installs all required Rust targets for cross-compilation.

```bash
cargo xtask setup-targets
```

**Targets installed:**
- iOS: `aarch64-apple-ios`, `x86_64-apple-ios`, `aarch64-apple-ios-sim`
- macOS: `aarch64-apple-darwin`, `x86_64-apple-darwin`
- Android: `aarch64-linux-android`, `armv7-linux-androideabi`, `x86_64-linux-android`

### `build-all` - Build All Platforms

Builds all platforms with one command. Skips platforms that can't be built on the current OS.

```bash
cargo xtask build-all --release
cargo xtask build-all --parallel --version 1.0.0
```

**Options:**
- `--release` - Build in release mode (default: true)
- `--debug` - Build in debug mode
- `--parallel` - Run builds concurrently (experimental)
- `--version <ver>` - Override version

### `package` - Package Distribution Artifacts

Creates distribution packages with checksums and manifest.

```bash
cargo xtask package --version 1.0.0
cargo xtask package --output-dir ./release --skip-flutter
```

**Options:**
- `--version <ver>` - Package version (default: from Cargo.toml or git tag)
- `--output-dir <path>` - Output directory (default: `dist/`)
- `--skip-apple` - Skip XCFramework packaging
- `--skip-android` - Skip Android .so packaging
- `--skip-flutter` - Skip Flutter plugin packaging

**Outputs:**
- `XybridFFI-<version>.xcframework.zip` - Apple XCFramework
- `xybrid-android-<version>.zip` - Android .so files
- `xybrid-flutter-<version>.tar.gz` - Flutter plugin
- `checksums.sha256` - SHA256 checksums
- `manifest.json` - Package manifest with metadata

## CI/CD Integration

The xtask commands are used by GitHub Actions workflows:

| Workflow | Command | Runner |
|----------|---------|--------|
| `build-flutter.yml` | `cargo xtask build-flutter --platform linux` | ubuntu-latest |
| `test-ci.yml` (apple) | `cargo xtask build-xcframework --release` | macos-14 |

The native build workflows (`build-apple.yml`, `build-android.yml`,
`build-react-native.yml`, `release-prep.yml`) invoke Bazel directly.

## Quick Start Examples

```bash
# First-time setup: install all cross-compilation targets
cargo xtask setup-targets

# Build for Android (Bazel brings its own NDK)
bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar

# Build for Apple platforms (macOS only)
cargo xtask build-xcframework --release

# Build Flutter plugin for current platform
cargo xtask build-flutter --platform macos --release

# Build everything possible on current OS
cargo xtask build-all --release

# Package a release
cargo xtask package --version 0.2.0
```

## Shell Scripts

For simpler tasks or CI pipelines that prefer shell scripts, helper scripts are provided in `scripts/`. These are thin wrappers that may call xtask commands internally.

## Related Documentation

- [FEATURE_MATRIX.md](../docs/FEATURE_MATRIX.md) - Feature flags and valid combinations
- [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) - SDK restructuring plan
