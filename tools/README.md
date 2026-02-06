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

### `build-uniffi` - Build UniFFI Library

Builds the xybrid-uniffi library for Swift/Kotlin FFI via UniFFI.

```bash
cargo xtask build-uniffi
cargo xtask build-uniffi --target aarch64-apple-darwin --release
```

**Options:**
- `--target <triple>` - Target triple (e.g., `aarch64-apple-darwin`, `aarch64-linux-android`)
- `--release` - Build in release mode

### `build-ffi` - Build C ABI Library

Builds the xybrid-ffi library for Unity/C++ integration.

```bash
cargo xtask build-ffi
cargo xtask build-ffi --target x86_64-unknown-linux-gnu --release
```

**Options:**
- `--target <triple>` - Target triple
- `--release` - Build in release mode

**Outputs:**
- Dynamic library: `target/<target>/<profile>/libxybrid_ffi.{dylib,so,dll}`
- Static library: `target/<target>/<profile>/libxybrid_ffi.a`
- C header: `crates/xybrid-ffi/include/xybrid.h`

### `generate-bindings` - Generate Swift/Kotlin Bindings

Generates platform-specific bindings from xybrid-uniffi using uniffi-bindgen.

```bash
cargo xtask generate-bindings
cargo xtask generate-bindings --language swift
cargo xtask generate-bindings --language kotlin --out-dir ./my-bindings
```

**Options:**
- `--language <lang>` - Language to generate: `swift`, `kotlin`, or `all` (default: `all`)
- `--out-dir <path>` - Output directory (default: platform-specific paths in `bindings/`)

**Default output locations:**
- Swift: `bindings/apple/Sources/Xybrid/`
- Kotlin: `bindings/kotlin/src/main/kotlin/ai/xybrid/`

### `build-xcframework` - Build Apple XCFramework (macOS only)

Builds universal XCFramework for iOS and macOS platforms.

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
- Optional: `ORT_IOS_XCFWK_LOCATION` for iOS targets (if not set, builds macOS-only)

**Architectures:**
- iOS arm64 (device)
- iOS Simulator arm64 + x86_64 (universal)
- macOS arm64 + x86_64 (universal)

**Output:** `bindings/apple/XCFrameworks/XybridFFI.xcframework`

### `build-android` - Build Android .so Files

Builds native .so files for Android ABIs.

```bash
cargo xtask build-android --release
cargo xtask build-android --abi arm64-v8a --abi x86_64
cargo xtask build-android --debug --version 1.0.0
```

**Options:**
- `--release` - Build in release mode (default: true)
- `--debug` - Build in debug mode (overrides --release)
- `--abi <abi>` - Build specific ABI(s): `armeabi-v7a`, `arm64-v8a`, `x86_64` (default: all)
- `--version <ver>` - Override version

**Requirements:**
- cargo-ndk (`cargo install cargo-ndk`) or `ANDROID_NDK_HOME` environment variable

**Output:** `bindings/kotlin/libs/<abi>/libxybrid_uniffi.so`

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
| `build-apple.yml` | `cargo xtask build-xcframework --release` | macos-14 |
| `build-android.yml` | `cargo xtask build-android --release` | ubuntu-latest |
| `build-flutter.yml` | `cargo xtask build-flutter --platform <plat>` | matrix (linux, macos, windows) |
| `release.yml` | All build commands + manual packaging | matrix |

## Quick Start Examples

```bash
# First-time setup: install all cross-compilation targets
cargo xtask setup-targets

# Build for Android (requires NDK)
cargo xtask build-android --release

# Build for Apple platforms (macOS only)
cargo xtask build-xcframework --release

# Build Flutter plugin for current platform
cargo xtask build-flutter --platform macos --release

# Build everything possible on current OS
cargo xtask build-all --release

# Package a release
cargo xtask package --version 0.1.0
```

## Shell Scripts

For simpler tasks or CI pipelines that prefer shell scripts, helper scripts are provided in `scripts/`. These are thin wrappers that may call xtask commands internally.

## Related Documentation

- [FEATURE_MATRIX.md](../docs/FEATURE_MATRIX.md) - Feature flags and valid combinations
- [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) - SDK restructuring plan
