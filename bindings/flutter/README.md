# Xybrid Flutter Binding

> **Status**: Placeholder

This directory contains the Flutter plugin for Xybrid, providing Dart bindings via flutter_rust_bridge (FRB).

## Prerequisites

### Required Tools

| Tool | Required Version | Purpose |
|------|------------------|---------|
| Rust | 1.70+ | Compile native libraries |
| Flutter | 3.10+ | Flutter SDK |
| flutter_rust_bridge_codegen | 2.0+ | Generate Dart bindings |
| LLVM/Clang | 15+ | FFI parsing (used by FRB) |

### Installing Flutter Rust Bridge (FRB)

This project uses FRB v2.x for generating Dart bindings from Rust code.

**Required Version:** `flutter_rust_bridge ^2.0.0`

**Option 1: Using Cargo (Recommended)**

```bash
cargo install flutter_rust_bridge_codegen
```

**Option 2: Using Dart**

```bash
dart pub global activate flutter_rust_bridge
```

**Verify installation:**

```bash
flutter_rust_bridge_codegen --version
# Should print: flutter_rust_bridge_codegen 2.x.x
```

### Installing LLVM/Clang

FRB requires LLVM/Clang for parsing Rust FFI types.

**macOS:**
```bash
brew install llvm

# Add to PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
```

**Ubuntu/Debian:**
```bash
sudo apt install llvm-dev libclang-dev clang
```

**Windows:**
```powershell
# Using Chocolatey
choco install llvm

# Or download from: https://releases.llvm.org/
```

**Verify installation:**
```bash
clang --version
# Should print: clang version 15.x.x or higher
```

### Installing Rust Targets

For cross-platform builds, install the required targets:

```bash
# From the xybrid repo root
cargo xtask setup-targets

# Or manually for specific platforms:
rustup target add aarch64-apple-ios aarch64-apple-ios-sim  # iOS
rustup target add aarch64-apple-darwin x86_64-apple-darwin # macOS
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android  # Android
rustup target add x86_64-pc-windows-msvc     # Windows
rustup target add x86_64-unknown-linux-gnu   # Linux
```

### Platform-Specific Requirements

| Platform | Additional Requirements |
|----------|-------------------------|
| iOS/macOS | Xcode 14+, Apple targets installed |
| Android | Android NDK r26+, `ANDROID_NDK_HOME` set, cargo-ndk |
| Windows | Visual Studio Build Tools 2019+ |
| Linux | `build-essential`, `pkg-config` |

## Building

Use the xtask command to build Flutter native libraries:

```bash
# Build for your current platform
cargo xtask build-flutter --platform macos

# Build for specific platforms
cargo xtask build-flutter --platform ios
cargo xtask build-flutter --platform android
cargo xtask build-flutter --platform linux
cargo xtask build-flutter --platform windows

# Debug build (unoptimized, with symbols)
cargo xtask build-flutter --platform macos --debug
```

This will:
1. Run `flutter_rust_bridge_codegen generate` to regenerate Dart bindings
2. Build the native Rust library for the specified platform
3. Output libraries to `rust/target/{target}/{profile}/`

### Build Output

| Platform | Output Path | Library Name |
|----------|-------------|--------------|
| iOS | `rust/target/aarch64-apple-ios/release/` | `libxybrid_flutter_ffi.a` |
| macOS | `rust/target/aarch64-apple-darwin/release/` | `libxybrid_flutter_ffi.a` |
| Android | `rust/target/{abi}/release/` | `libxybrid_flutter_ffi.so` |
| Linux | `rust/target/x86_64-unknown-linux-gnu/release/` | `libxybrid_flutter_ffi.so` |
| Windows | `rust/target/x86_64-pc-windows-msvc/release/` | `xybrid_flutter_ffi.dll` |

### FRB Codegen

The FRB codegen step runs automatically during `cargo xtask build-flutter`. To run it manually:

```bash
# From the bindings/flutter directory
cd bindings/flutter
flutter_rust_bridge_codegen generate
```

**Generated Files:**

```
lib/src/rust/              # All files in this directory are generated
├── frb_generated.dart     # Main generated bindings
├── frb_generated.io.dart  # Platform-specific (mobile)
└── frb_generated.web.dart # Web platform
```

These files are gitignored and regenerated on every build.

### Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `ANDROID_NDK_HOME` | Android NDK path | Android builds |
| `ANDROID_HOME` | Android SDK path | Android builds |
| `PATH` | Include LLVM/Clang | All platforms |
| `LIBCLANG_PATH` | LLVM lib path | macOS (if Homebrew LLVM) |

## Configuration

FRB configuration is defined in `flutter_rust_bridge.yaml`:

- **Input**: `rust/src/api/` - Rust types with `#[frb]` attributes
- **Output**: `lib/src/rust/` - Generated Dart bindings (gitignored)

## Project Structure

```
flutter/
├── pubspec.yaml              # Flutter plugin manifest
├── flutter_rust_bridge.yaml  # FRB configuration
├── lib/                      # Dart API wrappers
│   ├── xybrid.dart           # Main entry point
│   └── src/
│       ├── envelope.dart     # Envelope types
│       ├── model_loader.dart # Model loading API
│       ├── result.dart       # Result types
│       └── rust/             # Generated by FRB (gitignored)
├── rust/                     # FRB bridge (Rust code)
│   ├── Cargo.toml            # depends on xybrid-sdk
│   └── src/api/              # Thin #[frb] wrappers
├── ios/
├── android/
├── macos/
├── windows/
└── linux/
```

## Development Workflow

1. Modify Rust types in `rust/src/api/`
2. Run `cargo xtask build-flutter --platform <your-platform>`
3. FRB regenerates Dart bindings automatically
4. Update manual Dart wrappers in `lib/src/` if needed

## Troubleshooting

### "flutter_rust_bridge_codegen: command not found"

**Cause**: FRB codegen not installed or not in PATH.

**Fix**:
```bash
# Install with Cargo
cargo install flutter_rust_bridge_codegen

# Or add Dart pub cache to PATH
export PATH="$PATH:$HOME/.pub-cache/bin"
```

### "fatal error: 'clang-c/Index.h' file not found"

**Cause**: LLVM/Clang not installed or not found.

**Fix (macOS)**:
```bash
brew install llvm
export LIBCLANG_PATH="/opt/homebrew/opt/llvm/lib"
```

**Fix (Linux)**:
```bash
sudo apt install llvm-dev libclang-dev clang
```

### "error: failed to run custom build command for 'flutter_rust_bridge_codegen'"

**Cause**: Missing build dependencies.

**Fix**: Install LLVM development files:
```bash
# macOS
brew install llvm

# Ubuntu/Debian
sudo apt install llvm-dev libclang-dev
```

### "dart analyze" fails with type errors

**Cause**: Generated Dart code out of sync with Rust changes.

**Fix**: Regenerate bindings:
```bash
cd bindings/flutter
flutter_rust_bridge_codegen generate
flutter pub get
```

### "flutter pub get" fails with dependency conflicts

**Cause**: Mismatched flutter_rust_bridge versions.

**Fix**: Ensure pubspec.yaml uses the same major version as the codegen tool:
```yaml
dependencies:
  flutter_rust_bridge: ^2.0.0  # Match codegen version
```

### Android build fails with "linker not found"

**Cause**: Android NDK not configured.

**Fix**: See [Kotlin README](../kotlin/README.md#installing-android-ndk) for NDK setup.

### iOS build fails with "target not installed"

**Cause**: Missing iOS Rust targets.

**Fix**:
```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
```

### Windows build fails with "link.exe not found"

**Cause**: Visual Studio Build Tools not installed.

**Fix**: Install Visual Studio Build Tools 2019+ with "Desktop development with C++" workload.

### "FRB codegen warning: skipping..." messages

**Cause**: Some Rust types not compatible with FRB or missing `#[frb]` attributes.

**Fix**: Check the skipped types and add appropriate annotations or exclude them from the API.

## Current Location

The Flutter SDK currently lives in a separate repository:
- **Repository**: [xybrid-flutter](https://github.com/xybrid-ai/xybrid-flutter)

## Migration Plan

As part of the monorepo restructure, the Flutter binding will be migrated here with:
1. Rust business logic moved to `crates/xybrid-sdk/`
2. Flutter-specific code kept thin (FRB wrappers only)
3. Platform-specific code in respective directories

## Full Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the complete restructuring plan.
