# Tools

Build automation and scripts for the xybrid project.

## Directory Structure

```
tools/
├── scripts/        # Build, release, and codegen helpers (see below)
│   ├── natives-*.sh          # Prebuilt llama.cpp slices: fingerprint, push,
│   │                         #   pull, manifest, anonymous-pull verification
│   ├── gen_*_bolt*.py        # Generate the Kotlin / Python / Unity C# bolt
│   │                         #   bindings (CI byte-compares with --check)
│   ├── version-sync.sh       # Read or set the version across every manifest
│   ├── api-contract-check.sh # Soft-warning public SDK signature check
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

### Apple XCFramework — direct Bazel (no xtask command)

`build-xcframework` (boltffi pack) was removed: the release ships the
[rules_apple](https://github.com/bazelbuild/rules_apple)-built xcframework, so
local dev uses the same target (Mac + Xcode required; device + simulator
slices, min iOS 16):

```bash
bazel build --config=ios //bindings/apple:XybridFFI
# → bazel-bin/bindings/apple/XybridFFI.xcframework.zip
```

The C header and Swift wrapper are committed (`bindings/apple/include/`,
`bindings/apple/Sources/Xybrid/`) — no boltffi CLI needed.

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
`libxybrid_bolt.so` is a clean one-link output (16 KB-aligned,
`libc++_shared` in DT_NEEDED, no patchelf) with the ORT runtime bundled.

**Output:**
- `bindings/kotlin/libs/<abi>/libxybrid_bolt.so` (native library; 16 KB-aligned,
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

### Removed commands (Bazel-era)

`setup-targets`, `build-all`, `build-xcframework`, `build-android`,
`stage-react-native`, and `package` no longer exist:

- Bazel manages its own hermetic toolchains (Rust, NDK, clang), and cargokit
  installs its own rustup targets for the Flutter from-source path — nothing
  needs `setup-targets`.
- Native artifacts build with direct `bazel build` commands (see the sections
  above and `CONTRIBUTING.md`).
- Release artifact assembly (naming, checksums, attestation, draft release)
  lives in `.github/workflows/release-prep.yml`, which replaced `package`.

## CI/CD Integration

The xtask commands are used by GitHub Actions workflows:

| Workflow | Command | Runner |
|----------|---------|--------|
| `build-flutter.yml` | `cargo xtask build-flutter --platform linux` | ubuntu-latest |

The native build workflows (`build-apple.yml`, `build-android.yml`,
`build-react-native.yml`, `release-prep.yml`) invoke Bazel directly.

## Quick Start Examples

```bash
# Build for Android (Bazel brings its own NDK)
bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar

# Build the Apple XCFramework (macOS only)
bazel build --config=ios //bindings/apple:XybridFFI

# Build the CLI (macOS host; see .bazelrc for the other configs)
bazel build --config=macos-metal //crates/xybrid-cli:xybrid

# Build the Flutter plugin natives (contributor from-source path — cargo on purpose)
cargo xtask build-flutter --platform macos --release

# Build the Unity native library for the host platform (editor testing)
cargo xtask build-ffi --deploy-unity
```

## Shell Scripts

For simpler tasks or CI pipelines that prefer shell scripts, helper scripts are provided in `scripts/`. These are thin wrappers that may call xtask commands internally.

## Related Documentation

- [FEATURE_MATRIX.md](../docs/FEATURE_MATRIX.md) - Feature flags and valid combinations
- [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) - SDK restructuring plan
