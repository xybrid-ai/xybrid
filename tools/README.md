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
    └── main.rs
```

## xtask Pattern

The `xtask` crate (located at the **repo root**, not under `tools/`) follows the [cargo-xtask](https://github.com/matklad/cargo-xtask) pattern for build automation. This provides:

- **Cross-platform builds**: Run the same automation on macOS, Linux, and Windows
- **Type-safe scripting**: Use Rust instead of shell scripts for complex logic
- **IDE integration**: Get autocomplete and error checking in your editor
- **Dependency management**: Use crates.io ecosystem for common tasks

### Usage

```bash
# Run an xtask command
cargo xtask <command>

# Current commands:
cargo xtask setup-test-env              # Setup integration test environment
cargo xtask setup-test-env --registry <url>  # Use custom registry URL

# Planned commands (Phase 2):
cargo xtask build-xcframework --release
cargo xtask build-android --targets arm64-v8a
```

### Current Status

The xtask crate currently provides:
- `setup-test-env` - Download models for integration tests

Planned commands include:
- `build-xcframework` - Build XCFramework for iOS/macOS
- `build-android` - Build AAR for Android
- `generate-bindings` - Generate UniFFI bindings for all platforms
- `package` - Create release packages

## Shell Scripts

For simpler build tasks, shell scripts are provided in `scripts/`:

- **build-xcframework.sh** - Placeholder for XCFramework build script
- **build-android.sh** - Placeholder for Android AAR build script

These scripts are intended for CI/CD pipelines or developers who prefer shell scripts.

## Implementation Plan

See [DRAFT-PLATFORM-SDK-RESTRUCTURE.md](../docs/architecture/DRAFT-PLATFORM-SDK-RESTRUCTURE.md) for the full implementation plan including:

- Phase 2: Native bindings via UniFFI
- Build targets and architectures
- CI/CD integration strategy
