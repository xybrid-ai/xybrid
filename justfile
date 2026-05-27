# Xybrid - Development Tasks
# Run `just --list` to see all available commands

# Default recipe - show help
default:
    @just --list

# =============================================================================
# Build & Test
# =============================================================================

# Build all packages
build:
    cargo build --workspace

# Build in release mode
build-release:
    cargo build --workspace --release

# Run all tests
test:
    cargo test --workspace

# Run tests with output
test-verbose:
    cargo test --workspace -- --nocapture

# Check all packages compile
check:
    cargo check --workspace

# Run clippy lints
lint:
    cargo clippy --workspace -- -D warnings

# Format code
fmt:
    cargo fmt --all

# Format check (don't modify)
fmt-check:
    cargo fmt --all -- --check

# =============================================================================
# Test Models (Integration Tests)
# =============================================================================

mod integration-tests

# Run tests that require models (ignored by default)
test-models:
    cargo test --workspace --ignored

# =============================================================================
# Examples
# =============================================================================

# Run an example (e.g., just example asr_whisper)
example name *args:
    cargo run --example {{name}} -p xybrid-core {{args}}

# Run an example with candle feature
example-candle name *args:
    cargo run --example {{name}} -p xybrid-core --features candle {{args}}

# =============================================================================
# Version Management
# =============================================================================

# Show current version across all packages
version:
    @./tools/scripts/version-sync.sh --check

# Set version across all packages (e.g., just bump-version 0.2.0)
bump-version new_version:
    ./tools/scripts/version-sync.sh {{new_version}}
    @./tools/scripts/version-sync.sh --check

# Sync non-Rust packages to match Cargo workspace version
version-sync:
    ./tools/scripts/version-sync.sh
    @./tools/scripts/version-sync.sh --check

# Consumer-side resolution test for a published release. Generates minimal
# consumer projects in a tmp dir for SPM / Cargo / Flutter / Maven and
# verifies each registry resolves the requested version. Also runs
# xcodebuild against examples/ios/XybridExample (iOS Simulator, no
# codesign) for an end-to-end real-app SPM check.
#
# Examples:
#   just verify-release                    # uses current workspace version
#   just verify-release 0.1.0-rc4
#   just verify-release 0.1.0-rc4 --skip-ios
#   just verify-release 0.1.0-rc4 --only spm,cargo
verify-release *args:
    ./tools/scripts/verify-release.sh {{args}}

# =============================================================================
# Documentation
# =============================================================================

# Generate documentation
doc:
    cargo doc --workspace --no-deps

# Generate and open documentation
doc-open:
    cargo doc --workspace --no-deps --open
