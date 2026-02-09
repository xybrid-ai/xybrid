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

# =============================================================================
# Documentation
# =============================================================================

# Generate documentation
doc:
    cargo doc --workspace --no-deps

# Generate and open documentation
doc-open:
    cargo doc --workspace --no-deps --open
