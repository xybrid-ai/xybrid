#!/usr/bin/env bash
# Regression test: iOS may build the non-linking `llm-mlx` tier, but
# `llm-mlx-runtime` must remain a compile-time error until upstream MLX ships a
# Metal-enabled iOS slice.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

cd "$REPO_ROOT"

TARGET="aarch64-apple-ios"
OUT="$TMP_ROOT/ios-runtime-gate.out"

if ! rustup target list --installed | grep -qx "$TARGET"; then
    echo "rust target $TARGET is not installed; run: rustup target add $TARGET" >&2
    exit 1
fi

set +e
CARGO_TARGET_DIR="$TMP_ROOT/target" \
    cargo check -p xybrid-core \
    --target "$TARGET" \
    --no-default-features \
    --features llm-mlx-runtime \
    --lib > "$OUT" 2>&1
status=$?
set -e

if [ "$status" -eq 0 ]; then
    echo "expected llm-mlx-runtime to fail for $TARGET, but cargo check passed" >&2
    cat "$OUT" >&2
    exit 1
fi

if ! grep -q "Apple Silicon macOS" "$OUT"; then
    echo "expected compile error to explain the Apple Silicon macOS-only gate" >&2
    cat "$OUT" >&2
    exit 1
fi

if ! grep -q "iOS remains non-linking only" "$OUT"; then
    echo "expected compile error to explain that iOS remains non-linking only" >&2
    cat "$OUT" >&2
    exit 1
fi

echo "iOS llm-mlx-runtime gate is enforced"
