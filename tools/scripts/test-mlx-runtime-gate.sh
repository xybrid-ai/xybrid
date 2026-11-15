#!/usr/bin/env bash
# Regression test: `llm-mlx-runtime` must stay a compile-time error on every
# target but Apple Silicon macOS. The `llm-mlx` tier builds anywhere; linking
# MLX elsewhere would only fail later and less clearly.
#
# Usage:
#   tools/scripts/test-mlx-runtime-gate.sh [target]
#
# `target` is a Rust target triple. It defaults to the host, so ci.yml's Linux
# job proves the gate without a cross toolchain; on an Apple Silicon Mac,
# where the host is the one supported target, it defaults to aarch64-apple-ios.
# The guard is one `cfg`, so any unsupported target exercises it.
#
# The check builds into a temporary directory, so a local run leaves the
# workspace's `target/` alone. XYBRID_MLX_GATE_TARGET_DIR names another one
# (a path relative to the repository root is fine): ci.yml points it at the
# job's `target`, where the dependencies are already built.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

cd "$REPO_ROOT"

SUPPORTED="aarch64-apple-darwin"
HOST=$(rustc -vV | sed -n 's/^host: //p')
TARGET="${1:-$HOST}"
if [ "$TARGET" = "$SUPPORTED" ]; then
    if [ "$#" -gt 0 ]; then
        echo "$SUPPORTED is the one target llm-mlx-runtime supports; name another" >&2
        exit 2
    fi
    TARGET="aarch64-apple-ios"
fi

TARGET_ARGS=()
if [ "$TARGET" != "$HOST" ]; then
    if ! rustup target list --installed | grep -qx "$TARGET"; then
        echo "rust target $TARGET is not installed; run: rustup target add $TARGET" >&2
        exit 1
    fi
    TARGET_ARGS=(--target "$TARGET")
fi

OUT="$TMP_ROOT/runtime-gate.out"

set +e
CARGO_TARGET_DIR="${XYBRID_MLX_GATE_TARGET_DIR:-$TMP_ROOT/target}" \
    cargo check -p xybrid-core \
    ${TARGET_ARGS[@]+"${TARGET_ARGS[@]}"} \
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
    echo "expected the compile error to name the Apple Silicon macOS-only gate" >&2
    cat "$OUT" >&2
    exit 1
fi

if ! grep -q "iOS remains non-linking only" "$OUT"; then
    echo "expected the compile error to say that iOS remains non-linking only" >&2
    cat "$OUT" >&2
    exit 1
fi

echo "llm-mlx-runtime is refused for $TARGET"
