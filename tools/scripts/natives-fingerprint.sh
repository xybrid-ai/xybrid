#!/usr/bin/env bash
# natives-fingerprint.sh
#
# Computes the content fingerprint that identifies one prebuilt llama.cpp
# native slice (a per-target set of static archives + headers). The SAME
# script is run by the publisher (to TAG the upload) and by the consumer
# pre-step (to PULL by tag) so the two can never drift — a single source of
# truth. See .context/natives-prebuilt-plan.md.
#
# The fingerprint MUST change whenever the compiled bytes would change, or a
# consumer could link a stale/ABI-mismatched archive under a matching tag.
# So it folds in, besides the source identity and our own files, the
# toolchain versions that drive codegen (NDK revision on Android, cc/cmake
# elsewhere). Pilot scope: Android + a generic host path; Apple SDK / Windows
# CRT dimensions are deferred (see the plan doc).
#
# Usage: natives-fingerprint.sh <target-triple> <feature-set>
#   <feature-set>: "base" (shipped llm-llamacpp) | "vision" (+ mtmd)
# Prints the hex fingerprint to stdout.
set -euo pipefail

TARGET="${1:?usage: natives-fingerprint.sh <target-triple> <feature-set>}"
FEATURES="${2:-base}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYS="$ROOT/crates/llama-cpp-sys"

# Portable SHA-256: Linux has `sha256sum`, macOS has `shasum`. Hashes a file
# argument, or stdin when called with none.
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$@" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$@" | cut -d' ' -f1
  else
    echo "natives-fingerprint: no sha256sum/shasum found" >&2
    exit 1
  fi
}

# 1. llama.cpp source identity. CI does not init the submodule, so build.rs's
#    clone-at-const is the source of record; key on that pinned commit.
LLAMA_SHA="$(grep -E 'const LLAMA_CPP_COMMIT' "$SYS/build.rs" | grep -oE '[0-9a-f]{40}')"

# 2. First-party inputs that change the compiled objects / link surface.
WRAPPER_CPP="$(sha256 "$SYS/wrapper.cpp")"
WRAPPER_H="$(sha256 "$SYS/wrapper.h")"
BUILD_RS="$(sha256 "$SYS/build.rs")"

# 3. Toolchain identity — codegen depends on it, so a slice built by a
#    different toolchain must NOT collide on the same fingerprint.
CMAKE_VER="$(cmake --version 2>/dev/null | head -1 || true)"
CC_VER=""
NDK_REV=""
case "$TARGET" in
  *android*)
    NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
    if [ -n "$NDK" ] && [ -f "$NDK/source.properties" ]; then
      NDK_REV="$(grep -E '^Pkg.Revision' "$NDK/source.properties" | cut -d= -f2 | tr -d ' ')"
    fi
    ;;
  *)
    CC_VER="$({ "${CC:-cc}" --version 2>/dev/null || true; } | head -1)"
    ;;
esac

# 4. Combine with a record separator (0x1f) so no two values can collide by
#    concatenation. Bump the leading schema version when this formula changes
#    so old caches invalidate cleanly.
US=$'\x1f'
PAYLOAD="v1${US}llama=${LLAMA_SHA}${US}wrapper_cpp=${WRAPPER_CPP}${US}wrapper_h=${WRAPPER_H}${US}build_rs=${BUILD_RS}${US}target=${TARGET}${US}features=${FEATURES}${US}profile=release${US}cmake=${CMAKE_VER}${US}cc=${CC_VER}${US}ndk=${NDK_REV}"
printf '%s' "$PAYLOAD" | sha256
