#!/usr/bin/env bash
# natives-push.sh
#
# Publish one freshly-built llama.cpp native slice to ghcr, tagged by its
# fingerprint. Write-once: if the fingerprint tag already exists it is left
# untouched (never overwrite a published slice — mutable tags are a
# cache-poisoning vector). Run by the publisher workflow only.
#
# Expects the slice already exported by build.rs (XYBRID_NATIVES_EXPORT_DIR),
# i.e. <export>/<target-triple>/{lib,include}.
#
# Usage: natives-push.sh <target-triple> <feature-set> <export-base-dir>
# Requires: oras on PATH, logged in to ghcr with packages:write.
set -euo pipefail

TARGET="${1:?usage: natives-push.sh <target-triple> <feature-set> <export-base-dir>}"
FEATURES="${2:?}"
EXPORT="${3:?}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="${XYBRID_NATIVES_PKG:-ghcr.io/xybrid-ai/llama-natives}"

FP="$("$ROOT/tools/scripts/natives-fingerprint.sh" "$TARGET" "$FEATURES")"
LLAMA_SHA="$(grep -E 'const LLAMA_CPP_COMMIT' "$ROOT/crates/llama-cpp-sys/build.rs" | grep -oE '[0-9a-f]{40}')"
echo "natives-push: target=$TARGET features=$FEATURES fingerprint=$FP"

# Write-once: skip if this fingerprint is already published.
if oras manifest fetch --descriptor "$PKG:$FP" >/dev/null 2>&1; then
  echo "natives-push: $PKG:$FP already published — skip"
  exit 0
fi

SLICE="$EXPORT/$TARGET"
[ -d "$SLICE/lib" ] || { echo "natives-push: no built slice at $SLICE/lib" >&2; exit 1; }
[ -d "$SLICE/include" ] || { echo "natives-push: no headers at $SLICE/include" >&2; exit 1; }

slice_dirs=(lib include)
[ -d "$SLICE/lib64" ] && slice_dirs+=(lib64)
tar -C "$SLICE" -czf "$SLICE/native.tar.gz" "${slice_dirs[@]}"

oras push "$PKG:$FP" \
  --artifact-type application/vnd.xybrid.natives.layer.v1+gzip \
  --annotation "org.opencontainers.image.source=https://github.com/xybrid-ai/xybrid" \
  --annotation "dev.xybrid.triple=$TARGET" \
  --annotation "dev.xybrid.features=$FEATURES" \
  --annotation "dev.xybrid.llama-sha=$LLAMA_SHA" \
  "$SLICE/native.tar.gz:application/vnd.xybrid.natives.layer.v1.tar+gzip"

echo "natives-push: pushed $PKG:$FP"
