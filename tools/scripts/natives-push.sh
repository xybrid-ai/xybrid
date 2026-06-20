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

command -v oras >/dev/null 2>&1 || { echo "natives-push: oras not found" >&2; exit 1; }

FP="$("$ROOT/tools/scripts/natives-fingerprint.sh" "$TARGET" "$FEATURES")"
# `|| true`: the SHA is only a human-readable annotation, so a future reformat
# of the const line should degrade it gracefully rather than abort the publish.
LLAMA_SHA="$(grep -E 'const LLAMA_CPP_COMMIT' "$ROOT/crates/llama-cpp-sys/build.rs" | grep -oE '[0-9a-f]{40}' || true)"
echo "natives-push: target=$TARGET features=$FEATURES fingerprint=$FP"

# Write-once: skip if this fingerprint is already published.
if oras manifest fetch --descriptor "$PKG:$FP" >/dev/null 2>&1; then
  echo "natives-push: $PKG:$FP already published — skip"
  exit 0
fi

SLICE="$EXPORT/$TARGET"
[ -d "$SLICE/include" ] || { echo "natives-push: no headers at $SLICE/include" >&2; exit 1; }

# Never poison the write-once tag with an incomplete slice: verify the
# archives build.rs's resolve_prebuilt requires are present and non-empty in
# lib/ OR lib64/ before publishing. Mirror required_archives() in build.rs:
# MSVC names static libs `<name>.lib` (no prefix); every other target is
# Unix-style `lib<name>.a`. Base set, plus ggml-metal on Apple and mtmd on vision.
case "$TARGET" in
  *windows-msvc*) pfx=''; sfx='.lib' ;;
  *) pfx='lib'; sfx='.a' ;;
esac
archives=("${pfx}llama${sfx}" "${pfx}ggml${sfx}" "${pfx}ggml-base${sfx}" "${pfx}ggml-cpu${sfx}")
case "$TARGET" in
  *apple*) archives+=("${pfx}ggml-metal${sfx}") ;;
esac
[ "$FEATURES" = "vision" ] && archives+=("${pfx}mtmd${sfx}")
for a in "${archives[@]}"; do
  [ -s "$SLICE/lib/$a" ] || [ -s "$SLICE/lib64/$a" ] || {
    echo "natives-push: required archive $a missing — refusing to publish incomplete slice" >&2
    exit 1
  }
done

slice_dirs=(lib include)
[ -d "$SLICE/lib64" ] && slice_dirs+=(lib64)
tar -C "$SLICE" -czf "$SLICE/native.tar.gz" "${slice_dirs[@]}"

# Push from INSIDE the slice dir so the layer reference is a RELATIVE path:
# oras rejects absolute file paths (path-validation), and a bare
# `native.tar.gz` title is exactly what natives-pull.sh expects from
# `oras pull -o <dir>` (it writes <dir>/native.tar.gz).
(
  cd "$SLICE"
  oras push "$PKG:$FP" \
    --artifact-type application/vnd.xybrid.natives.layer.v1+gzip \
    --annotation "org.opencontainers.image.source=https://github.com/xybrid-ai/xybrid" \
    --annotation "dev.xybrid.triple=$TARGET" \
    --annotation "dev.xybrid.features=$FEATURES" \
    --annotation "dev.xybrid.llama-sha=$LLAMA_SHA" \
    "native.tar.gz:application/vnd.xybrid.natives.layer.v1.tar+gzip"
)

echo "natives-push: pushed $PKG:$FP"
