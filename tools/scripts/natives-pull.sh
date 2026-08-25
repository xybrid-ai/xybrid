#!/usr/bin/env bash
# natives-pull.sh
#
# Best-effort pull of one prebuilt llama.cpp native slice from ghcr into a
# target-keyed dir, for the consumer fast path. Prints the BASE dir to stdout
# on success (point XYBRID_NATIVES_PREBUILT_DIR at it). Exits non-zero on ANY
# miss/error so the caller falls back to a source compile — this is never the
# critical path; a miss is normal and must not fail the build.
#
# Layout produced:  <dest>/<target-triple>/{lib,include}   (build.rs reads
# XYBRID_NATIVES_PREBUILT_DIR/<target>; see resolve_prebuilt in build.rs).
#
# Usage: natives-pull.sh <target-triple> <feature-set> <dest-base-dir>
# Requires: oras on PATH, and read access to $XYBRID_NATIVES_PKG. Anonymous
# pulls only work once that package is public — until then callers must
# `oras login` first (our CI jobs do). tools/scripts/natives-verify-anon.sh
# checks the anonymous path, which is the one external cargo consumers use.
set -euo pipefail

TARGET="${1:?usage: natives-pull.sh <target-triple> <feature-set> <dest-base-dir>}"
FEATURES="${2:?}"
DEST="${3:?}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="${XYBRID_NATIVES_PKG:-ghcr.io/xybrid-ai/llama-natives}"

command -v oras >/dev/null 2>&1 || { echo "natives-pull: oras not found" >&2; exit 1; }

FP="$("$ROOT/tools/scripts/natives-fingerprint.sh" "$TARGET" "$FEATURES")" || exit 1

SLICE="$DEST/$TARGET"
mkdir -p "$SLICE"

# Pull is best-effort: anonymous works for a public package; any failure
# (absent tag, network, auth) drops us to the source build.
if ! oras pull "$PKG:$FP" -o "$SLICE" >/dev/null 2>&1; then
  echo "natives-pull: no prebuilt for $PKG:$FP" >&2
  exit 1
fi

# The artifact is a single native.tar.gz of {lib,include}; unpack in place.
if [ ! -f "$SLICE/native.tar.gz" ]; then
  echo "natives-pull: pulled artifact missing native.tar.gz" >&2
  exit 1
fi
# Relative -f from inside the dir: GNU tar parses a `D:` drive-letter prefix
# in -f as a remote host on the windows runner (same fix as natives-push.sh).
(cd "$SLICE" && tar -xzf native.tar.gz && rm -f native.tar.gz) || exit 1

# build.rs re-validates completeness (required archives + include/) before
# trusting the slice, so a partial unpack here just degrades to source.
echo "$DEST"
