#!/usr/bin/env bash
# build-python-bolt.sh
#
# Build the xybrid BoltFFI cdylib for the host and refresh the Python package's
# bundled native library copy.
#
# Why this exists:
#
# - The Python SDK uses a hand-ported ctypes wire layer
#   (`bindings/python/xybrid/_bolt.py`) pinned to the BoltFFI 0.25.3 ABI.
#   No BoltFFI Python generator is run here; the generator cannot express the
#   handle/fallible-function surface this package needs until the workspace
#   migrates to boltffi >= 0.26.
# - Cargo already emits the cdylib that ctypes loads, so this script only builds
#   `xybrid-bolt` with the correct host feature preset and copies the resulting
#   library into `bindings/python/xybrid/_native/`.
#
# Usage: ./tools/scripts/build-python-bolt.sh
# Optional env overrides:
#   XYBRID_FEATURES  Cargo features to enable
#   DEBUG=1          Build target/debug instead of target/release
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_NATIVE_DIR="$REPO_ROOT/bindings/python/xybrid/_native"

case "$(uname -s)" in
    Darwin)
        DEFAULT_FEATURES="platform-macos"
        LIB_NAME="libxybrid_bolt.dylib"
        ;;
    Linux)
        DEFAULT_FEATURES="platform-desktop"
        LIB_NAME="libxybrid_bolt.so"
        ;;
    *)
        echo "error: unsupported host OS '$(uname -s)'; expected Darwin or Linux" >&2
        exit 1
        ;;
esac

FEATURES="${XYBRID_FEATURES:-$DEFAULT_FEATURES}"
PROFILE="release"
PROFILE_FLAG="--release"
if [ "${DEBUG:-0}" = "1" ]; then
    PROFILE="debug"
    PROFILE_FLAG=""
fi

echo "==> Building Python bolt native library"
echo "    Features: $FEATURES"
echo "    Profile:  $PROFILE"

cd "$REPO_ROOT"
# shellcheck disable=SC2086  # deliberate word-split: empty PROFILE_FLAG = debug
cargo build -p xybrid-bolt $PROFILE_FLAG --features "$FEATURES"

SRC="$REPO_ROOT/target/$PROFILE/$LIB_NAME"
if [ ! -f "$SRC" ]; then
    echo "error: expected native library not found at $SRC" >&2
    exit 1
fi

mkdir -p "$PY_NATIVE_DIR"
cp "$SRC" "$PY_NATIVE_DIR/$LIB_NAME"

echo "==> Done. Copied $PY_NATIVE_DIR/$LIB_NAME"
