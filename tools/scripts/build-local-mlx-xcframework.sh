#!/usr/bin/env bash
# build-local-mlx-xcframework.sh — Build a macOS arm64 mlx.xcframework from
# the pinned upstream MLX sources without requiring a published GitHub Release.
#
# This is the release-free fallback used by runtime CI while
# vendor/mlx-apple/UPSTREAM_VERSIONS.txt has release=unpublished. It builds only
# the macOS arm64 slice because that is the only runtime-supported MLX target.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./tools/scripts/build-local-mlx-xcframework.sh [--vendor-dir <path>] [--work-dir <path>] [--keep-work]

Options:
  --vendor-dir <path>  Override vendor/mlx-apple.
  --work-dir <path>    Override temporary build directory.
  --keep-work          Keep the temporary build directory after completion.
  -h, --help           Show this help.

Environment:
  MLX_DEPLOYMENT_TARGET_MACOS       macOS deployment target (default: 14.0)
  XYBRID_MLX_BUILD_SLICE_SCRIPT     Override build-mlx-slice.sh path (tests)
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

VENDOR_DIR="$REPO_ROOT/vendor/mlx-apple"
WORK_DIR=""
KEEP_WORK=false
DEPLOYMENT_TARGET="${MLX_DEPLOYMENT_TARGET_MACOS:-14.0}"
BUILD_SLICE_SCRIPT="${XYBRID_MLX_BUILD_SLICE_SCRIPT:-$REPO_ROOT/tools/scripts/build-mlx-slice.sh}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --vendor-dir) VENDOR_DIR="$2"; shift 2 ;;
        --work-dir)   WORK_DIR="$2"; shift 2 ;;
        --keep-work)  KEEP_WORK=true; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)
            echo "unknown arg: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

PINS_FILE="$VENDOR_DIR/UPSTREAM_VERSIONS.txt"
XCFRAMEWORK_DIR="$VENDOR_DIR/mlx.xcframework"
SOURCE_MARKER="$VENDOR_DIR/.installed-source-pin"
SHA_MARKER="$VENDOR_DIR/.installed-sha256"

if [ ! -f "$PINS_FILE" ]; then
    echo "error: $PINS_FILE missing — cannot resolve pinned upstream MLX sources" >&2
    exit 1
fi

if [ ! -x "$BUILD_SLICE_SCRIPT" ]; then
    echo "error: build slice script is not executable: $BUILD_SLICE_SCRIPT" >&2
    exit 1
fi

for cmd in git xcodebuild; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "error: '$cmd' is required to build the local MLX xcframework" >&2
        exit 1
    fi
done

read_pin() {
    awk -v key="$1" '
        index($0, key "=") == 1 {
            value = substr($0, length(key) + 2)
            gsub(/[[:space:]]/, "", value)
            print value
            exit
        }
    ' "$PINS_FILE"
}

MLX_SHA=$(read_pin mlx)
MLX_C_SHA=$(read_pin mlx-c)

if [ -z "$MLX_SHA" ] || [ -z "$MLX_C_SHA" ]; then
    echo "error: UPSTREAM_VERSIONS.txt must define both 'mlx=' and 'mlx-c=' keys" >&2
    exit 1
fi

if [ -z "$WORK_DIR" ]; then
    WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/xybrid-mlx-build.XXXXXX")
else
    mkdir -p "$WORK_DIR"
fi

cleanup() {
    if [ "$KEEP_WORK" != true ]; then
        rm -rf "$WORK_DIR"
    fi
}
trap cleanup EXIT

echo "==> building local MLX xcframework from pinned sources"
echo "    vendor:  $VENDOR_DIR"
echo "    work:    $WORK_DIR"
echo "    mlx:     $MLX_SHA"
echo "    mlx-c:   $MLX_C_SHA"
echo "    target:  macos-arm64 (deployment ${DEPLOYMENT_TARGET})"

mkdir -p "$WORK_DIR/upstream" "$WORK_DIR/slices"
rm -rf "$WORK_DIR/upstream/mlx" "$WORK_DIR/upstream/mlx-c" "$WORK_DIR/slices/macos-arm64"

git clone --filter=blob:none https://github.com/ml-explore/mlx.git "$WORK_DIR/upstream/mlx"
git -C "$WORK_DIR/upstream/mlx" checkout "$MLX_SHA"
git clone --filter=blob:none https://github.com/ml-explore/mlx-c.git "$WORK_DIR/upstream/mlx-c"
git -C "$WORK_DIR/upstream/mlx-c" checkout "$MLX_C_SHA"

"$BUILD_SLICE_SCRIPT" \
    --slice macos-arm64 \
    --sysroot macosx \
    --arch arm64 \
    --system-name Darwin \
    --deployment-target "$DEPLOYMENT_TARGET" \
    --mlx-src "$WORK_DIR/upstream/mlx" \
    --mlx-c-src "$WORK_DIR/upstream/mlx-c" \
    --out "$WORK_DIR/slices/macos-arm64"

rm -rf "$XCFRAMEWORK_DIR"
mkdir -p "$VENDOR_DIR"
xcodebuild -create-xcframework \
    -library "$WORK_DIR/slices/macos-arm64/lib/libmlx-combined.a" \
      -headers "$WORK_DIR/slices/macos-arm64/include" \
    -output "$XCFRAMEWORK_DIR"

mkdir -p "$XCFRAMEWORK_DIR/macos-arm64/Resources"
if compgen -G "$WORK_DIR/slices/macos-arm64/lib/*.metallib" > /dev/null; then
    cp "$WORK_DIR"/slices/macos-arm64/lib/*.metallib "$XCFRAMEWORK_DIR/macos-arm64/Resources/"
else
    echo "error: no .metallib produced for macos-arm64; MLX runtime would not have Metal kernels" >&2
    exit 1
fi

rm -f "$SHA_MARKER"
{
    printf 'mlx=%s\n' "$MLX_SHA"
    printf 'mlx-c=%s\n' "$MLX_C_SHA"
    printf 'slice=macos-arm64\n'
} > "$SOURCE_MARKER"

echo "==> local mlx.xcframework installed at $XCFRAMEWORK_DIR"
find "$XCFRAMEWORK_DIR" -maxdepth 3 -print | sort
