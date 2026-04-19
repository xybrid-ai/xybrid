#!/usr/bin/env bash
# build-mlx-slice.sh — Build a single (sysroot, arch) slice of mlx + mlx-c.
#
# Produces, under --out:
#   include/...           merged public headers from mlx and mlx-c
#   lib/libmlx.a          MLX core static library
#   lib/libmlxc.a         MLX C wrapper static library
#   lib/libmlx-combined.a libmlx.a + libmlxc.a merged via libtool -static
#   lib/*.metallib        compiled Metal shader libraries (when produced)
#
# Invoked once per xcframework slice by .github/workflows/build-mlx-xcframework.yml.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build-mlx-slice.sh \
    --slice <name> \
    --sysroot <iphoneos|iphonesimulator|macosx> \
    --arch <arm64> \
    --system-name <iOS|Darwin> \
    --deployment-target <version> \
    --mlx-src <path> \
    --mlx-c-src <path> \
    --out <path>
EOF
}

SLICE=""
SYSROOT=""
ARCH=""
SYSTEM_NAME=""
DEPLOYMENT_TARGET=""
MLX_SRC=""
MLX_C_SRC=""
OUT=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --slice)              SLICE="$2"; shift 2 ;;
        --sysroot)            SYSROOT="$2"; shift 2 ;;
        --arch)               ARCH="$2"; shift 2 ;;
        --system-name)        SYSTEM_NAME="$2"; shift 2 ;;
        --deployment-target)  DEPLOYMENT_TARGET="$2"; shift 2 ;;
        --mlx-src)            MLX_SRC="$2"; shift 2 ;;
        --mlx-c-src)          MLX_C_SRC="$2"; shift 2 ;;
        --out)                OUT="$2"; shift 2 ;;
        -h|--help)            usage; exit 0 ;;
        *)                    echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for var in SLICE SYSROOT ARCH SYSTEM_NAME DEPLOYMENT_TARGET MLX_SRC MLX_C_SRC OUT; do
    if [ -z "${!var}" ]; then
        echo "missing required --${var,,}" >&2
        usage >&2
        exit 2
    fi
done

SDK_PATH=$(xcrun --sdk "$SYSROOT" --show-sdk-path)
echo "==> [${SLICE}] sysroot=${SYSROOT} arch=${ARCH} sdk=${SDK_PATH}"

mkdir -p "$OUT/lib" "$OUT/include"

# Common CMake flags for both mlx and mlx-c.
COMMON_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_SYSTEM_NAME="$SYSTEM_NAME"
    -DCMAKE_OSX_SYSROOT="$SDK_PATH"
    -DCMAKE_OSX_ARCHITECTURES="$ARCH"
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET"
    -DBUILD_SHARED_LIBS=OFF
    -DMLX_BUILD_METAL=ON
    -DMLX_BUILD_CPU=ON
    -DMLX_BUILD_TESTS=OFF
    -DMLX_BUILD_EXAMPLES=OFF
    -DMLX_BUILD_BENCHMARKS=OFF
    -DMLX_BUILD_PYTHON_BINDINGS=OFF
)

# 1. Build MLX.
MLX_BUILD="$MLX_SRC/build-${SLICE}"
cmake -S "$MLX_SRC" -B "$MLX_BUILD" -G Ninja "${COMMON_FLAGS[@]}"
cmake --build "$MLX_BUILD" --config Release --target mlx
cp "$MLX_BUILD"/libmlx.a "$OUT/lib/libmlx.a"

# Metal shaders are optional — only present when the Metal pipeline ran.
shopt -s nullglob
for METALLIB in "$MLX_BUILD"/*.metallib "$MLX_BUILD"/mlx/*.metallib; do
    cp "$METALLIB" "$OUT/lib/"
done
shopt -u nullglob

# 2. Build mlx-c, pointing it at the freshly built MLX.
MLX_C_BUILD="$MLX_C_SRC/build-${SLICE}"
cmake -S "$MLX_C_SRC" -B "$MLX_C_BUILD" -G Ninja "${COMMON_FLAGS[@]}" \
    -DMLX_DIR="$MLX_BUILD"
cmake --build "$MLX_C_BUILD" --config Release --target mlxc
cp "$MLX_C_BUILD"/libmlxc.a "$OUT/lib/libmlxc.a"

# 3. Combine into a single archive for xcframework consumption.
libtool -static -no_warning_for_no_symbols \
    -o "$OUT/lib/libmlx-combined.a" \
    "$OUT/lib/libmlx.a" \
    "$OUT/lib/libmlxc.a"

# 4. Stage public headers (merged into a single include tree).
if [ -d "$MLX_SRC/mlx" ]; then
    rsync -a --include='*/' --include='*.h' --include='*.hpp' --exclude='*' \
        "$MLX_SRC/mlx/" "$OUT/include/mlx/"
fi
if [ -d "$MLX_C_SRC/mlx/c" ]; then
    rsync -a --include='*/' --include='*.h' --exclude='*' \
        "$MLX_C_SRC/mlx/c/" "$OUT/include/mlx/c/"
fi

echo "==> [${SLICE}] done. Output:"
find "$OUT" -maxdepth 3 -type f | sort
