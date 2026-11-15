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
# MLX is built once, as part of the mlx-c build. mlx-c's CMake fetches the MLX
# release its own CMakeLists names unless told otherwise;
# FETCHCONTENT_SOURCE_DIR_MLX points it at --mlx-src instead, so the wrapper is
# compiled against exactly the MLX it is archived with, the pinned one.
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
        opt=$(printf '%s' "$var" | tr '[:upper:]_' '[:lower:]-')
        echo "missing required --${opt}" >&2
        usage >&2
        exit 2
    fi
done

CMAKE_GENERATOR="${CMAKE_GENERATOR:-}"
if [ -z "$CMAKE_GENERATOR" ]; then
    if command -v ninja >/dev/null 2>&1; then
        CMAKE_GENERATOR=Ninja
    else
        CMAKE_GENERATOR="Unix Makefiles"
    fi
fi

if [ "$CMAKE_GENERATOR" = "Ninja" ] && ! command -v ninja >/dev/null 2>&1; then
    echo "error: CMAKE_GENERATOR=Ninja but 'ninja' is not on PATH" >&2
    echo "       install ninja or set CMAKE_GENERATOR='Unix Makefiles'" >&2
    exit 1
fi

SDK_PATH=$(xcrun --sdk "$SYSROOT" --show-sdk-path)
echo "==> [${SLICE}] sysroot=${SYSROOT} arch=${ARCH} sdk=${SDK_PATH} generator=${CMAKE_GENERATOR}"

mkdir -p "$OUT/lib" "$OUT/include"

# Absolute paths from here on: CMake reads FETCHCONTENT_SOURCE_DIR_MLX relative
# to the build tree, not to where this script was started, and the compiler
# sees absolute source paths.
for var in MLX_SRC MLX_C_SRC OUT; do
    if [ ! -d "${!var}" ]; then
        echo "error: not a directory: ${!var}" >&2
        exit 1
    fi
done
MLX_SRC=$(cd "$MLX_SRC" && pwd)
MLX_C_SRC=$(cd "$MLX_C_SRC" && pwd)
OUT=$(cd "$OUT" && pwd)

# mlx-c ends every error message with the source file that raised it
# (`__FILE__`), so without a prefix map each MLX error a user sees would carry
# the build machine's checkout path. Map both checkouts to their names, under
# the path as given and under the path with symlinks resolved (a temporary
# directory on macOS has both).
PREFIX_MAP="-ffile-prefix-map=${MLX_C_SRC}=mlx-c -ffile-prefix-map=${MLX_SRC}=mlx"
MLX_SRC_REAL=$(cd "$MLX_SRC" && pwd -P)
MLX_C_SRC_REAL=$(cd "$MLX_C_SRC" && pwd -P)
if [ "$MLX_SRC_REAL" != "$MLX_SRC" ] || [ "$MLX_C_SRC_REAL" != "$MLX_C_SRC" ]; then
    PREFIX_MAP="$PREFIX_MAP -ffile-prefix-map=${MLX_C_SRC_REAL}=mlx-c -ffile-prefix-map=${MLX_SRC_REAL}=mlx"
fi

# Common CMake flags for both mlx and mlx-c.
COMMON_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_FLAGS="$PREFIX_MAP"
    -DCMAKE_CXX_FLAGS="$PREFIX_MAP"
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

# 1. Build mlx-c, and with it MLX from --mlx-src.
MLX_C_BUILD="$MLX_C_SRC/build-${SLICE}"
MLX_BUILD="$MLX_C_BUILD/_deps/mlx-build"
cmake -S "$MLX_C_SRC" -B "$MLX_C_BUILD" -G "$CMAKE_GENERATOR" "${COMMON_FLAGS[@]}" \
    -DFETCHCONTENT_SOURCE_DIR_MLX="$MLX_SRC"
cmake --build "$MLX_C_BUILD" --config Release --target mlxc

if [ ! -f "$MLX_BUILD/libmlx.a" ]; then
    echo "error: the mlx-c build did not produce $MLX_BUILD/libmlx.a" >&2
    echo "       (did mlx-c stop building MLX through FetchContent?)" >&2
    exit 1
fi
cp "$MLX_BUILD"/libmlx.a "$OUT/lib/libmlx.a"
cp "$MLX_C_BUILD"/libmlxc.a "$OUT/lib/libmlxc.a"

# 2. Metal shaders are optional, but when present MLX writes them below the
# generated backend tree (for example mlx/backend/metal/kernels/mlx.metallib).
METALLIB_COUNT=0
while IFS= read -r METALLIB; do
    [ -n "$METALLIB" ] || continue
    cp "$METALLIB" "$OUT/lib/"
    METALLIB_COUNT=$((METALLIB_COUNT + 1))
done < <(find "$MLX_BUILD" -type f -name '*.metallib')
if [ "$METALLIB_COUNT" -eq 0 ]; then
    echo "warning: no .metallib produced for ${SLICE}; Metal kernels may be unavailable" >&2
fi

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
