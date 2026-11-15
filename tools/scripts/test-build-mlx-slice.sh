#!/usr/bin/env bash
# Regression tests for build-mlx-slice.sh: shell compatibility, generator
# selection, and that MLX is built once, from --mlx-src, by the mlx-c build.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
BUILD_SCRIPT="$REPO_ROOT/tools/scripts/build-mlx-slice.sh"

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

TOOLS_DIR="$TMP_ROOT/tools"
mkdir -p "$TOOLS_DIR"

cat > "$TOOLS_DIR/xcrun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = "--sdk" ] && [ "$3" = "--show-sdk-path" ]; then
    echo "/fake/$2.sdk"
    exit 0
fi
echo "unexpected xcrun args: $*" >&2
exit 1
EOF

cat > "$TOOLS_DIR/cmake" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$XYBRID_TEST_CMAKE_LOG"
if [ "$1" = "--build" ]; then
    build_dir="$2"
    target=""
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "--target" ]; then
            target="$2"
            break
        fi
        shift
    done
    mkdir -p "$build_dir"
    case "$target" in
        mlxc)
            # mlx-c's build fetches MLX and builds it under _deps/.
            printf 'mlxc\n' > "$build_dir/libmlxc.a"
            if [ -z "${XYBRID_TEST_NO_MLX:-}" ]; then
                mkdir -p "$build_dir/_deps/mlx-build/mlx/backend/metal/kernels"
                printf 'mlx\n' > "$build_dir/_deps/mlx-build/libmlx.a"
                printf 'metallib\n' \
                    > "$build_dir/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib"
            fi
            ;;
        *)    echo "unexpected cmake target: $target" >&2; exit 1 ;;
    esac
    exit 0
fi
exit 0
EOF

cat > "$TOOLS_DIR/libtool" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
out=""
while [ "$#" -gt 0 ]; do
    if [ "$1" = "-o" ]; then
        out="$2"
        break
    fi
    shift
done
[ -n "$out" ] || { echo "missing libtool -o" >&2; exit 1; }
printf 'combined\n' > "$out"
EOF

cat > "$TOOLS_DIR/rsync" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

chmod +x "$TOOLS_DIR"/xcrun "$TOOLS_DIR"/cmake "$TOOLS_DIR"/libtool "$TOOLS_DIR"/rsync

mkdir -p "$TMP_ROOT/mlx/mlx" "$TMP_ROOT/mlx-c/mlx/c" "$TMP_ROOT/out"
cmake_log="$TMP_ROOT/cmake.log"

PATH="$TOOLS_DIR:/bin:/usr/bin:/usr/sbin:/sbin" \
XYBRID_TEST_CMAKE_LOG="$cmake_log" \
    "$BUILD_SCRIPT" \
    --slice macos-arm64 \
    --sysroot macosx \
    --arch arm64 \
    --system-name Darwin \
    --deployment-target 14.0 \
    --mlx-src "$TMP_ROOT/mlx" \
    --mlx-c-src "$TMP_ROOT/mlx-c" \
    --out "$TMP_ROOT/out" > "$TMP_ROOT/default.stdout"

grep -q -- 'generator=Unix Makefiles' "$TMP_ROOT/default.stdout"
grep -q -- '-G Unix Makefiles' "$cmake_log"
test -f "$TMP_ROOT/out/lib/libmlx-combined.a"
test -f "$TMP_ROOT/out/lib/mlx.metallib"

# MLX is configured once, by mlx-c, from the pinned checkout: the wrapper must
# be compiled against the MLX it is archived with, not the release mlx-c's own
# CMakeLists would fetch.
[ "$(grep -c -- '^-S ' "$cmake_log")" = "1" ] || {
    echo "expected exactly one cmake configure, got:" >&2
    cat "$cmake_log" >&2
    exit 1
}
grep -q -- "^-S $TMP_ROOT/mlx-c .*-DFETCHCONTENT_SOURCE_DIR_MLX=$TMP_ROOT/mlx" "$cmake_log"
# Error messages name their source file; the build machine's path stays out.
grep -q -- "-DCMAKE_CXX_FLAGS=-ffile-prefix-map=$TMP_ROOT/mlx-c=mlx-c -ffile-prefix-map=$TMP_ROOT/mlx=mlx" "$cmake_log"
if grep -q -- '-DMLX_DIR=' "$cmake_log"; then
    echo "mlx-c ignores MLX_DIR; the slice must not rely on it" >&2
    exit 1
fi
[ "$(grep -c -- '^--build ' "$cmake_log")" = "1" ]
[ "$(cat "$TMP_ROOT/out/lib/libmlx.a")" = "mlx" ]
[ "$(cat "$TMP_ROOT/out/lib/libmlxc.a")" = "mlxc" ]

# Relative checkouts, as build-mlx-xcframework.yml passes them, reach CMake as
# absolute paths: it would read a relative FETCHCONTENT_SOURCE_DIR_MLX against
# the build tree.
mkdir -p "$TMP_ROOT/rel/src/mlx/mlx" "$TMP_ROOT/rel/src/mlx-c/mlx/c"
(
    cd "$TMP_ROOT/rel"
    PATH="$TOOLS_DIR:/bin:/usr/bin:/usr/sbin:/sbin" \
    XYBRID_TEST_CMAKE_LOG="$TMP_ROOT/rel.log" \
        "$BUILD_SCRIPT" \
        --slice macos-arm64 \
        --sysroot macosx \
        --arch arm64 \
        --system-name Darwin \
        --deployment-target 14.0 \
        --mlx-src src/mlx \
        --mlx-c-src src/mlx-c \
        --out out > "$TMP_ROOT/rel.stdout"
)
grep -q -- "^-S $TMP_ROOT/rel/src/mlx-c -B $TMP_ROOT/rel/src/mlx-c/build-macos-arm64 " "$TMP_ROOT/rel.log"
grep -q -- "-DFETCHCONTENT_SOURCE_DIR_MLX=$TMP_ROOT/rel/src/mlx\$" "$TMP_ROOT/rel.log"
grep -q -- "-ffile-prefix-map=$TMP_ROOT/rel/src/mlx-c=mlx-c " "$TMP_ROOT/rel.log"
test -f "$TMP_ROOT/rel/out/lib/libmlx-combined.a"

# If mlx-c ever stops building MLX itself, fail instead of archiving a wrapper
# with no MLX behind it. (A fresh mlx-c checkout: the build tree lives in it.)
mkdir -p "$TMP_ROOT/mlx-c-bare/mlx/c"
if PATH="$TOOLS_DIR:/bin:/usr/bin:/usr/sbin:/sbin" \
    XYBRID_TEST_NO_MLX=1 \
    XYBRID_TEST_CMAKE_LOG="$TMP_ROOT/no-mlx.log" \
    "$BUILD_SCRIPT" \
    --slice macos-arm64 \
    --sysroot macosx \
    --arch arm64 \
    --system-name Darwin \
    --deployment-target 14.0 \
    --mlx-src "$TMP_ROOT/mlx" \
    --mlx-c-src "$TMP_ROOT/mlx-c-bare" \
    --out "$TMP_ROOT/out-no-mlx" > "$TMP_ROOT/no-mlx.stdout" 2> "$TMP_ROOT/no-mlx.stderr"; then
    echo "expected a build that produced no libmlx.a to fail" >&2
    exit 1
fi
grep -q -- 'did not produce' "$TMP_ROOT/no-mlx.stderr"

if PATH="$TOOLS_DIR:/bin:/usr/bin:/usr/sbin:/sbin" \
    CMAKE_GENERATOR=Ninja \
    XYBRID_TEST_CMAKE_LOG="$TMP_ROOT/ninja.log" \
    "$BUILD_SCRIPT" \
    --slice macos-arm64 \
    --sysroot macosx \
    --arch arm64 \
    --system-name Darwin \
    --deployment-target 14.0 \
    --mlx-src "$TMP_ROOT/mlx" \
    --mlx-c-src "$TMP_ROOT/mlx-c" \
    --out "$TMP_ROOT/out-ninja" > "$TMP_ROOT/ninja.stdout" 2> "$TMP_ROOT/ninja.stderr"; then
    echo "expected explicit Ninja generator to fail when ninja is missing" >&2
    exit 1
fi
grep -q -- "CMAKE_GENERATOR=Ninja but 'ninja' is not on PATH" "$TMP_ROOT/ninja.stderr"

echo "build-mlx-slice.sh tests passed"
