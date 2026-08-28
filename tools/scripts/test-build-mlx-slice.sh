#!/usr/bin/env bash
# Regression tests for build-mlx-slice.sh shell compatibility and generator selection.

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
        mlx)
            printf 'mlx\n' > "$build_dir/libmlx.a"
            mkdir -p "$build_dir/mlx/backend/metal/kernels"
            printf 'metallib\n' > "$build_dir/mlx/backend/metal/kernels/mlx.metallib"
            ;;
        mlxc) printf 'mlxc\n' > "$build_dir/libmlxc.a" ;;
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

echo "build-mlx-slice.sh generator tests passed"
