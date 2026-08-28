#!/usr/bin/env bash
# Regression tests for build-local-mlx-xcframework.sh orchestration.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
BUILD_LOCAL_SCRIPT="$REPO_ROOT/tools/scripts/build-local-mlx-xcframework.sh"

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

TOOLS_DIR="$TMP_ROOT/tools"
mkdir -p "$TOOLS_DIR"

cat > "$TOOLS_DIR/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$XYBRID_TEST_GIT_LOG"
if [ "$1" = "clone" ]; then
    dest="${@: -1}"
    mkdir -p "$dest"
    exit 0
fi
if [ "$1" = "-C" ] && [ "$3" = "checkout" ]; then
    exit 0
fi
echo "unexpected git args: $*" >&2
exit 1
EOF

cat > "$TOOLS_DIR/xcodebuild" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
out=""
while [ "$#" -gt 0 ]; do
    if [ "$1" = "-output" ]; then
        out="$2"
        break
    fi
    shift
done
[ -n "$out" ] || { echo "missing xcodebuild -output" >&2; exit 1; }
mkdir -p "$out/macos-arm64"
printf 'plist\n' > "$out/Info.plist"
EOF

FAKE_SLICE="$TMP_ROOT/build-mlx-slice.sh"
cat > "$FAKE_SLICE" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$XYBRID_TEST_SLICE_LOG"
out=""
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--out" ]; then
        out="$2"
        break
    fi
    shift
done
[ -n "$out" ] || { echo "missing --out" >&2; exit 1; }
mkdir -p "$out/lib" "$out/include"
printf 'lib\n' > "$out/lib/libmlx-combined.a"
printf 'metal\n' > "$out/lib/mlx.metallib"
EOF

chmod +x "$TOOLS_DIR"/git "$TOOLS_DIR"/xcodebuild "$FAKE_SLICE"

VENDOR_DIR="$TMP_ROOT/vendor/mlx-apple"
mkdir -p "$VENDOR_DIR"
cat > "$VENDOR_DIR/UPSTREAM_VERSIONS.txt" <<'EOF'
mlx=ce45c52505c8158ea48d2a54e8caae05efd86bfe
mlx-c=a0b5179b2a34f6441eb78efe39f3b36be757f5a6
release=unpublished
sha256=unpublished
EOF

touch "$VENDOR_DIR/.installed-sha256"
WORK_DIR="$TMP_ROOT/work"
git_log="$TMP_ROOT/git.log"
slice_log="$TMP_ROOT/slice.log"

PATH="$TOOLS_DIR:/bin:/usr/bin:/usr/sbin:/sbin" \
XYBRID_TEST_GIT_LOG="$git_log" \
XYBRID_TEST_SLICE_LOG="$slice_log" \
XYBRID_MLX_BUILD_SLICE_SCRIPT="$FAKE_SLICE" \
    "$BUILD_LOCAL_SCRIPT" \
    --vendor-dir "$VENDOR_DIR" \
    --work-dir "$WORK_DIR" > "$TMP_ROOT/stdout"

grep -q -- 'https://github.com/ml-explore/mlx.git' "$git_log"
grep -q -- 'https://github.com/ml-explore/mlx-c.git' "$git_log"
grep -q -- '--slice macos-arm64' "$slice_log"
test -f "$VENDOR_DIR/mlx.xcframework/macos-arm64/Resources/mlx.metallib"
test -f "$VENDOR_DIR/.installed-source-pin"
test ! -f "$VENDOR_DIR/.installed-sha256"
test ! -d "$WORK_DIR"

echo "build-local-mlx-xcframework.sh tests passed"
