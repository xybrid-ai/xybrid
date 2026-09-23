#!/usr/bin/env bash
# Regression tests for fetch-mlx-xcframework.sh pin parsing/status mode.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
FETCH_SCRIPT="$REPO_ROOT/tools/scripts/fetch-mlx-xcframework.sh"

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

make_vendor() {
    local name="$1"
    local release="$2"
    local sha256="$3"
    local dir="$TMP_ROOT/$name"
    mkdir -p "$dir"
    cat > "$dir/UPSTREAM_VERSIONS.txt" <<EOF
mlx=ce45c52505c8158ea48d2a54e8caae05efd86bfe
mlx-c=a0b5179b2a34f6441eb78efe39f3b36be757f5a6
release=${release}
sha256=${sha256}
EOF
    printf '%s\n' "$dir"
}

make_vendor_missing_release() {
    local dir="$TMP_ROOT/missing-release"
    mkdir -p "$dir"
    cat > "$dir/UPSTREAM_VERSIONS.txt" <<EOF
mlx=ce45c52505c8158ea48d2a54e8caae05efd86bfe
mlx-c=a0b5179b2a34f6441eb78efe39f3b36be757f5a6
sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
EOF
    printf '%s\n' "$dir"
}

assert_file_contains() {
    local file="$1"
    local pattern="$2"
    if ! grep -q -- "$pattern" "$file"; then
        echo "expected $file to contain: $pattern" >&2
        echo "--- $file ---" >&2
        cat "$file" >&2
        exit 1
    fi
}

unpublished_vendor=$(make_vendor unpublished unpublished unpublished)
unpublished_output="$TMP_ROOT/unpublished.github-output"
GITHUB_OUTPUT="$unpublished_output" \
    XYBRID_MLX_VENDOR_DIR="$unpublished_vendor" \
    "$FETCH_SCRIPT" --check-published > "$TMP_ROOT/unpublished.stdout" 2> "$TMP_ROOT/unpublished.stderr"
assert_file_contains "$unpublished_output" '^published=false$'
assert_file_contains "$unpublished_output" '^release=unpublished$'

missing_release_vendor=$(make_vendor_missing_release)
missing_release_output="$TMP_ROOT/missing-release.github-output"
if GITHUB_OUTPUT="$missing_release_output" \
    XYBRID_MLX_VENDOR_DIR="$missing_release_vendor" \
    "$FETCH_SCRIPT" --check-published > "$TMP_ROOT/missing-release.stdout" 2> "$TMP_ROOT/missing-release.stderr"; then
    echo "expected missing release pin to fail" >&2
    exit 1
fi
assert_file_contains "$missing_release_output" '^published=false$'
assert_file_contains "$TMP_ROOT/missing-release.stderr" "must define both 'release=' and 'sha256=' keys"

published_sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
published_vendor=$(make_vendor published mlx-v0.31.1 "$published_sha")
published_output="$TMP_ROOT/published.github-output"
GITHUB_OUTPUT="$published_output" \
    XYBRID_MLX_VENDOR_DIR="$published_vendor" \
    "$FETCH_SCRIPT" --check-published > "$TMP_ROOT/published.stdout" 2> "$TMP_ROOT/published.stderr"
assert_file_contains "$published_output" '^published=true$'
assert_file_contains "$published_output" '^release=mlx-v0.31.1$'
assert_file_contains "$published_output" "^sha256=${published_sha}$"
assert_file_contains "$published_output" '^asset_name=mlx-0.31.1.xcframework.zip$'

invalid_vendor=$(make_vendor invalid mlx-v0.31.1 not-a-sha)
if XYBRID_MLX_VENDOR_DIR="$invalid_vendor" \
    "$FETCH_SCRIPT" --check-published > "$TMP_ROOT/invalid.stdout" 2> "$TMP_ROOT/invalid.stderr"; then
    echo "expected invalid sha256 pin to fail" >&2
    exit 1
fi
assert_file_contains "$TMP_ROOT/invalid.stderr" 'sha256='

echo "fetch-mlx-xcframework.sh pin status tests passed"
