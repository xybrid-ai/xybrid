#!/usr/bin/env bash
# fetch-mlx-xcframework.sh — Install the prebuilt mlx.xcframework pinned in
# vendor/mlx-apple/UPSTREAM_VERSIONS.txt.
#
# Idempotent: if the currently-installed xcframework SHA matches the pin, the
# script exits 0 without touching anything. Otherwise it downloads the GitHub
# Release artifact, verifies its SHA256, and unpacks it into
# vendor/mlx-apple/mlx.xcframework/.
#
# Usage:
#   ./tools/scripts/fetch-mlx-xcframework.sh [--force]
#
# Options:
#   --force        Re-download and reinstall even if the local SHA matches.
#
# Environment:
#   XYBRID_RELEASE_REPO  Override the GitHub repo path (default: xybrid-ai/xybrid).

set -euo pipefail

usage() {
    cat <<'EOF'
fetch-mlx-xcframework.sh — install the prebuilt mlx.xcframework pinned in
vendor/mlx-apple/UPSTREAM_VERSIONS.txt.

Usage:
  ./tools/scripts/fetch-mlx-xcframework.sh [--force]

Options:
  --force        Re-download and reinstall even if the local SHA matches.

Environment:
  XYBRID_RELEASE_REPO  Override the GitHub repo path (default: xybrid-ai/xybrid).
EOF
}

RELEASE_REPO="${XYBRID_RELEASE_REPO:-xybrid-ai/xybrid}"
FORCE=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --force)    FORCE=true; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)
            echo "unknown arg: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# Resolve the repo root (this script lives at tools/scripts/).
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
VENDOR_DIR="$REPO_ROOT/vendor/mlx-apple"
PINS_FILE="$VENDOR_DIR/UPSTREAM_VERSIONS.txt"
XCFRAMEWORK_DIR="$VENDOR_DIR/mlx.xcframework"
INSTALLED_MARKER="$VENDOR_DIR/.installed-sha256"

if [ ! -f "$PINS_FILE" ]; then
    echo "error: $PINS_FILE missing — cannot resolve which xcframework to fetch" >&2
    exit 1
fi

read_pin() {
    # Reads `key=value` from UPSTREAM_VERSIONS.txt, ignoring comments/blank lines.
    grep "^$1=" "$PINS_FILE" | head -n1 | cut -d= -f2- | tr -d '[:space:]'
}

RELEASE_TAG=$(read_pin release)
EXPECTED_SHA=$(read_pin sha256)

if [ -z "$RELEASE_TAG" ] || [ -z "$EXPECTED_SHA" ]; then
    echo "error: UPSTREAM_VERSIONS.txt must define both 'release=' and 'sha256=' keys" >&2
    exit 1
fi

if [ "$RELEASE_TAG" = "unpublished" ] || [ "$EXPECTED_SHA" = "unpublished" ]; then
    cat >&2 <<EOF
error: MLX xcframework release is still 'unpublished' in $PINS_FILE.

Build the artifact first by triggering the GitHub Actions workflow:
    .github/workflows/build-mlx-xcframework.yml

Either bump and push 'vendor/mlx-apple/UPSTREAM_VERSIONS.txt' on master, or
cut a 'mlx-v<version>' tag. Once the release attaches the xcframework zip,
update 'release=' and 'sha256=' in UPSTREAM_VERSIONS.txt and re-run this script.
EOF
    exit 1
fi

# Strip an optional leading 'mlx-v' from the release tag to derive the version
# portion of the asset filename. The workflow names assets 'mlx-<version>.xcframework.zip'.
VERSION="${RELEASE_TAG#mlx-v}"
ASSET_NAME="mlx-${VERSION}.xcframework.zip"
ASSET_URL="https://github.com/${RELEASE_REPO}/releases/download/${RELEASE_TAG}/${ASSET_NAME}"

# Idempotency short-circuit.
if [ "$FORCE" != true ] && [ -d "$XCFRAMEWORK_DIR" ] && [ -f "$INSTALLED_MARKER" ]; then
    INSTALLED_SHA=$(tr -d '[:space:]' < "$INSTALLED_MARKER")
    if [ "$INSTALLED_SHA" = "$EXPECTED_SHA" ]; then
        echo "mlx.xcframework already installed (sha256 ${EXPECTED_SHA})"
        exit 0
    fi
    echo "installed SHA ${INSTALLED_SHA} does not match pin ${EXPECTED_SHA} — refetching"
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

ZIP_PATH="$TMP_DIR/$ASSET_NAME"

echo "==> downloading $ASSET_URL"
if ! curl --fail --location --show-error --silent --output "$ZIP_PATH" "$ASSET_URL"; then
    echo "error: failed to download $ASSET_URL" >&2
    echo "       verify the release exists at https://github.com/${RELEASE_REPO}/releases/tag/${RELEASE_TAG}" >&2
    exit 1
fi

echo "==> verifying SHA256"
if command -v shasum >/dev/null 2>&1; then
    ACTUAL_SHA=$(shasum -a 256 "$ZIP_PATH" | awk '{print $1}')
elif command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_SHA=$(sha256sum "$ZIP_PATH" | awk '{print $1}')
else
    echo "error: neither 'shasum' nor 'sha256sum' is available on PATH" >&2
    exit 1
fi

if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
    echo "error: SHA256 mismatch for $ASSET_NAME" >&2
    echo "       expected: $EXPECTED_SHA" >&2
    echo "       actual:   $ACTUAL_SHA" >&2
    exit 1
fi

echo "==> unpacking into $XCFRAMEWORK_DIR"
rm -rf "$XCFRAMEWORK_DIR"
mkdir -p "$VENDOR_DIR"
(cd "$VENDOR_DIR" && unzip -q "$ZIP_PATH")

if [ ! -d "$XCFRAMEWORK_DIR" ]; then
    echo "error: unzip did not produce the expected mlx.xcframework/ directory" >&2
    exit 1
fi

printf '%s\n' "$EXPECTED_SHA" > "$INSTALLED_MARKER"

echo "mlx.xcframework installed (${RELEASE_TAG}, sha256 ${EXPECTED_SHA})"
