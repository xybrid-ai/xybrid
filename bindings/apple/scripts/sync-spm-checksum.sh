#!/usr/bin/env bash
# =============================================================================
# sync-spm-checksum.sh
# =============================================================================
# Computes the SHA-256 checksum of an XybridFFI.xcframework zip and patches
# `xybridFFIChecksum` in the root Package.swift so SPM consumers pinned at
# the release tag can resolve the binary target.
#
# Usage:
#   bindings/apple/scripts/sync-spm-checksum.sh <path-to-zip>
#   bindings/apple/scripts/sync-spm-checksum.sh --check <path-to-zip>
#
# --check exits non-zero (without modifying Package.swift) if the manifest's
# checksum does NOT match the zip — used in CI to fail fast when the
# manifest at a tag is stale.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PACKAGE_FILE="${REPO_ROOT}/Package.swift"

CHECK_ONLY=false
ZIP_PATH=""

while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=true; shift ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        -*) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
        *)
            if [ -z "$ZIP_PATH" ]; then
                ZIP_PATH="$1"
            else
                echo "ERROR: multiple zip paths given" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$ZIP_PATH" ]; then
    echo "ERROR: zip path required" >&2
    sed -n '2,15p' "$0"
    exit 1
fi

if [ ! -f "$ZIP_PATH" ]; then
    echo "ERROR: zip not found: $ZIP_PATH" >&2
    exit 1
fi

if [ ! -f "$PACKAGE_FILE" ]; then
    echo "ERROR: Package.swift not found at $PACKAGE_FILE" >&2
    exit 1
fi

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        sha256sum "$1" | awk '{print $1}'
    fi
}

NEW_SUM="$(sha256_of "$ZIP_PATH")"
CURRENT_SUM="$(grep -E '^let xybridFFIChecksum' "$PACKAGE_FILE" | sed -E 's/.*"([0-9a-f]+)".*/\1/')"

echo "zip:             $ZIP_PATH"
echo "computed sha256: $NEW_SUM"
echo "manifest sha256: $CURRENT_SUM"

if [ "$NEW_SUM" = "$CURRENT_SUM" ]; then
    echo "Checksum already up to date."
    exit 0
fi

if [ "$CHECK_ONLY" = true ]; then
    echo "ERROR: Package.swift checksum is stale. Run without --check to update." >&2
    exit 1
fi

# Replace the checksum literal in the Package.swift line. The literal is
# 64 hex characters, matched specifically inside the xybridFFIChecksum line
# so we don't accidentally rewrite anything else.
python3 - "$PACKAGE_FILE" "$NEW_SUM" <<'PY'
import re, sys
path, new_sum = sys.argv[1], sys.argv[2]
with open(path) as f:
    src = f.read()
new_src, n = re.subn(
    r'(let xybridFFIChecksum\s*=\s*")[0-9a-f]{64}(")',
    r'\g<1>' + new_sum + r'\g<2>',
    src,
)
if n != 1:
    sys.stderr.write(f"ERROR: expected 1 xybridFFIChecksum match, found {n}\n")
    sys.exit(1)
with open(path, 'w') as f:
    f.write(new_src)
print(f"Updated xybridFFIChecksum in {path}")
PY
