#!/usr/bin/env bash
# =============================================================================
# set-natives-mode.sh
# =============================================================================
# Flips the `useLocalNatives` flag in the root Package.swift between
# local-development mode (uses bindings/apple/XCFrameworks/) and remote mode
# (downloads the published xcframework from a GitHub release).
#
# Usage:
#   bindings/apple/scripts/set-natives-mode.sh --set-local
#   bindings/apple/scripts/set-natives-mode.sh --set-remote
#   bindings/apple/scripts/set-natives-mode.sh --status
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PACKAGE_FILE="${REPO_ROOT}/Package.swift"

if [ ! -f "$PACKAGE_FILE" ]; then
    echo "ERROR: Package.swift not found at $PACKAGE_FILE" >&2
    exit 1
fi

usage() {
    sed -n '2,12p' "$0"
    exit "${1:-0}"
}

case "${1:-}" in
    --set-local)
        if grep -q '^let useLocalNatives = true' "$PACKAGE_FILE"; then
            echo "useLocalNatives is already true — nothing to do."
            exit 0
        fi
        sed -i.bak 's/^let useLocalNatives = false/let useLocalNatives = true/' "$PACKAGE_FILE"
        rm -f "${PACKAGE_FILE}.bak"
        if ! grep -q '^let useLocalNatives = true' "$PACKAGE_FILE"; then
            echo "ERROR: failed to set useLocalNatives = true" >&2
            exit 1
        fi
        echo "Set useLocalNatives = true (local xcframework mode)."
        ;;
    --set-remote)
        if grep -q '^let useLocalNatives = false' "$PACKAGE_FILE"; then
            echo "useLocalNatives is already false — nothing to do."
            exit 0
        fi
        sed -i.bak 's/^let useLocalNatives = true/let useLocalNatives = false/' "$PACKAGE_FILE"
        rm -f "${PACKAGE_FILE}.bak"
        if ! grep -q '^let useLocalNatives = false' "$PACKAGE_FILE"; then
            echo "ERROR: failed to set useLocalNatives = false" >&2
            exit 1
        fi
        echo "Set useLocalNatives = false (remote release-asset mode)."
        ;;
    --status|"")
        grep -E '^let (useLocalNatives|sdkVersion|xybridFFIChecksum)' "$PACKAGE_FILE"
        ;;
    -h|--help)
        usage 0
        ;;
    *)
        echo "ERROR: unknown option: $1" >&2
        usage 1
        ;;
esac
