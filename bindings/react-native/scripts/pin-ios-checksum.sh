#!/usr/bin/env bash
# Pins the SHA-256 of the release XCFramework zip in package.json
# (`xybrid.iosXcframeworkSha256`), which ios/xybrid_natives.rb verifies before
# using the asset it downloads at `pod install`. The React Native twin of
# bindings/apple/scripts/sync-spm-checksum.sh; release-prep.yml runs both on
# the same zip.
#
# Usage:
#   bindings/react-native/scripts/pin-ios-checksum.sh <path-to-zip>
#   bindings/react-native/scripts/pin-ios-checksum.sh --check <path-to-zip>
#
# --check exits non-zero, changing nothing, when the pinned checksum does not
# match the zip.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="$HERE/package.json"

CHECK=false
if [ "${1:-}" = "--check" ]; then
  CHECK=true
  shift
fi
ZIP="${1:?usage: pin-ios-checksum.sh [--check] <XybridFFI.xcframework.zip>}"
[ -f "$ZIP" ] || { echo "ERROR: zip not found: $ZIP" >&2; exit 1; }

SHA=$(shasum -a 256 "$ZIP" | awk '{print $1}')

python3 - "$PACKAGE" "$SHA" "$CHECK" <<'PY'
import json
import sys

path, sha, check = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
with open(path) as handle:
    package = json.load(handle)
pinned = package.get("xybrid", {}).get("iosXcframeworkSha256", "")
if check:
    if pinned != sha:
        sys.exit(f"ERROR: package.json pins {pinned or '<nothing>'}, the zip is {sha}")
    print(f"OK: package.json pins {sha}")
    sys.exit(0)
package.setdefault("xybrid", {})["iosXcframeworkSha256"] = sha
with open(path, "w") as handle:
    json.dump(package, handle, indent=2)
    handle.write("\n")
print(f"Pinned {sha} in {path}")
PY
