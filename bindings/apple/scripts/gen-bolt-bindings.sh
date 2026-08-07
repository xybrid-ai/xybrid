#!/usr/bin/env bash
# Regenerate the committed boltffi-generated Apple artifacts:
#   bindings/apple/Sources/Xybrid/xybrid_bolt.swift  (Swift bindings)
#   bindings/apple/include/xybrid-bolt.h             (C header the Bazel
#                                                     xcframework ships)
#
# No post-processing: boltffi >= 0.26 backtick-escapes Swift keywords itself
# (`XybridTelemetryConfig::init`), so the `public func init()` fix this script
# applied to 0.25.3 output is gone. The C header now comes out beside the Swift
# sources from `generate swift`; 0.29 has no separate `generate header` target.
set -euo pipefail

repo_root="$(git -C "$(cd "$(dirname "$0")" && pwd)" rev-parse --show-toplevel)"
bolt_dir="$repo_root/crates/xybrid-bolt"

(cd "$bolt_dir" && boltffi generate swift -q)

swift_src="$bolt_dir/dist/apple/Sources/XybridBoltBoltFFI.swift"
header_src="$bolt_dir/dist/apple/Sources/boltffi.h"

cp "$swift_src" "$repo_root/bindings/apple/Sources/Xybrid/xybrid_bolt.swift"
cp "$header_src" "$repo_root/bindings/apple/include/xybrid-bolt.h"

echo "regenerated: bindings/apple/Sources/Xybrid/xybrid_bolt.swift"
echo "regenerated: bindings/apple/include/xybrid-bolt.h"
