#!/usr/bin/env bash
# Regenerate the committed boltffi-generated Apple artifacts:
#   bindings/apple/Sources/Xybrid/xybrid_bolt.swift  (Swift bindings)
#   bindings/apple/include/xybrid-bolt.h             (C header the Bazel
#                                                     xcframework ships)
#
# boltffi 0.25.3 emits `public func init()` for Rust handle methods named
# `init` (XybridTelemetryConfig::init) — `init` is a Swift keyword, so the
# raw output does not compile. Mirroring the Unity generator post-process
# (tools/scripts/gen_unity_bolt_csharp.py), this script applies one
# deterministic fix: backtick-escape the method name. Drop the fix once the
# boltffi pin grows keyword escaping (boltffi >= 0.26).
set -euo pipefail

repo_root="$(git -C "$(cd "$(dirname "$0")" && pwd)" rev-parse --show-toplevel)"
bolt_dir="$repo_root/crates/xybrid-bolt"

(cd "$bolt_dir" && boltffi generate swift -q && boltffi generate header -q)

swift_src="$bolt_dir/dist/apple/Sources/Xybrid-boltBoltFFI.swift"
header_src="$bolt_dir/dist/apple/include/xybrid-bolt.h"

sed -e 's/public func init()/public func `init`()/' "$swift_src" \
  > "$repo_root/bindings/apple/Sources/Xybrid/xybrid_bolt.swift"
cp "$header_src" "$repo_root/bindings/apple/include/xybrid-bolt.h"

echo "regenerated: bindings/apple/Sources/Xybrid/xybrid_bolt.swift"
echo "regenerated: bindings/apple/include/xybrid-bolt.h"
