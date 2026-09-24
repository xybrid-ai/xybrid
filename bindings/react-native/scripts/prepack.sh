#!/usr/bin/env bash
# Runs before `npm pack` / `npm publish`: builds lib/ and stages the files the
# tarball ships but git does not track — the Swift SDK sources the pod
# compiles, and the license. The XCFramework is never packed: `pod install`
# fetches the release asset pinned in package.json (ios/xybrid_natives.rb).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

npm run build

mkdir -p ios/XybridSwift
cp "$REPO/bindings/apple/Sources/Xybrid/Xybrid.swift" \
   "$REPO/bindings/apple/Sources/Xybrid/xybrid_bolt.swift" ios/XybridSwift/
cp "$REPO/LICENSE" LICENSE
