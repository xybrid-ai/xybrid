#!/usr/bin/env bash
# iOS checks that need no React Native install, no pods and no Rust build:
#
#   1. Type-check the whole Swift half of the module (ios/*.swift) against
#      the Swift SDK it ships with (bindings/apple/Sources/Xybrid) for the iOS
#      simulator — the gate issue #589 asked for. React is stubbed with the
#      two block types the module uses; XybridFFI is the committed C header.
#   2. Run ios/XybridCodec.swift against the real bolt records on this Mac.
#
# Needs Xcode. Usage: scripts/test-ios.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLE="$HERE/../apple"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/XybridFFI" "$WORK/React"
cp "$APPLE/include/xybrid-bolt.h" "$WORK/XybridFFI/"
printf 'module XybridFFI {\n  header "xybrid-bolt.h"\n  export *\n}\n' > "$WORK/XybridFFI/module.modulemap"
cat > "$WORK/React/React.h" <<'H'
#import <Foundation/Foundation.h>
typedef void (^RCTPromiseResolveBlock)(id _Nullable result);
typedef void (^RCTPromiseRejectBlock)(NSString *_Nullable code, NSString *_Nullable message, NSError *_Nullable error);
H
printf 'module React {\n  header "React.h"\n  export *\n}\n' > "$WORK/React/module.modulemap"

SDK_SOURCES=("$APPLE/Sources/Xybrid/Xybrid.swift" "$APPLE/Sources/Xybrid/xybrid_bolt.swift")

echo "==> Type-checking ios/*.swift for the iOS simulator"
xcrun --sdk iphonesimulator swiftc -typecheck \
  -target arm64-apple-ios16.0-simulator \
  -swift-version 5 -warnings-as-errors \
  -module-name react_native_xybrid \
  -I "$WORK/XybridFFI" -I "$WORK/React" \
  "${SDK_SOURCES[@]}" "$HERE"/ios/*.swift

echo "==> Running the codec tests on this Mac"
# Only the pure codec is exercised, so the Rust symbols the SDK references
# are left undefined and never called.
xcrun swiftc -swift-version 5 \
  -module-name XybridCodecTests \
  -I "$WORK/XybridFFI" \
  -Xlinker -undefined -Xlinker dynamic_lookup \
  -o "$WORK/codec-tests" \
  "${SDK_SOURCES[@]}" "$HERE/ios/XybridCodec.swift" "$HERE/tests/swift/main.swift"
"$WORK/codec-tests"
