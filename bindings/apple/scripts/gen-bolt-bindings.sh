#!/usr/bin/env bash
# Regenerate the committed boltffi-generated Apple artifacts:
#   bindings/apple/Sources/Xybrid/xybrid_bolt.swift  (Swift bindings)
#   bindings/apple/include/xybrid-bolt.h             (C header the Bazel
#                                                     xcframework ships)
#
# The Swift source receives three compatibility transforms:
#   (a) XybridResult's append-only reasoning field defaults to nil and decodes
#       results from the merged tool-calling wire shape, which does not emit
#       that trailing field.
#   (b) appended cloud fallback options default to nil in the public initializer,
#       so existing five-argument Swift construction remains source-compatible.
#   (c) a fallible method taking a slice and returning unit is emitted as
#       `_ = x.withUnsafeBufferPointer { ... throw ... }`, which Swift rejects
#       for the missing `try`. Without this the binding does not compile.
#
# Sibling of tools/scripts/gen_kotlin_bolt.py, gen_python_bolt.py and
# gen_unity_bolt_csharp.py, and like them it has a --check mode so CI can fail
# on drift instead of discovering it when a human next regenerates.
#
# Usage:
#   bindings/apple/scripts/gen-bolt-bindings.sh            # regenerate + write
#   bindings/apple/scripts/gen-bolt-bindings.sh --check    # fail on drift
#
# This script only checks that the committed output matches a fresh generation
# — it does not compile it, and `swiftc -parse` will not catch a type error
# like a missing `try`. CI compiles the wrapper inside the XCFramework build,
# which is the slowest job on the board. To get the same answer in seconds:
#
#   mkdir -p /tmp/xybridmod && cp bindings/apple/include/xybrid-bolt.h /tmp/xybridmod/
#   printf 'module XybridFFI {\n  header "xybrid-bolt.h"\n  export *\n}\n' \
#       > /tmp/xybridmod/module.modulemap
#   swiftc -typecheck -target arm64-apple-macos13.0 \
#       -I /tmp/xybridmod bindings/apple/Sources/Xybrid/*.swift
set -euo pipefail

PINNED_BOLTFFI="0.30.1"

check=0
case "${1:-}" in
    --check) check=1 ;;
    "") ;;
    *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

if ! command -v boltffi >/dev/null 2>&1; then
    echo "error: \`boltffi\` CLI not found. Install the pinned version:" >&2
    echo "  cargo install boltffi_cli --version $PINNED_BOLTFFI --locked" >&2
    exit 1
fi
boltffi_version="$(boltffi --version)"
case "$boltffi_version" in
    *"$PINNED_BOLTFFI"*) ;;
    *) echo "warning: expected boltffi $PINNED_BOLTFFI, got '$boltffi_version'." \
            "Generated output may differ from the committed sources." >&2 ;;
esac

repo_root="$(git -C "$(cd "$(dirname "$0")" && pwd)" rev-parse --show-toplevel)"
bolt_dir="$repo_root/crates/xybrid-bolt"

(cd "$bolt_dir" && boltffi generate swift --deny-skipped -q)

swift_src="$bolt_dir/dist/apple/Sources/XybridBoltBoltFFI.swift"
header_src="$bolt_dir/dist/apple/Sources/boltffi.h"

swift_dest="$repo_root/bindings/apple/Sources/Xybrid/xybrid_bolt.swift"
header_dest="$repo_root/bindings/apple/include/xybrid-bolt.h"

# Post-process into a staging dir so --check compares without touching the tree.
stage_dir="$(mktemp -d)"
trap 'rm -rf "$stage_dir"' EXIT
swift_staged="$stage_dir/xybrid_bolt.swift"

python3 - "$swift_src" "$swift_staged" <<'PY'
import re
import sys
from pathlib import Path

source_path, destination_path = map(Path, sys.argv[1:])
source = source_path.read_text()

# Transform (b): preserve construction with the original five run options.
options_start = source.index("public struct XybridRunOptions:")
options_end = source.index("\npublic struct ", options_start + 1)
options = source[options_start:options_end]
for name in ("cloudProvider", "cloudModel", "cloudGatewayUrl"):
    original = f"        {name}: String?"
    if options.count(original) != 1:
        raise SystemExit(f"error: expected one XybridRunOptions {name} initializer parameter")
    options = options.replace(original, f"        {name}: String? = nil", 1)
source = source[:options_start] + options + source[options_end:]

initializer = "        reasoningContent: String?\n    ) {"
if source.count(initializer) != 1:
    raise SystemExit("error: expected one XybridResult reasoning initializer parameter")
source = source.replace(
    initializer,
    "        reasoningContent: String? = nil\n    ) {",
)

decoder = '''    @inlinable static func decode(from reader: inout WireReader) -> XybridResult {
        XybridResult(
            envelope: XybridEnvelope.decode(from: &reader),
            outputType: XybridOutputType(rawValue: reader.readI32())!,
            modelId: reader.readString(),
            latencyMs: reader.readU32(),
            executionTarget: XybridExecutionTarget(rawValue: reader.readI32())!,
            metrics: XybridInferenceMetrics.decode(from: &reader),
            toolCalls: reader.readArray { reader in XybridToolCall.decode(from: &reader) },
            reasoningContent: reader.readOptional { reader in reader.readString() }
        )
    }
'''
replacement = '''    @inlinable static func decode(from reader: inout WireReader) -> XybridResult {
        let envelope = XybridEnvelope.decode(from: &reader)
        let outputType = XybridOutputType(rawValue: reader.readI32())!
        let modelId = reader.readString()
        let latencyMs = reader.readU32()
        let executionTarget = XybridExecutionTarget(rawValue: reader.readI32())!
        let metrics = XybridInferenceMetrics.decode(from: &reader)
        let toolCalls = reader.readArray { reader in XybridToolCall.decode(from: &reader) }
        let reasoningContent = reader.position < reader.data.count
            ? reader.readOptional { reader in reader.readString() }
            : envelope.metadata.first { $0.key == "reasoning_content" }?.value
        return XybridResult(
            envelope: envelope,
            outputType: outputType,
            modelId: modelId,
            latencyMs: latencyMs,
            executionTarget: executionTarget,
            metrics: metrics,
            toolCalls: toolCalls,
            reasoningContent: reasoningContent
        )
    }
'''
if source.count(decoder) != 1:
    raise SystemExit("error: expected one generated XybridResult decoder")
source = source.replace(decoder, replacement)

# Transform (c): mark a throwing slice call with `try`.
#
# For a fallible method whose only parameter is a slice and whose result is
# unit, boltffi 0.30.1 emits the call as the function's first statement:
#
#     public func feed(samples: [Float]) throws {
#         _ = samples.withUnsafeBufferPointer { buffer in
#             ...
#             throw ...
#         }
#     }
#
# The closure throws, so `withUnsafeBufferPointer` is itself a throwing call
# and Swift rejects it without `try` ("call can throw but is not marked with
# 'try'"). `_ =` on a Void result is redundant besides, so `try` replaces it.
# This is a generator bug, not a style preference — the binding does not
# compile at all without the fix.
THROWING_SLICE_CALL = re.compile(
    r"^(    public func [^\n]*\bthrows\b[^\n]*\{\n)(        )_ = ",
    re.MULTILINE,
)
source, marked = THROWING_SLICE_CALL.subn(r"\1\2try ", source)

# Nothing may be left that the compiler would reject the same way. A miss here
# means the emitted shape moved and the pattern above needs re-reading.
for match in re.finditer(r"^ +_ = .*\.withUnsafe\w*BufferPointer \{", source, re.MULTILINE):
    raise SystemExit(
        "error: an untransformed `_ = ...BufferPointer {` remains, which will "
        f"not compile if its closure throws: {match.group(0).strip()}"
    )

if marked == 0:
    # Either boltffi fixed this upstream or no fallible slice method is
    # exported any more. Both want a human to re-read the transform before it
    # silently no-ops.
    print(
        "warning: no throwing slice call needed a `try` — verify the binding "
        "still compiles and drop this transform if boltffi now emits it.",
        file=sys.stderr,
    )

destination_path.write_text(source)
PY

if [ "$check" -eq 1 ]; then
    stale=""
    cmp -s "$swift_staged" "$swift_dest" ||
        stale="$stale  - bindings/apple/Sources/Xybrid/xybrid_bolt.swift"$'\n'
    cmp -s "$header_src" "$header_dest" ||
        stale="$stale  - bindings/apple/include/xybrid-bolt.h"$'\n'
    if [ -n "$stale" ]; then
        {
            echo "error: generated Apple bindings are out of date:"
            printf '%s' "$stale"
            echo "Run: bindings/apple/scripts/gen-bolt-bindings.sh"
        } >&2
        exit 1
    fi
    echo "Apple bolt bindings are up to date"
    exit 0
fi

cp "$swift_staged" "$swift_dest"
cp "$header_src" "$header_dest"

echo "regenerated: bindings/apple/Sources/Xybrid/xybrid_bolt.swift"
echo "regenerated: bindings/apple/include/xybrid-bolt.h"
