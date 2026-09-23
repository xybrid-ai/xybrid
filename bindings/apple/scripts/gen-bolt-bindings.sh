#!/usr/bin/env bash
# Regenerate the committed boltffi-generated Apple artifacts:
#   bindings/apple/Sources/Xybrid/xybrid_bolt.swift  (Swift bindings)
#   bindings/apple/include/xybrid-bolt.h             (C header the Bazel
#                                                     xcframework ships)
#
# The Swift source receives one compatibility transform: XybridResult's
# append-only reasoning field defaults to nil and decodes results from the
# merged tool-calling wire shape, which does not emit that trailing field.
#
# Sibling of tools/scripts/gen_kotlin_bolt.py, gen_python_bolt.py and
# gen_unity_bolt_csharp.py, and like them it has a --check mode so CI can fail
# on drift instead of discovering it when a human next regenerates.
#
# Usage:
#   bindings/apple/scripts/gen-bolt-bindings.sh            # regenerate + write
#   bindings/apple/scripts/gen-bolt-bindings.sh --check    # fail on drift
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
import sys
from pathlib import Path

source_path, destination_path = map(Path, sys.argv[1:])
source = source_path.read_text()

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
destination_path.write_text(source.replace(decoder, replacement))
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
