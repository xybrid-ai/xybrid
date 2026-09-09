#!/usr/bin/env bash
# Regenerate the committed boltffi-generated Apple artifacts:
#   bindings/apple/Sources/Xybrid/xybrid_bolt.swift  (Swift bindings)
#   bindings/apple/include/xybrid-bolt.h             (C header the Bazel
#                                                     xcframework ships)
#
# The Swift source receives one compatibility transform: XybridResult's
# append-only reasoning field defaults to nil and decodes results from the
# merged tool-calling wire shape, which does not emit that trailing field.
set -euo pipefail

repo_root="$(git -C "$(cd "$(dirname "$0")" && pwd)" rev-parse --show-toplevel)"
bolt_dir="$repo_root/crates/xybrid-bolt"

(cd "$bolt_dir" && boltffi generate swift -q)

swift_src="$bolt_dir/dist/apple/Sources/XybridBoltBoltFFI.swift"
header_src="$bolt_dir/dist/apple/Sources/boltffi.h"

swift_dest="$repo_root/bindings/apple/Sources/Xybrid/xybrid_bolt.swift"
python3 - "$swift_src" "$swift_dest" <<'PY'
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
source = source.replace(decoder, replacement)

# boltffi 0.29.3 omits `try` when a fallible method encodes `[Float]` through
# `withUnsafeBufferPointer`, making the generated Swift fail typecheck. Keep
# this narrow and counted so a generator fix or a second affected method trips
# the guard instead of silently rewriting an unexpected call site.
fallible_float_buffer = "        _ = samples.withUnsafeBufferPointer { boltffiSamplesBuffer in\n"
if source.count(fallible_float_buffer) != 1:
    raise SystemExit("error: expected one fallible Swift Float buffer call")
source = source.replace(
    fallible_float_buffer,
    "        try samples.withUnsafeBufferPointer { boltffiSamplesBuffer in\n",
)

destination_path.write_text(source)
PY
cp "$header_src" "$repo_root/bindings/apple/include/xybrid-bolt.h"

echo "regenerated: bindings/apple/Sources/Xybrid/xybrid_bolt.swift"
echo "regenerated: bindings/apple/include/xybrid-bolt.h"
