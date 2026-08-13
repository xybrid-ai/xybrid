#!/usr/bin/env python3
"""Regenerate the committed BoltFFI Kotlin binding for the Android SDK.

Sibling of `bindings/apple/scripts/gen-bolt-bindings.sh`, `gen_unity_bolt_csharp.py`
and `gen_python_bolt.py`. Until now Kotlin was the one binding refreshed by
hand — `boltffi generate kotlin` followed by a copy — which meant the one
post-process it needs was invisible and got lost on every regeneration.

The post-processes:

  boltffi 0.29 emits each `XybridError` variant as a data class holding its
  payload, and `XybridError` extends `Exception`. For the fourteen variants
  whose payload field is `message`, that collides with `Throwable.message`:

      error: 'message' hides member of supertype 'Throwable' and needs an
             'override' modifier.

  Kotlin narrowing `String?` to `String` on an override is legal, so adding the
  modifier is both sufficient and safe. Without it the binding does not
  compile at all — this is a generator bug, not a style preference, so the
  transform is applied deterministically rather than by hand.

  `XybridResult` gained an append-only `reasoning_content` wire field after
  tool calling landed. Its decoder probes for that tail before reading it and
  falls back to the existing reasoning metadata when the typed tail is absent.

Usage:
    python3 tools/scripts/gen_kotlin_bolt.py            # regenerate + write
    python3 tools/scripts/gen_kotlin_bolt.py --check    # fail on drift
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOLT_DIR = REPO_ROOT / "crates" / "xybrid-bolt"
RAW_FILE = BOLT_DIR / "dist" / "android" / "kotlin" / "ai" / "xybrid" / "XybridBolt.kt"
DEST_FILE = REPO_ROOT / "bindings" / "kotlin" / "src" / "main" / "kotlin" / "ai" / "xybrid" / "XybridBolt.kt"
PINNED_BOLTFFI = "0.29.3"

# `data class Foo(<params>) : XybridError() {` — payload lists carry no nested
# parentheses, so stopping at the first `)` is exact.
_ERROR_VARIANT = re.compile(r"data class \w+\([^)]*\) : XybridError\(\) \{")
_MESSAGE_FIELD = re.compile(r"\bval message:")


def check_boltffi_version() -> None:
    try:
        out = subprocess.run(["boltffi", "--version"], capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        sys.exit(
            "error: `boltffi` CLI not found. Install the pinned version:\n"
            f"  cargo install boltffi_cli --version {PINNED_BOLTFFI} --locked\n"
            f"({exc})"
        )
    if PINNED_BOLTFFI not in out:
        print(
            f"warning: expected boltffi {PINNED_BOLTFFI}, got '{out}'. "
            "Generated output may differ from the committed sources.",
            file=sys.stderr,
        )


def _add_message_override(source: str) -> tuple[str, int]:
    """Add the `override` modifier to every `message` payload field."""

    count = 0

    def fix_variant(match: re.Match[str]) -> str:
        nonlocal count
        patched, replaced = _MESSAGE_FIELD.subn("override val message:", match.group(0))
        count += replaced
        return patched

    return _ERROR_VARIANT.sub(fix_variant, source), count


def _add_result_wire_compatibility(source: str) -> str:
    reader_target = """internal class WireReader(private val bytes: ByteArray) {
    private var position = 0

    fun readBool(): Boolean = readI8() != 0.toByte()
"""
    reader_replacement = """internal class WireReader(private val bytes: ByteArray) {
    private var position = 0

    fun hasRemaining(): Boolean = position < bytes.size

    fun readBool(): Boolean = readI8() != 0.toByte()
"""
    if source.count(reader_target) != 1:
        sys.exit("error: expected one Kotlin WireReader declaration")
    source = source.replace(reader_target, reader_replacement, 1)

    decoder_target = """        internal fun fromReader(reader: WireReader): XybridResult {
            return XybridResult(
                XybridEnvelope.fromReader(reader),
                XybridOutputType.fromValue(reader.readI32()),
                reader.readString(),
                reader.readU32(),
                XybridExecutionTarget.fromValue(reader.readI32()),
                XybridInferenceMetrics.fromReader(reader),
                reader.readSequence({ reader -> XybridToolCall.fromReader(reader) }),
                reader.readOptionalValue({ reader -> reader.readString() })
            )
        }
"""
    decoder_replacement = """        internal fun fromReader(reader: WireReader): XybridResult {
            val envelope = XybridEnvelope.fromReader(reader)
            val outputType = XybridOutputType.fromValue(reader.readI32())
            val modelId = reader.readString()
            val latencyMs = reader.readU32()
            val executionTarget = XybridExecutionTarget.fromValue(reader.readI32())
            val metrics = XybridInferenceMetrics.fromReader(reader)
            val toolCalls = reader.readSequence({ reader -> XybridToolCall.fromReader(reader) })
            val reasoningContent = if (reader.hasRemaining()) {
                reader.readOptionalValue({ reader -> reader.readString() })
            } else {
                envelope.metadata.firstOrNull { it.key == "reasoning_content" }?.value
            }
            return XybridResult(
                envelope,
                outputType,
                modelId,
                latencyMs,
                executionTarget,
                metrics,
                toolCalls,
                reasoningContent
            )
        }
"""
    if source.count(decoder_target) != 1:
        sys.exit("error: expected one generated Kotlin XybridResult decoder")
    return source.replace(decoder_target, decoder_replacement, 1)


def render() -> str:
    subprocess.run(["boltffi", "generate", "kotlin"], cwd=BOLT_DIR, check=True)
    if not RAW_FILE.is_file():
        sys.exit(f"error: boltffi produced no Kotlin source at {RAW_FILE}")

    source, overrides = _add_message_override(RAW_FILE.read_text())
    result_field = "    val reasoningContent: String?\n) {"
    if source.count(result_field) != 1:
        sys.exit("error: expected one XybridResult reasoning constructor field")
    source = source.replace(
        result_field,
        "    val reasoningContent: String? = null\n) {",
    )
    source = _add_result_wire_compatibility(source)
    if overrides == 0:
        # Either boltffi fixed this upstream or the error shape moved. Both
        # want a human to re-read the transform before it silently no-ops.
        print(
            "warning: no `message` payload field needed an override — verify "
            "the binding still compiles and drop this transform if boltffi "
            "now emits it.",
            file=sys.stderr,
        )
    return source if source.endswith("\n") else source + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed binding differs from a fresh one")
    args = parser.parse_args()

    check_boltffi_version()
    rendered = render()

    if args.check:
        if not DEST_FILE.exists() or DEST_FILE.read_text() != rendered:
            print(
                f"error: {DEST_FILE.relative_to(REPO_ROOT)} is out of date.\n"
                "Run: python3 tools/scripts/gen_kotlin_bolt.py",
                file=sys.stderr,
            )
            return 1
        print("Kotlin binding up to date")
        return 0

    DEST_FILE.write_text(rendered)
    print(f"Wrote {DEST_FILE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
