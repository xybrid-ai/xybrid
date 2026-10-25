#!/usr/bin/env python3
"""Generate private companion metadata from an independently validated pin."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


DOMAIN = b"xybrid.android-tts-companion.v1\0"
TARGET = "aarch64-linux-android"


def load_verifier(path: Path):
    spec = importlib.util.spec_from_file_location("_xybrid_companion_verifier", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the pin verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def contract(verifier, manifest: Path) -> tuple[bytes, dict]:
    pin = verifier.select_pin(TARGET, manifest)
    entry = pin.entry
    for field, expected in {
        "receipt_schema": 5,
        "cpu": "baseline",
        "deployment_minimum": "29",
        "system_libraries": ["c", "m", "dl"],
    }.items():
        verifier.expect(entry, field, expected, "companion pin")
    # select_pin also validates the exact selected models, embed-only ABI1,
    # slice/archive naming, archive SHA256 and immutable notice map. Bind the
    # entire actual entry so any pin/notice/release change changes this ID.
    record = {"release_tag": pin.release_tag, "entry": entry}
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(DOMAIN + encoded).digest(), record


def header(identity: bytes) -> str:
    if len(identity) != 32:
        raise ValueError("invalid companion identity size")
    values = ", ".join(f"0x{value:02x}" for value in identity)
    return (
        "/* Generated from the reviewed schema-5 selected Android pin. */\n"
        "#ifndef XYBRID_TTS_COMPANION_METADATA_H\n"
        "#define XYBRID_TTS_COMPANION_METADATA_H\n"
        "#define XYBRID_TTS_ENGINE_ABI 1u\n"
        "#define XYBRID_TTS_ANDROID_API 29u\n"
        "#define XYBRID_TTS_MODELS 3u\n"
        f"#define XYBRID_TTS_CONTRACT_ID_BYTES {{{values}}}\n"
        "#endif\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        verifier = load_verifier(args.verifier)
        identity, _ = contract(verifier, args.manifest)
        args.output.write_text(header(identity), encoding="ascii")
    except Exception:
        # Do not disclose private manifest/receipt values or filesystem paths,
        # including through a verifier traceback.
        print("companion metadata: a valid notice-bearing selected Android pin is required", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
