#!/usr/bin/env python3
"""Check exact generated JNI exports in a linked (or stripped) Android ELF."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

JNI_PREFIX = "Java_ai_xybrid_Native_"
BOLT_SYMBOL = "boltffi_function_xybrid_bolt_set_binding"


def verify(glue: str, dynamic_symbols: str) -> int:
    expected = set(re.findall(r"\bJNICALL\s+(Java_ai_xybrid_Native_\w+)\s*\(", glue))
    if not expected:
        raise ValueError("generated JNI source contains no entry points")

    exported = set()
    for line in dynamic_symbols.splitlines():
        fields = line.split()
        # readelf --dyn-syms --wide: Num Value Size Type Bind Vis Ndx Name
        if (len(fields) >= 8 and fields[0].rstrip(":").isdigit()
                and fields[3] == "FUNC" and fields[4] in {"GLOBAL", "WEAK"}
                and fields[5] in {"DEFAULT", "PROTECTED"} and fields[6] != "UND"):
            exported.add(fields[7].split("@", 1)[0])

    actual = {symbol for symbol in exported if symbol.startswith(JNI_PREFIX)}
    missing, unexpected = expected - actual, actual - expected
    problems = []
    if missing:
        problems.append("missing JNI exports: " + ", ".join(sorted(missing)))
    if unexpected:
        problems.append("unexpected JNI exports: " + ", ".join(sorted(unexpected)))
    if BOLT_SYMBOL not in exported:
        problems.append("missing underlying Bolt C ABI: " + BOLT_SYMBOL)
    if problems:
        raise ValueError("\n".join(problems))
    return len(expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--glue", type=Path, required=True)
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    try:
        symbols = subprocess.run(
            [args.readelf, "--dyn-syms", "--wide", str(args.library)],
            capture_output=True, text=True, check=True,
        ).stdout
        count = verify(args.glue.read_text(), symbols)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {args.library}: {error}", file=sys.stderr)
        return 1
    print(f"{args.library}: all {count} generated JNI exports match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
