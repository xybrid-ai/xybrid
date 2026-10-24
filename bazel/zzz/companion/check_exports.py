#!/usr/bin/env python3
"""Audit a future actual Android companion with an explicit llvm-readelf tool."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


GETTER = "xybrid_tts_get_api_v1"
SONAME = "libxybrid_zzz_tts.so"
ALLOWED_NEEDED = {"libc.so", "libm.so", "libdl.so"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(header: str, symbols: str, dynamic: str, programs: str) -> None:
    require(re.search(r"Class:\s+ELF64\b", header) is not None, "ELF64 required")
    require(re.search(r"Data:\s+2's complement, little endian\b", header) is not None, "little-endian ELF required")
    require(re.search(r"Type:\s+DYN\b", header) is not None, "shared object required")
    require(re.search(r"Machine:\s+AArch64\b", header) is not None, "Android ARM64 required")
    require("Symbol table '.dynsym'" in symbols, "dynamic symbol table required")
    exports = []
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) < 8 or not re.fullmatch(r"\d+:", fields[0]):
            continue
        _, _, _, kind, binding, visibility, index, name, *rest = fields
        name = name.split("@", 1)[0]
        # Engine APIs must be statically resolved and hidden, never imported
        # or exported by this SDK-owned component. No public engine ABI leaks.
        require(not name.startswith("zzz_"), "engine ABI symbol leakage")
        if binding in ("GLOBAL", "WEAK") and index != "UND":
            require(binding == "GLOBAL" and kind == "FUNC" and visibility == "DEFAULT", "unexpected public symbol type")
            exports.append(name)
    require(exports == [GETTER], "exactly the SDK getter must be exported")
    sonames = re.findall(r"\(SONAME\).*?\[([^\]]+)\]", dynamic)
    require(sonames == [SONAME], "fixed companion SONAME required")
    needed = re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", dynamic)
    require(len(needed) == len(set(needed)) and set(needed) <= ALLOWED_NEEDED,
            "unexpected runtime dependency")
    loads = [line.split() for line in programs.splitlines() if line.strip().startswith("LOAD ")]
    require(bool(loads), "ELF load segments required")
    for fields in loads:
        require(len(fields) >= 8 and all(re.fullmatch(r"0x[0-9a-fA-F]+", value)
                                        for value in fields[1:3]), "invalid ELF load segment")
        require(re.fullmatch(r"0x[0-9a-fA-F]+", fields[-1]) is not None, "invalid ELF alignment")
        alignment = int(fields[-1], 16)
        require(alignment >= 16384 and alignment & (alignment - 1) == 0, "16-KiB load alignment required")
        require(int(fields[1], 16) % alignment == int(fields[2], 16) % alignment,
                "ELF load offset/address must be congruent for its alignment")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readelf", required=True, type=Path)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args(argv)
    try:
        reports = []
        for option in ("--file-header", "--dyn-syms", "--dynamic", "--program-headers"):
            result = subprocess.run([str(args.readelf), option, "--wide", str(args.binary)],
                                    check=True, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=30)
            reports.append(result.stdout)
        audit(*reports)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        reason = str(error) if isinstance(error, ValueError) else "readelf inspection failed"
        print(f"companion audit: {reason}", file=sys.stderr)
        return 1
    print("companion export/SONAME/dependency/alignment audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
