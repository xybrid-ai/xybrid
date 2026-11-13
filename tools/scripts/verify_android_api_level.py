#!/usr/bin/env python3
"""Check the Android API level that linked Android .so files were built for.

An NDK link takes crtbegin_so.o from the sysroot's usr/lib/<triple>/<api>/,
and that object carries a `.note.android.ident` ELF note whose description
starts with the API level as a little-endian int32 (every Android ABI is
little-endian). So the note records which toolchain linked the library:

- the Kotlin AAR's arm64-v8a libraries (libxybrid_bolt.so, libllama.so,
  libggml*.so, libmtmd.so): 29, from @androidndk_api29 (Android 10);
- every other Bazel Android native (the AAR's armeabi-v7a and x86_64 slices,
  the Flutter precompiled libraries, the Unity plugin): 28, from @androidndk.

A library linked for a higher level than its SDK promises may not load on the
older devices (native ELF TLS, newer libc symbols). Prebuilt libraries we only
bundle (libonnxruntime.so, libc++_shared.so) carry their vendor's level and
are not checked.

The note is read from `readelf -W -n` text; GNU readelf and llvm-readelf both
parse here. Every failure is reported, not just the first.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable

IDENT_SECTION = ".note.android.ident"
NOTE_OWNER = "Android"

# "Displaying notes found in: <section>" starts each section's block in both
# GNU and LLVM readelf; a block without section headers says "found at file
# offset ..." instead and is not used here.
_SECTION_RE = re.compile(r"^\s*Displaying notes found (?:in: (?P<section>\S+)|at )")
# The hex bytes after "description data:". GNU prints them on the owner line,
# LLVM on the next one.
_DATA_RE = re.compile(r"description data:\s*(?P<hex>(?:[0-9a-fA-F]{2}\s*)+)")


def parse_api_levels(text: str) -> list[int]:
    """API levels of every Android note in `.note.android.ident`, in order."""
    levels = []
    in_ident = False
    in_android_note = False
    for line in text.splitlines():
        section = _SECTION_RE.match(line)
        if section:
            in_ident = section.group("section") == IDENT_SECTION
            in_android_note = False
            continue
        if not in_ident:
            continue
        fields = line.split()
        # An owner row: "<owner> 0x<data size> <type...>".
        if len(fields) >= 2 and fields[1].startswith("0x"):
            in_android_note = fields[0] == NOTE_OWNER
        if not in_android_note:
            continue
        data = _DATA_RE.search(line)
        if data:
            desc = bytes.fromhex("".join(data.group("hex").split()))
            if len(desc) >= 4:
                levels.append(int.from_bytes(desc[:4], "little"))
            in_android_note = False
    return levels


def check(path: str, text: str, api: int) -> tuple[str | None, str]:
    """Return (failure, summary) for one library's `readelf -W -n` text."""
    levels = parse_api_levels(text)
    if not levels:
        return (
            f"{path}: no {IDENT_SECTION} note in readelf output; cannot tell "
            "which Android API level it was linked for",
            "",
        )
    wrong = sorted({level for level in levels if level != api})
    if wrong:
        found = ", ".join(str(level) for level in sorted(set(levels)))
        return (
            f"{path}: linked for Android API {found}, expected {api} "
            "(wrong NDK toolchain? see //bazel/toolchains/android)",
            "",
        )
    return None, f"OK: {path}: Android API {api}"


def readelf_notes(readelf: str, path: str) -> str:
    return subprocess.run(
        [readelf, "-W", "-n", path], capture_output=True, text=True, check=True
    ).stdout


def check_all(
    paths: list[str], api: int, read: Callable[[str], str]
) -> tuple[list[str], list[str]]:
    """Check every library; returns (failures, summaries)."""
    failures, summaries = [], []
    for path in paths:
        try:
            text = read(path)
        except subprocess.CalledProcessError as error:
            failures.append(f"{path}: readelf failed: {(error.stderr or '').strip()}")
            continue
        failure, summary = check(path, text, api)
        if failure:
            failures.append(failure)
        else:
            summaries.append(summary)
    return failures, summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--api", type=int, required=True,
        help="the Android API level every library must have been linked for",
    )
    parser.add_argument(
        "--readelf", default="readelf",
        help="GNU readelf or llvm-readelf (default: %(default)s)",
    )
    parser.add_argument("libs", nargs="+", help="linked Android .so files")
    args = parser.parse_args(argv)

    try:
        failures, summaries = check_all(
            args.libs, args.api, lambda path: readelf_notes(args.readelf, path)
        )
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for summary in summaries:
        print(summary)
    if failures:
        print(f"\n{len(failures)} problem(s):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
