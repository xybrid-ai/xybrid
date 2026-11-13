#!/usr/bin/env python3
"""Check that Apple static libraries need no newer OS than a podspec declares.

Every Mach-O object records the OS version it was compiled for: `minos` in an
LC_BUILD_VERSION load command, or `version` in an older LC_VERSION_MIN_*
one. The highest value across an archive's members is the oldest OS the
library can run on, and the podspec's `s.platform` must not promise less:
CocoaPods only rejects an app whose deployment target is below the declared
version, so a declaration lower than the objects lets an app build for an OS
the library was not built for.

The Flutter precompiled libraries take their floors from different places:
iOS and the simulator from `--ios_minimum_os` in .bazelrc, macOS from
hermetic-llvm's default (no `--macos_minimum_os` is set), and the bundled
ONNX Runtime from its vendor. This check measures the result instead of
trusting any of them. Every member must also target the podspec's platform
(macOS for `:osx`; iOS or the iOS simulator for `:ios`).

Archives are read directly (BSD and GNU `ar` layouts, thin Mach-O members),
so the check needs no otool and runs on any host. Every kind of failure is
reported, not just the first.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Iterator

AR_MAGIC = b"!<arch>\n"
AR_HEADER_SIZE = 60
# Archive members that are the archive's own symbol or name tables.
AR_INDEX_NAMES = {"/", "//", "/SYM64/", "__.SYMDEF", "__.SYMDEF SORTED",
                  "__.SYMDEF_64", "__.SYMDEF_64 SORTED"}

# mach-o/loader.h: (magic as stored, struct byte order, header size).
MACHO_MAGICS = {
    b"\xcf\xfa\xed\xfe": ("<", 32),  # MH_MAGIC_64, little-endian
    b"\xce\xfa\xed\xfe": ("<", 28),  # MH_MAGIC, little-endian
    b"\xfe\xed\xfa\xcf": (">", 32),  # MH_MAGIC_64, big-endian
    b"\xfe\xed\xfa\xce": (">", 28),  # MH_MAGIC, big-endian
}
LC_BUILD_VERSION = 0x32
# LC_VERSION_MIN_* carries no platform field; the command names it. The
# iPhoneOS one covers the simulator too, so it maps to the iOS family.
LC_VERSION_MIN = {0x24: "macOS", 0x25: "iOS", 0x2F: "tvOS", 0x30: "watchOS"}
# LC_BUILD_VERSION platform ids (mach-o/loader.h PLATFORM_*).
PLATFORMS = {1: "macOS", 2: "iOS", 3: "tvOS", 4: "watchOS", 5: "bridgeOS",
             6: "Mac Catalyst", 7: "iOS simulator", 8: "tvOS simulator",
             9: "watchOS simulator", 10: "DriverKit", 11: "visionOS",
             12: "visionOS simulator"}
# The platforms each podspec `s.platform` symbol may contain.
PODSPEC_PLATFORMS = {"osx": {"macOS"}, "ios": {"iOS", "iOS simulator"}}

# Object names quoted per failure line.
EXAMPLES = 3

_PODSPEC_RE = re.compile(
    r"^\s*s\.platform\s*=\s*:(\w+)\s*,\s*['\"]([0-9.]+)['\"]", re.M
)

Version = tuple[int, int, int]


def parse_version(text: str) -> Version:
    """`"16.0"` -> `(16, 0, 0)`: always three parts, so tuples compare."""
    parts = [int(part) for part in text.split(".")]
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"bad version {text!r}")
    return tuple(parts + [0] * (3 - len(parts)))


def format_version(version: Version) -> str:
    """`(16, 0, 0)` -> `"16.0"`, the way otool and podspecs write it."""
    return ".".join(str(part) for part in (version if version[2] else version[:2]))


def decode_version(packed: int) -> Version:
    """Mach-O packs version X.Y.Z as nibbles xxxx.yy.zz."""
    return (packed >> 16, (packed >> 8) & 0xFF, packed & 0xFF)


def podspec_platform(text: str) -> tuple[str, Version]:
    """The `s.platform = :<os>, '<version>'` declaration of a podspec."""
    match = _PODSPEC_RE.search(text)
    if not match:
        raise ValueError("no `s.platform = :<os>, '<version>'` line")
    if match.group(1) not in PODSPEC_PLATFORMS:
        raise ValueError(f"unsupported platform :{match.group(1)}")
    return match.group(1), parse_version(match.group(2))


def macho_versions(header: bytes) -> list[tuple[str, Version]] | None:
    """(platform, minimum OS) per version load command, or None if not Mach-O.

    `header` must hold the Mach-O header and all of its load commands.
    """
    layout = MACHO_MAGICS.get(header[:4])
    if layout is None:
        return None
    order, offset = layout
    ncmds, sizeofcmds = struct.unpack_from(order + "II", header, 16)
    end = offset + sizeofcmds
    if len(header) < end:
        raise ValueError("truncated Mach-O load commands")
    versions = []
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from(order + "II", header, offset)
        if cmdsize < 8 or offset + cmdsize > end:
            raise ValueError(f"malformed load command 0x{cmd:x}")
        if cmd == LC_BUILD_VERSION:
            platform, minos = struct.unpack_from(order + "II", header, offset + 8)
            versions.append((PLATFORMS.get(platform, f"platform {platform}"),
                             decode_version(minos)))
        elif cmd in LC_VERSION_MIN:
            (version,) = struct.unpack_from(order + "I", header, offset + 8)
            versions.append((LC_VERSION_MIN[cmd], decode_version(version)))
        offset += cmdsize
    return versions


def _read_macho_header(stream: BinaryIO, size: int) -> bytes:
    """The Mach-O header plus load commands at `stream`'s position, if any."""
    head = stream.read(min(size, 32))
    layout = MACHO_MAGICS.get(head[:4])
    if layout is None or len(head) < 24:
        return head
    (sizeofcmds,) = struct.unpack_from(layout[0] + "I", head, 20)
    want = min(size, layout[1] + sizeofcmds)
    return head + stream.read(max(0, want - len(head)))


def archive_members(stream: BinaryIO) -> Iterator[tuple[str, bytes]]:
    """(name, Mach-O header bytes) for each object in an `ar` archive."""
    long_names = b""
    while True:
        header = stream.read(AR_HEADER_SIZE)
        if not header:
            return
        if len(header) < AR_HEADER_SIZE or header[58:60] != b"`\n":
            raise ValueError("malformed ar member header")
        raw_name = header[:16].decode("ascii", "replace").rstrip(" ")
        size = int(header[48:58])
        start = stream.tell()
        if raw_name.startswith("#1/"):  # BSD: the name leads the member data
            name_len = int(raw_name[3:])
            name = stream.read(name_len).rstrip(b"\0").decode("utf-8", "replace")
            data_size = size - name_len
        elif raw_name.startswith("/") and raw_name[1:].isdigit():  # GNU long name
            at = int(raw_name[1:])
            name = long_names[at:long_names.index(b"/\n", at)].decode("utf-8", "replace")
            data_size = size
        else:  # GNU ends short names with "/"; its index names keep theirs
            name = raw_name
            if name.endswith("/") and name not in AR_INDEX_NAMES:
                name = name[:-1]
            data_size = size
        if raw_name == "//":
            long_names = stream.read(size)
        elif name not in AR_INDEX_NAMES:
            yield name, _read_macho_header(stream, data_size)
        stream.seek(start + size + (size & 1))  # members are 2-byte aligned


def library_objects(path: Path) -> Iterator[tuple[str, bytes]]:
    """(member label, Mach-O header bytes) for an archive or a lone Mach-O file."""
    with path.open("rb") as stream:
        if stream.read(len(AR_MAGIC)) == AR_MAGIC:
            yield from archive_members(stream)
            return
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(0)
        yield path.name, _read_macho_header(stream, size)


def check(path: Path, os_name: str, floor: Version) -> tuple[list[str], str]:
    """Return (failures, summary) for one library against a podspec floor.

    Objects that fail the same way share one failure line, which names the
    first few: a shipped archive holds thousands of objects.
    """
    allowed = PODSPEC_PLATFORMS[os_name]
    problems: dict[str, list[str]] = {}
    highest, objects = None, 0
    for member, header in library_objects(path):
        try:
            versions = macho_versions(header)
        except (ValueError, struct.error) as error:
            problems.setdefault(f"malformed ({error})", []).append(member)
            continue
        if versions is None:
            problems.setdefault("not thin Mach-O, so their OS is unknown", []).append(member)
            continue
        objects += 1
        for platform, version in versions:
            if platform not in allowed:
                problem = f"built for {platform}, but the podspec declares :{os_name}"
            elif version > floor:
                problem = (f"built for {platform} {format_version(version)}, "
                           f"above the podspec's {format_version(floor)}")
            else:
                problem = None
            if problem:
                problems.setdefault(problem, []).append(member)
            if highest is None or version > highest:
                highest = version
    failures = []
    for problem, members in problems.items():
        names = ", ".join(members[:EXAMPLES]) + (", ..." if len(members) > EXAMPLES else "")
        failures.append(f"{path}: {len(members)} object(s) {problem}: {names}")
    if not objects and not problems:
        failures.append(f"{path}: no Mach-O objects")
    found = "none recorded" if highest is None else format_version(highest)
    return failures, (f"{path}: {objects} object(s), highest minimum OS {found}, "
                      f"podspec :{os_name} {format_version(floor)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--podspec", type=Path, required=True,
        help="the podspec whose `s.platform` version the libraries must not exceed",
    )
    parser.add_argument("libs", nargs="+", type=Path,
                        help="static libraries (ar archives) or Mach-O files")
    args = parser.parse_args(argv)

    try:
        os_name, floor = podspec_platform(args.podspec.read_text())
    except (OSError, ValueError) as error:
        print(f"error: {args.podspec}: {error}", file=sys.stderr)
        return 1
    failures = []
    for lib in args.libs:
        try:
            lib_failures, summary = check(lib, os_name, floor)
        except (OSError, ValueError) as error:
            failures.append(f"{lib}: {error}")
            continue
        failures.extend(lib_failures)
        print(("FAIL: " if lib_failures else "OK: ") + summary)
    if failures:
        print(f"\n{len(failures)} problem(s):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
