#!/usr/bin/env python3
"""Package or verify canonical zzz notices in the explicit Apple SDK ZIP."""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import plistlib
import re
import stat
import struct
import sys
import tempfile
import zipfile


# The tool and contract are declared Bazel inputs from this reviewed checkout.
# Never import a candidate module through cwd or sys.path.
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "xybrid_apple_notice_contract", ROOT / "crates/zzz-sys/zzz_pull.py")
PULL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PULL
SPEC.loader.exec_module(PULL)
FRAMEWORK = "XybridFFI.xcframework"
NOTICE_ROOT = FRAMEWORK + "/Notices/zzz/"
BASE_FILES = {FRAMEWORK + "/Info.plist"} | {
    FRAMEWORK + "/" + platform + "/XybridFFI.framework/" + name
    for platform in ("ios-arm64", "ios-arm64-simulator")
    for name in ("XybridFFI", "Info.plist", "Headers/xybrid-bolt.h", "Modules/module.modulemap")
}
MAX_ZIP = 512 * 1024 * 1024
MAX_MEMBER = 512 * 1024 * 1024
MAX_TOTAL = 1024 * 1024 * 1024
MAX_MEMBERS = 4096
MAX_CENTRAL = 2 * 1024 * 1024
MAX_NOTICE = 1024 * 1024
MAX_RATIO = 100
CHUNK = 64 * 1024
END = struct.Struct("<4s4H2IH")


class PackageError(ValueError):
    """A fixed message safe for build logs containing private input paths."""


def require(condition, message):
    if not condition:
        raise PackageError(message)


class PrivateParser(argparse.ArgumentParser):
    def error(self, _message):
        raise PackageError("invalid command-line arguments")


def require_host():
    require(os.name == "posix" and all(type(getattr(os, name, None)) is int and
            getattr(os, name) > 0 for name in ("O_NOFOLLOW", "O_NONBLOCK")),
            "Apple notice packaging requires POSIX regular-file checks")


def identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def regular_input(path, limit):
    # Bazel's declared action inputs and external repository files are commonly
    # symlinks. Resolve that infrastructure path, then hold/check its regular
    # target; only independently pinned notice bytes can reach the output.
    resolved = Path(path).resolve(strict=True)
    descriptor = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 < before.st_size <= limit, "input is not a bounded single-link regular file")
        yield stream
        require(identity(before) == identity(os.fstat(stream.fileno())) and
                identity(before) == identity(os.stat(resolved, follow_symlinks=False)),
                "input changed while packaging")


def notice_inputs(values):
    paths = {}
    for value in values:
        name, separator, path = value.partition("=")
        require(separator and path and name not in paths and name in PULL.SCHEMA5_NOTICE_FILES,
                "notice input names differ from the exact schema-5 inventory")
        paths[name] = Path(path)
    require(set(paths) == set(PULL.SCHEMA5_NOTICE_FILES),
            "notice input names differ from the exact schema-5 inventory")
    result = {}
    for name, expected in PULL.SCHEMA5_NOTICE_FILES.items():
        with regular_input(paths[name], MAX_NOTICE) as stream:
            data = stream.read(MAX_NOTICE + 1)
            require(hashlib.sha256(data).hexdigest() == expected,
                    "notice bytes differ from the immutable schema-5 contract")
            result[name] = data
    return result


def envelope(stream):
    size = os.fstat(stream.fileno()).st_size
    require(size >= END.size, "invalid XCFramework ZIP envelope")
    stream.seek(size - END.size)
    signature, disk, cd_disk, disk_count, count, cd_size, cd_start, comment = END.unpack(stream.read(END.size))
    # Bound directory allocation before zipfile parses it. ZIP64, comments,
    # multi-disk files and trailing data are outside this actual target format.
    require(signature == b"PK\x05\x06" and disk == cd_disk == comment == 0 and
            disk_count == count and 0 < count <= MAX_MEMBERS and
            cd_size <= MAX_CENTRAL and cd_start + cd_size + END.size == size,
            "unsupported or beyond-bound XCFramework ZIP envelope")
    return count


def member_index(archive, count, require_notices):
    infos = archive.infolist()
    require(len(infos) == count, "ZIP directory count differs")
    files, seen, total = {}, set(), 0
    expected = BASE_FILES | {NOTICE_ROOT + name for name in PULL.SCHEMA5_NOTICE_FILES}
    directories = {name[:index] for name in expected for index, char in enumerate(name) if char == "/"}
    for info in infos:
        name = info.filename
        key = name.rstrip("/")
        require(name == info.orig_filename and name.isascii() and len(name) <= 512 and
                re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*/?", name) and
                all(part not in (".", "..") for part in key.split("/")) and key.lower() not in seen,
                "unsafe, aliased or duplicate XCFramework ZIP member")
        seen.add(key.lower())
        mode = info.external_attr >> 16
        require(info.create_system == 3 and info.flag_bits & ~0x800 == 0 and
                info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) and
                not info.extra and not info.comment and mode & 0o7000 == 0,
                "unsupported XCFramework ZIP member metadata")
        if info.is_dir():
            require(key in directories and stat.S_ISDIR(mode) and info.file_size == 0,
                    "unexpected XCFramework ZIP directory")
        else:
            require(name in expected and stat.S_ISREG(mode) and 0 <= info.file_size <= MAX_MEMBER and
                    info.file_size <= MAX_RATIO * max(1, info.compress_size),
                    "unexpected, linked or beyond-bound XCFramework ZIP payload")
            total += info.file_size
            require(total <= MAX_TOTAL, "XCFramework ZIP total exceeds its byte bound")
            files[name] = info
    allowed = [expected] if require_notices else [BASE_FILES, expected]
    require(set(files) in allowed, "XCFramework payload or complete notice inventory is missing")
    return files


def validate_plist(archive, files):
    info = files[FRAMEWORK + "/Info.plist"]
    require(info.file_size <= MAX_NOTICE, "XCFramework metadata exceeds its byte bound")
    value = plistlib.loads(archive.read(info))
    require(type(value) is dict and value.get("CFBundlePackageType") == "XFWK" and
            type(value.get("AvailableLibraries")) is list and len(value["AvailableLibraries"]) == 2,
            "XCFramework metadata does not describe the actual Apple target")
    expected = {"ios-arm64": None, "ios-arm64-simulator": "simulator"}
    seen = set()
    for item in value["AvailableLibraries"]:
        require(type(item) is dict and item.get("LibraryIdentifier") in expected and
                item["LibraryIdentifier"] not in seen and item.get("LibraryPath") == "XybridFFI.framework" and
                item.get("SupportedArchitectures") == ["arm64"] and item.get("SupportedPlatform") == "ios" and
                item.get("SupportedPlatformVariant") == expected[item["LibraryIdentifier"]],
                "XCFramework device or simulator metadata differs")
        seen.add(item["LibraryIdentifier"])
    require(seen == set(expected), "XCFramework platform metadata is incomplete")


def copy_member(archive, info, target=None):
    digest, count = hashlib.sha256(), 0
    with archive.open(info) as source:
        while True:
            block = source.read(min(CHUNK, info.file_size + 1 - count))
            if not block:
                break
            count += len(block)
            require(count <= info.file_size, "ZIP payload grew beyond its declared size")
            digest.update(block)
            if target is not None:
                target.write(block)
    require(count == info.file_size, "ZIP payload size differs")
    return digest.hexdigest()


def validate_notices(archive, files):
    for name, expected in PULL.SCHEMA5_NOTICE_FILES.items():
        member = files.get(NOTICE_ROOT + name)
        if member is not None:
            require(member.file_size <= MAX_NOTICE and copy_member(archive, member) == expected,
                    "packaged notice bytes differ from the immutable schema-5 contract")


def output_info(name, original=None):
    info = zipfile.ZipInfo(name, original.date_time if original else (1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    mode = (original.external_attr >> 16) & 0o777 if original else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def package_archive(archive_path, device, simulator, output):
    require_host()
    device, simulator = notice_inputs(device), notice_inputs(simulator)
    require(device == simulator, "device and simulator notices conflict")
    output = Path(output)
    temporary = None
    try:
        with regular_input(archive_path, MAX_ZIP) as source:
            count = envelope(source)
            with zipfile.ZipFile(source) as archive:
                files = member_index(archive, count, require_notices=False)
                validate_plist(archive, files)
                validate_notices(archive, files)
                descriptor, temporary = tempfile.mkstemp(prefix=".zzz-apple-", suffix=".zip", dir=output.parent)
                with os.fdopen(descriptor, "w+b") as stream:
                    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as result:
                        for name in sorted(BASE_FILES):
                            with result.open(output_info(name, files[name]), "w") as destination:
                                copy_member(archive, files[name], destination)
                        for name, data in sorted(device.items()):
                            result.writestr(output_info(NOTICE_ROOT + name), data)
                    require(stream.tell() <= MAX_ZIP, "notice-bearing XCFramework exceeds its ZIP byte bound")
        # Publish without overwriting any existing path, even if it appeared
        # after validation. The temporary and destination share a filesystem.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def verify_archive(path):
    require_host()
    with regular_input(path, MAX_ZIP) as source:
        count = envelope(source)
        with zipfile.ZipFile(source) as archive:
            files = member_index(archive, count, require_notices=True)
            validate_plist(archive, files)
            validate_notices(archive, files)
            for name in BASE_FILES:
                copy_member(archive, files[name])


def main(argv=None):
    parser = PrivateParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--archive", required=True, type=Path)
    pack.add_argument("--device-notice", action="append", default=[])
    pack.add_argument("--simulator-notice", action="append", default=[])
    pack.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--archive", required=True, type=Path)
    try:
        arguments = parser.parse_args(argv)
        if arguments.command == "pack":
            package_archive(arguments.archive, arguments.device_notice, arguments.simulator_notice, arguments.output)
        else:
            verify_archive(arguments.archive)
        return 0
    except Exception as error:
        message = str(error) if isinstance(error, PackageError) else "invalid or unavailable package input"
        sys.stderr.write("Apple zzz notices: " + message + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
