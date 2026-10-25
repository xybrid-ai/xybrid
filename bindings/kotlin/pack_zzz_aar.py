#!/usr/bin/env python3
"""Package or verify the explicit notice-bearing zzz Android AAR."""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile
import zipfile


# The tool and contract are declared Bazel runfiles. Preserve the launcher's
# runfiles path: resolving its symlink may reach a detached physical source
# file whose siblings are not declared runtime inputs. Direct source invocations
# have the same relative layout. Never import through cwd or sys.path.
ROOT = Path(os.path.abspath(__file__)).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "xybrid_android_notice_contract", ROOT / "crates/zzz-sys/zzz_pull.py")
PULL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PULL
SPEC.loader.exec_module(PULL)

ABIS = ("arm64-v8a", "armeabi-v7a", "x86_64")
ROOT_FILES = {"AndroidManifest.xml", "R.txt", "classes.jar"}
COMPANION = "jni/arm64-v8a/libxybrid_zzz_tts.so"
NOTICE_ROOT = "assets/xybrid/zzz/"
MIN_SDK = "24"
MAX_ZIP = 512 * 1024 * 1024
MAX_MEMBER = 256 * 1024 * 1024
MAX_TOTAL = 1024 * 1024 * 1024
MAX_MEMBERS = 256
MAX_CENTRAL = 256 * 1024
MAX_COMPANION = 64 * 1024 * 1024
MAX_NOTICE = 1024 * 1024
MAX_MANIFEST = 64 * 1024
MAX_RATIO = 100
CHUNK = 64 * 1024
END = struct.Struct("<4s4H2IH")
# ELF64, little endian, version 1, ET_DYN, EM_AARCH64.
ELF_HEADER = struct.Struct("<4sBBB9xHHI")


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
            "Android notice packaging requires POSIX regular-file checks")


def identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def regular_input(path, limit):
    # Bazel's declared action inputs and external repository files are commonly
    # symlinks. Resolve that infrastructure path, then hold/check its regular
    # target; only checked bytes can reach the output.
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


def base_inventory(values):
    """The reviewed base AAR payload, declared by the Bazel graph that builds it."""
    names = set(values)
    require(len(names) == len(values) and len(names) <= MAX_MEMBERS, "duplicate base AAR inventory entry")
    jni = names - ROOT_FILES
    require(ROOT_FILES <= names and
            all(re.fullmatch(r"jni/(?:%s)/lib[A-Za-z0-9_.+-]+\.so" % "|".join(map(re.escape, ABIS)), name)
                for name in jni) and
            all("jni/%s/%s" % (abi, lib) in names for abi in ABIS
                for lib in ("libxybrid_bolt.so", "libc++_shared.so")) and
            COMPANION not in names,
            "base AAR inventory differs from the reviewed three-ABI layout")
    return names


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


def check_companion_header(data):
    require(len(data) >= ELF_HEADER.size, "companion is not an ELF shared object")
    magic, elf_class, order, version, kind, machine, _ = ELF_HEADER.unpack_from(data)
    require(magic == b"\x7fELF" and elf_class == 2 and order == 1 and version == 1 and
            kind == 3 and machine == 183,
            "companion must be a little-endian ARM64 ELF shared object")


def companion_input(path):
    with regular_input(path, MAX_COMPANION) as stream:
        data = stream.read(MAX_COMPANION + 1)
    check_companion_header(data)
    return data


def envelope(stream):
    size = os.fstat(stream.fileno()).st_size
    require(size >= END.size, "invalid AAR ZIP envelope")
    stream.seek(size - END.size)
    signature, disk, cd_disk, disk_count, count, cd_size, cd_start, comment = END.unpack(stream.read(END.size))
    # Bound directory allocation before zipfile parses it. ZIP64, comments,
    # multi-disk files and trailing data are outside this actual target format.
    require(signature == b"PK\x05\x06" and disk == cd_disk == comment == 0 and
            disk_count == count and 0 < count <= MAX_MEMBERS and
            cd_size <= MAX_CENTRAL and cd_start + cd_size + END.size == size,
            "unsupported or beyond-bound AAR ZIP envelope")
    return count


def member_index(archive, count, base, final):
    infos = archive.infolist()
    require(len(infos) == count, "ZIP directory count differs")
    expected = set(base)
    if final:
        expected |= {COMPANION} | {NOTICE_ROOT + name for name in PULL.SCHEMA5_NOTICE_FILES}
    directories = {name[:index] for name in expected for index, char in enumerate(name) if char == "/"}
    files, seen, total = {}, set(), 0
    for info in infos:
        name = info.filename
        key = name.rstrip("/")
        require(name == info.orig_filename and name.isascii() and len(name) <= 512 and
                re.fullmatch(r"[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*/?", name) and
                all(part not in (".", "..") for part in key.split("/")) and key.lower() not in seen,
                "unsafe, aliased or duplicate AAR ZIP member")
        seen.add(key.lower())
        mode = info.external_attr >> 16
        require(info.create_system == 3 and info.flag_bits & ~0x800 == 0 and
                info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) and
                not info.extra and not info.comment and mode & 0o7000 == 0,
                "unsupported AAR ZIP member metadata")
        if info.is_dir():
            require(key in directories and stat.S_ISDIR(mode) and info.file_size == 0,
                    "unexpected AAR ZIP directory")
        else:
            require(name in expected and stat.S_ISREG(mode) and 0 <= info.file_size <= MAX_MEMBER and
                    info.file_size <= MAX_RATIO * max(1, info.compress_size),
                    "unexpected, linked or beyond-bound AAR ZIP payload")
            total += info.file_size
            require(total <= MAX_TOTAL, "AAR ZIP total exceeds its byte bound")
            files[name] = info
    require(set(files) == expected, "AAR payload, companion or complete notice inventory differs")
    return files


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


def validate_manifest(archive, files):
    info = files["AndroidManifest.xml"]
    require(info.file_size <= MAX_MANIFEST, "AAR manifest exceeds its byte bound")
    text = re.sub(r"<!--.*?-->", "", archive.read(info).decode("utf-8"), flags=re.DOTALL)
    # The wrapper adds an ARM64/API-29 component loaded only at runtime; the
    # package floor must stay the base SDK's, never raised to hide a failure.
    require(re.findall(r"android:minSdkVersion\s*=\s*[\"']([^\"']*)[\"']", text) == [MIN_SDK] and
            re.findall(r"\bpackage\s*=\s*[\"']([^\"']*)[\"']", text) == ["ai.xybrid"],
            "AAR manifest must keep the ai.xybrid package and minSdk 24")


def audit_companion(data, readelf):
    """Run the companion's export/SONAME/dependency/alignment audit on packaged bytes."""
    spec = importlib.util.spec_from_file_location(
        "xybrid_android_companion_audit", ROOT / "bazel/zzz/companion/check_exports.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    with tempfile.TemporaryDirectory(prefix=".zzz-android-audit-") as directory:
        path = Path(directory) / "libxybrid_zzz_tts.so"
        path.write_bytes(data)
        require(checker.main(["--readelf", str(readelf), str(path)]) == 0,
                "packaged companion failed its ELF audit")


def validate_additions(archive, files, readelf=None):
    for name, expected in PULL.SCHEMA5_NOTICE_FILES.items():
        member = files[NOTICE_ROOT + name]
        require(member.file_size <= MAX_NOTICE and copy_member(archive, member) == expected,
                "packaged notice bytes differ from the immutable schema-5 contract")
    member = files[COMPANION]
    require(0 < member.file_size <= MAX_COMPANION, "packaged companion exceeds its byte bound")
    with archive.open(member) as stream:
        data = stream.read(MAX_COMPANION + 1)
    require(len(data) == member.file_size, "ZIP payload size differs")
    check_companion_header(data)
    if readelf is not None:
        audit_companion(data, readelf)


def output_info(name, original=None):
    info = zipfile.ZipInfo(name, original.date_time if original else (1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    if original is not None and original.is_dir():
        info.external_attr = original.external_attr
        info.compress_type = zipfile.ZIP_STORED
        return info
    mode = (original.external_attr >> 16) & 0o777 if original else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def package_archive(archive_path, companion_path, base_files, notice_values, output):
    require_host()
    base = base_inventory(base_files)
    notices = notice_inputs(notice_values)
    companion = companion_input(companion_path)
    output = Path(output)
    temporary = None
    try:
        with regular_input(archive_path, MAX_ZIP) as source:
            count = envelope(source)
            with zipfile.ZipFile(source) as archive:
                files = member_index(archive, count, base, final=False)
                validate_manifest(archive, files)
                descriptor, temporary = tempfile.mkstemp(prefix=".zzz-android-", suffix=".aar", dir=output.parent)
                with os.fdopen(descriptor, "w+b") as stream:
                    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as result:
                        # Preserve the base order and every payload byte;
                        # only the companion and notices are added.
                        for info in archive.infolist():
                            if info.is_dir():
                                result.writestr(output_info(info.filename, info), b"")
                                continue
                            with result.open(output_info(info.filename, info), "w") as destination:
                                copy_member(archive, info, destination)
                        result.writestr(output_info(COMPANION), companion)
                        for name, data in sorted(notices.items()):
                            result.writestr(output_info(NOTICE_ROOT + name), data)
                    require(stream.tell() <= MAX_ZIP, "notice-bearing AAR exceeds its ZIP byte bound")
        # Publish without overwriting any existing path, even if it appeared
        # after validation. The temporary and destination share a filesystem.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def verify_archive(path, base_files, readelf=None):
    require_host()
    base = base_inventory(base_files)
    with regular_input(path, MAX_ZIP) as source:
        count = envelope(source)
        with zipfile.ZipFile(source) as archive:
            files = member_index(archive, count, base, final=True)
            validate_manifest(archive, files)
            validate_additions(archive, files, readelf)
            for name in base:
                copy_member(archive, files[name])


def main(argv=None):
    parser = PrivateParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--archive", required=True, type=Path)
    pack.add_argument("--companion", required=True, type=Path)
    pack.add_argument("--base-file", action="append", default=[])
    pack.add_argument("--notice", action="append", default=[])
    pack.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--archive", required=True, type=Path)
    verify.add_argument("--base-file", action="append", default=[])
    # The pinned NDK llvm-readelf; adds the companion's full ELF audit.
    verify.add_argument("--readelf", type=Path)
    try:
        arguments = parser.parse_args(argv)
        if arguments.command == "pack":
            package_archive(arguments.archive, arguments.companion, arguments.base_file,
                            arguments.notice, arguments.output)
        else:
            verify_archive(arguments.archive, arguments.base_file, arguments.readelf)
        return 0
    except Exception as error:
        message = str(error) if isinstance(error, PackageError) else "invalid or unavailable package input"
        sys.stderr.write("Android zzz package: " + message + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
