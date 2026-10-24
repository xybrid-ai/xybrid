#!/usr/bin/env python3
"""Draft bounded final-SDK receiver under independently supplied policies."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import struct
import sys
import zipfile


# Import only this reviewed sibling, never a candidate path or sys.path module.
_SPEC = importlib.util.spec_from_file_location(
    "xybrid_receiver_producer_verify", Path(__file__).with_name("zzz_producer_verify.py"))
PRODUCER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(PRODUCER)
Error = PRODUCER.VerificationError
require = PRODUCER.require
MAX_CONTRACT = 128 * 1024
MAX_MEMBERS = 256
MAX_CENTRAL = 1024 * 1024
MAX_UNPACKED = 512 * 1024 * 1024
MAX_PATH = 512
MAX_PATH_DEPTH = 8
MAX_RATIO = 100
MAX_BINDING = 16 * 1024
SDK_SUFFIXES = {"apple": (".zip",), "android": (".aar",), "python": (".whl",),
                "web": (".tgz",), "react-native": (".tgz",), "unity": (".zip",),
                "flutter": (".tar.gz",), "rust": (".crate",)}
CENTRAL = struct.Struct("<4s6H3I5H2I")
LOCAL = struct.Struct("<4s5H3I2H")
END = struct.Struct("<4s4H2IH")


def require_host():
    PRODUCER.require_posix_host()
    directory = getattr(os, "O_DIRECTORY", None)
    require(type(directory) is int and directory > 0 and
            directory not in (os.O_NOFOLLOW, os.O_NONBLOCK) and
            all(fn in os.supports_dir_fd for fn in (os.open, os.mkdir, os.stat, os.rmdir)) and
            os.scandir in os.supports_fd and
            getattr(shutil.rmtree, "avoids_symlink_attacks", False),
            "receiver requires POSIX directory descriptors and safe cleanup")


def safe_path(value):
    require(type(value) is str and value.isascii() and 0 < len(value) <= MAX_PATH and
            len(value.split("/")) <= MAX_PATH_DEPTH and
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*", value) and
            all(part not in (".", "..") for part in value.split("/")),
            "invalid archive member path")
    return value


def contract_value(value, policy):
    PRODUCER.object_fields(value, {"schema_version", "artifact_role", "trusted_master_commit",
                                  "candidate_commit", "source_binding_member", "members"})
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "unsupported extraction contract schema")
    role = value["artifact_role"]
    require(type(role) is str and role in SDK_SUFFIXES and role == policy["artifact_role"],
            "only an independently approved final SDK role is accepted")
    PRODUCER.hex_string(value["trusted_master_commit"], 40)
    PRODUCER.hex_string(value["candidate_commit"], 40)
    require(value["trusted_master_commit"] == policy["approved_head_sha"] and
            value["source_binding_member"] == "source-binding.json",
            "external source-binding context does not match the producer policy")
    members = value["members"]
    require(type(members) is dict and 2 <= len(members) <= MAX_MEMBERS,
            "invalid exact member allowlist")
    total, names = 0, set()
    for name, entry in members.items():
        safe_path(name)
        folded = name.lower()
        require(folded not in names, "member paths alias by case")
        names.add(folded)
        PRODUCER.object_fields(entry, {"kind", "size", "sha256"})
        size = PRODUCER.integer(entry["size"])
        PRODUCER.hex_string(entry["sha256"], 64)
        require(size <= MAX_UNPACKED, "member byte bound exceeded")
        total += size
        if name == value["source_binding_member"]:
            require(entry["kind"] == "source-binding" and size <= MAX_BINDING,
                    "invalid externally pinned source-binding member")
        else:
            base = name.rsplit("/", 1)[-1].lower()
            require(entry["kind"] == "sdk-package" and base.startswith("xybrid") and
                    base.endswith(SDK_SUFFIXES[role]) and
                    not any(part.lower() in {"private", "staging", "natives", "target", "zig-out"}
                            for part in name.split("/")),
                    "raw engine, staging or non-SDK members are outside this role")
    require(value["source_binding_member"] in members and total <= MAX_UNPACKED,
            "source-binding member is missing or total byte bound exceeded")
    for name in names:
        require(not any(name.startswith(other + "/") for other in names if name != other),
                "file and directory member paths overlap")
    return value


def identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def regular(value, limit):
    require(stat.S_ISREG(value.st_mode) and value.st_nlink == 1 and 0 < value.st_size <= limit,
            "input must be a bounded single-link regular file")


def read_json(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        regular(before, limit)
        data = stream.read(limit + 1)
        require(len(data) == before.st_size and identity(before) == identity(os.fstat(stream.fileno())),
                "JSON input changed during reading")
    return PRODUCER.strict_json(data, limit)


def parent_directory(destination):
    require(destination.is_absolute() and destination.name not in ("", ".", "..") and
            all(part not in (".", "..") for part in destination.parts[1:]),
            "destination must be an absolute path without dot components")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open("/", flags)
    try:
        for part in destination.parent.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        try:
            os.stat(destination.name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return fd
        raise Error("destination already exists")
    except BaseException:
        os.close(fd)
        raise


def snapshot(archive, scratch_fd):
    source_fd = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(source_fd, "rb") as source:
        before = os.fstat(source.fileno())
        regular(before, PRODUCER.MAX_ARCHIVE)
        target_fd = os.open("archive.zip", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o600, dir_fd=scratch_fd)
        with os.fdopen(target_fd, "wb") as target:
            count = 0
            while True:
                block = source.read(min(PRODUCER.CHUNK, before.st_size + 1 - count))
                if not block:
                    break
                count += len(block)
                require(count <= before.st_size, "archive changed while snapshotting")
                target.write(block)
            target.flush()
            require(count == before.st_size and identity(before) == identity(os.fstat(source.fileno())),
                    "archive changed while snapshotting")
            os.fchmod(target.fileno(), 0o400)
    return os.open("archive.zip", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=scratch_fd)


def held_digest(stream, path, expected):
    before = os.fstat(stream.fileno())
    regular(before, PRODUCER.MAX_ARCHIVE)
    require(before.st_uid == os.geteuid() and stat.S_IMODE(before.st_mode) == 0o400 and
            identity(before) == identity(os.stat(path, follow_symlinks=False)),
            "private archive snapshot identity changed")
    stream.seek(0)
    digest, count = hashlib.sha256(), 0
    while True:
        block = stream.read(PRODUCER.CHUNK)
        if not block:
            break
        count += len(block)
        require(count <= before.st_size, "private archive snapshot changed")
        digest.update(block)
    require(count == before.st_size and digest.hexdigest() == expected and
            identity(before) == identity(os.fstat(stream.fileno())) and
            identity(before) == identity(os.stat(path, follow_symlinks=False)),
            "private archive snapshot digest or identity changed")


def read_exact(stream, offset, size):
    stream.seek(offset)
    data = stream.read(size)
    require(len(data) == size, "truncated ZIP structure")
    return data


def same_directory(descriptor, parent_fd, name):
    expected = os.fstat(descriptor)
    actual = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    require(stat.S_ISDIR(actual.st_mode) and (actual.st_dev, actual.st_ino) ==
            (expected.st_dev, expected.st_ino), "receiver output directory identity changed")


def zip_members(stream, contract):
    size = os.fstat(stream.fileno()).st_size
    require(size >= END.size, "invalid ZIP envelope")
    signature, disk, cd_disk, disk_count, count, cd_size, cd_start, comment = END.unpack(
        read_exact(stream, size - END.size, END.size))
    require(signature == b"PK\x05\x06" and disk == cd_disk == comment == 0 and
            disk_count == count and 0 < count <= MAX_MEMBERS and
            cd_size <= MAX_CENTRAL and cd_start + cd_size + END.size == size,
            "unsupported or beyond-bound ZIP envelope")
    central = read_exact(stream, cd_start, cd_size)
    records, position, total = {}, 0, 0
    while position < len(central):
        require(len(records) < MAX_MEMBERS and position + CENTRAL.size <= len(central),
                "ZIP central-directory count or byte bound exceeded")
        fields = CENTRAL.unpack_from(central, position)
        (sig, made, needed, flags, method, _, _, crc, packed, unpacked,
         name_size, extra, member_comment, start_disk, internal, external, offset) = fields
        end = position + CENTRAL.size + name_size
        require(sig == b"PK\x01\x02" and end <= len(central) and
                extra == member_comment == start_disk == 0 and internal in (0, 1) and
                made >> 8 == 3 and 10 <= needed <= 20 and
                stat.S_IFMT(external >> 16) == stat.S_IFREG and
                external & 0xffff == 0 and (external >> 16) & 0o7000 == 0 and
                flags & ~0x80e == 0 and method in (0, 8) and (method == 8 or flags & 6 == 0),
                "unsupported ZIP compression, flags or non-regular metadata")
        name = safe_path(central[position + CENTRAL.size:end].decode("ascii"))
        require(name not in records and name in contract["members"],
                "duplicate or unlisted ZIP member")
        require(unpacked == contract["members"][name]["size"] and 0 < packed <= PRODUCER.MAX_ARCHIVE and
                unpacked <= MAX_RATIO * packed,
                "ZIP member size or compression ratio is outside bounds")
        total += unpacked
        require(total <= MAX_UNPACKED, "ZIP uncompressed total exceeds its bound")
        records[name] = (offset, flags, method, crc, packed, unpacked,
                         central[position + CENTRAL.size:end])
        position = end
    require(len(records) == count and set(records) == set(contract["members"]),
            "ZIP member allowlist or declared count differs")
    # Bind local headers/data descriptors and reject prefixes, gaps and overlaps.
    ordered = sorted(records.values())
    require(ordered[0][0] == 0, "ZIP prefix is outside this contract")
    for index, record in enumerate(ordered):
        offset, flags, method, crc, packed, unpacked, raw_name = record
        header = LOCAL.unpack(read_exact(stream, offset, LOCAL.size))
        sig, needed, local_flags, local_method, _, _, local_crc, local_packed, local_size, names, extra = header
        require(sig == b"PK\x03\x04" and 10 <= needed <= 20 and local_flags == flags and
                local_method == method and names == len(raw_name) and extra == 0 and
                read_exact(stream, offset + LOCAL.size, names) == raw_name,
                "ZIP local header differs from its approved central member")
        data_end = offset + LOCAL.size + names + packed
        next_offset = ordered[index + 1][0] if index + 1 < len(ordered) else cd_start
        if flags & 8:
            require(local_crc in (0, crc) and local_packed in (0, packed) and local_size in (0, unpacked)
                    and next_offset - data_end in (12, 16), "invalid ZIP data descriptor layout")
            descriptor = read_exact(stream, data_end, next_offset - data_end)
            if len(descriptor) == 16:
                require(descriptor[:4] == b"PK\x07\x08", "invalid ZIP data descriptor signature")
                descriptor = descriptor[4:]
            require(struct.unpack("<III", descriptor) == (crc, packed, unpacked),
                    "ZIP data descriptor differs")
        else:
            require((local_crc, local_packed, local_size) == (crc, packed, unpacked) and
                    data_end == next_offset, "ZIP data overlaps or disagrees with its header")
    return records


def extract_members(stream, root_fd, contract):
    directories = {"": root_fd}
    try:
        with zipfile.ZipFile(stream) as archive:
            require(len(archive.infolist()) == len(contract["members"]), "ZIP parser member count differs")
            for name, entry in contract["members"].items():
                parts, parent = name.split("/"), ""
                for part in parts[:-1]:
                    current = parent + "/" + part if parent else part
                    if current not in directories:
                        os.mkdir(part, 0o700, dir_fd=directories[parent])
                        directories[current] = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                                       dir_fd=directories[parent])
                    parent = current
                fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
                             0o600, dir_fd=directories[parent])
                digest, count, prefix = hashlib.sha256(), 0, b""
                with os.fdopen(fd, "wb") as target, archive.open(name) as source:
                    while True:
                        data = source.read(min(PRODUCER.CHUNK, entry["size"] + 1 - count))
                        if not data:
                            break
                        count += len(data)
                        require(count <= entry["size"], "decompressed member exceeds its approved size")
                        prefix = (prefix + data)[:4]
                        digest.update(data)
                        target.write(data)
                    require(count == entry["size"] and digest.hexdigest() == entry["sha256"],
                            "decompressed member bytes differ from the external allowlist")
                    if entry["kind"] == "sdk-package":
                        magic = b"PK\x03\x04" if name.lower().endswith((".zip", ".aar", ".whl")) else b"\x1f\x8b"
                        require(prefix.startswith(magic), "member is not a supported SDK package container")
                    os.fchmod(target.fileno(), 0o400)
    finally:
        for path, fd in reversed(list(directories.items())):
            if path:
                os.close(fd)


def verify_output_file(fd, entry):
    actual = os.fstat(fd)
    regular(actual, entry["size"])
    require(actual.st_size == entry["size"] and actual.st_uid == os.geteuid() and
            stat.S_IMODE(actual.st_mode) == 0o400, "receiver output member metadata changed")
    digest, count = hashlib.sha256(), 0
    while True:
        data = os.read(fd, min(PRODUCER.CHUNK, entry["size"] + 1 - count))
        if not data:
            break
        count += len(data)
        require(count <= entry["size"], "receiver output member grew beyond its bound")
        digest.update(data)
    require(count == entry["size"] and digest.hexdigest() == entry["sha256"],
            "receiver output member bytes changed")


def verify_output(root_fd, contract):
    tree = {}
    for name, entry in contract["members"].items():
        node = tree
        parts = name.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = (entry,)

    def walk(fd, expected):
        before = os.fstat(fd)
        require(stat.S_ISDIR(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o700 and
                before.st_uid == os.geteuid(), "receiver output directory metadata changed")
        seen = set()
        with os.scandir(fd) as entries:
            for entry in entries:
                require(len(seen) < len(expected) and entry.name in expected and entry.name not in seen,
                        "receiver output inventory differs from its exact allowlist")
                seen.add(entry.name)
                node = expected[entry.name]
                metadata = entry.stat(follow_symlinks=False)
                is_file = type(node) is tuple
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if not is_file:
                    flags |= os.O_DIRECTORY
                child = os.open(entry.name, flags, dir_fd=fd)
                try:
                    actual = os.fstat(child)
                    require(identity(metadata) == identity(actual), "receiver output member identity changed")
                    if is_file:
                        verify_output_file(child, node[0])
                    else:
                        walk(child, node)
                    require(identity(actual) == identity(os.fstat(child)) and
                            identity(actual) == identity(os.stat(entry.name, dir_fd=fd, follow_symlinks=False)),
                            "receiver output member changed during revalidation")
                finally:
                    os.close(child)
        require(seen == set(expected) and identity(before) == identity(os.fstat(fd)),
                "receiver output directory changed during revalidation")
    walk(root_fd, tree)


def receive(policy, contract, run_id, artifact_id, archive, destination, api=None, now=None):
    PRODUCER.refuse_credentials()
    require_host()
    # Defensive copies preserve the independently supplied expected values.
    policy = PRODUCER.policy_value(PRODUCER.strict_json(json.dumps(policy).encode(), PRODUCER.MAX_POLICY))
    contract = contract_value(PRODUCER.strict_json(json.dumps(contract).encode(), MAX_CONTRACT), policy)
    require(PRODUCER.integer(run_id) == policy["run_id"] and
            PRODUCER.integer(artifact_id) == policy["artifact_id"], "candidate selectors differ from trusted policy")
    destination = Path(destination)
    parent_fd = parent_directory(destination)
    scratch_name = ".zzz-receive-" + secrets.token_hex(16)
    scratch_fd, output_fd, held, result, failure = None, None, None, None, None
    scratch_created, created = False, False
    try:
        os.mkdir(scratch_name, 0o700, dir_fd=parent_fd)
        scratch_created = True
        scratch_fd = os.open(scratch_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        snapshot_path = destination.parent / scratch_name / "archive.zip"
        held = os.fdopen(snapshot(archive, scratch_fd), "rb")
        expected = policy["artifact_sha256"]
        held_digest(held, snapshot_path, expected)
        PRODUCER.verify(policy, run_id, artifact_id, snapshot_path, api=api, now=now)
        held_digest(held, snapshot_path, expected)
        zip_members(held, contract)
        os.mkdir(destination.name, 0o700, dir_fd=parent_fd)  # Atomic no-overwrite reservation.
        created = True
        output_fd = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        extract_members(held, output_fd, contract)
        same_directory(output_fd, parent_fd, destination.name)
        held_digest(held, snapshot_path, expected)
        PRODUCER.verify(policy, run_id, artifact_id, snapshot_path, api=api, now=now)
        held_digest(held, snapshot_path, expected)
        same_directory(output_fd, parent_fd, destination.name)
        verify_output(output_fd, contract)
        same_directory(output_fd, parent_fd, destination.name)
        result = {"schema_version": 1, "extraction": "EXACT_EXTERNAL_ALLOWLIST_AND_REPEATED_PRODUCER_CHECKS",
                  "artifact_role": contract["artifact_role"], "run_id": run_id, "artifact_id": artifact_id,
                  "archive_sha256": expected, "candidate_commit": contract["candidate_commit"],
                  "trusted_master_commit": contract["trusted_master_commit"],
                  "members": contract["members"], "source_policy": "NOT_RECOMPUTED_BY_THIS_HELPER",
                  "activation": "NOT_AUTHORIZED"}
    except BaseException as error:
        failure = error
    finally:
        if held is not None:
            held.close()
        if scratch_created:
            try:
                require(scratch_fd is not None, "private snapshot directory identity was not captured")
                same_directory(scratch_fd, parent_fd, scratch_name)
                shutil.rmtree(scratch_name, dir_fd=parent_fd)
            except (OSError, Error):
                failure = Error("receiver private snapshot cleanup failed")
        if failure is not None and created:
            try:
                require(output_fd is not None, "receiver output directory identity was not captured")
                same_directory(output_fd, parent_fd, destination.name)
                shutil.rmtree(destination.name, dir_fd=parent_fd)
            except (OSError, Error):
                failure = Error("receiver output rollback failed")
        if output_fd is not None:
            os.close(output_fd)
        if scratch_fd is not None:
            os.close(scratch_fd)
        os.close(parent_fd)
    if failure is not None:
        raise failure
    return result


def main():
    parser = PRODUCER.PrivateArgumentParser(description=__doc__)
    parser.add_argument("--trusted-producer-policy", required=True, type=Path)
    parser.add_argument("--trusted-extraction-contract", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--artifact-id", required=True, type=int)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    try:
        arguments = parser.parse_args()
        PRODUCER.refuse_credentials()
        require_host()
        result = receive(read_json(arguments.trusted_producer_policy, PRODUCER.MAX_POLICY),
                         read_json(arguments.trusted_extraction_contract, MAX_CONTRACT),
                         arguments.run_id, arguments.artifact_id, arguments.archive, arguments.destination)
        sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        return 0
    except Exception as error:
        message = str(error) if isinstance(error, Error) else "input/archive failure"
        sys.stderr.write("candidate extraction failed: " + message + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
