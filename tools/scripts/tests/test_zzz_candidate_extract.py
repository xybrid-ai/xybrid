"""Synthetic producer/archive fixtures; never download or execute payloads."""

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


RECEIVE = module("zzz_candidate_extract", ROOT / "zzz_candidate_extract.py")
FIXTURES = module("producer_fixtures_for_receiver", HERE / "test_zzz_producer_verify.py")


def sdk_fixture():
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("fixture.txt", b"synthetic SDK package; no native execution")
    return inner.getvalue()


@unittest.skipUnless(RECEIVE.PRODUCER.posix_host() and hasattr(os, "O_DIRECTORY") and
                     getattr(RECEIVE.shutil.rmtree, "avoids_symlink_attacks", False),
                     "receiver archive fixtures require supported POSIX descriptor primitives")
class CandidateExtractionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.archive = self.directory / "candidate.zip"
        self.destination = self.directory / "received"
        self.members = {"packages/xybrid.aar": sdk_fixture(),
                        "source-binding.json": b'{"candidate_commit":"untrusted self-claim"}\n'}
        self.contract = {
            "schema_version": 1, "artifact_role": "android",
            "trusted_master_commit": "a" * 40, "candidate_commit": "b" * 40,
            "source_binding_member": "source-binding.json",
            "members": {name: {"kind": "source-binding" if name.endswith(".json") else "sdk-package",
                                "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                        for name, data in self.members.items()},
        }
        self.publish(self.zip_bytes())

    def zip_bytes(self, members=None, mode=stat.S_IFREG, extra=b"", compression=zipfile.ZIP_DEFLATED,
                  streaming=False):
        class NoSeek(io.BytesIO):
            def seek(self, *args):
                raise io.UnsupportedOperation("streaming fixture")
        stream = NoSeek() if streaming else io.BytesIO()
        with warnings.catch_warnings(), zipfile.ZipFile(stream, "w") as archive:
            warnings.simplefilter("ignore", UserWarning)  # Deliberate duplicate-name fixtures.
            for name, data in (members if members is not None else list(self.members.items())):
                info = zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (mode | 0o644) << 16
                info.compress_type = compression
                info.extra = extra
                archive.writestr(info, data)
        return stream.getvalue()

    def publish(self, raw):
        self.archive.write_bytes(raw)
        self.policy, responses, _ = FIXTURES.fixtures()
        self.policy.update(artifact_role="android", artifact_sha256=hashlib.sha256(raw).hexdigest())
        for value in responses.values():
            if type(value) is dict and "size_in_bytes" in value:
                value.update(name=RECEIVE.PRODUCER.artifact_name(self.policy), size_in_bytes=len(raw),
                             digest="sha256:" + self.policy["artifact_sha256"])
        responses[RECEIVE.PRODUCER.API_ROOT + "/actions/runs/300/artifacts?per_page=100&page=1"][
            "artifacts"] = [responses[RECEIVE.PRODUCER.API_ROOT + "/actions/artifacts/400"]]
        self.api = FIXTURES.FakeAPI(responses)

    def receive(self):
        return RECEIVE.receive(self.policy, self.contract, 300, 400, self.archive,
                               self.destination, api=self.api, now=FIXTURES.NOW)

    def rejected(self, destination_remains=False):
        with self.assertRaises((RECEIVE.Error, OSError, ValueError, zipfile.BadZipFile)):
            self.receive()
        if not destination_remains:
            self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.directory.glob(".zzz-receive-*")), [])

    def test_exact_members_use_real_producer_verification_before_and_after_extraction(self):
        result = self.receive()
        for name, data in self.members.items():
            self.assertEqual((self.destination / name).read_bytes(), data)
        self.assertEqual(result["source_policy"], "NOT_RECOMPUTED_BY_THIS_HELPER")
        self.assertEqual(result["activation"], "NOT_AUTHORIZED")
        run_path = RECEIVE.PRODUCER.API_ROOT + "/actions/runs/300"
        self.assertEqual(self.api.paths.count(run_path), 4)
        self.assertEqual(list(self.directory.glob(".zzz-receive-*")), [])
        for name in self.members:
            self.assertEqual(stat.S_IMODE((self.destination / name).stat().st_mode), 0o400)
        self.assertEqual(result["candidate_commit"], "b" * 40)  # Never the member's self-claim.

    def test_stored_members_and_streaming_data_descriptors_are_supported(self):
        for compression, streaming in [(zipfile.ZIP_STORED, False), (zipfile.ZIP_DEFLATED, True)]:
            with self.subTest(compression=compression, streaming=streaming):
                self.publish(self.zip_bytes(compression=compression, streaming=streaming))
                self.receive()
                self.assertEqual((self.destination / "packages/xybrid.aar").read_bytes(),
                                 self.members["packages/xybrid.aar"])
                RECEIVE.shutil.rmtree(self.destination)

    def test_external_roles_kinds_context_and_digests_are_not_read_from_archive(self):
        changes = [("artifact_role", "engine"), ("artifact_role", "apple"),
                   ("trusted_master_commit", "c" * 40), ("candidate_commit", "master"),
                   ("source_binding_member", "receipt.json"), ("schema_version", True),
                   ("verification", "saved success")]
        original = copy.deepcopy(self.contract)
        for field, value in changes:
            with self.subTest(field=field, value=value):
                self.contract = dict(copy.deepcopy(original), **{field: value})
                self.rejected()
                self.assertEqual(self.api.paths, [])
        self.contract = copy.deepcopy(original)
        self.contract["members"]["packages/xybrid.aar"]["kind"] = "raw-engine"
        self.rejected()
        self.assertEqual(self.api.paths, [])

    def test_member_types_numbers_unknown_fields_and_missing_binding_fail_closed(self):
        original = copy.deepcopy(self.contract)
        for change in [dict(size=True), dict(size=0), dict(size=-1), dict(size=1.5),
                       dict(sha256="claimed"), dict(extra="unapproved")]:
            with self.subTest(change=change):
                self.contract = copy.deepcopy(original)
                self.contract["members"]["packages/xybrid.aar"].update(change)
                self.rejected()
        self.contract = copy.deepcopy(original)
        del self.contract["members"]["source-binding.json"]
        self.rejected()
        self.assertEqual(self.api.paths, [])

    def test_raw_archives_static_libraries_receipts_and_private_staging_are_not_sdk_members(self):
        original = copy.deepcopy(self.contract)
        for name in ["zzz-phonon.tar.gz", "libzzz_embed.a", "xybrid.rlib", "receipt.json",
                     "staging/xybrid.aar", "target/xybrid.aar", "natives/xybrid.aar"]:
            with self.subTest(name=name):
                self.contract = copy.deepcopy(original)
                entry = self.contract["members"].pop("packages/xybrid.aar")
                self.contract["members"][name] = entry
                self.rejected()
        self.assertEqual(self.api.paths, [])

    def test_duplicate_traversal_backslash_absolute_nul_and_case_alias_paths_fail(self):
        for name in ["../xybrid.aar", "/xybrid.aar", "packages\\xybrid.aar", "a/../xybrid.aar",
                     "C:/xybrid.aar", "packages/xybrid.aar\0hidden", "PACKAGES/xybrid.aar"]:
            with self.subTest(name=name):
                self.publish(self.zip_bytes(members=list(self.members.items()) + [(name, b"x")]))
                self.rejected()
        self.publish(self.zip_bytes(members=list(self.members.items()) + [list(self.members.items())[0]]))
        self.rejected()
        original = self.contract["members"]["packages/xybrid.aar"]
        self.contract["members"]["PACKAGES/xybrid.aar"] = copy.deepcopy(original)
        self.rejected()

    def test_contract_path_length_depth_and_file_directory_overlap_fail(self):
        original = copy.deepcopy(self.contract)
        for name in ["x" * (RECEIVE.MAX_PATH + 1), "/".join(["a"] * 9) + "/xybrid.aar",
                     "source-binding.json/xybrid.aar"]:
            with self.subTest(name=name):
                self.contract = copy.deepcopy(original)
                entry = self.contract["members"].pop("packages/xybrid.aar")
                self.contract["members"][name] = entry
                self.rejected()

    def test_links_directories_devices_and_unix_hardlink_extra_metadata_fail(self):
        for mode in [stat.S_IFLNK, stat.S_IFDIR, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFSOCK, 0]:
            with self.subTest(mode=mode):
                self.publish(self.zip_bytes(mode=mode))
                self.rejected()
        for extra in [b"\x0d\x00\x00\x00", b"\x6e\x75\x00\x00", b"\x01\x00\x00\x00"]:
            with self.subTest(extra=extra):
                self.publish(self.zip_bytes(extra=extra))
                self.rejected()

    def test_encryption_unsupported_compression_and_dos_metadata_fail(self):
        raw = bytearray(self.zip_bytes())
        central = raw.index(b"PK\x01\x02")
        struct.pack_into("<H", raw, 6, 1)
        struct.pack_into("<H", raw, central + 8, 1)
        self.publish(bytes(raw))
        self.rejected()
        self.publish(self.zip_bytes(compression=zipfile.ZIP_BZIP2))
        self.rejected()
        raw = bytearray(self.zip_bytes())
        raw[raw.index(b"PK\x01\x02") + 5] = 0  # MS-DOS create-system.
        self.publish(bytes(raw))
        self.rejected()

    def test_prefix_trailing_data_multidisk_and_inconsistent_local_headers_fail(self):
        raw = self.zip_bytes()
        for altered in [b"prefix" + raw, raw + b"trailer", raw[:-1],
                        raw[:30] + b"X" + raw[31:]]:
            with self.subTest(size=len(altered)):
                self.publish(altered)
                self.rejected()
        altered = bytearray(raw)
        struct.pack_into("<H", altered, len(altered) - 18, 1)
        self.publish(bytes(altered))
        self.rejected()

    def test_raw_nul_names_zip64_counts_and_false_central_counts_fail(self):
        raw = self.zip_bytes()
        central = raw.index(b"PK\x01\x02")
        nul = bytearray(raw)
        nul[30 + len("packages")] = 0
        nul[central + 46 + len("packages")] = 0
        self.publish(bytes(nul))
        self.rejected()
        for count in [1, 0xffff]:
            changed = bytearray(raw)
            struct.pack_into("<HH", changed, len(changed) - 14, count, count)
            self.publish(bytes(changed))
            self.rejected()

    def test_corrupt_member_crc_rolls_back_without_leaving_partial_files(self):
        raw = bytearray(self.zip_bytes())
        central = raw.index(b"PK\x01\x02")
        bad_crc = struct.unpack_from("<I", raw, central + 16)[0] ^ 1
        struct.pack_into("<I", raw, 14, bad_crc)
        struct.pack_into("<I", raw, central + 16, bad_crc)
        self.publish(bytes(raw))
        self.rejected()

    def test_member_digest_failure_rolls_back_an_already_created_destination(self):
        self.contract["members"]["source-binding.json"]["sha256"] = "0" * 64
        self.rejected()
        self.assertGreater(len(self.api.paths), 0)

    def test_sdk_package_magic_rejects_a_renamed_raw_static_library(self):
        self.members["packages/xybrid.aar"] = b"!<arch>\nraw static engine"
        data = self.members["packages/xybrid.aar"]
        self.contract["members"]["packages/xybrid.aar"].update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.publish(self.zip_bytes())
        self.rejected()

    def test_compressed_wrapper_central_member_total_and_ratio_bounds(self):
        with patch.object(RECEIVE.PRODUCER, "MAX_ARCHIVE", self.archive.stat().st_size - 1):
            self.rejected()
        with patch.object(RECEIVE, "MAX_CENTRAL", 1):
            self.rejected()
        with patch.object(RECEIVE, "MAX_MEMBERS", 1):
            self.rejected()
        with patch.object(RECEIVE, "MAX_UNPACKED", max(len(x) for x in self.members.values())):
            self.rejected()
        data = b"PK\x03\x04" + b"x" * 16000
        self.members["packages/xybrid.aar"] = data
        self.contract["members"]["packages/xybrid.aar"].update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.publish(self.zip_bytes())
        self.rejected()

    def test_existing_file_directory_symlink_and_fifo_destinations_are_never_overwritten(self):
        for kind in ["file", "directory", "symlink", "fifo"]:
            with self.subTest(kind=kind):
                if kind == "directory":
                    self.destination.mkdir()
                elif kind == "symlink":
                    self.destination.symlink_to(self.archive)
                elif kind == "fifo":
                    os.mkfifo(self.destination)
                else:
                    self.destination.write_bytes(b"leave alone")
                self.rejected(destination_remains=True)
                self.assertTrue(os.path.lexists(self.destination))
                if kind == "directory":
                    self.destination.rmdir()
                else:
                    self.destination.unlink()
        self.assertEqual(self.api.paths, [])

    def test_symlinked_destination_parent_is_rejected_before_reads(self):
        link = self.directory / "linked-parent"
        link.symlink_to(self.directory, target_is_directory=True)
        self.destination = link / "received"
        self.rejected()
        self.assertEqual(self.api.paths, [])

    def test_source_archive_links_fifos_and_hardlinks_fail_without_blocking(self):
        original = self.archive
        for kind in ["symlink", "fifo", "hardlink"]:
            with self.subTest(kind=kind):
                self.archive = self.directory / "untrusted.zip"
                if kind == "symlink":
                    self.archive.symlink_to(original)
                elif kind == "fifo":
                    os.mkfifo(self.archive)
                else:
                    os.link(original, self.archive)
                self.rejected()
                self.archive.unlink()
        self.archive = original

    def test_api_failure_before_any_member_read_and_after_extraction_rolls_back(self):
        root = RECEIVE.PRODUCER.API_ROOT
        self.api.responses[root]["id"] = 101
        with patch.object(RECEIVE, "extract_members", side_effect=AssertionError("must not extract")):
            self.rejected()
        self.publish(self.zip_bytes())
        run_path = root + "/actions/runs/300"
        original = copy.deepcopy(self.api.responses[run_path])
        def changed(count):
            if count == 3:
                self.assertTrue((self.destination / "packages/xybrid.aar").is_file())
                return dict(original, conclusion="failure")
            return original
        self.api.responses[run_path] = changed
        self.rejected()
        self.assertEqual(self.api.paths.count(run_path), 3)

    def test_original_path_replacement_cannot_change_the_verified_held_bytes(self):
        root = RECEIVE.PRODUCER.API_ROOT
        original = copy.deepcopy(self.api.responses[root])
        def replace(_count):
            self.archive.write_bytes(b"replacement is never extracted")
            return original
        self.api.responses[root] = replace
        self.receive()
        self.assertEqual((self.destination / "packages/xybrid.aar").read_bytes(),
                         self.members["packages/xybrid.aar"])

    def test_private_snapshot_replacement_after_api_hash_fails_before_extraction(self):
        run_path = RECEIVE.PRODUCER.API_ROOT + "/actions/runs/300"
        original = copy.deepcopy(self.api.responses[run_path])
        def replace(count):
            if count == 2:
                snapshot = next(self.directory.glob(".zzz-receive-*/archive.zip"))
                snapshot.unlink()
                snapshot.write_bytes(b"replacement")
            return original
        self.api.responses[run_path] = replace
        self.rejected()

    def test_whole_scratch_replacement_cannot_delete_an_unrelated_directory(self):
        run_path = RECEIVE.PRODUCER.API_ROOT + "/actions/runs/300"
        original = copy.deepcopy(self.api.responses[run_path])
        replacement = []
        def replace(count):
            if count == 2:
                scratch = next(self.directory.glob(".zzz-receive-*"))
                scratch.rename(self.directory / "moved-owned-snapshot")
                scratch.mkdir()
                sentinel = scratch / "other-owner"
                sentinel.write_bytes(b"leave alone")
                replacement.append(sentinel)
            return original
        self.api.responses[run_path] = replace
        with self.assertRaisesRegex(RECEIVE.Error, "private snapshot cleanup failed"):
            self.receive()
        self.assertEqual(replacement[0].read_bytes(), b"leave alone")
        self.assertFalse(self.destination.exists())

    def test_unacquired_output_identity_cannot_authorize_recursive_rollback(self):
        open_file = RECEIVE.os.open
        def replace(path, flags, *args, **kwargs):
            if path == self.destination.name and flags & os.O_DIRECTORY:
                self.destination.rename(self.directory / "moved-empty-output")
                self.destination.mkdir()
                (self.destination / "other-owner").write_bytes(b"leave alone")
                raise OSError("synthetic output open failure")
            return open_file(path, flags, *args, **kwargs)
        RECEIVE.require_host()
        with patch.object(RECEIVE.os, "open", side_effect=replace), patch.object(RECEIVE, "require_host"):
            with self.assertRaisesRegex(RECEIVE.Error, "output rollback failed"):
                self.receive()
        self.assertEqual((self.destination / "other-owner").read_bytes(), b"leave alone")
        self.assertEqual(list(self.directory.glob(".zzz-receive-*")), [])

    def test_unacquired_scratch_identity_cannot_authorize_recursive_cleanup(self):
        open_file = RECEIVE.os.open
        replacement = []
        def replace(path, flags, *args, **kwargs):
            if str(path).startswith(".zzz-receive-") and flags & os.O_DIRECTORY:
                scratch = self.directory / path
                scratch.rename(self.directory / "moved-empty-snapshot")
                scratch.mkdir()
                sentinel = scratch / "other-owner"
                sentinel.write_bytes(b"leave alone")
                replacement.append(sentinel)
                raise OSError("synthetic scratch open failure")
            return open_file(path, flags, *args, **kwargs)
        RECEIVE.require_host()
        with patch.object(RECEIVE.os, "open", side_effect=replace), patch.object(RECEIVE, "require_host"):
            with self.assertRaisesRegex(RECEIVE.Error, "private snapshot cleanup failed"):
                self.receive()
        self.assertEqual(replacement[0].read_bytes(), b"leave alone")
        self.assertFalse(self.destination.exists())

    def test_private_snapshot_mutation_during_extraction_rolls_back(self):
        extract = RECEIVE.extract_members
        def mutate(*args):
            extract(*args)
            snapshot = next(self.directory.glob(".zzz-receive-*/archive.zip"))
            snapshot.chmod(0o600)
            snapshot.write_bytes(b"changed after extraction")
            snapshot.chmod(0o400)
        with patch.object(RECEIVE, "extract_members", side_effect=mutate):
            self.rejected()

    def test_changed_output_identity_cannot_delete_an_unrelated_replacement(self):
        run_path = RECEIVE.PRODUCER.API_ROOT + "/actions/runs/300"
        original = copy.deepcopy(self.api.responses[run_path])
        def replace(count):
            if count == 3:
                self.destination.rename(self.directory / "moved-owned-output")
                self.destination.mkdir()
                (self.destination / "other-owner").write_bytes(b"leave alone")
            return original
        self.api.responses[run_path] = replace
        with self.assertRaisesRegex(RECEIVE.Error, "rollback failed"):
            self.receive()
        self.assertEqual((self.destination / "other-owner").read_bytes(), b"leave alone")
        self.assertEqual(list(self.directory.glob(".zzz-receive-*")), [])

    def test_final_api_check_cannot_hide_changed_member_bytes_or_extra_inventory(self):
        run_path = RECEIVE.PRODUCER.API_ROOT + "/actions/runs/300"
        for kind in ["changed-bytes", "extra-file", "extra-directory", "hardlink", "writable", "symlink", "missing"]:
            with self.subTest(kind=kind):
                self.publish(self.zip_bytes())
                original = copy.deepcopy(self.api.responses[run_path])
                def mutate(count):
                    if count == 4:
                        member = self.destination / "packages/xybrid.aar"
                        if kind == "changed-bytes":
                            member.chmod(0o600)
                            data = member.read_bytes()
                            member.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
                            member.chmod(0o400)
                        elif kind == "extra-file":
                            (self.destination / "unlisted").write_bytes(b"unexpected")
                        elif kind == "extra-directory":
                            (self.destination / "packages/unlisted").mkdir()
                        elif kind == "hardlink":
                            os.link(member, self.directory / "outside-hardlink")
                        elif kind == "writable":
                            member.chmod(0o600)
                        elif kind == "symlink":
                            member.unlink()
                            member.symlink_to(self.archive)
                        else:
                            member.unlink()
                    return original
                self.api.responses[run_path] = mutate
                self.rejected()
                outside = self.directory / "outside-hardlink"
                if outside.exists():
                    outside.unlink()

    def test_destination_created_during_verification_is_not_removed_or_overwritten(self):
        run_path = RECEIVE.PRODUCER.API_ROOT + "/actions/runs/300"
        original = copy.deepcopy(self.api.responses[run_path])
        def reserve(count):
            if count == 2:
                self.destination.mkdir()
                (self.destination / "existing").write_bytes(b"other owner")
            return original
        self.api.responses[run_path] = reserve
        self.rejected(destination_remains=True)
        self.assertEqual((self.destination / "existing").read_bytes(), b"other owner")

    def test_snapshot_name_collision_cannot_delete_a_directory_it_did_not_create(self):
        collision = self.directory / (".zzz-receive-" + "c" * 32)
        collision.mkdir()
        sentinel = collision / "existing"
        sentinel.write_bytes(b"other owner")
        with patch.object(RECEIVE.secrets, "token_hex", return_value="c" * 32):
            with self.assertRaises((RECEIVE.Error, OSError)):
                self.receive()
        self.assertEqual(sentinel.read_bytes(), b"other owner")


class ReceiverCliBoundaryTests(unittest.TestCase):
    def arguments(self, extra=()):
        return ["receiver", "--trusted-producer-policy", "/private-sentinel/policy.json",
                "--trusted-extraction-contract", "/private-sentinel/contract.json",
                "--run-id", "300", "--artifact-id", "400",
                "--archive", "/private-sentinel/archive.zip",
                "--destination", "/private-sentinel/output", *extra]

    def main(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", arguments), patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            result = RECEIVE.main()
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("private-sentinel", stderr.getvalue())
        self.assertLess(len(stderr.getvalue()), 512)
        return result, stderr.getvalue()

    def test_credential_presence_is_rejected_before_policy_archive_api_or_destination_reads(self):
        for name in RECEIVE.PRODUCER.FORBIDDEN_ENV:
            with self.subTest(name=name), patch.dict(os.environ, {name: "DO-NOT-ECHO-SECRET"}, clear=True), \
                    patch.object(RECEIVE, "read_json") as read, \
                    patch.object(RECEIVE.PRODUCER, "GitHub") as api:
                result, message = self.main(self.arguments())
                self.assertEqual(result, 1)
                self.assertIn("credential-free", message)
                self.assertNotIn("DO-NOT-ECHO-SECRET", message)
                read.assert_not_called()
                api.assert_not_called()

    def test_unsupported_host_is_rejected_before_input_reads(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(RECEIVE.PRODUCER, "posix_host", return_value=False), \
                patch.object(RECEIVE, "read_json") as read:
            result, message = self.main(self.arguments())
            self.assertEqual(result, 1)
            self.assertIn("POSIX", message)
            read.assert_not_called()

    def test_missing_directory_primitive_fails_without_weaker_fallback(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(RECEIVE.PRODUCER, "require_posix_host"), \
                patch.object(RECEIVE.os, "O_DIRECTORY", None, create=True), \
                patch.object(RECEIVE, "read_json") as read:
            result, message = self.main(self.arguments())
            self.assertEqual(result, 1)
            self.assertIn("directory descriptors", message)
            read.assert_not_called()

    def test_saved_success_json_or_other_unknown_cli_arguments_cannot_authorize_reads(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(RECEIVE, "read_json") as read:
            result, message = self.main(self.arguments(("--verified-record", "/private-sentinel/success.json")))
            self.assertEqual(result, 1)
            self.assertIn("invalid command-line arguments", message)
            read.assert_not_called()

    def test_help_is_portable_and_does_not_touch_inputs(self):
        with patch.object(sys, "argv", ["receiver", "--help"]), \
                patch("sys.stdout", io.StringIO()), patch.object(RECEIVE, "read_json") as read, \
                self.assertRaises(SystemExit) as raised:
            RECEIVE.main()
        self.assertEqual(raised.exception.code, 0)
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
