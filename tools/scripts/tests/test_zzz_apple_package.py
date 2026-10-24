"""Offline Apple final-package tests; no native build, downloads or qualification."""

from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import plistlib
import stat
import tempfile
import types
import unittest
from unittest import mock
import warnings
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("apple_notice_packager", ROOT / "bindings/apple/pack_zzz_notices.py")
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)
CONTRACT = dict(PACK.PULL.SCHEMA5_NOTICE_FILES)


@unittest.skipUnless(os.name == "posix", "packager requires POSIX regular-file checks")
class ApplePackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = {name: ("offline test notice: " + name + "\n").encode() for name in CONTRACT}
        self.contract = {name: hashlib.sha256(data).hexdigest() for name, data in self.payload.items()}
        self.patch = mock.patch.object(PACK.PULL, "SCHEMA5_NOTICE_FILES", self.contract)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.device = self.notices("device")
        self.simulator = self.notices("simulator")
        self.archive = self.root / "input.zip"
        self.output = self.root / "output.zip"
        self.base = {name: ("native fixture: " + name).encode() for name in PACK.BASE_FILES}
        self.base[PACK.FRAMEWORK + "/Info.plist"] = plistlib.dumps({
            "CFBundlePackageType": "XFWK", "AvailableLibraries": [
                {"LibraryIdentifier": "ios-arm64", "LibraryPath": "XybridFFI.framework",
                 "SupportedArchitectures": ["arm64"], "SupportedPlatform": "ios"},
                {"LibraryIdentifier": "ios-arm64-simulator", "LibraryPath": "XybridFFI.framework",
                 "SupportedArchitectures": ["arm64"], "SupportedPlatform": "ios",
                 "SupportedPlatformVariant": "simulator"},
            ]})
        self.write_zip(self.base)

    def notices(self, directory):
        values = []
        for name, data in self.payload.items():
            path = self.root / directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            values.append(name + "=" + str(path))
        return values

    def write_zip(self, contents, *, mutate=None):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, data in contents.items():
                info = PACK.output_info(name)
                if mutate:
                    mutate(info)
                archive.writestr(info, data)

    def package(self):
        PACK.package_archive(self.archive, self.device, self.simulator, self.output)

    def rejected(self, expected=None):
        with self.assertRaisesRegex((PACK.PackageError, zipfile.BadZipFile, OSError, zlib.error), expected or ".*"):
            self.package()
        self.assertFalse(self.output.exists())
        self.assertEqual([], list(self.root.glob(".zzz-apple-*")))

    def test_reviewed_contract_is_immutable_and_has_fourteen_files(self):
        self.assertEqual(14, len(CONTRACT))
        self.assertEqual("a634e1fb5e0cdfa87efea950c72ff9bb522c6620cec16924395820e24b06c9ce",
                         hashlib.sha256(json.dumps(CONTRACT, sort_keys=True, separators=(",", ":")).encode()).hexdigest())

    def test_packages_both_inventories_preserves_native_bytes_and_verifies(self):
        original = self.archive.read_bytes()
        self.package()
        PACK.verify_archive(self.output)
        with zipfile.ZipFile(self.output) as archive:
            expected = self.base | {PACK.NOTICE_ROOT + name: data for name, data in self.payload.items()}
            self.assertEqual(expected, {name: archive.read(name) for name in archive.namelist()})
            self.assertEqual(23, len(archive.namelist()))
        self.assertEqual(original, self.archive.read_bytes())
        self.assertEqual(0, PACK.main(["verify", "--archive", str(self.output)]))

    def test_bazel_symlink_inputs_resolve_to_checked_regular_targets(self):
        link = self.root / "bazel-input.zip"
        link.symlink_to(self.archive)
        self.archive = link
        name, _, path = self.device[0].partition("=")
        notice = self.root / "bazel-notice"
        notice.symlink_to(path)
        self.device[0] = name + "=" + str(notice)
        self.package()
        PACK.verify_archive(self.output)

    def test_every_changed_slice_notice_fails_even_when_both_slices_agree(self):
        for name, data in self.payload.items():
            for directory in ("device", "simulator"):
                (self.root / directory / name).write_bytes(data + b" changed")
            self.rejected("immutable schema-5")
            for directory in ("device", "simulator"):
                (self.root / directory / name).write_bytes(data)

    def test_correct_device_cannot_authorize_changed_simulator(self):
        (self.root / "simulator" / "RUNTIME_TERMS.md").write_bytes(b"changed")
        self.rejected("immutable schema-5")

    def test_missing_extra_duplicate_and_alias_notice_input_names_fail(self):
        original = list(self.simulator)
        for values in (original[:-1], original + ["extra=private"], original + original[:1],
                       [original[0].replace("LICENSES/", "licenses/", 1)] + original[1:]):
            self.simulator = values
            self.rejected("exact schema-5 inventory")

    def test_missing_hardlinked_fifo_and_oversized_notice_files_fail(self):
        path = self.root / "simulator" / "RUNTIME_TERMS.md"
        data = path.read_bytes()
        path.unlink()
        self.rejected()
        path.write_bytes(data)
        linked = self.root / "hardlink"
        os.link(path, linked)
        self.rejected("single-link regular")
        linked.unlink()
        path.unlink()
        os.mkfifo(path)
        self.rejected("single-link regular")
        path.unlink()
        path.write_bytes(b"x" * (PACK.MAX_NOTICE + 1))
        self.rejected("single-link regular")

    def test_existing_complete_correct_notices_are_idempotent(self):
        self.package()
        self.archive = self.output
        self.output = self.root / "second.zip"
        self.package()
        PACK.verify_archive(self.output)
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(23, len(archive.namelist()))

    def test_partial_conflicting_and_extra_existing_notices_fail(self):
        notices = {PACK.NOTICE_ROOT + name: data for name, data in self.payload.items()}
        for values in ({next(iter(notices)): b"partial"}, notices | {PACK.NOTICE_ROOT + "RUNTIME_TERMS.md": b"conflict"},
                       notices | {PACK.NOTICE_ROOT + "extra.txt": b"extra"}):
            self.write_zip(self.base | values)
            self.rejected()

    def test_verify_requires_complete_notices_and_checks_every_hash(self):
        with self.assertRaisesRegex(PACK.PackageError, "inventory is missing"):
            PACK.verify_archive(self.archive)
        self.write_zip(self.base | {PACK.NOTICE_ROOT + name: data for name, data in self.payload.items()})
        for name, data in self.payload.items():
            contents = self.base | {PACK.NOTICE_ROOT + n: d for n, d in self.payload.items()}
            contents[PACK.NOTICE_ROOT + name] = data + b"changed"
            self.write_zip(contents)
            with self.assertRaisesRegex(PACK.PackageError, "immutable schema-5"):
                PACK.verify_archive(self.archive)

    def test_missing_native_payload_or_unexpected_root_fails(self):
        contents = dict(self.base)
        contents.pop(next(iter(contents)))
        self.write_zip(contents)
        self.rejected()
        self.write_zip(self.base | {"other.framework/file": b"extra"})
        self.rejected()

    def test_traversal_alias_duplicate_and_link_members_fail(self):
        for name in ("../escape", PACK.FRAMEWORK + "/../escape", PACK.FRAMEWORK.lower() + "/Info.plist"):
            self.write_zip(self.base | {name: b"extra"})
            self.rejected()
        self.write_zip(self.base)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.archive, "a") as archive:
                archive.writestr(PACK.output_info(next(iter(self.base))), b"duplicate")
        self.rejected("duplicate")
        self.write_zip(self.base, mutate=lambda info: setattr(info, "external_attr", (stat.S_IFLNK | 0o777) << 16))
        self.rejected("linked")

    def test_wrong_simulator_plist_fails(self):
        name = PACK.FRAMEWORK + "/Info.plist"
        metadata = plistlib.loads(self.base[name])
        metadata["AvailableLibraries"][1]["SupportedPlatformVariant"] = "device"
        self.write_zip(self.base | {name: plistlib.dumps(metadata)})
        self.rejected("metadata differs")

    def test_envelope_bounds_reject_before_zipfile_parses(self):
        raw = bytearray(self.archive.read_bytes())
        raw[-12:-8] = (PACK.MAX_CENTRAL + 1).to_bytes(4, "little")
        self.archive.write_bytes(raw)
        with mock.patch.object(PACK.zipfile, "ZipFile", side_effect=AssertionError("must not parse")):
            self.rejected("envelope")

    def test_crc_failure_removes_partial_output(self):
        self.write_zip(self.base)
        raw = bytearray(self.archive.read_bytes())
        with zipfile.ZipFile(self.archive) as archive:
            info = archive.getinfo(next(name for name in self.base if not name.endswith("Info.plist")))
            start = info.header_offset
            name_length = int.from_bytes(raw[start + 26:start + 28], "little")
            extra_length = int.from_bytes(raw[start + 28:start + 30], "little")
            raw[start + 30 + name_length + extra_length + info.compress_size // 2] ^= 1
        self.archive.write_bytes(raw)
        self.rejected()

    def test_existing_output_is_never_overwritten(self):
        self.output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.package()
        self.assertEqual(b"keep", self.output.read_bytes())
        self.assertEqual([], list(self.root.glob(".zzz-apple-*")))

    def test_safe_cli_errors_do_not_echo_paths_or_tracebacks(self):
        for args in (["verify", "--archive", "/private/customer/path"], ["--private-secret"]):
            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(1, PACK.main(args))
            self.assertNotIn("private", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())


class AppleRuleTests(unittest.TestCase):
    def rule_source(self):
        namespace = {"rule": lambda **kw: kw["implementation"],
                     "attr": types.SimpleNamespace(label=lambda **kw: None, bool=lambda **kw: None),
                     "DefaultInfo": lambda **kw: kw, "depset": lambda value: value,
                     "fail": lambda message: (_ for _ in ()).throw(ValueError(message))}
        path = ROOT / "bindings/apple/zzz_notices.bzl"
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        return namespace

    def test_real_rule_declares_both_slice_inventories_and_archive_as_action_inputs(self):
        namespace = self.rule_source()
        self.assertEqual(sorted(CONTRACT), namespace["_NOTICE_PATHS"])
        device = [types.SimpleNamespace(path="external/zzz_ios/notices/" + name) for name in CONTRACT]
        simulator = [types.SimpleNamespace(path="external/zzz_ios_sim/notices/" + name) for name in CONTRACT]
        arguments = []
        args = types.SimpleNamespace(add=lambda *values: arguments.extend(values))
        tool = mock.MagicMock()
        ctx = types.SimpleNamespace(
            attr=types.SimpleNamespace(enabled=True, _packager=tool),
            files=types.SimpleNamespace(device_notices=device, simulator_notices=simulator),
            file=types.SimpleNamespace(archive="native.zip"), executable=types.SimpleNamespace(_packager="tool"),
            label=types.SimpleNamespace(name="XybridFFI_zzz"), actions=types.SimpleNamespace(
                declare_file=lambda name: name, args=lambda: args, run=mock.Mock()))
        result = namespace["zzz_xcframework"](ctx)
        self.assertEqual([{"files": ["XybridFFI_zzz.xcframework.zip"]}], result)
        self.assertEqual(["native.zip"] + device + simulator, ctx.actions.run.call_args.kwargs["inputs"])
        self.assertEqual(14, arguments.count("--device-notice"))
        self.assertEqual(14, arguments.count("--simulator-notice"))
        self.assertIn("RUNTIME_TERMS.md=external/zzz_ios_sim/notices/RUNTIME_TERMS.md", arguments)
        for files in (device[:-1], device + device[:1], device + [types.SimpleNamespace(path="extra")]):
            ctx.files.device_notices = files
            with self.assertRaises(ValueError):
                namespace["zzz_xcframework"](ctx)
        ctx.attr.enabled = False
        with self.assertRaisesRegex(ValueError, "explicit"):
            namespace["zzz_xcframework"](ctx)
