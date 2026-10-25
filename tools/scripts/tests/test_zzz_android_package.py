"""Offline Android final-package tests; no native build, downloads or qualification."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock
import warnings
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("android_zzz_packager", ROOT / "bindings/kotlin/pack_zzz_aar.py")
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)
CONTRACT = dict(PACK.PULL.SCHEMA5_NOTICE_FILES)
MANIFEST = (ROOT / "bindings/kotlin/bazel/AndroidManifest.xml").read_bytes()


class Select:
    """A select() stand-in that, like Starlark's, concatenates with lists."""

    def __init__(self, value):
        self.value = value

    def __add__(self, _other):
        return self

    __radd__ = __add__


def kotlin_build():
    """Evaluate bindings/kotlin/BUILD.bazel with recording stubs."""
    targets = {}

    def record(kind):
        def save(**kwargs):
            targets[kwargs["name"]] = (kind, kwargs)
        return save

    ggml = {}
    path = ROOT / "bazel/ggml/android.bzl"
    exec(compile(path.read_text(), str(path), "exec"), ggml)
    namespace = {"load": lambda *args: None, "select": Select, "glob": lambda *args, **kwargs: [],
                 "ANDROID_ARM64_GGML_LIBS": ggml["ANDROID_ARM64_GGML_LIBS"],
                 "ANDROID_ARM64_CPU_VARIANTS": ggml["ANDROID_ARM64_CPU_VARIANTS"]}
    for kind in ("define_kt_toolchain", "cc_import", "cc_binary", "platform_transition_filegroup",
                 "pkg_files", "filegroup", "kt_jvm_library", "pkg_zip", "py_binary", "zzz_aar"):
        namespace[kind] = record(kind)
    path = ROOT / "bindings/kotlin/BUILD.bazel"
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace, targets, ggml


BUILD, TARGETS, GGML = kotlin_build()
BASE_FILES = BUILD["XYBRID_KOTLIN_AAR_FILES"]


def elf(machine=183, elf_class=2, order=1, kind=3):
    return PACK.ELF_HEADER.pack(b"\x7fELF", elf_class, order, 1, kind, machine, 1) + b"\0" * 64


def zip_info(name, directory=False):
    # The metadata rules_pkg's pkg_zip writes for the real AAR.
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    if directory:
        info.external_attr = (stat.S_IFDIR | 0o755) << 16
        info.compress_type = zipfile.ZIP_STORED
    else:
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
    return info


@unittest.skipUnless(os.name == "posix", "packager requires POSIX regular-file checks")
class AndroidPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = {name: ("offline test notice: " + name + "\n").encode() for name in CONTRACT}
        self.contract = {name: hashlib.sha256(data).hexdigest() for name, data in self.payload.items()}
        self.patch = mock.patch.object(PACK.PULL, "SCHEMA5_NOTICE_FILES", self.contract)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.notices = []
        for name, data in self.payload.items():
            path = self.root / "notices" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.notices.append(name + "=" + str(path))
        self.companion = self.root / "libxybrid_zzz_tts.so"
        self.companion.write_bytes(elf())
        self.archive = self.root / "xybrid-kotlin.aar"
        self.output = self.root / "xybrid-kotlin-zzz.aar"
        self.base = {name: ("native fixture: " + name).encode() for name in BASE_FILES}
        self.base["AndroidManifest.xml"] = MANIFEST
        self.base["R.txt"] = b""
        self.base_files = list(BASE_FILES)
        self.write_base(self.base)

    def write_base(self, contents, *, mutate=None, comment=b""):
        directories = sorted({name[:index + 1] for name in contents for index, char in enumerate(name) if char == "/"})
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name in directories:
                archive.writestr(zip_info(name, directory=True), b"")
            for name, data in contents.items():
                info = zip_info(name)
                if mutate:
                    mutate(name, info)
                archive.writestr(info, data)
            archive.comment = comment

    def set_flag(self, name, bit):
        data = bytearray(self.archive.read_bytes())
        with zipfile.ZipFile(self.archive) as archive:
            local = archive.getinfo(name).header_offset
        data[local + 6] |= bit
        central = data.find(b"PK\x01\x02")
        while central != -1:
            length = struct.unpack_from("<H", data, central + 28)[0]
            if data[central + 46:central + 46 + length] == name.encode():
                data[central + 8] |= bit
                break
            central = data.find(b"PK\x01\x02", central + 4)
        self.assertNotEqual(-1, central)
        self.archive.write_bytes(bytes(data))

    def package(self):
        PACK.package_archive(self.archive, self.companion, self.base_files, self.notices, self.output)

    def rejected(self, expected=None):
        with self.assertRaisesRegex((PACK.PackageError, zipfile.BadZipFile, OSError, zlib.error, ValueError),
                                    expected or ".*"):
            self.package()
        self.assertFalse(self.output.exists())
        self.assertEqual([], list(self.root.glob(".zzz-android-*")))

    def test_build_declares_the_eighteen_file_three_abi_base(self):
        self.assertEqual(18, len(BASE_FILES))
        self.assertEqual(len(BASE_FILES), len(set(BASE_FILES)))
        for lib in GGML["ANDROID_ARM64_GGML_LIBS"] + GGML["ANDROID_ARM64_CPU_VARIANTS"]:
            self.assertIn("jni/arm64-v8a/" + lib, BASE_FILES)
        self.assertNotIn("jni/armeabi-v7a/libonnxruntime.so", BASE_FILES)
        self.assertNotIn(PACK.COMPANION, BASE_FILES)
        self.assertEqual(PACK.base_inventory(BASE_FILES), set(BASE_FILES))

    def test_build_wires_an_opt_in_manual_wrapper_from_reviewed_inputs(self):
        kind, wrapper = TARGETS["xybrid_kotlin_zzz_aar"]
        self.assertEqual("zzz_aar", kind)
        self.assertEqual(":xybrid_kotlin_aar", wrapper["base"])
        self.assertEqual("//bazel/zzz/companion:companion", wrapper["companion"])
        self.assertEqual("@zzz_android//:notices", wrapper["notices"])
        self.assertEqual(BASE_FILES, wrapper["base_files"])
        self.assertEqual({"//:tts_zzz_android_companion_enabled": True, "//conditions:default": False},
                         wrapper["enabled"].value)
        self.assertEqual(["manual"], wrapper["tags"])
        self.assertEqual("xybrid-kotlin.aar", TARGETS["xybrid_kotlin_aar"][1]["out"])
        self.assertNotIn("tags", TARGETS["xybrid_kotlin_aar"][1])

    def test_packages_companion_and_notices_and_preserves_base_bytes(self):
        original = self.archive.read_bytes()
        self.package()
        PACK.verify_archive(self.output, self.base_files)
        with zipfile.ZipFile(self.archive) as base, zipfile.ZipFile(self.output) as result:
            files = {info.filename: result.read(info) for info in result.infolist() if not info.is_dir()}
            expected = dict(self.base)
            expected[PACK.COMPANION] = elf()
            expected.update({PACK.NOTICE_ROOT + name: data for name, data in self.payload.items()})
            self.assertEqual(expected, files)
            self.assertEqual(33, len(files))
            # Base entries, directories included, keep their order and metadata.
            count = len(base.infolist())
            self.assertEqual([i.filename for i in base.infolist()], [i.filename for i in result.infolist()][:count])
            for before, after in zip(base.infolist(), result.infolist()):
                self.assertEqual((before.date_time, before.external_attr, before.create_system),
                                 (after.date_time, after.external_attr, after.create_system))
        self.assertEqual(original, self.archive.read_bytes())
        arguments = ["verify", "--archive", str(self.output)]
        for name in self.base_files:
            arguments += ["--base-file", name]
        self.assertEqual(0, PACK.main(arguments))

    def test_bazel_symlink_inputs_resolve_to_checked_regular_targets(self):
        for attribute in ("archive", "companion"):
            link = self.root / ("link-" + attribute)
            link.symlink_to(getattr(self, attribute))
            setattr(self, attribute, link)
        self.package()
        PACK.verify_archive(self.output, self.base_files)

    def test_base_must_match_the_declared_inventory_exactly(self):
        missing = dict(self.base)
        del missing["jni/armeabi-v7a/libxybrid_bolt.so"]
        self.write_base(missing)
        self.rejected("AAR payload, companion or complete notice inventory differs")
        # Undeclared members, including a pre-placed companion, fail as
        # unexpected payload before the inventory comparison.
        for name in ("jni/x86_64/libextra.so", PACK.COMPANION, "assets/other.txt"):
            with self.subTest(name=name):
                self.write_base(dict(self.base, **{name: elf()}))
                self.rejected("unexpected(?: AAR ZIP directory|, linked or beyond-bound AAR ZIP payload)")

    def test_declared_inventory_must_keep_the_three_abi_layout(self):
        # Each archive matches its declared list, so only the layout check can reject it.
        for files in (BASE_FILES[1:], BASE_FILES + [PACK.COMPANION], BASE_FILES + ["jni/riscv64/libx.so"],
                      [name for name in BASE_FILES if name != "jni/x86_64/libc++_shared.so"],
                      BASE_FILES + ["assets/extra.txt"]):
            with self.subTest(files=files):
                self.base_files = files
                self.write_base({name: self.base.get(name, b"x") for name in files})
                self.rejected("reviewed three-ABI layout")
        self.base_files = BASE_FILES + BASE_FILES[:1]
        self.rejected("duplicate base AAR inventory entry")

    def test_every_changed_missing_or_duplicate_notice_fails(self):
        notices = list(self.notices)
        for values in (notices[:-1], notices + notices[:1],
                       notices[:-1] + ["UNKNOWN.txt=" + notices[-1].partition("=")[2]]):
            with self.subTest(values=len(values)):
                self.notices = values
                self.rejected("schema-5 inventory")
        self.notices = notices
        Path(notices[0].partition("=")[2]).write_bytes(b"changed")
        self.rejected("immutable schema-5 contract")

    def test_companion_must_be_a_little_endian_arm64_shared_object(self):
        for data in (b"not elf", elf(machine=62), elf(elf_class=1), elf(order=2), elf(kind=2)):
            with self.subTest(data=data[:20]):
                self.companion.write_bytes(data)
                self.rejected("ARM64 ELF shared object|not an ELF")

    def test_manifest_must_keep_package_and_min_sdk(self):
        commented = MANIFEST.replace(b'<uses-sdk android:minSdkVersion="24" />',
                                     b"<!-- <uses-sdk android:minSdkVersion=\"24\" /> -->"
                                     b"<uses-sdk android:minSdkVersion='29' />")
        for manifest in (MANIFEST.replace(b'"24"', b'"29"'), MANIFEST.replace(b"ai.xybrid", b"ai.other"),
                         MANIFEST.replace(b'<uses-sdk android:minSdkVersion="24" />', b""), commented,
                         MANIFEST.replace(b'<uses-sdk android:minSdkVersion="24" />',
                                          b'<!-- <uses-sdk android:minSdkVersion="24" /> -->')):
            with self.subTest(manifest=manifest[-80:]):
                self.write_base(dict(self.base, **{"AndroidManifest.xml": manifest}))
                self.rejected("minSdk 24")

    def test_unsafe_names_metadata_and_envelopes_fail(self):
        def link(name, info):
            if name == "R.txt":
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
        def extra(name, info):
            if name == "R.txt":
                info.extra = struct.pack("<HH", 0xCAFE, 0)
        def setuid(name, info):
            if name == "classes.jar":
                info.external_attr = (stat.S_IFREG | 0o4755) << 16
        def dos(name, info):
            if name == "classes.jar":
                info.create_system = 0
        for mutate in (link, extra, setuid, dos):
            with self.subTest(mutate=mutate.__name__):
                self.write_base(self.base, mutate=mutate)
                self.rejected()
        # zipfile resets flag bits on write, so set them in the written headers.
        for bit in (0x8, 0x1):
            with self.subTest(flag=bit):
                self.write_base(self.base)
                self.set_flag("R.txt", bit)
                self.rejected("unsupported AAR ZIP member metadata")
        self.write_base(self.base, comment=b"comment")
        self.rejected("envelope")
        self.write_base(self.base)
        with self.archive.open("ab") as stream:
            stream.write(b"trailing")
        self.rejected("envelope")
        for name in ("../escape.so", "jni/arm64-v8a/../../x", "jni/ARM64-V8A/libxybrid_bolt.so"):
            with self.subTest(name=name):
                self.write_base(dict(self.base, **{name: b"x"}))
                self.rejected()

    def test_compression_ratio_directories_and_duplicates_are_bounded(self):
        self.write_base(dict(self.base, **{"classes.jar": b"\0" * (1024 * 1024)}))
        self.rejected("unexpected, linked or beyond-bound AAR ZIP payload")
        self.write_base(self.base)
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr(zip_info("jni/riscv64/", directory=True), b"")
        self.rejected("unexpected AAR ZIP directory")
        self.write_base(self.base)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zipfile.ZipFile(self.archive, "a") as archive:
                archive.writestr(zip_info("R.txt"), b"second")
        self.rejected("unsafe, aliased or duplicate AAR ZIP member")
        self.write_base(dict(self.base, **{"r.TXT": b"alias"}))
        self.rejected()

    def test_existing_output_is_never_overwritten(self):
        self.output.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            self.package()
        self.assertEqual(b"existing", self.output.read_bytes())
        self.assertEqual([], list(self.root.glob(".zzz-android-*")))

    def test_verify_rejects_incomplete_or_changed_final_packages(self):
        self.package()
        with zipfile.ZipFile(self.output) as archive:
            final = {info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()}
        cases = {
            "missing companion": {k: v for k, v in final.items() if k != PACK.COMPANION},
            "missing notice": {k: v for k, v in final.items() if not k.endswith("RUNTIME_TERMS.md")},
            "changed notice": dict(final, **{PACK.NOTICE_ROOT + "RUNTIME_TERMS.md": b"changed"}),
            "wrong companion": dict(final, **{PACK.COMPANION: elf(machine=62)}),
        }
        for label, contents in cases.items():
            with self.subTest(label=label):
                candidate = self.root / (label.replace(" ", "-") + ".aar")
                with zipfile.ZipFile(candidate, "w") as archive:
                    for name, data in contents.items():
                        archive.writestr(zip_info(name), data)
                with self.assertRaises(PACK.PackageError):
                    PACK.verify_archive(candidate, self.base_files)

    def test_verify_detects_corrupted_base_payload(self):
        self.package()
        data = bytearray(self.output.read_bytes())
        with zipfile.ZipFile(self.output) as archive:
            info = archive.getinfo("jni/x86_64/libxybrid_bolt.so")
        start = info.header_offset + 30 + len(info.filename)
        data[start] ^= 0xFF
        self.output.write_bytes(bytes(data))
        with self.assertRaises((PACK.PackageError, zipfile.BadZipFile, zlib.error)):
            PACK.verify_archive(self.output, self.base_files)

    def readelf(self, symbols):
        reports = {"--file-header": "Class: ELF64\nData: 2's complement, little endian\n"
                                    "Type: DYN (Shared object file)\nMachine: AArch64\n",
                   "--dyn-syms": "Symbol table '.dynsym' contains 2 entries:\n" + symbols,
                   "--dynamic": "0x1 (NEEDED) Shared library: [libc.so]\n"
                                "0xe (SONAME) Library soname: [libxybrid_zzz_tts.so]\n",
                   "--program-headers": "  LOAD 0x0 0x0 0x0 0x100 0x100 R E 0x4000\n"
                                        "  GNU_STACK 0x0 0x0 0x0 0x0 0x0 RW 0x0\n"}
        tool = self.root / "fake-readelf"
        tool.write_text("#!" + sys.executable + "\nimport sys\nprint(" + repr(reports) + "[sys.argv[1]])\n")
        tool.chmod(0o755)
        return tool

    def test_verify_runs_the_companion_elf_audit_when_given_readelf(self):
        self.package()
        getter = " 1: 00000100 32 FUNC GLOBAL DEFAULT 10 xybrid_tts_get_api_v1\n"
        with redirect_stdout(io.StringIO()):
            PACK.verify_archive(self.output, self.base_files, self.readelf(getter))
        leaked = getter + " 2: 00000200 16 FUNC GLOBAL DEFAULT 10 zzz_embed_open\n"
        with redirect_stderr(io.StringIO()), self.assertRaisesRegex(PACK.PackageError, "ELF audit"):
            PACK.verify_archive(self.output, self.base_files, self.readelf(leaked))

    def test_errors_never_disclose_private_paths(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(1, PACK.main(["pack", "--archive", str(self.root / "private-missing.aar"),
                                           "--companion", str(self.companion), "--output", str(self.output)]))
        self.assertNotIn(str(self.root), stderr.getvalue())
        self.assertTrue(stderr.getvalue().startswith("Android zzz package: "))


class AndroidRuleTests(unittest.TestCase):
    def rule(self):
        namespace = {"rule": lambda **kw: kw["implementation"],
                     "attr": types.SimpleNamespace(label=lambda **kw: None, bool=lambda **kw: None,
                                                   string=lambda **kw: None, string_list=lambda **kw: None),
                     "DefaultInfo": lambda **kw: kw, "depset": lambda value: value,
                     "fail": lambda message: (_ for _ in ()).throw(ValueError(message))}
        path = ROOT / "bindings/kotlin/zzz_aar.bzl"
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        return namespace["zzz_aar"]

    def context(self, notices, enabled=True):
        arguments = []
        args = types.SimpleNamespace(
            add=lambda *values: arguments.extend(str(value) for value in values),
            add_all=lambda values, before_each: [arguments.extend([before_each, value]) for value in values])
        ctx = types.SimpleNamespace(
            attr=types.SimpleNamespace(enabled=enabled, out="xybrid-kotlin-zzz.aar", base_files=BASE_FILES,
                                       _packager=mock.MagicMock()),
            file=types.SimpleNamespace(base="base.aar", companion="libxybrid_zzz_tts.so"),
            files=types.SimpleNamespace(notices=notices), executable=types.SimpleNamespace(_packager="tool"),
            actions=types.SimpleNamespace(declare_file=lambda name: name, args=lambda: args, run=mock.Mock()))
        return ctx, arguments

    def test_rule_passes_base_inventory_companion_and_named_notices(self):
        notices = [types.SimpleNamespace(path="external/+zzz+zzz_android/notices/" + name) for name in CONTRACT]
        ctx, arguments = self.context(notices)
        result = self.rule()(ctx)
        self.assertEqual([{"files": ["xybrid-kotlin-zzz.aar"]}], result)
        self.assertEqual(["base.aar", "libxybrid_zzz_tts.so"] + notices, ctx.actions.run.call_args.kwargs["inputs"])
        self.assertEqual(len(BASE_FILES), arguments.count("--base-file"))
        self.assertEqual(14, arguments.count("--notice"))
        self.assertIn("RUNTIME_TERMS.md=external/+zzz+zzz_android/notices/RUNTIME_TERMS.md", arguments)
        self.assertEqual(["pack", "--archive", "base.aar", "--companion", "libxybrid_zzz_tts.so"], arguments[:5])

    def test_rule_rejects_disabled_mode_and_ambiguous_notices(self):
        notices = [types.SimpleNamespace(path="external/zzz_android/notices/" + name) for name in CONTRACT]
        ctx, _ = self.context(notices, enabled=False)
        with self.assertRaisesRegex(ValueError, "tts_zzz_android_companion"):
            self.rule()(ctx)
        for files in (notices + notices[:1], notices + [types.SimpleNamespace(path="external/x/other.txt")],
                      notices + [types.SimpleNamespace(path="a/notices/b/notices/c")]):
            ctx, _ = self.context(files)
            with self.assertRaisesRegex(ValueError, "canonical notice exports"):
                self.rule()(ctx)


if __name__ == "__main__":
    unittest.main()
