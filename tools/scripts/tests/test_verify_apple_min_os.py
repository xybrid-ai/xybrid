"""Unit tests for verify_apple_min_os.py over synthetic Mach-O objects and archives."""

import contextlib
import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verify_apple_min_os as verifier

REPO = Path(__file__).resolve().parents[3]

MACOS, IOS, IOS_SIMULATOR = 1, 2, 7
LC_SEGMENT_64, LC_VERSION_MIN_MACOSX, LC_VERSION_MIN_IPHONEOS = 0x19, 0x24, 0x25


def packed(version):
    major, minor, patch = verifier.parse_version(version)
    return major << 16 | minor << 8 | patch


def build_version(platform, minos, order="<"):
    return struct.pack(order + "6I", 0x32, 24, platform, packed(minos), packed("26.0"), 0)


def version_min(cmd, version):
    return struct.pack("<4I", cmd, 16, packed(version), packed("26.0"))


def segment():
    """A load command the check must step over."""
    return struct.pack("<2I", LC_SEGMENT_64, 72) + bytes(64)


def macho(*commands, bits=64, order="<"):
    """A Mach-O object: header, load commands, then some section bytes."""
    body = b"".join(commands)
    magic = 0xFEEDFACF if bits == 64 else 0xFEEDFACE
    fields = (magic, 0x0100000C, 0, 1, len(commands), len(body), 0)
    header = struct.pack(order + "7I", *fields) + (bytes(4) if bits == 64 else b"")
    return header + body + b"\x90" * 33


def ar_header(name, size):
    return f"{name:<16}{0:<12}{0:<6}{0:<6}{644:<8}{size:<10}`\n".encode()


def pad(data):
    return data + (b"\n" if len(data) % 2 else b"")


def bsd_archive(members):
    """The layout Apple's and llvm-ar's darwin format write: `#1/<len>` names."""
    out = b"!<arch>\n"
    for name, data in [("__.SYMDEF SORTED", bytes(8))] + list(members):
        raw = name.encode().ljust((len(name) + 8) // 8 * 8, b"\0")
        out += ar_header(f"#1/{len(raw)}", len(raw) + len(data)) + pad(raw + data)
    return out


def gnu_archive(members):
    """The GNU layout: `/` symbol table, `//` long names, `name/` short names."""
    members = list(members)
    long_names, offsets = b"", {}
    for name, _ in members:
        if len(name) > 15:
            offsets[name] = len(long_names)
            long_names += name.encode() + b"/\n"
    out = b"!<arch>\n" + ar_header("/", 4) + bytes(4) + ar_header("//", len(long_names)) + pad(long_names)
    for name, data in members:
        label = f"/{offsets[name]}" if name in offsets else f"{name}/"
        out += ar_header(label, len(data)) + pad(data)
    return out


class VersionTests(unittest.TestCase):
    def test_parse_format_and_decode(self):
        self.assertEqual(verifier.parse_version("16"), (16, 0, 0))
        self.assertEqual(verifier.parse_version("13.4"), (13, 4, 0))
        self.assertEqual(verifier.format_version((14, 0, 0)), "14.0")
        self.assertEqual(verifier.format_version((15, 1, 2)), "15.1.2")
        self.assertEqual(verifier.decode_version(0x000E0000), (14, 0, 0))
        self.assertEqual(verifier.decode_version(0x000D0401), (13, 4, 1))
        self.assertGreater(verifier.parse_version("16.0.1"), verifier.parse_version("16"))
        with self.assertRaises(ValueError):
            verifier.parse_version("1.2.3.4")

    def test_podspec_platform(self):
        spec = "Pod::Spec.new do |s|\n  s.name = 'x'\n  s.platform = :osx, '14.0'\nend\n"
        self.assertEqual(verifier.podspec_platform(spec), ("osx", (14, 0, 0)))
        self.assertEqual(verifier.podspec_platform('  s.platform = :ios, "16.0"'), ("ios", (16, 0, 0)))
        with self.assertRaisesRegex(ValueError, "no `s.platform"):
            verifier.podspec_platform("  # s.platform = :ios, '13.0'\n")
        with self.assertRaisesRegex(ValueError, "unsupported platform :tvos"):
            verifier.podspec_platform("  s.platform = :tvos, '17.0'\n")

    def test_the_flutter_podspecs_parse(self):
        """release-prep reads these; a reworded declaration must still parse."""
        for path, os_name in (("bindings/flutter/ios/xybrid_flutter.podspec", "ios"),
                              ("bindings/flutter/macos/xybrid_flutter.podspec", "osx")):
            with self.subTest(path=path):
                self.assertEqual(verifier.podspec_platform((REPO / path).read_text())[0], os_name)


class MachOTests(unittest.TestCase):
    def test_build_version_and_version_min(self):
        self.assertEqual(verifier.macho_versions(macho(segment(), build_version(MACOS, "14.0"))),
                         [("macOS", (14, 0, 0))])
        self.assertEqual(verifier.macho_versions(macho(version_min(LC_VERSION_MIN_IPHONEOS, "10.0"))),
                         [("iOS", (10, 0, 0))])
        self.assertEqual(verifier.macho_versions(macho(build_version(IOS_SIMULATOR, "16.0"))),
                         [("iOS simulator", (16, 0, 0))])

    def test_32_bit_and_big_endian_headers(self):
        self.assertEqual(verifier.macho_versions(macho(build_version(IOS, "9.0"), bits=32)),
                         [("iOS", (9, 0, 0))])
        self.assertEqual(
            verifier.macho_versions(macho(build_version(MACOS, "10.15", ">"), order=">")),
            [("macOS", (10, 15, 0))],
        )

    def test_object_without_version_commands(self):
        self.assertEqual(verifier.macho_versions(macho(segment())), [])

    def test_not_macho(self):
        self.assertIsNone(verifier.macho_versions(b"\x7fELF" + bytes(60)))
        self.assertIsNone(verifier.macho_versions(b"BC\xc0\xde" + bytes(60)))  # LLVM bitcode
        self.assertIsNone(verifier.macho_versions(b"\xca\xfe\xba\xbe" + bytes(60)))  # universal

    def test_truncated_and_malformed_commands(self):
        whole = macho(build_version(MACOS, "14.0"))
        with self.assertRaisesRegex(ValueError, "truncated"):
            verifier.macho_versions(whole[:40])
        bad = bytearray(whole)
        struct.pack_into("<I", bad, 32 + 4, 4)  # cmdsize below the 8-byte minimum
        with self.assertRaisesRegex(ValueError, "malformed load command 0x32"):
            verifier.macho_versions(bytes(bad))


class CheckTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, data):
        path = self.tmp / name
        path.write_bytes(data)
        return path

    # The v0.16.0 darwin staticlib's three kinds of object: Rust, ONNX Runtime
    # and llama.cpp / whisper.cpp.
    DARWIN = [
        ("xybrid_flutter_ffi.xybrid_flutter_ffi.3ff9d27ad1bb25e7-cgu.00.rcgu.o",
         macho(segment(), build_version(MACOS, "11.0"))),
        ("IOBinding.cc.o", macho(build_version(MACOS, "13.4"))),
        ("odd.o", macho(build_version(MACOS, "13.4"))[:-1]),  # odd size: padded member
        ("whisper.pic.o", macho(segment(), build_version(MACOS, "14.0"))),
    ]

    def test_bsd_archive_at_the_floor_passes(self):
        lib = self.write("libdarwin.a", bsd_archive(self.DARWIN))
        failures, summary = verifier.check(lib, "osx", (14, 0, 0))
        self.assertEqual(failures, [])
        self.assertEqual(summary, f"{lib}: 4 object(s), highest minimum OS 14.0, podspec :osx 14.0")

    def test_gnu_archive_resolves_long_names(self):
        lib = self.write("libdarwin.a", gnu_archive(self.DARWIN))
        failures, _ = verifier.check(lib, "osx", (13, 4, 0))
        self.assertEqual(failures, [f"{lib}: 1 object(s) built for macOS 14.0, above the podspec's 13.4: "
                                    "whisper.pic.o"])
        failures, _ = verifier.check(lib, "osx", (11, 0, 0))
        self.assertIn(f"{lib}: 2 object(s) built for macOS 13.4, above the podspec's 11.0: "
                      "IOBinding.cc.o, odd.o", failures)

    def test_objects_above_the_floor_are_grouped(self):
        objects = [(f"ggml{i}.o", macho(build_version(MACOS, "14.0"))) for i in range(5)]
        lib = self.write("libdarwin.a", bsd_archive(objects))
        failures, summary = verifier.check(lib, "osx", (13, 3, 0))
        self.assertEqual(failures, [f"{lib}: 5 object(s) built for macOS 14.0, above the podspec's 13.3: "
                                    "ggml0.o, ggml1.o, ggml2.o, ..."])
        self.assertIn("highest minimum OS 14.0, podspec :osx 13.3", summary)

    def test_ios_podspec_accepts_device_and_simulator(self):
        device = self.write("libios.a", bsd_archive([
            ("rust.o", macho(version_min(LC_VERSION_MIN_IPHONEOS, "10.0"))),
            ("llama.o", macho(build_version(IOS, "16.0"))),
        ]))
        sim = self.write("libsim.a", bsd_archive([("llama.o", macho(build_version(IOS_SIMULATOR, "16.0")))]))
        for lib in (device, sim):
            with self.subTest(lib=lib.name):
                self.assertEqual(verifier.check(lib, "ios", (16, 0, 0))[0], [])

    def test_wrong_platform_is_rejected(self):
        lib = self.write("libios.a", bsd_archive([
            ("llama.o", macho(build_version(IOS, "16.0"))),
            ("old.o", macho(version_min(LC_VERSION_MIN_MACOSX, "10.15"))),
        ]))
        failures, _ = verifier.check(lib, "ios", (16, 0, 0))
        self.assertEqual(failures, [f"{lib}: 1 object(s) built for macOS, but the podspec declares :ios: old.o"])

    def test_members_that_are_not_macho_are_rejected(self):
        lib = self.write("libmixed.a", bsd_archive([
            ("ok.o", macho(build_version(MACOS, "14.0"))),
            ("lto.o", b"BC\xc0\xde" + bytes(40)),
            ("cut.o", macho(build_version(MACOS, "14.0"))[:40]),
        ]))
        failures, _ = verifier.check(lib, "osx", (14, 0, 0))
        self.assertEqual(failures, [
            f"{lib}: 1 object(s) not thin Mach-O, so their OS is unknown: lto.o",
            f"{lib}: 1 object(s) malformed (truncated Mach-O load commands): cut.o",
        ])

    def test_lone_macho_file_and_empty_archive(self):
        dylib = self.write("libxybrid.dylib", macho(build_version(MACOS, "14.0")))
        self.assertEqual(verifier.check(dylib, "osx", (14, 0, 0))[0], [])
        self.assertEqual(len(verifier.check(dylib, "osx", (13, 0, 0))[0]), 1)
        empty = self.write("libempty.a", bsd_archive([]))
        self.assertEqual(verifier.check(empty, "osx", (14, 0, 0))[0], [f"{empty}: no Mach-O objects"])


class MainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.lib = self.tmp / "libxybrid_flutter_ffi.a"
        self.lib.write_bytes(bsd_archive([("whisper.pic.o", macho(build_version(MACOS, "14.0")))]))

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, podspec_text, *libs):
        podspec = self.tmp / "xybrid_flutter.podspec"
        podspec.write_text(podspec_text)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = verifier.main(["--podspec", str(podspec), *map(str, libs)])
        return code, out.getvalue() + err.getvalue()

    def test_exit_status(self):
        self.assertEqual(self.run_main("  s.platform = :osx, '14.0'\n", self.lib)[0], 0)
        code, output = self.run_main("  s.platform = :osx, '13.3'\n", self.lib)
        self.assertEqual(code, 1)
        self.assertIn("above the podspec's 13.3", output)

    def test_every_library_is_checked(self):
        code, output = self.run_main("  s.platform = :osx, '14.0'\n", self.tmp / "missing.a", self.lib)
        self.assertEqual(code, 1)
        self.assertIn("missing.a", output)
        self.assertIn(f"OK: {self.lib}: 1 object(s)", output)

    def test_podspec_without_platform_fails(self):
        code, output = self.run_main("Pod::Spec.new do |s|\nend\n", self.lib)
        self.assertEqual(code, 1)
        self.assertIn("no `s.platform", output)


if __name__ == "__main__":
    unittest.main()
