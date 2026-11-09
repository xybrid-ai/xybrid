"""Unit tests for verify_android_api_level.py over canned readelf -n output."""

import contextlib
import io
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verify_android_api_level as verifier


def desc_hex(api):
    """The NDK r27 crtbegin_so.o note description: API level, NDK version, build."""
    desc = api.to_bytes(4, "little") + b"r27".ljust(64, b"\0") + b"12077973".ljust(64, b"\0")
    return " ".join(f"{byte:02x}" for byte in desc)


# The layouts of GNU readelf 2.41 and llvm-readelf 18 (NDK r27) `-W -n` on a
# library linked with `aarch64-linux-android<api>-clang -Wl,--build-id`.
def gnu_notes(api):
    return (
        "\nDisplaying notes found in: .note.android.ident\n"
        "  Owner                Data size \tDescription\n"
        f"  Android              0x00000084\tNT_VERSION (version)\t   description data: {desc_hex(api)} \n"
        "\nDisplaying notes found in: .note.gnu.build-id\n"
        "  Owner                Data size \tDescription\n"
        "  GNU                  0x00000014\tNT_GNU_BUILD_ID (unique build ID bitstring)\t"
        "    Build ID: 0d977f8694289a4ee7e7166e9fa450447d3c7123\n"
    )


def llvm_notes(api):
    return (
        "Displaying notes found in: .note.android.ident\n"
        "  Owner                Data size \tDescription\n"
        "  Android              0x00000084\tNT_ANDROID_TYPE_IDENT\n"
        f"   description data: {desc_hex(api)}\n"
        "\nDisplaying notes found in: .note.gnu.build-id\n"
        "  Owner                Data size \tDescription\n"
        "  GNU                  0x00000014\tNT_GNU_BUILD_ID (unique build ID bitstring)\n"
        "    Build ID: 0d977f8694289a4ee7e7166e9fa450447d3c7123\n"
    )


BUILD_ID_ONLY = (
    "Displaying notes found in: .note.gnu.build-id\n"
    "  Owner                Data size \tDescription\n"
    "  GNU                  0x00000014\tNT_GNU_BUILD_ID (unique build ID bitstring)\n"
    "    Build ID: 0d977f8694289a4ee7e7166e9fa450447d3c7123\n"
)


class ParseTests(unittest.TestCase):
    def test_gnu_and_llvm_layouts(self):
        for layout in (gnu_notes, llvm_notes):
            for api in (28, 29):
                with self.subTest(layout=layout.__name__, api=api):
                    self.assertEqual(verifier.parse_api_levels(layout(api)), [api])

    def test_no_android_note(self):
        self.assertEqual(verifier.parse_api_levels(BUILD_ID_ONLY), [])
        self.assertEqual(verifier.parse_api_levels(""), [])

    def test_android_owner_in_another_section_is_ignored(self):
        memtag = (
            "Displaying notes found in: .note.android.memtag\n"
            "  Owner                Data size \tDescription\n"
            "  Android              0x00000004\tNT_ANDROID_TYPE_MEMTAG\n"
            "   description data: 1d 00 00 00\n"
        )
        self.assertEqual(verifier.parse_api_levels(memtag), [])
        self.assertEqual(verifier.parse_api_levels(memtag + llvm_notes(28)), [28])

    def test_notes_without_section_headers_are_ignored(self):
        # What readelf prints for a library stripped of its section headers.
        text = llvm_notes(29).replace(
            "Displaying notes found in: .note.android.ident",
            "Displaying notes found at file offset 0x00000238 with length 0x00000098:",
        )
        self.assertEqual(verifier.parse_api_levels(text), [])

    def test_other_readelf_sections_after_the_notes_are_ignored(self):
        text = gnu_notes(29) + (
            "\nProgram Headers:\n"
            "  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align\n"
            "  LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x1000 0x1000 R E 0x4000\n"
        )
        self.assertEqual(verifier.parse_api_levels(text), [29])


class CheckTests(unittest.TestCase):
    def test_matching_level_passes(self):
        self.assertEqual(
            verifier.check("arm64-v8a/libggml.so", llvm_notes(29), 29),
            (None, "OK: arm64-v8a/libggml.so: Android API 29"),
        )

    def test_wrong_level_is_rejected(self):
        failure, summary = verifier.check("x86_64/libxybrid_bolt.so", gnu_notes(29), 28)
        self.assertEqual(summary, "")
        self.assertIn("x86_64/libxybrid_bolt.so: linked for Android API 29, expected 28", failure)

    def test_missing_note_is_rejected(self):
        failure, _ = verifier.check("libfoo.so", BUILD_ID_ONLY, 29)
        self.assertIn("libfoo.so: no .note.android.ident note", failure)

    def test_check_all_reports_every_failure(self):
        texts = {"a.so": llvm_notes(29), "b.so": llvm_notes(28), "c.so": BUILD_ID_ONLY}

        def read(path):
            if path == "d.so":
                raise subprocess.CalledProcessError(1, ["readelf"], stderr="readelf: Error: Not an ELF file\n")
            return texts[path]

        failures, summaries = verifier.check_all(["a.so", "b.so", "c.so", "d.so"], 29, read)
        self.assertEqual(summaries, ["OK: a.so: Android API 29"])
        self.assertEqual(len(failures), 3, failures)
        self.assertIn("b.so: linked for Android API 28, expected 29", failures[0])
        self.assertIn("c.so: no .note.android.ident note", failures[1])
        self.assertEqual(failures[2], "d.so: readelf failed: readelf: Error: Not an ELF file")


class MainTests(unittest.TestCase):
    """Drive main() through a stub readelf that replays canned text per file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.log = self.tmp / "readelf.log"
        self.readelf = self.tmp / "readelf"
        self.readelf.write_text(
            '#!/bin/sh\necho "$@" >> "%s"\nfor last; do :; done\ncat "$last.txt"\n' % self.log
        )
        self.readelf.chmod(self.readelf.stat().st_mode | stat.S_IXUSR)

    def tearDown(self):
        self._tmp.cleanup()

    def add(self, name, text):
        path = self.tmp / name
        path.write_bytes(b"\x7fELF")
        Path(f"{path}.txt").write_text(text)
        return str(path)

    def run_main(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return verifier.main(list(argv))

    @unittest.skipIf(os.name == "nt", "the stub readelf is a POSIX shell script")
    def test_reads_notes_wide_and_checks_every_library(self):
        good = self.add("libllama.so", gnu_notes(29))
        bolt = self.add("libxybrid_bolt.so", llvm_notes(29))
        self.assertEqual(self.run_main("--readelf", str(self.readelf), "--api", "29", good, bolt), 0)
        self.assertEqual(self.log.read_text().splitlines(), [f"-W -n {good}", f"-W -n {bolt}"])
        self.assertEqual(self.run_main("--readelf", str(self.readelf), "--api", "28", good, bolt), 1)

    def test_missing_readelf_fails(self):
        lib = self.add("libxybrid_bolt.so", "")
        self.assertEqual(self.run_main("--readelf", str(self.tmp / "no-such-readelf"), "--api", "28", lib), 1)


if __name__ == "__main__":
    unittest.main()
