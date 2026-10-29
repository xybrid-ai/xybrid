"""Unit tests for verify_unity_android_natives.py over canned readelf output."""

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

import verify_unity_android_natives as verifier

BOLT = verifier.BOLT
CXX_SHARED = verifier.CXX_SHARED
APPEND = "_ZNSt6__ndk112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6appendEPKcm"

# Verbatim lines from GNU readelf 2.41 and llvm-readelf 15 (-W) on the
# v0.13.0 arm64-v8a libxybrid_bolt.so.
GNU_DYNAMIC = """\
Dynamic section at offset 0x122f638 contains 28 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [liblog.so]
 0x0000000000000001 (NEEDED)             Shared library: [libc++_shared.so]
 0x000000000000001e (FLAGS)              BIND_NOW
"""
LLVM_DYNAMIC = """\
Dynamic section at offset 0x122f638 contains 28 entries:
  Tag                Type           Name/Value
  0x0000000000000001 (NEEDED)       Shared library: [liblog.so]
  0x0000000000000001 (NEEDED)       Shared library: [libc++_shared.so]
  0x000000000000001e (FLAGS)        BIND_NOW
"""
GNU_LOADS_64 = """\
Program Headers:
  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
  PHDR           0x000040 0x0000000000000040 0x0000000000000040 0x0001f8 0x0001f8 R   0x8
  LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x1176ee0 0x1176ee0 R E 0x4000
  LOAD           0x1178000 0x0000000001178000 0x0000000001178000 0x0b98a0 0x0ba000 RW  0x4000
  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0

 Section to Segment mapping:
  Segment Sections...
   00
   01     .note.android.ident .note.gnu.build-id .dynsym .gnu.version .text .plt
"""
LLVM_LOADS_32 = """\
Program Headers:
  Type           Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align
  LOAD           0x000000 0x00000000 0x00000000 0xb63b50 0xb63b50 R E 0x1000
  LOAD           0xb64000 0x00b64000 0x00b64000 0x62ae8 0x63000 RW  0x4000
  GNU_STACK      0x000000 0x00000000 0x00000000 0x00000 0x00000 RW  0x0
"""
# GNU readelf without -W wraps each 64-bit row over two lines.
GNU_LOADS_NARROW = """\
  Type           Offset             VirtAddr           PhysAddr
                 FileSiz            MemSiz              Flags  Align
  LOAD           0x0000000000000000 0x0000000000000000 0x0000000000000000
                 0x0000000001176ee0 0x0000000001176ee0  R E    0x4000
"""
GNU_SYMBOLS = """\
Symbol table '.dynsym' contains 503 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
     2: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __cxa_atexit@LIBC (2)
    24: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND _ZNSt6__ndk112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED2Ev
   334: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND getrandom@LIBC_P (5)
   400: 00000000000a1b20   116 FUNC    GLOBAL DEFAULT [VARIANT_PCS]    12 boltffi_free@@XYBRID
"""
LLVM_SYMBOLS = """\
Symbol table '.dynsym' contains 503 entries:
   Num:    Value          Size Type    Bind   Vis       Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
     2: 0000000000000000     0 FUNC    GLOBAL DEFAULT   UND __cxa_atexit@LIBC
    24: 0000000000000000     0 FUNC    GLOBAL DEFAULT   UND _ZNSt6__ndk112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED2Ev
   334: 0000000000000000     0 FUNC    WEAK   DEFAULT   UND getrandom@LIBC_P
   400: 00000000000a1b20   116 FUNC    GLOBAL DEFAULT    12 boltffi_free
"""

MACHINES = {"arm64-v8a": "AArch64", "armeabi-v7a": "ARM", "x86_64": "Advanced Micro Devices X86-64"}
SYSTEM_NEEDED = ("liblog.so", "libdl.so", "libm.so", "libc.so")
CLEAN_SYMBOLS = (
    ("__cxa_atexit@LIBC", "UND"),
    ("__cxa_finalize@LIBC", "UND"),
    ("memcpy@LIBC", "UND"),
    ("__android_log_write", "UND"),
    ("boltffi_function_xybrid_bolt_set_binding", "12"),
)


def sym(name, ndx="UND", bind="GLOBAL", vis="DEFAULT", kind="FUNC", style="gnu"):
    """One --dyn-syms row; GNU prints `@LIBC (2)` where LLVM prints `@LIBC`."""
    if style == "gnu" and "@" in name and "@@" not in name:
        name += " (2)"
    return f"    7: 0000000000000000     0 {kind:7} {bind:6} {vis:8} {ndx:>4} {name}\n"


def notes(api, style="gnu"):
    """`readelf -W -n` for an NDK r27 link at `api`: GNU puts the data on the owner row."""
    data = " ".join(f"{b:02x}" for b in api.to_bytes(4, "little") + b"r27".ljust(128, b"\0"))
    owner = "  Android              0x00000084\t"
    if style == "gnu":
        row = f"{owner}NT_VERSION (version)\t   description data: {data} \n"
    else:
        row = f"{owner}NT_ANDROID_TYPE_IDENT\n   description data: {data}\n"
    return (
        "\nDisplaying notes found in: .note.android.ident\n"
        "  Owner                Data size \tDescription\n" + row
    )


def elf(machine, needed=(), aligns=(0x4000,), symbols=(), style="gnu", api=None):
    """Combined `readelf -W -h -d -l [-n --dyn-syms]` text in GNU or LLVM layout."""
    pad = 13 if style == "gnu" else 7
    text = f"ELF Header:\n  Class:                             ELF64\n  Machine:                           {machine}\n\n"
    text += "Dynamic section at offset 0x1000 contains 9 entries:\n"
    for lib in needed:
        text += f" 0x0000000000000001 (NEEDED){' ' * pad}Shared library: [{lib}]\n"
    text += "\nProgram Headers:\n  Type Offset VirtAddr PhysAddr FileSiz MemSiz Flg Align\n"
    for align in aligns:
        text += f"  LOAD 0x000000 0x0000000000000000 0x0000000000000000 0x1000 0x1000 R E {align:#x}\n"
    text += "\nSymbol table '.dynsym' contains 9 entries:\n"
    text += "".join(sym(*row, style=style) if isinstance(row, tuple) else row for row in symbols)
    if api is not None:
        text += notes(api, style)
    return text


def clean_bolt(abi="arm64-v8a", style="gnu", **overrides):
    fields = dict(needed=SYSTEM_NEEDED, symbols=CLEAN_SYMBOLS, style=style, api=verifier.ANDROID_API)
    fields.update(overrides)
    return elf(MACHINES[abi], **fields)


def sibling(abi="arm64-v8a", needed=SYSTEM_NEEDED, aligns=(0x4000,)):
    return elf(MACHINES[abi], needed=needed, aligns=aligns, symbols=())


class ParseTests(unittest.TestCase):
    def test_needed_in_gnu_and_llvm_layouts(self):
        for text in (GNU_DYNAMIC, LLVM_DYNAMIC):
            with self.subTest(text=text[:60]):
                self.assertEqual(verifier.parse_needed(text), ["liblog.so", CXX_SHARED])

    def test_machine(self):
        self.assertEqual(verifier.parse_machine(clean_bolt("x86_64")), "Advanced Micro Devices X86-64")
        self.assertIsNone(verifier.parse_machine(GNU_DYNAMIC))

    def test_load_aligns_wide_layouts(self):
        self.assertEqual(verifier.parse_load_aligns(GNU_LOADS_64), [0x4000, 0x4000])
        self.assertEqual(verifier.parse_load_aligns(LLVM_LOADS_32), [0x1000, 0x4000])

    def test_load_aligns_without_wide_mode_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "readelf -W"):
            verifier.parse_load_aligns(GNU_LOADS_NARROW)

    def test_symbols_in_gnu_and_llvm_layouts(self):
        for text in (GNU_SYMBOLS, LLVM_SYMBOLS):
            with self.subTest(text=text[80:120]):
                symbols = {s.name: s for s in verifier.parse_symbols(text)}
                self.assertEqual(set(symbols), {
                    "__cxa_atexit", "getrandom", "boltffi_free",
                    "_ZNSt6__ndk112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED2Ev",
                })
                self.assertEqual(symbols["__cxa_atexit"].version, "LIBC")
                self.assertEqual(symbols["getrandom"].version, "LIBC_P")
                self.assertEqual(symbols["getrandom"].bind, "WEAK")
                self.assertTrue(symbols["boltffi_free"].defined)
                self.assertFalse(symbols["__cxa_atexit"].defined)

    def test_default_version_and_st_other_annotation(self):
        (symbol,) = verifier.parse_symbols(GNU_SYMBOLS.splitlines()[-1])
        self.assertEqual((symbol.name, symbol.version, symbol.ndx), ("boltffi_free", "XYBRID", "12"))


class SymbolDecisionTests(unittest.TestCase):
    def imports(self, *rows, style="gnu"):
        return verifier.cxx_runtime_imports(verifier.parse_symbols("".join(sym(*r, style=style) for r in rows)))

    def exports(self, *rows):
        return verifier.cxx_exports(verifier.parse_symbols("".join(sym(*r) for r in rows)))

    def test_bionic_versioned_symbols_are_allowed(self):
        for style in ("gnu", "llvm"):
            for name in ("__cxa_atexit@LIBC", "__cxa_finalize@LIBC",
                         "__cxa_thread_atexit_impl@LIBC_N", "_Znwm@LIBC", "_ZdlPv@LIBC"):
                with self.subTest(style=style, name=name):
                    self.assertEqual(self.imports((name,), style=style), [])

    def test_unversioned_bionic_cxa_is_allowed(self):
        self.assertEqual(self.imports(("__cxa_thread_atexit_impl", "UND", "WEAK")), [])

    def test_weak_tls_init_reference_is_allowed(self):
        # Verbatim from the vendored static-libc++ libonnxruntime.so.
        self.assertEqual(self.imports(("_ZTH15ThreadedBufSize", "UND", "WEAK", "DEFAULT", "NOTYPE")), [])
        self.assertEqual(self.imports(("_ZTH15ThreadedBufSize",)), ["_ZTH15ThreadedBufSize"])

    def test_ndk_libcxx_import_is_rejected(self):
        for style in ("gnu", "llvm"):
            with self.subTest(style=style):
                self.assertEqual(self.imports((APPEND,), style=style), [APPEND])

    def test_libcxxabi_imports_are_rejected(self):
        for name in ("__cxa_throw", "__cxa_begin_catch", "__cxa_end_catch", "__cxa_allocate_exception",
                     "__cxa_rethrow", "__cxa_guard_acquire", "__cxa_end_cleanup", "__gxx_personality_v0",
                     "__dynamic_cast", "_ZTISt9exception", "_ZTSSt9exception",
                     "_ZTVN10__cxxabiv117__class_type_infoE", "_ZSt9terminatev", "_Znwm"):
            with self.subTest(name=name):
                self.assertEqual(self.imports((name,)), [name])

    def test_c_imports_and_defined_symbols_are_not_cxx_imports(self):
        self.assertEqual(self.imports(("memcpy@LIBC",), ("__android_log_write",),
                                      ("ZSTD_trace_decompress_end", "UND", "WEAK"),
                                      ("_ZN6xybrid4core3run17h0123456789abcdefE", "12")), [])

    def test_exported_cxx_symbol_is_rejected(self):
        for bind in ("GLOBAL", "WEAK"):
            with self.subTest(bind=bind):
                self.assertEqual(self.exports((APPEND, "12", bind)), [APPEND])

    def test_exported_c_named_runtime_symbol_is_rejected(self):
        # libc++abi and the unwinder export C names; leaked, they interpose too.
        self.assertEqual(
            self.exports(("__cxa_throw", "12"), ("_Unwind_Resume", "12"),
                         ("boltffi_function_xybrid_bolt_set_binding", "12")),
            ["_Unwind_Resume", "__cxa_throw"],
        )

    def test_local_hidden_or_undefined_cxx_is_not_exported(self):
        self.assertEqual(self.exports((APPEND, "12", "LOCAL"), (APPEND, "12", "GLOBAL", "HIDDEN"),
                                      (APPEND,), ("boltffi_free", "12")), [])


class CheckAbiTests(unittest.TestCase):
    def test_static_build_with_ort_passes(self):
        for style in ("gnu", "llvm"):
            with self.subTest(style=style):
                failures, summary = verifier.check_abi("arm64-v8a", {
                    BOLT: clean_bolt(style=style), "libonnxruntime.so": sibling(),
                })
                self.assertEqual(failures, [])
                self.assertTrue(summary.startswith("arm64-v8a: OK"), summary)
                self.assertIn("LOAD align 0x4000", summary)
                self.assertIn("Android API 28", summary)

    def test_v0_13_layout_reports_every_failure(self):
        bolt = clean_bolt("x86_64", needed=SYSTEM_NEEDED + (CXX_SHARED,),
                          symbols=CLEAN_SYMBOLS + ((APPEND,), ("__cxa_throw",)))
        failures, summary = verifier.check_abi("x86_64", {
            BOLT: bolt, CXX_SHARED: sibling("arm64-v8a"), "libonnxruntime.so": sibling("x86_64"),
        })
        self.assertEqual(summary, "")
        text = "\n".join(failures)
        self.assertEqual(len(failures), 4, text)
        self.assertIn(f"x86_64/{CXX_SHARED}: shipped", text)
        self.assertIn(f"x86_64/{CXX_SHARED}: ELF machine is AArch64, expected Advanced Micro Devices X86-64", text)
        self.assertIn(f"x86_64/{BOLT}: NEEDs {CXX_SHARED}", text)
        self.assertIn("imports 2 C++ runtime symbol(s)", text)

    def test_needed_cxx_shared_is_rejected(self):
        failures, _ = verifier.check_abi("armeabi-v7a", {
            BOLT: clean_bolt("armeabi-v7a", needed=SYSTEM_NEEDED + (CXX_SHARED,)),
        })
        self.assertEqual(failures, [f"armeabi-v7a/{BOLT}: NEEDs {CXX_SHARED}; libc++ must be linked statically"])

    def test_sibling_needing_cxx_shared_is_rejected(self):
        failures, _ = verifier.check_abi("arm64-v8a", {
            BOLT: clean_bolt(), "libonnxruntime.so": sibling(needed=("libc.so", CXX_SHARED)),
        })
        self.assertEqual(failures, [f"arm64-v8a/libonnxruntime.so: NEEDs {CXX_SHARED}; libc++ must be linked statically"])

    def test_needed_must_be_system_or_shipped_beside(self):
        ok, _ = verifier.check_abi("arm64-v8a", {
            BOLT: clean_bolt(needed=SYSTEM_NEEDED + ("libandroid.so", "libz.so", "libonnxruntime.so")),
            "libonnxruntime.so": sibling(),
        })
        self.assertEqual(ok, [])
        failures, _ = verifier.check_abi("arm64-v8a", {BOLT: clean_bolt(needed=("libc.so", "libllama.so"))})
        self.assertEqual(len(failures), 1)
        self.assertIn("NEEDs libllama.so, which is neither", failures[0])

    def test_sibling_4k_alignment_is_rejected(self):
        failures, _ = verifier.check_abi("x86_64", {
            BOLT: clean_bolt("x86_64"), "libonnxruntime.so": sibling("x86_64", aligns=(0x1000,)),
        })
        self.assertEqual(len(failures), 1)
        self.assertIn("x86_64/libonnxruntime.so: LOAD segment aligned to 0x1000", failures[0])

    def test_unwinder_imports_are_rejected(self):
        # A static armv7 link that dropped libunwind.a; bionic's own copy is versioned.
        symbols = CLEAN_SYMBOLS + (
            ("_Unwind_Resume", "UND"), ("__aeabi_unwind_cpp_pr0", "UND"),
            ("__gnu_Unwind_Find_exidx@LIBC_N", "UND"),
        )
        failures, _ = verifier.check_abi("armeabi-v7a", {BOLT: clean_bolt("armeabi-v7a", symbols=symbols)})
        self.assertEqual(len(failures), 1)
        self.assertIn("imports 2 C++ runtime symbol(s)", failures[0])
        self.assertIn("_Unwind_Resume", failures[0])
        self.assertIn("__aeabi_unwind_cpp_pr0", failures[0])

    def test_statically_linked_bionic_is_rejected(self):
        # What linking every NDK sysroot archive produced: only liblog.so left.
        failures, _ = verifier.check_abi("arm64-v8a", {BOLT: clean_bolt(needed=("liblog.so",))})
        self.assertEqual(failures, [
            f"arm64-v8a/{BOLT}: does not NEED libc.so, so bionic is linked into it "
            "statically; link only libc++ statically",
        ])

    def test_exported_cxx_symbol_is_rejected(self):
        failures, _ = verifier.check_abi("arm64-v8a", {
            BOLT: clean_bolt(symbols=CLEAN_SYMBOLS + ((APPEND, "12"),)),
        })
        self.assertEqual(len(failures), 1)
        self.assertIn("exports 1 C++ symbol(s)", failures[0])

    def test_4k_alignment_is_rejected(self):
        failures, _ = verifier.check_abi("arm64-v8a", {BOLT: clean_bolt(aligns=(0x4000, 0x1000))})
        self.assertEqual(len(failures), 1)
        self.assertIn("aligned to 0x1000, below 16 KB", failures[0])

    def test_missing_load_headers_fail_closed(self):
        failures, _ = verifier.check_abi("arm64-v8a", {BOLT: clean_bolt(aligns=())})
        self.assertEqual(failures, [f"arm64-v8a/{BOLT}: no LOAD program headers in readelf output"])

    def test_android_10_link_is_rejected(self):
        # What a fallback to the Kotlin AAR's arm64 toolchain would produce.
        for style in ("gnu", "llvm"):
            with self.subTest(style=style):
                failures, summary = verifier.check_abi("arm64-v8a", {BOLT: clean_bolt(style=style, api=29)})
                self.assertEqual(summary, "")
                self.assertEqual(failures, [
                    f"arm64-v8a/{BOLT}: linked for Android API 29, expected 28; "
                    "the Unity plugin must keep loading on Android 9",
                ])

    def test_missing_api_note_fails_closed(self):
        failures, _ = verifier.check_abi("x86_64", {BOLT: clean_bolt("x86_64", api=None)})
        self.assertEqual(len(failures), 1)
        self.assertIn(f"x86_64/{BOLT}: no .note.android.ident note", failures[0])

    def test_missing_bolt_is_rejected(self):
        failures, _ = verifier.check_abi("x86_64", {"libonnxruntime.so": sibling("x86_64")})
        self.assertEqual(failures, [f"x86_64/{BOLT}: missing"])

    def test_wrong_machine_is_rejected(self):
        failures, _ = verifier.check_abi("armeabi-v7a", {BOLT: clean_bolt("arm64-v8a")})
        self.assertEqual(failures, [f"armeabi-v7a/{BOLT}: ELF machine is AArch64, expected ARM"])

    def test_unreadable_library_suppresses_ok_summary(self):
        failures, summary = verifier.check_abi("arm64-v8a", {BOLT: None})
        self.assertEqual((failures, summary), ([], ""))


class CheckTreeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.texts = {}

    def tearDown(self):
        self._tmp.cleanup()

    def add(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELF")
        self.texts[path] = text
        return path

    def read(self, path):
        return self.texts[path]

    def add_clean(self, abi):
        self.add(f"{abi}/{BOLT}", clean_bolt(abi))

    def test_every_abi_clean_passes(self):
        for abi in verifier.ABI_MACHINES:
            self.add_clean(abi)
        (self.root / "arm64-v8a.meta").write_text("folderAsset: yes\n")
        failures, summaries = verifier.check_tree(self.root, self.read, require_abis=True)
        self.assertEqual(failures, [])
        self.assertEqual([s.split(":")[0] for s in summaries], ["arm64-v8a", "armeabi-v7a", "x86_64"])

    def test_require_abis_reports_missing_directories(self):
        self.add_clean("arm64-v8a")
        self.assertEqual(verifier.check_tree(self.root, self.read)[0], [])
        failures, _ = verifier.check_tree(self.root, self.read, require_abis=True)
        self.assertEqual(failures, ["armeabi-v7a: ABI directory missing", "x86_64: ABI directory missing"])

    def test_libraries_outside_abi_directories_are_rejected(self):
        self.add_clean("arm64-v8a")
        for stray in (CXX_SHARED, "x86/libxybrid_bolt.so", "arm64-v8a/nested/libfoo.so"):
            self.add(stray, "")
        failures, _ = verifier.check_tree(self.root, self.read)
        self.assertEqual(sorted(failures), sorted([
            f"{CXX_SHARED}: native library outside a known ABI directory",
            "x86/libxybrid_bolt.so: native library outside a known ABI directory",
            "arm64-v8a/nested/libfoo.so: native library outside a known ABI directory",
        ]))

    def test_empty_or_missing_root_fails(self):
        self.assertIn("no ABI directories", verifier.check_tree(self.root, self.read)[0][0])
        self.assertIn("not a directory", verifier.check_tree(self.root / "nope", self.read)[0][0])

    def test_readelf_error_is_reported(self):
        self.add_clean("x86_64")
        broken = self.add("x86_64/libonnxruntime.so", "")

        def read(path):
            if path == broken:
                raise subprocess.CalledProcessError(1, ["readelf"], stderr="readelf: Error: Not an ELF file\n")
            return self.read(path)

        failures, summaries = verifier.check_tree(self.root, read)
        self.assertEqual(failures, ["x86_64/libonnxruntime.so: readelf failed: readelf: Error: Not an ELF file"])
        self.assertEqual(summaries, [])


class MainTests(unittest.TestCase):
    """Drive main() through a stub readelf that replays canned text per file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.root = tmp / "Android"
        self.log = tmp / "readelf.log"
        self.readelf = tmp / "readelf"
        self.readelf.write_text(
            '#!/bin/sh\necho "$@" >> "%s"\nfor last; do :; done\ncat "$last.txt"\n' % self.log
        )
        self.readelf.chmod(self.readelf.stat().st_mode | stat.S_IXUSR)

    def tearDown(self):
        self._tmp.cleanup()

    def add(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELF")
        Path(f"{path}.txt").write_text(text)

    def run_main(self, *extra, readelf=None):
        argv = ["--readelf", str(readelf or self.readelf), *extra, str(self.root)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return verifier.main(argv)

    @unittest.skipIf(os.name == "nt", "the stub readelf is a POSIX shell script")
    def test_passes_wide_flags_and_reads_notes_and_symbols_only_for_bolt(self):
        self.add(f"x86_64/{BOLT}", clean_bolt("x86_64"))
        self.add("x86_64/libonnxruntime.so", sibling("x86_64"))
        self.assertEqual(self.run_main(), 0)
        calls = sorted(self.log.read_text().splitlines())
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0].startswith("-W -h -d -l -n --dyn-syms ") and calls[0].endswith(BOLT), calls)
        self.assertTrue(calls[1].startswith("-W -h -d -l ") and "--dyn-syms" not in calls[1]
                        and " -n " not in calls[1], calls)

    @unittest.skipIf(os.name == "nt", "the stub readelf is a POSIX shell script")
    def test_fails_on_problems_and_missing_abis(self):
        self.add(f"x86_64/{BOLT}", clean_bolt("x86_64"))
        self.assertEqual(self.run_main("--require-abis"), 1)
        self.add(f"x86_64/{CXX_SHARED}", sibling("x86_64"))
        self.assertEqual(self.run_main(), 1)

    def test_missing_readelf_fails(self):
        self.add(f"x86_64/{BOLT}", "")
        self.assertEqual(self.run_main(readelf=self.root / "no-such-readelf"), 1)


if __name__ == "__main__":
    unittest.main()
