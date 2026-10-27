"""Offline tests for verify_zzz_android.py and the workflows that run it.

Libraries are JSON specs behind a magic prefix. FakeReadelf renders the text
GNU binutils readelf or llvm-readelf prints for the flags it is asked for, in
the layouts both print for the real Android libraries (captured from the
master AAR). No subprocess runs; the notice contract is patched with three
small fake notices. Python 3.9+, standard library only.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "verify_zzz_android_under_test", ROOT / "tools/scripts/verify_zzz_android.py")
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)

NOTICES = {
    "RUNTIME_TERMS.md": b"runtime terms\n",
    "THIRD_PARTY_NOTICES.md": b"third party\n",
    "LICENSES/Zig-MIT.txt": b"zig mit\n",
}
NOTICE_PINS = {name: hashlib.sha256(data).hexdigest() for name, data in NOTICES.items()}
JNI_EXPORTS = ["Java_ai_xybrid_Native_run", "Java_ai_xybrid_Native_version"]
GLUE = "".join(f"JNIEXPORT void JNICALL {name}(JNIEnv *env) {{}}\n" for name in JNI_EXPORTS)
EXPECTED = GATE.expected_natives()
MACHINES = {"arm64-v8a": "AArch64", "armeabi-v7a": "ARM", "x86_64": "Advanced Micro Devices X86-64"}
SYSTEM = ["liblog.so", "libdl.so", "libm.so", "libc.so"]
LLAMA = ["libllama.so", "libggml.so", "libggml-base.so", "libmtmd.so"]
PRIVATE_SYMBOL = "zzz_embed_private_internal"
FAKE = b"FAKE-ELF\n"
MANIFEST = (
    b'<?xml version="1.0" encoding="utf-8"?>\n'
    b'<manifest xmlns:android="http://schemas.android.com/apk/res/android"\n'
    b'    package="ai.xybrid">\n'
    b'    <uses-sdk android:minSdkVersion="29" />\n'
    b'</manifest>\n'
)
# AGP writes the merged library manifest with more namespaces and attributes.
GRADLE_MANIFEST = (
    b'<?xml version="1.0" encoding="utf-8"?>\n'
    b'<manifest xmlns:android="http://schemas.android.com/apk/res/android"\n'
    b'    xmlns:tools="http://schemas.android.com/tools"\n'
    b'    package="ai.xybrid" >\n\n'
    b'    <uses-sdk\n        android:minSdkVersion="29"\n        tools:overrideLibrary="x" />\n\n'
    b'</manifest>\n'
)


def symbol(name, ndx="14", kind="FUNC", bind="GLOBAL", vis="DEFAULT"):
    return [name, ndx, kind, bind, vis]


def lib_spec(abi, name, speech=False):
    spec = {"machine": MACHINES[abi], "aligns": [0x4000, 0x4000], "tls": False,
            "marker": False, "needed": ["libc.so", "libm.so", "libdl.so"],
            "api": GATE.API_LEVELS[abi], "dynsyms": []}
    if name in GATE.BUNDLED:
        # Vendor levels and 4 KB alignment: not checked for bundled libraries.
        spec.update(api=21 if name == GATE.CXX else 24, aligns=[0x1000])
        if name == GATE.ORT:
            spec["needed"].append("liblog.so")
        return spec
    if name == GATE.BOLT:
        spec["needed"] = [GATE.CXX] + (LLAMA if abi == GATE.SPEECH_ABI else []) + SYSTEM
        spec["dynsyms"] = [symbol(n) for n in JNI_EXPORTS + [GATE.JNI.BOLT_SYMBOL]]
        spec["dynsyms"].append(symbol("malloc@LIBC", "UND"))
        if abi == GATE.SPEECH_ABI:
            spec["dynsyms"] += [symbol("llama_backend_init", "UND"), symbol("ggml_init", "UND")]
        spec["tls"] = spec["marker"] = speech and abi == GATE.SPEECH_ABI
        return spec
    spec["needed"] += [GATE.CXX] + ([] if name == "libggml-base.so" else ["libggml-base.so"])
    return spec


def elf_bytes(spec):
    data = FAKE + json.dumps(spec, sort_keys=True).encode() + b"\n"
    if spec.get("marker"):
        data += b"..." + GATE.MARKER + b": %s\n"
    return data


def spec_of(data):
    return json.loads(data[len(FAKE):].split(b"\n", 1)[0])


def gnu_or_llvm(program):
    return "llvm" if os.path.basename(program).startswith("llvm") else "gnu"


def render_headers(spec, flavor):
    lines = ["ELF Header:", "  Class:                             ELF64",
             f"  Machine:                           {spec['machine']}", "",
             "Program Headers:",
             "  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align"]
    for align in spec["aligns"]:
        lines.append("  LOAD           0x000000 0x0000000000000000 0x0000000000000000 "
                     f"0x001000 0x001000 R E {align:#x}")
    if spec["tls"]:
        lines.append("  TLS            0x1f7898 0x00000000001f8898 0x00000000001f8898 "
                     "0x000028 0x000088 R   0x8")
    lines.append("  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 "
                 "0x000000 0x000000 RW  " + ("0x0" if flavor == "llvm" else "0"))
    lines += ["", " Section to Segment mapping:", "  Segment Sections...", "   00     .text",
              "   01     .tdata .tbss", "",
              f"Dynamic section at offset 0xf744d8 contains {len(spec['needed'])} entries:"]
    if flavor == "gnu":
        lines.append("  Tag        Type                         Name/Value")
        lines += [f" 0x0000000000000001 (NEEDED)             Shared library: [{n}]"
                  for n in spec["needed"]]
    else:
        lines.append("  Tag                Type           Name/Value")
        lines += [f"  0x0000000000000001 (NEEDED)       Shared library: [{n}]"
                  for n in spec["needed"]]
    return "\n".join(lines) + "\n"


def render_notes(api, flavor):
    build_id = ("\nDisplaying notes found in: .note.gnu.build-id\n"
                "  Owner                Data size \tDescription\n"
                "  GNU                  0x00000014\tNT_GNU_BUILD_ID (unique build ID bitstring)\n"
                "    Build ID: 0d977f8694289a4ee7e7166e9fa450447d3c7123\n")
    if api is None:
        return build_id
    desc = api.to_bytes(4, "little") + b"r27".ljust(64, b"\0") + b"12077973".ljust(64, b"\0")
    data = " ".join(f"{byte:02x}" for byte in desc)
    head = "\nDisplaying notes found in: .note.android.ident\n  Owner                Data size \tDescription\n"
    if flavor == "gnu":
        return head + f"  Android              0x00000084\tNT_VERSION (version)\t   description data: {data} \n" + build_id
    return head + "  Android              0x00000084\tNT_ANDROID_TYPE_IDENT\n" + f"   description data: {data}\n" + build_id


def render_symbols(symbols, flavor, table=".dynsym"):
    header = ("   Num:    Value          Size Type    Bind   Vis      Ndx Name" if flavor == "gnu"
              else "   Num:    Value          Size Type    Bind   Vis       Ndx Name")
    lines = ["", f"Symbol table '{table}' contains {len(symbols) + 1} entries:", header,
             "     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND "]
    for number, (name, ndx, kind, bind, vis) in enumerate(symbols, 1):
        if flavor == "gnu" and "@" in name:
            name += " (2)"  # GNU prints the version index; LLVM does not
        lines.append(f"{number:6}: 0000000000001000    12 {kind:7} {bind:6} {vis:8} {ndx:>4} {name}")
    return "\n".join(lines) + "\n"


def render_archive(members, flavor):
    text = ""
    for member, symbols in members:
        text += f"\nFile: libzzz_embed.a({member})\n" + render_symbols(symbols, flavor, ".symtab")
    return text


class FakeReadelf:
    """An injectable runner: renders readelf text from the file's JSON spec."""

    def __init__(self):
        self.fail = set()
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        path, flags, flavor = Path(argv[-1]), argv[1:-1], gnu_or_llvm(argv[0])
        data = path.read_bytes()
        if path.name in self.fail or not data.startswith(FAKE):
            raise GATE.GateError("readelf failed")
        spec = spec_of(data)
        if "-s" in flags:
            return render_archive(spec["archive"], flavor)
        if "--dyn-syms" in flags:
            return render_symbols(spec["dynsyms"], flavor)
        if "-n" in flags:
            return render_notes(spec["api"], flavor)
        return render_headers(spec, flavor)


def write_zip(path, entries):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the duplicate-name fixtures
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in entries:
                archive.writestr(name, data)


def natives(speech=True, prefix="jni/"):
    return {f"{prefix}{abi}/{name}": elf_bytes(lib_spec(abi, name, speech))
            for abi in GATE.ABIS for name in sorted(EXPECTED[abi])}


def notice_entries():
    return {GATE.NOTICE_ROOT + name: data for name, data in NOTICES.items()}


def aar_entries(speech=True):
    entries = {"AndroidManifest.xml": MANIFEST, "classes.jar": b"jar", "R.txt": b""}
    entries.update(natives(speech))
    if speech:
        entries.update(notice_entries())
    return entries


def dist_entries():
    entries = natives(True, prefix="")
    entries.update(notice_entries())
    return entries


def mutate(entries, key, **changes):
    spec = spec_of(entries[key])
    spec.update(changes)
    entries[key] = elf_bytes(spec)


def set_central_flag(path, name, bit):
    """Set a general-purpose flag bit on one central-directory record."""
    data = bytearray(Path(path).read_bytes())
    offset = data.find(b"PK\x01\x02")
    while offset != -1:
        length = struct.unpack_from("<H", data, offset + 28)[0]
        if data[offset + 46:offset + 46 + length] == name.encode():
            flags = struct.unpack_from("<H", data, offset + 8)[0]
            struct.pack_into("<H", data, offset + 8, flags | bit)
        offset = data.find(b"PK\x01\x02", offset + 4)
    Path(path).write_bytes(bytes(data))


class GateCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        notices = mock.patch.dict(GATE.PULL.SCHEMA5_NOTICE_FILES, NOTICE_PINS, clear=True)
        notices.start()
        self.addCleanup(notices.stop)
        self.glue = self.tmp / "jni_glue.c"
        self.glue.write_text(GLUE)
        self.readelf = FakeReadelf()

    def package(self, entries, name="xybrid-kotlin.aar"):
        path = self.tmp / name
        if path.exists():
            path.unlink()
        write_zip(path, entries.items() if isinstance(entries, dict) else entries)
        return path

    def run_gate(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = GATE.main([str(arg) for arg in argv], runner=self.readelf, glue_path=self.glue)
        return code, out.getvalue(), err.getvalue()

    def aar(self, entries, speech=True, *extra):
        return self.run_gate("aar", self.package(entries), "--speech" if speech else "--no-speech", *extra)

    def assertPasses(self, result):
        code, out, err = result
        self.assertEqual(code, 0, out + err)
        return out

    def assertFails(self, result, *fragments):
        code, out, err = result
        self.assertEqual(code, 1, out + err)
        for fragment in fragments:
            self.assertIn(fragment, err)
        self.assertNotIn(PRIVATE_SYMBOL, out + err)
        self.assertNotIn(str(self.tmp), err)
        return err


class AarTests(GateCase):
    def test_speech_aar_passes_with_gnu_and_llvm_readelf(self):
        for readelf in ("readelf", "/opt/ndk/bin/llvm-readelf"):
            with self.subTest(readelf=readelf):
                out = self.assertPasses(self.aar(aar_entries(), True, "--readelf", readelf, "--min-sdk", "29"))
                self.assertIn("arm64-v8a/libxybrid_bolt.so: speech engine linked statically", out)
                self.assertIn("x86_64/libxybrid_bolt.so: speech-free", out)
                self.assertIn("all 3 engine notices match", out)
                self.assertIn("minSdkVersion 29", out)

    def test_default_aar_passes_without_speech(self):
        for readelf in ("readelf", "llvm-readelf"):
            with self.subTest(readelf=readelf):
                out = self.assertPasses(self.aar(aar_entries(False), False, "--readelf", readelf))
                self.assertIn("arm64-v8a/libxybrid_bolt.so: speech-free", out)

    def test_each_modes_rejects_the_other_build(self):
        self.assertFails(self.aar(aar_entries(False), True),
                         "lacks the speech runtime's marker string", "has no PT_TLS segment",
                         "assets/xybrid/zzz/RUNTIME_TERMS.md: missing engine notice")
        self.assertFails(self.aar(aar_entries(True), False),
                         "arm64-v8a/libxybrid_bolt.so: must ship without speech",
                         "3 engine notice file(s) in a package built without speech")

    def test_every_engine_signal_is_required_on_arm64(self):
        bolt = "jni/arm64-v8a/libxybrid_bolt.so"
        cases = {
            "marker": dict(marker=False),
            "tls": dict(tls=False),
            "undefined engine import": dict(dynsyms=lib_spec("arm64-v8a", GATE.BOLT)["dynsyms"]
                                            + [symbol(PRIVATE_SYMBOL, "UND")]),
            "exported engine symbol": dict(dynsyms=lib_spec("arm64-v8a", GATE.BOLT)["dynsyms"]
                                           + [symbol(PRIVATE_SYMBOL)]),
        }
        messages = {"marker": "marker string", "tls": "PT_TLS",
                    "undefined engine import": "imports 1 engine symbol(s)",
                    "exported engine symbol": "exports 1 engine symbol(s)"}
        for case, changes in cases.items():
            with self.subTest(case=case):
                entries = aar_entries()
                mutate(entries, bolt, **changes)
                self.assertFails(self.aar(entries), messages[case])

    def test_speech_free_abis_stay_speech_free(self):
        for abi in ("armeabi-v7a", "x86_64"):
            key = f"jni/{abi}/libxybrid_bolt.so"
            base = lib_spec(abi, GATE.BOLT)
            for changes, message in ((dict(marker=True), "marker string"), (dict(tls=True), "PT_TLS"),
                                     (dict(dynsyms=base["dynsyms"] + [symbol(PRIVATE_SYMBOL, "UND")]),
                                      "references 1 engine symbol(s)")):
                with self.subTest(abi=abi, changes=sorted(changes)):
                    entries = aar_entries()
                    mutate(entries, key, **changes)
                    self.assertFails(self.aar(entries), f"{abi}/libxybrid_bolt.so: must ship without speech",
                                     message)

    def test_notices_must_match_the_contract_exactly(self):
        cases = []
        entries = aar_entries()
        del entries[GATE.NOTICE_ROOT + "LICENSES/Zig-MIT.txt"]
        cases.append((entries, "LICENSES/Zig-MIT.txt: missing engine notice"))
        entries = aar_entries()
        entries[GATE.NOTICE_ROOT + "RUNTIME_TERMS.md"] = b"edited\n"
        cases.append((entries, "RUNTIME_TERMS.md: bytes differ from the schema-5 contract"))
        entries = aar_entries()
        entries[GATE.NOTICE_ROOT + "LICENSES/Extra.txt"] = b"x"
        cases.append((entries, "unexpected file among the engine notices"))
        entries = aar_entries()
        entries["assets/fonts/x.ttf"] = b"x"
        cases.append((entries, "unexpected asset"))
        for entries, message in cases:
            with self.subTest(message=message):
                self.assertFails(self.aar(entries), message)
        # Directory entries: only the notice tree, and none without speech.
        self.assertPasses(self.aar(dict(aar_entries(), **{"assets/": b"", "assets/xybrid/zzz/LICENSES/": b""})))
        self.assertFails(self.aar(dict(aar_entries(), **{"assets/fonts/": b""})), "unexpected asset directory")
        self.assertFails(self.aar(dict(aar_entries(False), **{"assets/": b""}), False),
                         "unexpected asset directory")

    def test_manifest_is_a_text_manifest_with_the_min_sdk(self):
        good = self.aar(dict(aar_entries(), **{"AndroidManifest.xml": GRADLE_MANIFEST}))
        self.assertPasses(good)
        cases = {
            MANIFEST.replace(b'"29"', b'"28"'): "minSdkVersion is '28', expected 29",
            MANIFEST.replace(b'    <uses-sdk android:minSdkVersion="29" />\n', b""): "exactly one <uses-sdk>",
            MANIFEST.replace(b"</manifest>", b'<uses-sdk android:minSdkVersion="29" /></manifest>'):
                "exactly one <uses-sdk>",
            b"\x03\x00\x08\x00binary": "binary XML",
            b'<!DOCTYPE m [<!ENTITY a "a">]>' + MANIFEST: "DTD",
            b"<manifest": "not well-formed",
        }
        for manifest, message in cases.items():
            with self.subTest(message=message):
                self.assertFails(self.aar(dict(aar_entries(), **{"AndroidManifest.xml": manifest})), message)
        for required in ("AndroidManifest.xml", "classes.jar"):
            with self.subTest(missing=required):
                entries = aar_entries()
                del entries[required]
                self.assertFails(self.aar(entries), f"{required}: missing from the AAR")

    def test_min_sdk_cannot_undercut_the_arm64_link_level(self):
        self.assertFails(self.aar(dict(aar_entries(), **{"AndroidManifest.xml": MANIFEST.replace(b"29", b"28")}),
                                  True, "--min-sdk", "28"),
                         "--min-sdk 28 is below the arm64-v8a libraries' Android API 29")

    def test_api_levels_per_abi(self):
        cases = (("jni/arm64-v8a/libllama.so", dict(api=28), "arm64-v8a/libllama.so: linked for Android API 28, expected 29"),
                 ("jni/x86_64/libxybrid_bolt.so", dict(api=29), "x86_64/libxybrid_bolt.so: linked for Android API 29, expected 28"),
                 ("jni/armeabi-v7a/libxybrid_bolt.so", dict(api=29), "linked for Android API 29, expected 28"),
                 ("jni/arm64-v8a/libxybrid_bolt.so", dict(api=None), "linked for Android API none, expected 29"))
        for key, changes, message in cases:
            with self.subTest(key=key):
                entries = aar_entries()
                mutate(entries, key, **changes)
                self.assertFails(self.aar(entries), message)

    def test_needed_closure_alignment_machine_and_jni(self):
        bolt = lib_spec("arm64-v8a", GATE.BOLT, True)
        cases = (
            ("jni/arm64-v8a/libxybrid_bolt.so", dict(needed=bolt["needed"] + ["libfoo.so"]),
             "NEEDs 'libfoo.so', which is neither an Android system library nor shipped"),
            ("jni/x86_64/libxybrid_bolt.so", dict(needed=SYSTEM), "does not NEED libc++_shared.so"),
            ("jni/arm64-v8a/libggml.so", dict(aligns=[0x4000, 0x1000]), "aligned below 16 KB"),
            ("jni/x86_64/libxybrid_bolt.so", dict(machine="AArch64"),
             "ELF machine is 'AArch64', expected Advanced Micro Devices X86-64"),
            ("jni/arm64-v8a/libxybrid_bolt.so",
             dict(dynsyms=[s for s in bolt["dynsyms"] if s[0] != JNI_EXPORTS[0]]),
             "missing JNI exports: " + JNI_EXPORTS[0]),
        )
        for key, changes, message in cases:
            with self.subTest(message=message):
                entries = aar_entries()
                mutate(entries, key, **changes)
                self.assertFails(self.aar(entries), message)
        entries = aar_entries()  # armeabi-v7a is not a 16 KB-page ABI
        mutate(entries, "jni/armeabi-v7a/libxybrid_bolt.so", aligns=[0x1000])
        self.assertPasses(self.aar(entries))

    def test_native_inventory_is_exact(self):
        cases = []
        entries = aar_entries()
        del entries["jni/arm64-v8a/libmtmd.so"]
        cases.append((entries, "arm64-v8a/libmtmd.so: missing"))
        entries = aar_entries()
        entries["jni/x86_64/libfoo.so"] = elf_bytes(lib_spec("x86_64", "libfoo.so"))
        cases.append((entries, "x86_64/libfoo.so: unexpected native library"))
        for extra in ("jni/x86/libxybrid_bolt.so", "jni/arm64-v8a/README.txt",
                      "jni/arm64-v8a/sub/libx.so", "jni/libx.so"):
            entries = aar_entries()
            entries[extra] = b"x"
            cases.append((entries, f"{extra!r}: unexpected native payload"))
        for entries, message in cases:
            with self.subTest(message=message):
                self.assertFails(self.aar(entries), message)

    def test_private_staging_files_are_rejected_anywhere(self):
        for name in ("jni/arm64-v8a/libzzz_embed.a", "META-INF/RECEIPT.json", "res/raw/zzz_embed.h",
                     GATE.NOTICE_ROOT + ".pinned-archive.tar.gz", "META-INF/share/zzz_embed.json",
                     "jni/arm64-v8a/libxybrid_zzz_tts.so"):
            with self.subTest(name=name):
                self.assertFails(self.aar(dict(aar_entries(), **{name: b"private"})),
                                 f"{name!r}: private engine staging file")

    def test_unexpected_top_level_entries(self):
        for name, message in (("notes.txt", "unexpected top-level AAR entry"),
                              ("src/x.kt", "unexpected top-level AAR directory")):
            with self.subTest(name=name):
                self.assertFails(self.aar(dict(aar_entries(), **{name: b"x"})), message)
        self.assertPasses(self.aar(dict(aar_entries(), **{
            "META-INF/com/android/build/gradle/aar-metadata.properties": b"aarFormatVersion=1.0\n",
            "proguard.txt": b"", "res/values/values.xml": b"<resources/>"})))

    def test_readelf_failure_is_reported_without_its_output(self):
        self.readelf.fail.add("libllama.so")
        self.assertFails(self.aar(aar_entries()), "arm64-v8a/libllama.so: readelf failed")


class ZipSafetyTests(GateCase):
    def test_unsafe_names_are_rejected(self):
        for name in ("/abs.so", "../x.so", "jni/../x.so", "jni//x.so", "jni/./x.so",
                     "jni\\x.so", "jni/x y.so"):
            with self.subTest(name=name):
                self.assertFails(self.aar(list(aar_entries().items()) + [(name, b"x")]), "unsafe entry name")

    def test_duplicates_including_case_folded_ones(self):
        entries = list(aar_entries().items())
        for extra in (("classes.jar", b"other"), ("CLASSES.JAR", b"other")):
            with self.subTest(extra=extra[0]):
                self.assertFails(self.aar(entries + [extra]), "duplicate entry")

    def test_links_special_bits_and_conflicts(self):
        cases = []
        link = zipfile.ZipInfo("jni/arm64-v8a/libextra.so")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        cases.append(((link, "/etc/passwd"), "is a link or special file"))
        setuid = zipfile.ZipInfo("classes2.jar")
        setuid.external_attr = (stat.S_IFREG | 0o4755) << 16
        cases.append(((setuid, b"x"), "setuid"))
        cases.append((("R.txt/inner", b"x"), "both a file and a directory"))
        bz = zipfile.ZipInfo("proguard.txt")
        bz.compress_type = zipfile.ZIP_BZIP2
        cases.append(((bz, b"x"), "unsupported compression"))
        cases.append((("api.jar", b"\0" * (2 * 1024 * 1024)), "exceeds its size or compression bound"))
        for extra, message in cases:
            with self.subTest(message=message):
                self.assertFails(self.aar(list(aar_entries().items()) + [extra]), message)

    def test_encrypted_entries_are_rejected(self):
        path = self.package(aar_entries())
        set_central_flag(path, "classes.jar", 0x1)
        self.assertFails(self.run_gate("aar", path, "--speech"), "encrypted entry")

    def test_unreadable_inputs_get_fixed_messages(self):
        not_zip = self.tmp / "broken.aar"
        not_zip.write_bytes(b"not a zip")
        self.assertFails(self.run_gate("aar", not_zip, "--speech"), "broken.aar: not a readable zip")
        self.assertFails(self.run_gate("aar", self.tmp / "missing.aar", "--speech"), "missing.aar: cannot open")
        target = self.package(aar_entries())
        link = self.tmp / "link.aar"
        link.symlink_to(target)
        self.assertFails(self.run_gate("aar", link, "--speech"), "link.aar: cannot open")


class DistTests(GateCase):
    def test_pack_dist_keeps_only_abi_dirs_and_notices_and_gates(self):
        aar = self.package(aar_entries())
        dist = self.tmp / "xybrid-android-v9.9.9.zip"
        self.assertPasses(self.run_gate("pack-dist", aar, dist))
        with zipfile.ZipFile(dist) as archive:
            self.assertEqual(set(archive.namelist()), set(dist_entries()))
            for info in archive.infolist():
                self.assertEqual(info.external_attr >> 16, stat.S_IFREG | 0o644)
        for readelf in ("readelf", "llvm-readelf"):
            with self.subTest(readelf=readelf):
                out = self.assertPasses(self.run_gate("dist", dist, "--readelf", readelf))
                self.assertIn("arm64-v8a/libxybrid_bolt.so: speech engine linked statically", out)

    def test_pack_dist_never_overwrites_and_creates_nothing_on_error(self):
        aar = self.package(aar_entries())
        existing = self.tmp / "out.zip"
        existing.write_bytes(b"keep")
        self.assertFails(self.run_gate("pack-dist", aar, existing), "cannot create")
        self.assertEqual(existing.read_bytes(), b"keep")
        empty = self.package({"AndroidManifest.xml": MANIFEST}, "empty.aar")
        self.assertFails(self.run_gate("pack-dist", empty, self.tmp / "new.zip"), "no jni/")
        self.assertFalse((self.tmp / "new.zip").exists())

    def test_dist_layout_is_exact(self):
        zip_r = dict(dist_entries())  # `zip -r` also writes directory entries
        for directory in ("arm64-v8a/", "assets/", "assets/xybrid/", "assets/xybrid/zzz/",
                          "assets/xybrid/zzz/LICENSES/"):
            zip_r[directory] = b""
        self.assertPasses(self.run_gate("dist", self.package(zip_r, "d.zip")))
        cases = (({**dist_entries(), "AndroidManifest.xml": MANIFEST}, "unexpected entry"),
                 ({**dist_entries(), "x86/": b""}, "unexpected native directory"),
                 (aar_entries(), "unexpected entry"))
        for entries, message in cases:
            with self.subTest(message=message):
                self.assertFails(self.run_gate("dist", self.package(entries, "d.zip")), message)
        entries = dist_entries()
        del entries[GATE.NOTICE_ROOT + "RUNTIME_TERMS.md"]
        self.assertFails(self.run_gate("dist", self.package(entries, "d.zip")), "missing engine notice")


class StageGradleTests(GateCase):
    def setUp(self):
        super().setUp()
        self.kotlin = self.tmp / "kotlin"
        (self.kotlin / "src/main/kotlin").mkdir(parents=True)
        (self.kotlin / "libs").mkdir()
        (self.kotlin / "libs/.gitkeep").write_bytes(b"")
        (self.kotlin / "build.gradle.kts").write_text("android {}\n")
        self.zip = self.package(dist_entries(), "xybrid-android-v9.9.9.zip")

    def stage(self, zip_path=None):
        return self.run_gate("stage-gradle", zip_path or self.zip, "--kotlin-dir", self.kotlin)

    def tree(self):
        return {path.relative_to(self.kotlin).as_posix() for path in self.kotlin.rglob("*") if path.is_file()}

    def test_stages_exactly_the_natives_and_the_notices(self):
        before = self.tree()
        self.assertPasses(self.stage())
        expected = {f"libs/{abi}/{name}" for abi in GATE.ABIS for name in EXPECTED[abi]}
        expected |= {"src/main/" + GATE.NOTICE_ROOT + name for name in NOTICES}
        self.assertEqual(self.tree() - before, expected)
        self.assertEqual((self.kotlin / "src/main/assets/xybrid/zzz/LICENSES/Zig-MIT.txt").read_bytes(),
                         NOTICES["LICENSES/Zig-MIT.txt"])
        self.assertEqual((self.kotlin / "libs/x86_64/libxybrid_bolt.so").read_bytes(),
                         dist_entries()["x86_64/libxybrid_bolt.so"])

    def test_refuses_stale_natives(self):
        (self.kotlin / "libs/arm64-v8a").mkdir()
        (self.kotlin / "libs/arm64-v8a/libold.so").write_bytes(b"old")
        self.assertFails(self.stage(), "libs/arm64-v8a: already holds files")
        self.assertFalse((self.kotlin / "libs/x86_64").exists())

    def test_existing_notices_must_be_exact(self):
        notices = self.kotlin / "src/main/assets/xybrid/zzz"
        (notices / "LICENSES").mkdir(parents=True)
        (notices / "RUNTIME_TERMS.md").write_bytes(NOTICES["RUNTIME_TERMS.md"])
        self.assertPasses(self.stage())  # a partial, exact set is completed
        shutil.rmtree(self.kotlin / "libs")
        (self.kotlin / "libs").mkdir()
        self.assertPasses(self.stage())  # and an exact full set is accepted as is
        for path, data in (("RUNTIME_TERMS.md", b"edited"), ("EXTRA.md", b"x")):
            with self.subTest(path=path):
                shutil.rmtree(self.kotlin / "libs")
                (self.kotlin / "libs").mkdir()
                original = (notices / path).read_bytes() if (notices / path).exists() else None
                (notices / path).write_bytes(data)
                self.assertFails(self.stage(), "existing directory holds unexpected content")
                if original is None:
                    (notices / path).unlink()
                else:
                    (notices / path).write_bytes(original)

    def test_refuses_other_assets_and_links(self):
        (self.kotlin / "src/main/assets/fonts").mkdir(parents=True)
        self.assertFails(self.stage(), "src/main/assets: holds other assets")
        shutil.rmtree(self.kotlin / "src/main/assets")
        (self.kotlin / "src/main/assets/xybrid").mkdir(parents=True)
        outside = self.tmp / "outside"
        outside.mkdir()
        (self.kotlin / "src/main/assets/xybrid/zzz").symlink_to(outside, target_is_directory=True)
        self.assertFails(self.stage(), "holds a link")
        self.assertEqual(list(outside.iterdir()), [])
        shutil.rmtree(self.kotlin / "src/main/assets")
        shutil.rmtree(self.kotlin / "libs")
        (self.kotlin / "libs").symlink_to(outside, target_is_directory=True)
        self.assertFails(self.stage(), "libs: not a plain directory")
        self.assertEqual(list(outside.iterdir()), [])

    def test_refuses_a_wrong_kotlin_dir_and_bad_zips_without_writing(self):
        (self.kotlin / "build.gradle.kts").unlink()
        self.assertFails(self.stage(), "--kotlin-dir must be the Kotlin binding")
        (self.kotlin / "build.gradle.kts").write_text("android {}\n")
        before = self.tree()
        entries = dist_entries()
        entries[GATE.NOTICE_ROOT + "RUNTIME_TERMS.md"] = b"edited"
        self.assertFails(self.stage(self.package(entries, "bad.zip")), "bytes differ")
        entries = dist_entries()
        entries["x86_64/libfoo.so"] = b"x"
        self.assertFails(self.stage(self.package(entries, "bad.zip")), "unexpected native library")
        self.assertEqual(self.tree(), before)
        self.assertFalse((self.kotlin / "src/main/assets").exists())
        shutil.rmtree(self.kotlin / "libs")
        self.assertFails(self.stage(self.package(entries, "bad.zip")), "unexpected native library")
        self.assertFalse((self.kotlin / "libs").exists())
        self.assertPasses(self.stage())  # a missing libs/ is created once the zip validates

    def test_create_refuses_paths_outside_its_root(self):
        with self.assertRaisesRegex(GATE.GateError, "outside the staging roots"):
            GATE._create(self.tmp / "escape.so", self.kotlin / "libs")


class SymbolTests(GateCase):
    def archive(self, symbols):
        path = self.tmp / "libzzz_embed.a"
        path.write_bytes(elf_bytes({"archive": [["engine.o", symbols]]}))
        return path

    def library(self, name, exports):
        path = self.tmp / name
        path.write_bytes(elf_bytes({"dynsyms": [symbol(n) for n in exports] + [symbol("memcpy@LIBC", "UND")]}))
        return path

    def test_overlap_prints_only_publicly_exported_names(self):
        archive = self.archive([symbol("ggml_init"), symbol("llama_backend_init", vis="HIDDEN"),
                                symbol("gguf_init_empty", bind="WEAK"), symbol(PRIVATE_SYMBOL),
                                symbol("memcpy", "UND"), symbol("local_helper", bind="LOCAL"),
                                symbol("engine.zig", "ABS", kind="FILE", bind="LOCAL")])
        libs = [self.library("libggml-base.so", ["ggml_init", "gguf_init_empty", "local_helper"]),
                self.library("libllama.so", ["llama_backend_init"]),
                self.library("libc++_shared.so", ["memcpy_like"])]
        for readelf in ("readelf", "llvm-readelf"):
            with self.subTest(readelf=readelf):
                code, out, err = self.run_gate("symbol-overlap", "--readelf", readelf, "--archive", archive, *libs)
                self.assertEqual(code, 1, out + err)
                self.assertIn("3 engine symbol(s) also exported by 3 public libraries", out)
                for name in ("ggml_init", "gguf_init_empty", "llama_backend_init"):
                    self.assertIn(name, out)
                self.assertNotIn(PRIVATE_SYMBOL, out + err)
                self.assertNotIn("local_helper", out)

    def test_no_overlap_passes_and_failures_hide_the_archive_path(self):
        archive = self.archive([symbol(PRIVATE_SYMBOL)])
        libs = [self.library("libggml.so", ["ggml_init"])]
        code, out, err = self.run_gate("symbol-overlap", "--archive", archive, *libs)
        self.assertEqual((code, out.strip()), (0, "symbol-overlap: 0 engine symbol(s) also exported by 1 public library"))
        self.readelf.fail.add("libzzz_embed.a")
        code, out, err = self.run_gate("symbol-overlap", "--archive", archive, *libs)
        self.assertEqual(code, 1)
        self.assertEqual(err.strip(), "symbol-overlap: readelf could not read the engine archive")
        empty = self.archive([symbol("memcpy", "UND")])
        self.readelf.fail.clear()
        self.assertFails(self.run_gate("symbol-overlap", "--archive", empty, *libs), "wrong file?")

    def test_compare_imports(self):
        default = self.tmp / "default.so"
        default.write_bytes(elf_bytes({"dynsyms": [symbol("llama_backend_init", "UND"),
                                                   symbol("ggml_init", "UND"), symbol("malloc@LIBC", "UND")]}))
        same = self.tmp / "speech.so"
        same.write_bytes(elf_bytes({"dynsyms": [symbol("llama_backend_init", "UND"), symbol("ggml_init", "UND"),
                                                symbol("mtmd_init", "UND")]}))
        out = self.assertPasses(self.run_gate("compare-imports", default, same))
        self.assertIn("all 2 shared llama.cpp imports are still imported (1 new)", out)
        interposed = self.tmp / "interposed.so"
        interposed.write_bytes(elf_bytes({"dynsyms": [symbol("llama_backend_init", "UND"), symbol("ggml_init")]}))
        err = self.assertFails(self.run_gate("compare-imports", default, interposed), "1 of 2", "ggml_init")
        self.assertNotIn("llama_backend_init", err)
        bare = self.tmp / "bare.so"
        bare.write_bytes(elf_bytes({"dynsyms": [symbol("malloc@LIBC", "UND")]}))
        self.assertFails(self.run_gate("compare-imports", bare, same), "imports no ggml_/gguf_/llama_/mtmd_ symbols")


class CliTests(GateCase):
    def test_usage_errors_do_not_echo_arguments(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as raised:
            GATE.main(["symbol-overlap", "--archive", "/private/zzz-inputs/libzzz_embed.a", "--bogus"])
        self.assertEqual(raised.exception.code, 2)
        self.assertNotIn("/private", err.getvalue())
        self.assertIn("invalid arguments", err.getvalue())

    def test_bzl_lists_are_read_without_comments(self):
        bzl = self.tmp / "android.bzl"
        bzl.write_text('# "libcomment.so"\nANDROID_ARM64_GGML_LIBS = [\n    "liba.so",  # "libnot.so"\n]\n'
                       'ANDROID_ARM64_CPU_VARIANTS = [\n    "libb.so",\n]\n')
        self.assertEqual(GATE.expected_natives(bzl)["arm64-v8a"],
                         {GATE.BOLT, GATE.CXX, GATE.ORT, "liba.so", "libb.so"})
        self.assertEqual(GATE.expected_natives(bzl)["armeabi-v7a"], {GATE.BOLT, GATE.CXX})


# --- workflow hygiene (string level; no YAML parser needed) -----------------


def workflow(name):
    return (ROOT / ".github/workflows" / name).read_text()


def job(text, job_id):
    lines = text.splitlines()
    start = lines.index(f"  {job_id}:")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.match(r"^  [A-Za-z0-9_-]+:\s*$", lines[index]) or re.match(r"^\S", lines[index]):
            end = index
            break
    return "\n".join(lines[start:end])


def header(job_text):
    return job_text.split("\n    steps:\n", 1)[0]


def steps(job_text):
    """Each step's text, full-line comments dropped."""
    blocks = []
    for line in job_text.split("\n    steps:\n", 1)[1].splitlines():
        if line.startswith("      - "):
            blocks.append([line])
        elif blocks and not line.lstrip().startswith("#"):
            blocks[-1].append(line)
    return ["\n".join(block) for block in blocks]


def first(blocks, *needles):
    return min(i for i, block in enumerate(blocks) if any(needle in block for needle in needles))


TOOLING = ("bazelisk", "bazel-repo-cache", "write-buildbuddy-rc", "actions/cache@", "setup-java",
           "setup-android", "rust-toolchain", "gradlew")


class WorkflowHygieneTests(unittest.TestCase):
    def assert_single_token_step(self, text, job_text, flag):
        blocks = steps(job_text)
        with_secrets = [i for i, block in enumerate(blocks) if "secrets.ZZZ_" in block]
        self.assertEqual(len(with_secrets), 1)
        stage = blocks[with_secrets[0]]
        self.assertEqual(with_secrets[0], 1, "staging must follow checkout directly")
        self.assertIn("uses: actions/checkout@", blocks[0])
        self.assertIn("persist-credentials: false", blocks[0])
        self.assertIn("submodules: true", blocks[0])
        self.assertIn("python3 -I tools/scripts/zzz_pull.py", stage)
        self.assertIn('mkdir -m 0700 "$inputs"', stage)
        self.assertIn('mv -T -- "$slice" "$inputs/$target"', stage)
        self.assertIn('rm -rf "$RUNNER_TEMP/zzz-cache" "$RUNNER_TEMP/zzz-staging"', stage)
        for forbidden in TOOLING + ("XYBRID_ZZZ_PREBUILT_DIR",):
            self.assertNotIn(forbidden, stage)
        self.assertLess(with_secrets[0], first(blocks, *TOOLING))
        # Job env: a boolean presence flag only; nowhere else in the file.
        comparisons = re.findall(r"secrets\.ZZZ_\w+( != '')?", header(job_text))
        self.assertTrue(comparisons)
        self.assertTrue(all(comparisons), "job env may only compare the secrets with ''")
        self.assertIn(f"      {flag}: ${{{{ secrets.ZZZ_GITHUB_TOKEN != ''", header(job_text))
        total = text.count("secrets.ZZZ_")
        self.assertEqual(total, stage.count("secrets.ZZZ_") + header(job_text).count("secrets.ZZZ_"))
        self.assertEqual(stage.count("${{ secrets.ZZZ_GITHUB_TOKEN }}"), 1)
        self.assertEqual(stage.count("${{ secrets.ZZZ_RELEASE_REPOSITORY }}"), 1)
        self.assertNotIn("set -x", text)
        return blocks, with_secrets[0]

    def assert_inputs_deleted_before_packaging(self, blocks, last_private_use):
        delete = first(blocks, 'rm -rf "$RUNNER_TEMP"/zzz-*')
        self.assertIn("if: always()", blocks[delete])
        self.assertGreater(delete, last_private_use)
        later = blocks[delete + 1:]
        self.assertFalse(any("zzz-inputs" in block for block in later))
        for name in ("upload-artifact", "attest-build-provenance", "gradlew", "stage-gradle", "pack-dist"):
            uses = [i for i, block in enumerate(blocks) if name in block]
            self.assertTrue(all(i > delete for i in uses), name)
        for block in blocks:
            if "upload-artifact" in block:
                self.assertNotIn("zzz-", block.split("path:", 1)[1].split("\n")[0])
                self.assertIn("retention-days:", block)
        return delete

    def test_release_prep_build_android(self):
        text = workflow("release-prep.yml")
        job_text = job(text, "build-android")
        self.assertIn("    environment: zzz-release\n", header(job_text))
        self.assertEqual(text.count("environment: zzz-release"), 1)
        self.assertNotIn("zzz-engine\n", "\n".join(steps(job_text)))
        self.assertIn("    name: Build Android\n", header(job_text))
        self.assertIn("needs.check-release-branch.outputs.dry_run != 'true'", header(job_text))
        self.assertIn("id-token: write", header(job_text))
        blocks, stage = self.assert_single_token_step(text, job_text, "ZZZ_RELEASE_ENABLED")
        # Fails closed: no `if:`, an ::error:: and exit when the flag is off.
        self.assertNotIn("        if:", blocks[stage])
        self.assertIn('if [ "$ZZZ_RELEASE_ENABLED" != "true" ]; then', blocks[stage])
        self.assertIn("::error", blocks[stage])
        build = first(blocks, "bazelisk build")
        for flag in ("--config=remote", "-c opt", "--//:tts_zzz=true",
                     '"--repo_env=XYBRID_ZZZ_PREBUILT_DIR=$RUNNER_TEMP/zzz-inputs"',
                     '"--repo_env=XYBRID_ZZZ_PYTHON=$(command -v python3)"',
                     "//bindings/kotlin:xybrid_kotlin_aar", "tools/scripts/bazel-output.sh", "cp -L"):
            self.assertIn(flag, blocks[build])
        delete = self.assert_inputs_deleted_before_packaging(blocks, build)
        self.assertEqual(delete, build + 1)
        gate = first(blocks, "verify_zzz_android.py aar")
        self.assertIn("--readelf readelf --speech --min-sdk 29", blocks[gate])
        pack = first(blocks, "verify_zzz_android.py pack-dist")
        self.assertIn("verify_zzz_android.py dist", blocks[pack])
        attest, upload = first(blocks, "attest-build-provenance"), first(blocks, "upload-artifact")
        self.assertLess(delete, gate)
        self.assertLess(gate, pack)
        self.assertLess(pack, attest)
        self.assertLess(attest, upload)
        self.assertIn("name: android-libs", blocks[upload])
        self.assertIn("path: dist/xybrid-android-*.zip", blocks[upload])
        # precompile-flutter-linux still waits for build-android (warm RBE cache).
        self.assertIn("needs: [check-release-branch, build-android]", job(text, "precompile-flutter-linux"))

    def test_release_publish_gates_before_maven(self):
        text = workflow("release-publish.yml")
        job_text = job(text, "publish-kotlin")
        self.assertNotIn("environment:", header(job_text))
        self.assertNotIn("ZZZ_", job_text)
        blocks = steps(job_text)
        self.assertFalse(any("unzip" in block for block in blocks))
        order = [first(blocks, needle) for needle in (
            "gh release download", "verify_zzz_android.py dist", "verify_zzz_android.py stage-gradle",
            "./gradlew bundleReleaseAar", "verify_zzz_android.py aar",
            "publishAllPublicationsToMavenCentralRepository")]
        self.assertEqual(order, sorted(order))
        self.assertEqual(len(set(order)), 5)  # dist + stage-gradle share one step
        gate = blocks[first(blocks, "verify_zzz_android.py aar")]
        self.assertIn("bindings/kotlin/build/outputs/aar/xybrid-kotlin-release.aar", gate)
        self.assertIn("--readelf readelf --speech --min-sdk 29", gate)

    def test_proof_lane(self):
        text = workflow("test-zzz-android.yml")
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("# Purpose: "))
        self.assertTrue(lines[1].startswith('name: "CI: '))
        self.assertIn("  group: test-zzz-android-${{ github.ref }}\n", text)
        on = text.split("\non:\n", 1)[1].split("\nconcurrency:", 1)[0]
        self.assertNotIn("pull_request", on)
        self.assertIn("    branches: [master]", on)
        self.assertIn("'!**.md'", on)
        speech = job(text, "speech-aar")
        self.assertIn("    if: github.ref == 'refs/heads/master'\n", header(speech))
        self.assertIn("    environment: zzz-engine\n", header(speech))
        self.assertEqual(text.count("environment:"), 1)
        blocks, stage = self.assert_single_token_step(text, speech, "ZZZ_CI_ENABLED")
        self.assertIn("if: env.ZZZ_CI_ENABLED == 'true'", blocks[stage])
        self.assertTrue(any("::notice" in block and "ZZZ_CI_ENABLED != 'true'" in block for block in blocks))
        overlap = first(blocks, "verify_zzz_android.py symbol-overlap")
        self.assertIn("zzz-inputs/aarch64-linux-android/lib/libzzz_embed.a", blocks[overlap])
        build = first(blocks, "bazelisk build")
        self.assertIn("--//:tts_zzz=true", blocks[build])
        delete = self.assert_inputs_deleted_before_packaging(blocks, overlap)
        for needle in ("aar \"$RUNNER_TEMP/aars/default.aar\"", "--no-speech", "compare-imports",
                       "cmp -s", "::warning"):
            self.assertLess(first(blocks, needle), delete)
        order = [first(blocks, needle) for needle in (
            "verify_zzz_android.py pack-dist", "./gradlew bundleReleaseAar",
            "build/outputs/aar/xybrid-kotlin-release.aar \\", "zip -q -d", "name: zzz-android-dlopen-gate",
            "name: xybrid-kotlin-zzz-aar")]
        self.assertEqual(order, sorted(order))
        dlopen = blocks[first(blocks, "zip -q -d")]
        self.assertIn("'jni/arm64-v8a/*' 'jni/armeabi-v7a/*'", dlopen)
        self.assertIn("grep -c", dlopen)
        aar_upload = blocks[first(blocks, "name: xybrid-kotlin-zzz-aar")]
        self.assertIn("retention-days: 1", aar_upload)
        self.assertIn("path: bindings/kotlin/build/outputs/aar/xybrid-kotlin-release.aar", aar_upload)
        gate = job(text, "dlopen-gate")
        self.assertNotIn("environment:", header(gate))
        self.assertNotIn("secrets.", gate)
        self.assertIn("    needs: speech-aar\n", header(gate))
        self.assertIn("needs.speech-aar.outputs.staged == 'true'", header(gate))
        self.assertIn("name: zzz-android-dlopen-gate", gate)
        self.assertNotIn("xybrid-kotlin-zzz-aar", gate)
        self.assertIn("api-level: 35", gate)

    def test_ci_environments_stay_master_only(self):
        for name in ("test-zzz-engine.yml", "trusted-zzz-sdk.yml"):
            with self.subTest(name=name):
                text = workflow(name)
                self.assertIn("environment: zzz-engine", text)
                self.assertNotIn("zzz-release", text)


if __name__ == "__main__":
    unittest.main()
