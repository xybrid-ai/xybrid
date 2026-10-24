"""Offline companion preparation checks; no engine build, linking or execution."""

import copy
import hashlib
import importlib.util
import io
import json
import contextlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "bazel/zzz/companion"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


metadata = load("_test_companion_metadata", DIRECTORY / "metadata.py")
checker = load("_test_companion_exports", DIRECTORY / "check_exports.py")
verifier = metadata.load_verifier(ROOT / "crates/zzz-sys/zzz_pull.py")

# Deliberately synthetic declarations, never an engine library or staged pin.
SYNTHETIC_HEADER = r"""
#include <stddef.h>
#include <stdint.h>
#define ZZZ_EMBED_ABI_VERSION 1u
typedef struct zzz_embed_session zzz_embed_session;
typedef struct { int32_t code; char message[192]; } zzz_embed_error;
typedef struct { uint32_t struct_size,max_frames_per_chunk,max_chunk_tokens;
 float temperature; uint64_t seed; } zzz_embed_options;
typedef struct { uint32_t sample_rate,channels,chunks,limited_chunks;
 uint64_t samples; } zzz_embed_result;
typedef struct { uint32_t struct_size,threads,max_tokens,flags; uint64_t seed;
 const char *language_model_path,*decoder_model_path,*language_voice_path,
 *decoder_voice_path; } zzz_embed_kitten_options;
typedef int32_t (*zzz_embed_audio_callback)(const float *,size_t,uint32_t,
 uint32_t,uint64_t,void *);
uint32_t zzz_embed_abi_version(void);
int32_t zzz_embed_supports_model(const char *);
void zzz_embed_default_options(zzz_embed_options *);
int32_t zzz_embed_open(const char *,const char *,const char *,
 const zzz_embed_options *,zzz_embed_session **,zzz_embed_error *);
int32_t zzz_embed_open_kitten(const zzz_embed_kitten_options *,
 zzz_embed_session **,zzz_embed_error *);
int32_t zzz_embed_synthesize(zzz_embed_session *,const char *,const char *,
 zzz_embed_audio_callback,void *,zzz_embed_result *,zzz_embed_error *);
void zzz_embed_cancel(zzz_embed_session *);
void zzz_embed_close(zzz_embed_session *);
"""


class CompanionMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        original = json.loads(verifier.DEFAULT_MANIFEST.read_text())
        entry = copy.deepcopy(next(e for e in original["archives"] if e["target"] == metadata.TARGET))
        entry.update(receipt_schema=5, sha256="a" * 64,
                     notice_files=copy.deepcopy(verifier.SCHEMA5_NOTICE_FILES),
                     archive="zzz-phonon-kitten-tts2-android-arm64-v0.1.0-rc.9.tar.gz")
        self.data = dict(original, release_tag="v0.1.0-rc.9", archives=[entry])
        self.manifest = self.root / "synthetic.json"

    def identity(self):
        self.manifest.write_text(json.dumps(self.data))
        return metadata.contract(verifier, self.manifest)

    def test_identity_binds_canonical_entire_actual_pin_and_release(self):
        identity, record = self.identity()
        expected = hashlib.sha256(
            b"xybrid.android-tts-companion.v1\0" +
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).digest()
        self.assertEqual(expected, identity)
        self.assertEqual(32, len(identity))
        self.data["archives"][0]["sha256"] = "b" * 64
        archive_identity = self.identity()[0]
        self.assertNotEqual(identity, archive_identity)
        self.data["release_tag"] = "v0.1.0-rc.10"
        self.data["archives"][0]["archive"] = self.data["archives"][0]["archive"].replace("rc.9", "rc.10")
        release_identity = self.identity()[0]
        self.assertNotEqual(archive_identity, release_identity)
        # A different independently fixed notice contract must also change the
        # identity, even when every other selected pin field stays unchanged.
        notices = dict(verifier.SCHEMA5_NOTICE_FILES, **{"RUNTIME_TERMS.md": "c" * 64})
        self.data["archives"][0]["notice_files"] = notices
        with mock.patch.object(verifier, "SCHEMA5_NOTICE_FILES", notices):
            self.assertNotEqual(release_identity, self.identity()[0])
        # Notices are part of the independently validated record being hashed.
        self.assertEqual(verifier.SCHEMA5_NOTICE_FILES, record["entry"]["notice_files"])
        self.assertEqual(record, json.loads(json.dumps(record, sort_keys=True)))

    def test_real_rc8_is_rejected_without_writing_a_header(self):
        output = self.root / "companion_metadata.h"
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(1, metadata.main([
                "--verifier", str(ROOT / "crates/zzz-sys/zzz_pull.py"),
                "--manifest", str(verifier.DEFAULT_MANIFEST), "--output", str(output)]))
        self.assertFalse(output.exists())
        self.assertIn("notice-bearing selected Android pin is required", error.getvalue())
        self.assertIn("EXPECTED_CONTRACT: Option<[u8; 32]> = None", (ROOT / "crates/zzz-sys/src/companion.rs").read_text())

    def test_invalid_or_incompatible_pins_cannot_generate_metadata(self):
        original = copy.deepcopy(self.data)
        for field, value in (("cpu", "apple_m1"), ("deployment_minimum", "28"),
                             ("system_libraries", ["c", "m", "dl", "log"]),
                             ("models", ["phonon"]), ("apis", {"zzz_embed": 2}),
                             ("notice_files", {}), ("sha256", "not-a-pin")):
            with self.subTest(field=field):
                self.data = copy.deepcopy(original)
                self.data["archives"][0][field] = value
                with self.assertRaises(ValueError):
                    self.identity()
        for entries in ([], original["archives"] * 2):
            self.data = dict(original, archives=entries)
            with self.assertRaises(ValueError):
                self.identity()

    def test_generated_header_has_only_fixed_contract_values(self):
        identity, _ = self.identity()
        text = metadata.header(identity)
        self.assertIn("XYBRID_TTS_ANDROID_API 29u", text)
        self.assertIn("XYBRID_TTS_MODELS 3u", text)
        self.assertEqual(list(identity), [int(value, 16) for value in re.findall(r"0x([0-9a-f]{2})", text)])
        for private_value in (str(self.root), "RECEIPT", "source_commit", "fake-private-token"):
            self.assertNotIn(private_value, text)


class CompanionAdapterTests(unittest.TestCase):
    def test_fixed_table_syntax_and_actual_compile_guard_with_synthetic_headers(self):
        compiler = shutil.which("clang")
        self.assertIsNotNone(compiler, "clang required for the bounded syntax-only C ABI check")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "zzz_embed.h").write_text(SYNTHETIC_HEADER)
            (root / "companion_metadata.h").write_text(metadata.header(bytes(range(32))))
            command = [compiler, "-std=c11", "-fsyntax-only", "-Wall", "-Wextra", "-Werror",
                       "-I", str(root), "-I", str(DIRECTORY), str(DIRECTORY / "xybrid_tts_companion.c")]
            for macros, success in (
                (["-D__ANDROID__", "-D__aarch64__", "-D__ANDROID_API__=29"], True),
                (["-D__ANDROID__", "-D__aarch64__", "-D__ANDROID_API__=28"], False),
                (["-D__ANDROID__", "-D__aarch64__", "-D__ANDROID_API__=30"], False),
                (["-U__ANDROID__", "-D__aarch64__", "-D__ANDROID_API__=29"], False),
                (["-D__ANDROID__", "-U__aarch64__", "-D__ANDROID_API__=29"], False),
            ):
                with self.subTest(macros=macros):
                    result = subprocess.run(command + macros, text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, timeout=20)
                    self.assertEqual(success, result.returncode == 0, result.stderr)
                    if not success:
                        self.assertIn("separate Android ARM64 API29 toolchain", result.stderr)
            self.assertFalse(list(root.glob("*.o")))

    def test_bazel_seam_requires_explicit_toolchain_and_exact_delivery_notices(self):
        targets = {}
        def capture(kind):
            def save(**kwargs):
                targets[kwargs["name"]] = (kind, kwargs)
            return save
        namespace = {"load": lambda *args: None}
        for kind in ("constraint_setting", "constraint_value", "platform", "toolchain", "py_binary",
                     "genrule", "cc_binary", "platform_transition_filegroup", "filegroup"):
            namespace[kind] = capture(kind)
        path = DIRECTORY / "BUILD.bazel"
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        toolchain = targets["api29_toolchain"][1]
        required = set(toolchain["target_compatible_with"])
        base = {"@platforms//os:android", "@platforms//cpu:aarch64", ":base_api"}
        marked = base - {":base_api"} | {":api29"}
        self.assertFalse(required <= base)
        self.assertTrue(required <= marked)
        # The inspected original NDK toolchain requires only Android+ARM64, so
        # it matches both. Selection needs explicit extra-toolchain priority;
        # the syntax-only guard above proves fallback API28 cannot compile.
        old_required = {"@platforms//os:android", "@platforms//cpu:aarch64"}
        self.assertTrue(old_required <= base and old_required <= marked)
        self.assertIn("--extra_toolchains=//bazel/zzz/companion:api29_toolchain", (DIRECTORY / "README.md").read_text())
        self.assertEqual(":base_api", targets["android_api"][1]["default_constraint_value"])
        self.assertEqual([":api29"], targets["android_arm64_api29"][1]["constraint_values"])
        self.assertEqual(":android_arm64_api29", targets["companion"][1]["target_platform"])
        self.assertEqual(["@zzz_android//:zzz"], targets["libxybrid_zzz_tts.so"][1]["deps"])
        self.assertEqual([":companion", "@zzz_android//:notices"], targets["delivery_inputs"][1]["srcs"])
        self.assertIn("-Wl,--exclude-libs,ALL", targets["libxybrid_zzz_tts.so"][1]["linkopts"])
        self.assertIn("-Wl,-soname,libxybrid_zzz_tts.so", targets["libxybrid_zzz_tts.so"][1]["linkopts"])
        module = (ROOT / "MODULE.bazel").read_text()
        self.assertIn('android_ndk_api29(name = "androidndk_zzz_api29", api_level = 29)', module)
        self.assertNotRegex(module, r'register_toolchains\([^)]*(?:api29|companion)')
        self.assertIn("api_level = 28", module)
        self.assertEqual(["xybrid_tts_get_api_v1"], re.findall(r"\b(xybrid_\w+)\s*;", (DIRECTORY / "exports.map").read_text()))
        self.assertIn("local:\n        *;", (DIRECTORY / "exports.map").read_text())


class CompanionExportTests(unittest.TestCase):
    HEADER = "Class: ELF64\nData: 2's complement, little endian\nType: DYN (Shared object file)\nMachine: AArch64\n"
    SYMBOLS = "Symbol table '.dynsym' contains 2 entries:\n 1: 00000100 32 FUNC GLOBAL DEFAULT 10 xybrid_tts_get_api_v1\n"
    DYNAMIC = "0x1 (NEEDED) Shared library: [libc.so]\n0xe (SONAME) Library soname: [libxybrid_zzz_tts.so]\n"
    PROGRAMS = "  LOAD 0x0 0x0 0x0 0x100 0x100 R E 0x4000\n"

    def test_exact_getter_only_report_is_accepted(self):
        checker.audit(self.HEADER, self.SYMBOLS, self.DYNAMIC, self.PROGRAMS)

    def test_leaked_engine_apis_and_other_exports_are_rejected(self):
        for symbol in (" 2: 00000200 16 FUNC GLOBAL DEFAULT 10 zzz_embed_abi_version\n",
                       " 2: 00000000 0 FUNC GLOBAL DEFAULT UND zzz_embed_synthesize\n",
                       " 2: 00000200 16 FUNC GLOBAL DEFAULT 10 zzz_engine_open\n",
                       " 2: 00000200 16 FUNC GLOBAL DEFAULT 10 incidental_runtime_export\n"):
            with self.subTest(symbol=symbol), self.assertRaises(ValueError):
                checker.audit(self.HEADER, self.SYMBOLS + symbol, self.DYNAMIC, self.PROGRAMS)

    def test_wrong_machine_soname_dependencies_alignment_or_missing_reports_fail(self):
        for reports in (
            (self.HEADER.replace("AArch64", "X86-64"), self.SYMBOLS, self.DYNAMIC, self.PROGRAMS),
            (self.HEADER.replace("little endian", "big endian"), self.SYMBOLS, self.DYNAMIC, self.PROGRAMS),
            (self.HEADER, "", self.DYNAMIC, self.PROGRAMS),
            (self.HEADER, self.SYMBOLS.replace("FUNC GLOBAL DEFAULT", "FUNC WEAK DEFAULT"), self.DYNAMIC, self.PROGRAMS),
            (self.HEADER, self.SYMBOLS, self.DYNAMIC.replace(checker.SONAME, "libwrong.so"), self.PROGRAMS),
            (self.HEADER, self.SYMBOLS, self.DYNAMIC + "(NEEDED) [libc++_shared.so]\n", self.PROGRAMS),
            (self.HEADER, self.SYMBOLS, self.DYNAMIC, self.PROGRAMS.replace("0x4000", "0x1000")),
            (self.HEADER, self.SYMBOLS, self.DYNAMIC, self.PROGRAMS.replace("LOAD 0x0 0x0", "LOAD 0x1000 0x0")),
            (self.HEADER, self.SYMBOLS, self.DYNAMIC, self.PROGRAMS.replace("LOAD 0x0 0x0", "LOAD 0x0 0x1000")),
            (self.HEADER, self.SYMBOLS, self.DYNAMIC, ""),
        ):
            with self.subTest(reports=reports), self.assertRaises(ValueError):
                checker.audit(*reports)
