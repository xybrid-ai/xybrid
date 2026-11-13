"""Offline checks of the speech engine's isolation (bazel/zzz/isolate.bzl).

No Bazel, NDK or engine bytes. The API list must match the Rust declarations,
the Android slice must reach zzz-sys only through the isolation, the rule's
Starlark (a Python-compatible subset) must pick exactly one static archive,
and isolate_engine.sh runs against fake ld/objcopy/nm/ar to prove its checks
fail closed without printing private symbol names. The real tools run in
bazel.yml's self-test (bazel/zzz/selftest). Python 3.9+, standard library only.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "bazel/zzz/isolate_engine.sh"
ANDROID_ENGINE = "//bazel/zzz:android_engine"
SECRET = "secret_engine_symbol"
SECRET_HELPER = SECRET + "_helper"


class RuleFailure(Exception):
    pass


def fail(message):
    raise RuleFailure(message)


def isolate_bzl():
    namespace = {
        "load": lambda *a, **kw: None, "fail": fail, "struct": types.SimpleNamespace,
        "rule": lambda **kw: kw, "Label": str, "use_cc_toolchain": lambda: [],
        "attr": types.SimpleNamespace(string_list=lambda **kw: kw, label=lambda **kw: kw),
        "CcInfo": "CcInfo",
    }
    path = ROOT / "bazel/zzz/isolate.bzl"
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


API = isolate_bzl()["ZZZ_EMBED_API"]


def rust_api():
    """The extern "C" functions of static_functions() in crates/zzz-sys/src/api.rs."""
    text = (ROOT / "crates/zzz-sys/src/api.rs").read_text()
    body = text[text.index("fn static_functions() -> Functions {"):]
    block = body[body.index('extern "C" {'):body.index("\n    }\n")]
    return re.findall(r"\bfn (\w+)\(", block)


class Select:
    def __init__(self, arms):
        self.arms = arms

    def __radd__(self, other):
        return list(other) + [self]


def build_calls(relative, **names):
    """Runs a BUILD file; returns {target name: kwargs} of every rule call."""
    calls = {}

    def capture(**kwargs):
        calls[kwargs["name"]] = kwargs

    namespace = {"load": lambda *a, **kw: None, "glob": lambda *a, **kw: [], "select": Select,
                 "exports_files": lambda *a, **kw: None, "alias": capture, "XYBRID_VERSION": "test",
                 "all_crate_deps": lambda **kw: [], "ZZZ_EMBED_API": API,
                 "selects": types.SimpleNamespace(config_setting_group=capture)}
    for rule in ("config_setting", "rust_library", "cc_library", "cc_binary", "zzz_isolated_engine",
                 "isolation_selftest"):
        namespace[rule] = capture
    namespace.update(names)
    path = ROOT / relative
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return calls


class WiringTests(unittest.TestCase):
    def test_api_is_exactly_the_rust_declarations(self):
        self.assertEqual(8, len(API))
        self.assertEqual(len(API), len(set(API)))
        self.assertTrue(all(name.startswith("zzz_embed_") for name in API))
        self.assertEqual(sorted(rust_api()), sorted(API))

    def test_android_engine_isolates_the_pinned_slice(self):
        engine = build_calls("bazel/zzz/BUILD.bazel")["android_engine"]
        self.assertEqual("@zzz_android//:zzz", engine["engine"])
        self.assertIs(API, engine["api"])
        # Wildcard analysis must never reach @zzz_android without a staged slice.
        self.assertEqual(["manual"], engine["tags"])
        # rules_rust also configures zzz-sys's deps in the exec configuration;
        # an incompatible dep there would make the flag-on AAR incompatible.
        self.assertNotIn("target_compatible_with", engine)
        self.assertEqual(["//crates/zzz-sys:__pkg__"], engine["visibility"])

    def test_zzz_sys_links_android_only_through_the_isolation(self):
        deps = build_calls("crates/zzz-sys/BUILD.bazel")["xybrid_zzz_sys"]["deps"]
        arms = next(part for part in deps if isinstance(part, Select)).arms
        self.assertEqual([ANDROID_ENGINE], arms["//bazel/zzz:android_enabled"])
        for setting, repository in (("linux", "zzz_linux"), ("macos", "zzz_macos"), ("ios", "zzz_ios"),
                                    ("ios_sim", "zzz_ios_sim"), ("linux_arm64", "zzz_linux_arm64")):
            self.assertEqual([f"@{repository}//:zzz"], arms[f"//bazel/zzz:{setting}_enabled"])
        # No other BUILD or .bzl file links the raw Android archive.
        tracked = subprocess.run(["git", "ls-files", "-z", "--", "*BUILD.bazel", "*BUILD", "*.bzl"], cwd=ROOT,
                                 capture_output=True, check=True).stdout.decode().split("\0")
        raw = [relative for relative in filter(None, tracked)
               if '"@zzz_android//:zzz"' in (ROOT / relative).read_text(errors="replace")]
        self.assertEqual(["bazel/zzz/BUILD.bazel"], raw)

    def test_selftest_exercises_the_real_api_and_the_canary_names(self):
        calls = build_calls("bazel/zzz/selftest/BUILD.bazel")
        selftest = calls["selftest"]
        self.assertIs(API, selftest["api"])
        self.assertIs(API, calls["fake_engine_isolated"]["api"])
        for name in ("memcpy", "strlen", "expf"):
            self.assertIn(name, selftest["local"])
            self.assertIn(name, selftest["imports"])
        self.assertIn("write", selftest["undefined"])
        source = (ROOT / "bazel/zzz/selftest/fake_engine.c").read_text()
        for name in API + ["memcpy", "strlen", "expf"]:
            self.assertRegex(source, rf"\b{name}\([^;]*\)\s*\{{", name)
        # The dead member: its plain .text would merge with fake_engine.c's
        # live one without ld -r --unique, and its unresolved call would then
        # fail the --gc-sections/--no-undefined link of the library.
        engine = calls["fake_engine"]
        self.assertEqual(["fake_engine.c", "fake_engine_unused.c"], engine["srcs"])
        for flag in ("-fno-function-sections", "-fno-data-sections", "-faddrsig"):
            self.assertIn(flag, engine["copts"])
        for flag in ("-Wl,--gc-sections", "-Wl,--no-undefined"):
            self.assertIn(flag, calls["libselftest.so"]["linkopts"])
        self.assertEqual(["fake_engine_unresolved"], selftest["collected"])
        self.assertIn("fake_engine_unused", selftest["local"])
        unused = (ROOT / "bazel/zzz/selftest/fake_engine_unused.c").read_text()
        self.assertIn("extern void fake_engine_unresolved(void);", unused)
        self.assertIn("void fake_engine_unused(void) { fake_engine_unresolved(); }", unused)
        for other in ("fake_engine.c", "host.c"):
            self.assertNotIn("fake_engine_unresolved", (ROOT / "bazel/zzz/selftest" / other).read_text())
        # Android 10's native TLS only: never in a host or API-28 build.
        for name in ("fake_engine", "fake_engine_isolated", "libselftest.so", "selftest"):
            self.assertIn("//bazel/toolchains/android:api29", calls[name]["target_compatible_with"])

    def test_scripts_are_executable_in_git(self):
        # Bazel runs both as action executables, so a checkout must keep the bit.
        for relative in ("bazel/zzz/isolate_engine.sh", "bazel/zzz/selftest/check.sh"):
            mode = subprocess.run(["git", "ls-files", "-s", "--", relative], cwd=ROOT, capture_output=True,
                                  check=True).stdout.decode().split(" ", 1)[0]
            self.assertEqual("100755", mode, relative)
            self.assertTrue(os.access(ROOT / relative, os.X_OK), relative)

    def test_bazel_yml_builds_the_selftest_without_secrets(self):
        text = (ROOT / ".github/workflows/bazel.yml").read_text()
        steps = text.split("\n      - name: ")
        step = next(block for block in steps if "//bazel/zzz/selftest" in block)
        self.assertIn("tools/scripts/tests/test_zzz_isolation.py", step)
        # The AAR's arm64 platform (API 29, native TLS), built on the runner:
        # fork PRs have no RBE secret, and the step has no condition on one.
        self.assertIn("\n          bazelisk build --platforms=//bazel/ggml:android_arm64_cpu_variants "
                      "//bazel/zzz/selftest\n", step)
        self.assertNotIn("--config=remote", step)
        self.assertNotIn("        if:", step)
        # (b) proves the arm64 slice reaches the engine through the isolation.
        self.assertIn('"//bazel/zzz:android_engine"', step)


def library(static=None, pic=None, dynamic=None, interface=None, alwayslink=False):
    return types.SimpleNamespace(static_library=static, pic_static_library=pic, dynamic_library=dynamic,
                                 interface_library=interface, alwayslink=alwayslink)


def linker_input(libraries=(), flags=(), extra=()):
    return types.SimpleNamespace(libraries=list(libraries), user_link_flags=list(flags),
                                 additional_inputs=list(extra))


class EngineLibraryTests(unittest.TestCase):
    pick = staticmethod(isolate_bzl()["zzz_engine_library"])

    def test_cc_import_archive_and_its_link_flags(self):
        # repositories.bzl: cc_import(:archive) under cc_library(:zzz, linkopts).
        engine = self.pick([linker_input(flags=["-lc", "-lm", "-ldl"]),
                            linker_input([library(static="libzzz_embed.a")], extra=["script.ld"])])
        self.assertEqual(("libzzz_embed.a", False, False), (engine.archive, engine.pic, engine.alwayslink))
        self.assertEqual(["-lc", "-lm", "-ldl"], engine.user_link_flags)
        self.assertEqual(["script.ld"], engine.additional_inputs)

    def test_the_pic_archive_wins_and_alwayslink_is_kept(self):
        engine = self.pick([linker_input([library(static="a.a", pic="a.pic.a", alwayslink=True)])])
        self.assertEqual(("a.pic.a", True, True), (engine.archive, engine.pic, engine.alwayslink))

    def test_anything_but_one_static_archive_fails_analysis(self):
        cases = {
            "found 0": [linker_input(flags=["-lc"])],
            "found 2": [linker_input([library(static="a.a")]), linker_input([library(static="b.a")])],
            "not a shared library": [linker_input([library(static="a.a", dynamic="a.so")])],
            "no static archive": [linker_input([library(interface=None)])],
        }
        for message, inputs in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(RuleFailure, message):
                self.pick(inputs)


FAKE_TOOL = r"""
import json, sys

def load(path):
    with open(path) as stream:
        return json.load(stream)

def save(path, obj):
    with open(path, "w") as stream:
        json.dump(obj, stream)

# llvm-nm's letters: U undefined, w weak undefined, upper case a global
# definition (W weak), lower case a local one. "u!" stands for an undefined
# symbol made local, which -u lists and -g does not.
UNDEFINED = ("U", "w", "u!")
# What llvm-nm prints once a definition is local: the section's lowercase
# letter (a weak definition in .text becomes "t", not "w").
LOCAL = {"T": "t", "W": "t", "D": "d", "B": "b", "R": "r", "V": "d"}

def is_global(symbol):
    return symbol["type"] in ("U", "w") or symbol["type"].isupper()

tool, args = sys.argv[1], sys.argv[2:]
if tool == "ld":
    assert args[:4] == ["-r", "--unique", "--fatal-warnings", "--whole-archive"], args
    assert args[5:7] == ["--no-whole-archive", "-o"] and len(args) == 8, args
    archive = load(args[4])
    if "ld_error" in archive:
        sys.stderr.write(archive["ld_error"] + "\n")
        sys.exit(1)
    if "ld_warning" in archive:
        # What -r prints where a final link fails (lld: "relocation refers to
        # a discarded section"); --fatal-warnings makes it an error.
        sys.stderr.write(archive["ld_warning"] + "\n")
        if "--fatal-warnings" in args:
            sys.exit(1)
    # -r resolves one member's reference to another member's global definition.
    defined = {s["name"] for s in archive["symbols"] if s["type"] not in UNDEFINED and is_global(s)}
    symbols = []
    for symbol in archive["symbols"]:
        if symbol["type"] in UNDEFINED and symbol["name"] in defined:
            continue
        if symbol["name"] in archive.get("ld_drops", []):
            continue
        if symbol["name"] in archive.get("ld_undefines", []):
            symbol = {"name": symbol["name"], "type": "U"}
        symbols.append(symbol)
    symbols += [{"name": name, "type": "U"} for name in archive.get("ld_adds", [])]
    save(args[7], dict(archive, symbols=symbols))
elif tool == "objcopy":
    options = [arg for arg in args if arg.startswith("--")]
    source, target = [arg for arg in args if not arg.startswith("--")]
    keep_file = next(arg for arg in options if arg.startswith("--keep-global-symbols=")).split("=", 1)[1]
    keep = set(open(keep_file).read().split())
    obj = load(source)
    removed = {arg.split("=", 1)[1] for arg in options if arg.startswith("--remove-section=")}
    obj["sections"] = [name for name in obj.get("sections", []) if name not in removed]
    for symbol in obj["symbols"]:
        stays = symbol["name"] in keep or symbol["name"] in obj.get("objcopy_misses", [])
        if symbol["type"] in LOCAL and not stays:
            symbol["type"] = LOCAL[symbol["type"]]
        if symbol["name"] in obj.get("objcopy_localizes", []):
            symbol["type"] = "u!"
    obj["symbols"] = [s for s in obj["symbols"] if s["name"] not in obj.get("objcopy_drops", [])]
    save(target, obj)
elif tool == "nm":
    path, flags = args[-1], args[:-1]
    # -A: "file: name", or "archive:member: name" (the fake archive is one member).
    prefix = (path + (":engine.o: " if path.endswith(".a") else ": ")) if "-A" in flags else ""
    for symbol in load(path)["symbols"]:
        undefined = symbol["type"] in UNDEFINED
        if "-u" in flags and not undefined:
            continue
        if "--defined-only" in flags and undefined:
            continue
        if "-g" in flags and not is_global(symbol):
            continue
        print(prefix + (symbol["name"] if "-j" in flags else "%s %s 0 0" % (symbol["name"], symbol["type"])))
elif tool == "ar":
    assert args[0] == "rcsD", args
    save(args[1], load(args[2]))
"""


def engine_symbols():
    """A fake archive: the API, its own runtime and data, imports, and one
    member's reference to another member's definition (SECRET_HELPER)."""
    symbols = [{"name": name, "type": "T"} for name in API]
    symbols += [{"name": "memcpy", "type": "W"}, {"name": "expf", "type": "T"},
                {"name": SECRET, "type": "B"}, {"name": "secret_local", "type": "t"},
                {"name": SECRET_HELPER, "type": "U"}, {"name": SECRET_HELPER, "type": "T"},
                {"name": "write", "type": "U"}, {"name": "pthread_create", "type": "w"}]
    return symbols


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class IsolateScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.scratch = self.tmp / "scratch"
        self.scratch.mkdir()
        (self.tmp / "fake_tool.py").write_text(FAKE_TOOL)
        self.tools = []
        for tool in ("ld", "objcopy", "nm", "ar"):
            path = self.tmp / tool
            path.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{self.tmp / "fake_tool.py"}" {tool} "$@"\n')
            path.chmod(0o755)
            self.tools.append(str(path))
        self.api = self.tmp / "api.txt"
        self.api.write_text("".join(name + "\n" for name in reversed(API)))
        self.output = self.tmp / "out" / "libzzz_embed_isolated.a"
        self.output.parent.mkdir()

    def isolate(self, symbols, tools=None, **extra):
        archive = self.tmp / "libzzz_embed.a"
        extra.setdefault("sections", [".text", ".tbss", ".init_array", ".eh_frame", ".llvm_addrsig"])
        archive.write_text(json.dumps(dict(symbols=symbols, **extra)))
        result = subprocess.run(["bash", str(SCRIPT), *(tools or self.tools), str(archive), str(self.api),
                                 str(self.output)], capture_output=True, text=True,
                                env=dict(os.environ, TMPDIR=str(self.scratch)))
        self.assertEqual([], list(self.scratch.iterdir()), "the scratch directory is left behind")
        self.assertNotIn(SECRET, result.stdout + result.stderr)
        return result

    def assertIsolationFails(self, result, message):
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertIn("zzz engine isolation: " + message, result.stderr)
        self.assertFalse(self.output.exists())

    def test_only_the_api_stays_global(self):
        result = self.isolate(engine_symbols())
        self.assertEqual(0, result.returncode, result.stderr)
        # The member-to-member reference is no import: write and pthread_create are.
        self.assertEqual("zzz engine isolation: 8 API symbol(s) kept global, 4 other global or weak "
                         "definition(s) made local, 2 undefined import(s) unchanged\n", result.stdout)
        types_by_name = {s["name"]: s["type"] for s in json.loads(self.output.read_text())["symbols"]}
        self.assertEqual({name: "T" for name in API}, {n: t for n, t in types_by_name.items() if t in "TWDB"})
        self.assertEqual(("t", "t", "b", "t", "U", "w"), tuple(types_by_name[name] for name in (
            "memcpy", "expf", SECRET, SECRET_HELPER, "write", "pthread_create")))
        # ld -r leaves the address-significance table stale: it goes, the rest stays.
        self.assertEqual([".text", ".tbss", ".init_array", ".eh_frame"],
                         json.loads(self.output.read_text())["sections"])

    def test_a_tool_failure_withholds_its_diagnostics(self):
        result = self.isolate(engine_symbols(), ld_error=f"ld.lld: error: duplicate symbol: {SECRET}")
        self.assertIsolationFails(result, "ld failed; 1 line(s) of its diagnostics withheld")

    def test_a_prelink_warning_fails_too(self):
        # -r writes R_*_NONE where a final link refuses a relocation into a
        # discarded section, and only warns: the warning must fail the action.
        result = self.isolate(engine_symbols(), ld_warning=(
            f"ld.lld: warning: relocation refers to a discarded section: .text.{SECRET}"))
        self.assertIsolationFails(result, "ld failed; 1 line(s) of its diagnostics withheld")

    def test_common_symbols_are_refused(self):
        result = self.isolate(engine_symbols() + [{"name": SECRET + "_common", "type": "C"}])
        self.assertIsolationFails(result, "1 common symbol(s), which a relocatable link cannot allocate")

    def test_a_missing_api_symbol_is_named(self):
        result = self.isolate([s for s in engine_symbols() if s["name"] != "zzz_embed_close"])
        self.assertIsolationFails(result, "the engine does not define its API: zzz_embed_close")

    def test_the_prelink_may_import_only_what_the_archive_imports(self):
        # A definition of the engine's own left undefined would bind outside it.
        self.assertIsolationFails(self.isolate(engine_symbols(), ld_undefines=[SECRET]),
                                  "1 symbol(s) left undefined by ld -r that the archive does not import")
        self.assertIsolationFails(self.isolate(engine_symbols(), ld_adds=[SECRET + "_new"]),
                                  "1 symbol(s) left undefined by ld -r that the archive does not import")
        # An import that went would be harmless: the check is a subset one.
        result = self.isolate(engine_symbols(), ld_drops=["pthread_create"])
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue(result.stdout.endswith(", 1 undefined import(s) unchanged\n"), result.stdout)

    def test_a_global_left_behind_is_counted_not_named(self):
        result = self.isolate(engine_symbols(), objcopy_misses=[SECRET])
        self.assertIsolationFails(result, "9 global or weak definition(s) left; expected exactly the 8 API symbols")

    def test_changed_imports_or_definitions_fail(self):
        self.assertIsolationFails(self.isolate(engine_symbols(), objcopy_drops=["write"]),
                                  "the undefined imports changed (2 before, 1 after)")
        # An undefined symbol made local is no import any linker resolves.
        self.assertIsolationFails(self.isolate(engine_symbols(), objcopy_localizes=["write"]),
                                  "the undefined imports changed (2 before, 1 after)")
        self.assertIsolationFails(self.isolate(engine_symbols(), objcopy_drops=[SECRET]),
                                  "the set of defined symbols changed (13 before, 12 after)")

    def test_a_toolchain_without_the_tools_fails(self):
        result = self.isolate(engine_symbols(), tools=["", *self.tools[1:]])
        self.assertIsolationFails(result, "the C++ toolchain must declare ld, objcopy, nm and ar; got ''")


if __name__ == "__main__":
    unittest.main()
