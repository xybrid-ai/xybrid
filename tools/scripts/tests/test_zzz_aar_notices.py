"""Offline checks of the zzz notices the Kotlin AAR packages; no Bazel or engine bytes.

Runs the actual Starlark of //bazel/zzz:notices.bzl (a Python-compatible
subset) with fake File objects, and the Kotlin BUILD file with a recording
select(). Real Bazel analysis against a staged slice happens in CI.
"""

import ast
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREFIX = "assets/xybrid/zzz"


class RuleFailure(Exception):
    pass


def fail(message):
    raise RuleFailure(message)


def contract_paths():
    source = ast.parse((ROOT / "crates/zzz-sys/zzz_pull.py").read_text())
    notices = next(ast.literal_eval(node.value) for node in source.body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                   t.id == "SCHEMA5_NOTICE_FILES" for t in node.targets))
    return sorted(notices)


def bzl(relative):
    namespace = {
        "load": lambda *a, **kw: None, "fail": fail,
        "rule": lambda **kw: kw["implementation"],
        "attr": types.SimpleNamespace(label=lambda **kw: None, string=lambda **kw: None,
                                      bool=lambda **kw: None),
        "PackageFilesInfo": lambda **kw: ("PackageFilesInfo", kw),
        "DefaultInfo": lambda **kw: ("DefaultInfo", kw), "depset": list,
    }
    path = ROOT / relative
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


def exported(names, repository="external/+zzz_prebuilt_repository+zzz_android"):
    return [types.SimpleNamespace(path=f"{repository}/notices/{name}", is_directory=False) for name in names]


class Select:
    def __init__(self, arms):
        self.arms = arms

    def __add__(self, other):
        return [self, other]

    def __radd__(self, other):
        return list(other) + [self]


def kotlin_targets():
    targets = {}

    def capture(**kwargs):
        targets[kwargs["name"]] = kwargs

    namespace = {"load": lambda *a, **kw: None, "glob": lambda *a, **kw: [], "select": Select,
                 "ANDROID_ARM64_CPU_VARIANTS": [], "ANDROID_ARM64_GGML_LIBS": []}
    for rule in ("cc_import", "cc_binary", "filegroup", "pkg_files", "pkg_zip", "kt_jvm_library",
                 "define_kt_toolchain", "platform_transition_filegroup", "zzz_notice_files"):
        namespace[rule] = capture
    path = ROOT / "bindings/kotlin/BUILD.bazel"
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return targets


class AarNoticeTests(unittest.TestCase):
    def test_one_path_list_matches_the_contract_and_the_apple_rule(self):
        paths = bzl("bazel/zzz/notices.bzl")["ZZZ_NOTICE_PATHS"]
        self.assertEqual(14, len(paths))
        self.assertEqual(contract_paths(), paths)
        # The Apple rule keeps its own private copy (it stays untouched).
        self.assertEqual(paths, bzl("bindings/apple/zzz_notices.bzl")["_NOTICE_PATHS"])

    def test_map_places_exactly_the_canonical_export_under_the_prefix(self):
        namespace = bzl("bazel/zzz/notices.bzl")
        paths = namespace["ZZZ_NOTICE_PATHS"]
        files = exported(reversed(paths))
        for prefix in (PREFIX, PREFIX + "/", "/" + PREFIX):
            mapping = namespace["zzz_notice_map"](files, prefix)
            self.assertEqual({f"{PREFIX}/{name}" for name in paths}, set(mapping))
            for destination, file in mapping.items():
                self.assertTrue(file.path.endswith("/notices/" + destination[len(PREFIX) + 1:]))
        self.assertIn(f"{PREFIX}/LICENSES/Zig-MIT.txt", mapping)
        self.assertIn(f"{PREFIX}/RUNTIME_TERMS.md", mapping)

    def test_map_rejects_any_other_export(self):
        namespace = bzl("bazel/zzz/notices.bzl")
        paths = namespace["ZZZ_NOTICE_PATHS"]
        files = exported(paths)
        directory = types.SimpleNamespace(path=files[0].path, is_directory=True)
        cases = {
            "missing": (files[1:], "lacks LICENSES/Go-BSD-3-Clause.txt"),
            "extra": (files + exported(["LICENSES/Extra.txt"]), "not one of the fourteen"),
            "outside notices": (files[:-1] + [types.SimpleNamespace(
                path="external/r/THIRD_PARTY_NOTICES.md", is_directory=False)], "not one of the fourteen"),
            "duplicate": (files + exported(["RUNTIME_TERMS.md"], "external/r/extra"), "more than once"),
            "directory": ([directory] + files[1:], "not one of the fourteen"),
            "empty": ([], "lacks"),
        }
        for name, (candidate, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(RuleFailure, message):
                namespace["zzz_notice_map"](candidate, PREFIX)

    def test_rule_provides_rules_pkg_files_with_plain_file_modes(self):
        namespace = bzl("bazel/zzz/notices.bzl")
        files = exported(namespace["ZZZ_NOTICE_PATHS"])
        ctx = types.SimpleNamespace(files=types.SimpleNamespace(notices=files),
                                    attr=types.SimpleNamespace(prefix=PREFIX))
        (kind, info), (default_kind, default) = namespace["zzz_notice_files"](ctx)
        self.assertEqual(("PackageFilesInfo", "DefaultInfo"), (kind, default_kind))
        self.assertEqual({"mode": "0644"}, info["attributes"])
        self.assertEqual(14, len(info["dest_src_map"]))
        self.assertEqual(files, default["files"])

    def test_aar_packages_the_android_notices_only_with_the_flag(self):
        targets = kotlin_targets()
        notices = targets["aar_zzz_notices"]
        self.assertEqual("@zzz_android//:notices", notices["notices"])
        self.assertEqual(PREFIX, notices["prefix"])
        # Wildcard analysis must never reach @zzz_android without a staged slice.
        self.assertEqual(["manual"], notices["tags"])
        srcs = targets["xybrid_kotlin_aar"]["srcs"]
        selects = [part for part in srcs if isinstance(part, Select)]
        self.assertEqual(1, len(selects))
        self.assertEqual({"//:tts_zzz_enabled": [":aar_zzz_notices"], "//conditions:default": []}, selects[0].arms)
        self.assertNotIn(":aar_zzz_notices", [part for part in srcs if isinstance(part, str)])
        self.assertEqual("xybrid-kotlin.aar", targets["xybrid_kotlin_aar"]["out"])


if __name__ == "__main__":
    unittest.main()
