"""Offline contracts for unadopted static platforms; no Bazel fetch or engine build.

The repository helper uses a Python-compatible Starlark subset. Execute that
actual source with a bounded fake repository context to exercise lazy failures.
This does not replace a real Bazel analysis after producer pins are available.
"""

import json
import re
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
NEW_TARGETS = {
    "aarch64-apple-ios": "zzz_ios",
    "aarch64-apple-ios-sim": "zzz_ios_sim",
    "aarch64-unknown-linux-gnu": "zzz_linux_arm64",
}
OLD_TARGETS = (
    "aarch64-apple-darwin", "aarch64-linux-android", "x86_64-unknown-linux-gnu"
)


class RuleFailure(Exception):
    pass


def fail(message):
    raise RuleFailure(message)


class FakePath:
    def __init__(self, name):
        self.name = name
        self.exists = False

    def get_child(self, name):
        return FakePath(f"{self.name}/{name}")

    def __str__(self):
        return self.name


def repository_source():
    namespace = {
        "json": types.SimpleNamespace(decode=json.loads),
        "fail": fail,
        "attr": types.SimpleNamespace(string=lambda **kw: None, label=lambda **kw: None),
        "repository_rule": lambda **kw: kw["implementation"],
    }
    path = ROOT / "bazel/zzz/repositories.bzl"
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


def context(target, entries, *, staged=True, verify=True):
    return types.SimpleNamespace(
        attr=types.SimpleNamespace(target=target, manifest="manifest", verifier="verifier"),
        os=types.SimpleNamespace(environ={"XYBRID_ZZZ_PREBUILT_DIR": "/staged"} if staged else {}),
        read=mock.Mock(return_value=json.dumps({"archives": entries})),
        path=FakePath,
        file=mock.Mock(), symlink=mock.Mock(),
        execute=mock.Mock(return_value=types.SimpleNamespace(
            return_code=0 if verify else 1, stdout="/verified\n", stderr="invalid pin")),
    )


class ZzzPlatformTests(unittest.TestCase):
    def test_absent_new_targets_load_without_staging_then_fail_when_selected(self):
        implementation = repository_source()["zzz_prebuilt_repository"]
        for target in NEW_TARGETS:
            with self.subTest(target=target):
                ctx = context(target, [{"target": OLD_TARGETS[0], "profile": "phonon-kitten-tts2"}], staged=False)
                implementation(ctx)
                ctx.execute.assert_not_called()
                ctx.symlink.assert_not_called()
                files = {call.args[0]: call.args[1] for call in ctx.file.call_args_list}
                self.assertIn('unavailable(name = "zzz"', files["BUILD.bazel"])
                self.assertIn('unavailable(name = "notices"', files["BUILD.bazel"])
                namespace = {"fail": fail, "rule": lambda **kw: kw["implementation"]}
                exec(files["unavailable.bzl"], namespace)
                with self.assertRaisesRegex(RuleFailure, "no pinned zzz engine slice.*schema-5 pin"):
                    namespace["unavailable"](None)

    def test_attempted_pins_never_downgrade_to_placeholder(self):
        implementation = repository_source()["zzz_prebuilt_repository"]
        for target in (*OLD_TARGETS, *NEW_TARGETS):
            for entries in (
                [{"target": target, "profile": "phonon-kitten-tts2"}],
                [{"target": target, "profile": "phonon-kitten-tts2"}] * 2,
                [{"target": target, "profile": "wrong-profile"}],
                [{"target": target}],
            ):
                with self.subTest(target=target, entries=entries):
                    ctx = context(target, entries, verify=False)
                    with self.assertRaisesRegex(RuleFailure, "verification failed: invalid pin"):
                        implementation(ctx)
                    self.assertEqual(1, ctx.execute.call_count)
                    self.assertEqual(["python3", "verifier", "--target", target, "--verify-only"], ctx.execute.call_args.args[0])
                    ctx.file.assert_not_called()
                    ctx.symlink.assert_not_called()

    def test_existing_targets_keep_strict_failure_with_absent_rows(self):
        implementation = repository_source()["zzz_prebuilt_repository"]
        for target in OLD_TARGETS:
            ctx = context(target, [], staged=False)
            with self.subTest(target=target), self.assertRaisesRegex(RuleFailure, "Stage the pinned slice"):
                implementation(ctx)
            ctx.file.assert_not_called()
            ctx = context(target, [], verify=False)
            with self.assertRaisesRegex(RuleFailure, "verification failed"):
                implementation(ctx)
            ctx.execute.assert_called_once()

    def test_verified_new_targets_emit_only_the_contract_system_libraries(self):
        implementation = repository_source()["zzz_prebuilt_repository"]
        for target in NEW_TARGETS:
            ctx = context(target, [{"target": target, "profile": "phonon-kitten-tts2"}])
            implementation(ctx)
            build = ctx.file.call_args.args[1]
            captured = []
            exec(build, {"load": lambda *args: None, "cc_import": lambda **kw: None,
                         "cc_library": lambda **kw: captured.append(kw), "unavailable": lambda **kw: None})
            expected = ["-lc", "-lm", "-lpthread", "-ldl"] if "linux" in target else ["-lSystem"]
            self.assertEqual(expected, captured[0]["linkopts"])
            self.assertEqual(2, ctx.symlink.call_count)

    def test_schema5_exports_only_verified_notice_files_with_canonical_paths(self):
        # Execute the actual repository helper after its verifier success. The
        # verifier's immutable-map checks have their own archive tests.
        import ast
        source = ast.parse((ROOT / "crates/zzz-sys/zzz_pull.py").read_text())
        names = next(ast.literal_eval(node.value) for node in source.body
                     if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                     t.id == "SCHEMA5_NOTICE_FILES" for t in node.targets))
        for target in (*NEW_TARGETS, "aarch64-linux-android"):
            ctx = context(target, [{"target": target, "receipt_schema": 5, "notice_files": names}])
            repository_source()["zzz_prebuilt_repository"](ctx)
            captured = []
            exec(ctx.file.call_args.args[1], {"load": lambda *args: None,
                 "cc_import": lambda **kw: None, "cc_library": lambda **kw: None,
                 "filegroup": lambda **kw: captured.append(kw)})
            self.assertEqual([{"name": "notices", "srcs": ["notices/" + name for name in sorted(names)],
                               "visibility": ["//visibility:public"]}], captured)
            self.assertEqual({(f"/verified/{name}", "notices/" + name) for name in names},
                             {(str(call.args[0]), call.args[1]) for call in ctx.symlink.call_args_list[2:]})
            self.assertEqual(16, ctx.symlink.call_count)

    def test_schema4_notice_export_fails_analysis_without_changing_native_inputs(self):
        ctx = context(OLD_TARGETS[0], [{"target": OLD_TARGETS[0], "receipt_schema": 4}])
        repository_source()["zzz_prebuilt_repository"](ctx)
        files = {call.args[0]: call.args[1] for call in ctx.file.call_args_list}
        self.assertIn('unavailable(name = "notices"', files["BUILD.bazel"])
        namespace = {"fail": fail, "rule": lambda **kw: kw["implementation"]}
        exec(files["notices_unavailable.bzl"], namespace)
        with self.assertRaisesRegex(RuleFailure, "require a qualified schema-5 pin"):
            namespace["unavailable"](None)
        self.assertEqual(2, ctx.symlink.call_count)

    def test_target_maps_and_apple_environment_selects_agree(self):
        settings = {}
        groups = []
        namespace = {
            "load": lambda *args: None,
            "config_setting": lambda **kw: settings.update({kw["name"]: kw}),
            "selects": types.SimpleNamespace(config_setting_group=lambda **kw: groups.append(kw)),
            # The Android engine's isolation (test_zzz_isolation.py covers it).
            "zzz_isolated_engine": lambda **kw: None, "ZZZ_EMBED_API": [],
        }
        path = ROOT / "bazel/zzz/BUILD.bazel"
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        platforms = {
            "ios_enabled": {"@platforms//os:ios", "@platforms//cpu:aarch64", "@build_bazel_apple_support//constraints:device"},
            "ios_sim_enabled": {"@platforms//os:ios", "@platforms//cpu:aarch64", "@build_bazel_apple_support//constraints:simulator"},
            "linux_arm64_enabled": {"@platforms//os:linux", "@platforms//cpu:aarch64", "@rules_rs//rs/platforms/constraints:glibc"},
        }
        for expected, constraints in platforms.items():
            matches = [name for name, setting in settings.items()
                       if set(setting["constraint_values"]) <= constraints]
            self.assertEqual([expected], matches)
            self.assertEqual({"//:tts_zzz": "true"}, settings[expected]["flag_values"])
            self.assertIn(":" + expected, groups[0]["match_any"])
        for constraints in (
            {"@platforms//os:ios", "@platforms//cpu:x86_64", "@build_bazel_apple_support//constraints:simulator"},
            {"@platforms//os:linux", "@platforms//cpu:aarch64", "@rules_rs//rs/platforms/constraints:musl"},
        ):
            self.assertFalse(any(set(setting["constraint_values"]) <= constraints for setting in settings.values()))
        module = (ROOT / "MODULE.bazel").read_text()
        build_rs = (ROOT / "crates/zzz-sys/build.rs").read_text()
        crate_build = (ROOT / "crates/zzz-sys/BUILD.bazel").read_text()
        for target, repository in NEW_TARGETS.items():
            self.assertIn(f'zzz_prebuilt_repository(name = "{repository}", target = "{target}")', module)
            self.assertIn(f'"{target}"', build_rs.split("fn main()", 1)[0])
            self.assertIn(f'"@{repository}//:zzz"', crate_build)
        triples = re.search(r"platform_triples = \[(.*?)\]", module, re.S).group(1)
        self.assertIn('"aarch64-unknown-linux-gnu"', triples)
