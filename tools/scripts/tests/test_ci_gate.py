"""Regression tests for docs-only selection and the required CI gate."""

import importlib.util
import io
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci_gate import OPTIONAL_JOBS, PATH_FILTERS, check_jobs, classify


def pr_files(*paths: str) -> list:
    return [[{"filename": path} for path in paths]]


class ChangeSelectionTests(unittest.TestCase):
    def test_docs_only_changes_skip_the_matrix(self):
        outputs = classify("pull_request", pr_files(
            "README.md", "docs/guide.json", "examples/hello.rs",
            ".github/ISSUE_TEMPLATE/bug.yml", "LICENSE", "tools/scripts/README.md",
        ))
        self.assertTrue(all(value == "false" for value in outputs.values()))

    def test_mixed_code_and_changelog_requires_full_ci(self):
        outputs = classify("pull_request", pr_files("CHANGELOG.md", "vendor/llama-cpp"))
        self.assertEqual(outputs["full_ci"], "true")

    def test_pagination_includes_later_code_changes(self):
        outputs = classify("pull_request", pr_files("README.md") + pr_files("Cargo.toml"))
        self.assertEqual(outputs["full_ci"], "true")

    def test_renaming_code_to_docs_requires_full_ci(self):
        outputs = classify("pull_request", [[{
            "filename": "docs/old-script.md", "previous_filename": "tools/scripts/old.py",
        }]])
        self.assertEqual(outputs["full_ci"], "true")
        self.assertEqual(outputs["relevant"], "true")

    def test_push_uses_the_same_docs_rule(self):
        self.assertEqual(classify("push", {"files": pr_files("README.md")[0]})["full_ci"], "false")
        self.assertEqual(classify("push", {"files": pr_files("Cargo.lock")[0]})["full_ci"], "true")

    def test_zzz_contract_and_gated_workflows_run_tooling_tests(self):
        for path in (
            "crates/zzz-sys/zzz_pull.py", "crates/zzz-sys/natives-manifest.json",
            ".github/workflows/release-prep.yml", ".github/workflows/release-publish.yml",
            ".github/workflows/test-zzz-android.yml", ".github/workflows/test-zzz-engine.yml",
            ".github/workflows/trusted-zzz-sdk.yml",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["relevant"], "true")
        for path in (
            "crates/zzz-sys/src/lib.rs", "crates/zzz-sys/zzz_pull.py.orig",
            ".github/workflows/build-android.yml", ".github/workflows/release-prep.yml.bak",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["relevant"], "false")

    def test_js_release_surfaces_run_tooling_and_full_ci(self):
        for path in (
            "bindings/web/scripts/check-pack.mjs",
            "bindings/web/scripts/check-pack.test.mjs",
            "bindings/web/package.json", "bindings/web/pnpm-lock.yaml",
            "bindings/web/pnpm-workspace.yaml",
            "bindings/observability-js/scripts/build.ts",
            "bindings/observability-js/package.json",
            "bindings/observability-js/pnpm-lock.yaml",
            "bindings/observability-js/pnpm-workspace.yaml",
            "bindings/observability-js/src/version.ts",
        ):
            with self.subTest(path=path):
                outputs = classify("pull_request", pr_files(path))
                self.assertEqual(outputs["full_ci"], "true")
                self.assertEqual(outputs["relevant"], "true")
        for path in (
            "bindings/web/README.md", "bindings/observability-js/RELEASING.md",
            "bindings/web/package.json.bak", "bindings/web/src/llm.ts",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["relevant"], "false")

    def test_flutter_platform_floors_run_tooling_tests(self):
        for path in (
            "bindings/flutter/ios/xybrid_flutter.podspec",
            "bindings/flutter/macos/xybrid_flutter.podspec",
            "bindings/flutter/android/build.gradle", ".bazelrc",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["relevant"], "true")
        for path in (
            "bindings/flutter/example/android/app/build.gradle.kts",
            "bindings/flutter/ios/Classes/dummy_file.c", "bindings/.bazelrc",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["relevant"], "false")

    def test_every_file_the_generation_gate_tests_read_runs_tooling_tests(self):
        # test_generation_gates.py asserts on build-flutter.yml, the FRB
        # wrapper, ci.yml and every generator it globs. Editing any of them
        # (say, deleting the FRB drift step) must run that test, or the PR
        # skips it and still passes CI Success. Record what the suite really
        # reads instead of restating it here, so the two cannot drift.
        tests_dir = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "generation_gates_under_ci_gate", tests_dir / "test_generation_gates.py"
        )
        gates = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gates)
        repo = gates.REPO_ROOT.resolve()
        read, globbed = set(), []
        read_text, glob = Path.read_text, Path.glob

        def recording_read(path, *args, **kwargs):
            read.add(path.resolve())
            return read_text(path, *args, **kwargs)

        def recording_glob(path, pattern, *args, **kwargs):
            globbed.append(pattern)
            return glob(path, pattern, *args, **kwargs)

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(gates.GenerationGateTests)
        with mock.patch.object(Path, "read_text", recording_read), \
                mock.patch.object(Path, "glob", recording_glob):
            result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)

        inputs = sorted(
            path.relative_to(repo).as_posix() for path in read if path.is_relative_to(repo)
        )
        for expected in (".github/workflows/build-flutter.yml", "bindings/flutter/generate-frb.sh"):
            self.assertIn(expected, inputs, "the generation gate tests no longer read this file")
        # A listed generator (deleting one fails the suite) and any file its
        # globs would discover (a new generator without a gate) must select
        # the tooling tests too.
        self.assertTrue(globbed, "the generation gate tests no longer glob for generators")
        new_generators = [pattern.replace("*", "new") for pattern in globbed]
        for path in inputs + sorted(gates.GATES) + new_generators:
            with self.subTest(path=path):
                outputs = classify("pull_request", pr_files(path))
                self.assertEqual(outputs["relevant"], "true")
                self.assertEqual(OPTIONAL_JOBS["tooling-tests"], "relevant")
        for path in (
            ".github/workflows/build-flutter.yml.bak", "bindings/flutter/generate-frb.sh.orig",
            "bindings/flutter/lib/src/pipeline.dart", "bindings/apple/scripts/set-natives-mode.sh",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["relevant"], "false")

    def test_bolt_and_facade_sources_run_every_generated_binding_gate(self):
        # Each committed binding is generated from the bolt crate (exports and
        # doc comments) over facade types, so a source edit must run its drift
        # check, and the Unity Api layer must recompile on top of it.
        for path in (
            "crates/xybrid-bolt/src/lib.rs", "crates/xybrid-bolt/Cargo.toml",
            "crates/xybrid-ffi-facade/src/lib.rs",
            "crates/xybrid-ffi-facade/src/observability_operations.rs",
        ):
            outputs = classify("pull_request", pr_files(path))
            for flag in ("unity_bolt", "python_sdk", "kotlin_jni", "apple_bolt"):
                with self.subTest(path=path, flag=flag):
                    self.assertEqual(outputs[flag], "true")
        for path in ("crates/xybrid-sdk/src/lib.rs", "bindings/unity/Tests/Editor/ObservationTests.cs"):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["unity_bolt"], "false")

    def test_observability_scenario_inputs_run_their_gates(self):
        # test_observability_capture.py requires every lane's scenario source to
        # configure the omit rule and record the content it drops, so editing a
        # lane must run the tooling tests. The Python suite runs the scenario
        # end to end over the harness, the SDK and the observability crate.
        tests_dir = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "observability_capture_under_ci_gate", tests_dir / "test_observability_capture.py"
        )
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        for path, _, _ in capture.LANE_SOURCES.values():
            with self.subTest(path=path):
                self.assertTrue((capture.REPO / path).is_file(), path)
                self.assertEqual(classify("pull_request", pr_files(path))["relevant"], "true")
        for path in (
            "tools/scripts/observability_capture/check_capture.py",
            "crates/xybrid-observability/src/exporter.rs",
            "crates/xybrid-sdk/src/observability.rs",
            "bindings/python/tests/observability_scenario.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["python_sdk"], "true")
        for path in (
            "bindings/flutter/test/observability_test.dart",
            "bindings/apple/Tests/XybridTests/ObservationTests.swift",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["relevant"], "false")

    def test_unity_drift_inputs_select_both_unity_jobs(self):
        for path in (
            "bindings/unity/Runtime/Bolt/XybridError.cs",
            "bindings/unity/Runtime/Api/Observation.cs",
            "tools/scripts/gen_unity_bolt_csharp.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["unity_bolt"], "true")

    def test_workflow_changes_run_every_optional_gate(self):
        outputs = classify("pull_request", pr_files(".github/workflows/ci.yml"))
        self.assertTrue(all(value == "true" for value in outputs.values()))

    def test_empty_or_capped_results_run_every_gate(self):
        for event, payload in (
            ("pull_request", []), ("push", {"files": []}),
            ("pull_request", pr_files(*(["README.md"] * 3000))),
            ("push", {"files": pr_files(*(["README.md"] * 300))[0]}),
        ):
            with self.subTest(event=event, count=len(str(payload))):
                self.assertTrue(all(value == "true" for value in classify(event, payload).values()))

    def test_invalid_api_response_fails(self):
        for payload in ([{}], [[{}]], [[{"filename": ""}]], {"message": "API failed"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                classify("pull_request", payload)


class RequiredGateTests(unittest.TestCase):
    def needs(self, *paths):
        return {
            "tooling-changes": {"result": "success", "outputs": classify("pull_request", pr_files(*paths))},
            "fmt": {"result": "success"},
            "clippy": {"result": "success"},
            "tooling-tests": {"result": "skipped"},
        }

    def test_docs_only_accepts_skipped_matrix(self):
        needs = self.needs("README.md")
        needs["clippy"]["result"] = "skipped"
        check_jobs(needs)

    def test_code_accepts_irrelevant_tooling_skips(self):
        check_jobs(self.needs("crates/xybrid-core/src/lib.rs"))

    def test_mixed_change_cannot_mask_a_failed_required_check(self):
        for result in ("failure", "cancelled", "skipped"):
            needs = self.needs("CHANGELOG.md", "vendor/llama-cpp")
            needs["clippy"]["result"] = result
            with self.subTest(result=result), self.assertRaises(ValueError):
                check_jobs(needs)

    def test_docs_only_runs_and_cannot_mask_the_always_on_jobs(self):
        # Docs-only PRs skip every gated job, and they are where the docs'
        # install pins get edited, so Version sync (like Workflow index) has no
        # path gate: it must stay ungated, required, and fail the gate.
        workflow = (Path(__file__).resolve().parents[3] / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        required = re.search(r"^  ci-success:\n(?:    .*\n)*?    needs: \[([^\]]*)\]", workflow, re.M)[1]
        for job in ("workflow-index", "version-sync"):
            block = re.search(rf"^  {job}:\n((?:    .*\n|\n)*)", workflow, re.M)[1]
            with self.subTest(job=job):
                self.assertNotRegex(block, r"(?m)^    (if|needs):", "an always-on job has no path gate")
                self.assertIn(job, [name.strip() for name in required.split(",")])
                for result in ("failure", "cancelled"):
                    needs = self.needs("README.md", "docs/content/docs/quickstart.mdx")
                    needs[job] = {"result": result}
                    with self.assertRaises(ValueError):
                        check_jobs(needs)

    def test_unity_drift_is_optional_only_when_unity_inputs_are_untouched(self):
        needs = self.needs("crates/xybrid-core/src/lib.rs")
        needs["unity-bolt-drift"] = {"result": "skipped"}
        check_jobs(needs)
        for path in ("crates/xybrid-bolt/src/lib.rs", "bindings/unity/Runtime/Bolt/XybridError.cs"):
            needs = self.needs(path)
            needs["unity-bolt-drift"] = {"result": "skipped"}
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_jobs(needs)

    def test_every_optional_job_is_required_by_ci_success_and_gated_by_its_flag(self):
        workflow = (Path(__file__).resolve().parents[3] / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        required = re.search(r"^  ci-success:\n(?:    .*\n)*?    needs: \[([^\]]*)\]", workflow, re.M)[1]
        required = [name.strip() for name in required.split(",")]
        outputs = re.search(r"^  tooling-changes:\n((?:    .*\n|\n)*)", workflow, re.M)[1]
        for job, flag in OPTIONAL_JOBS.items():
            with self.subTest(job=job):
                self.assertIn(job, required)
                block = re.search(rf"^  {job}:\n((?:    .*\n|\n)*)", workflow, re.M)[1]
                self.assertIn(f"if: needs.tooling-changes.outputs.{flag} == 'true'", block)
                self.assertIn(f"{flag}: ${{{{ steps.filter.outputs.{flag} }}}}", outputs)

    def test_relevant_tooling_cannot_be_skipped(self):
        with self.assertRaises(ValueError):
            check_jobs(self.needs("tools/scripts/ci_gate.py"))

    def test_missing_or_failed_detection_cannot_pass(self):
        for detector in ({}, {"result": "failure"}, {"result": "skipped"}, {"result": "success"}):
            with self.subTest(detector=detector), self.assertRaises(ValueError):
                check_jobs({"tooling-changes": detector})


if __name__ == "__main__":
    unittest.main()
