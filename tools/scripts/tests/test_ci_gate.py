"""Regression tests for the CI path registry and the required CI Success gate."""

import contextlib
import io
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gen_workflow_index
import llamacpp_update
from ci_gate import (
    ADVISORY_JOBS,
    ALWAYS_JOBS,
    LANE_PATHS,
    OPTIONAL_JOBS,
    PATH_FILTERS,
    PIPELINE_PATHS,
    check_jobs,
    classify,
    dispatch,
    glob_regex,
    path_selected,
)

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO / ".github" / "workflows"


def pr_files(*paths: str) -> list:
    return [[{"filename": path} for path in paths]]


def ci_jobs() -> dict:
    return gen_workflow_index.read_jobs(WORKFLOWS / "ci.yml")


def check_names(workflow: str) -> set:
    """The check-run names a workflow's jobs report; a job calling a lane file
    reports `<caller> / <job>` for each job of the callee."""
    names = set()
    for job in gen_workflow_index.read_jobs(WORKFLOWS / workflow).values():
        if job.uses and job.uses.startswith("./.github/workflows/"):
            callee = job.uses.rsplit("/", 1)[1]
            names |= {f"{job.check_name} / {name}" for name in check_names(callee)}
        else:
            names.add(job.check_name)
    return names


class GlobTests(unittest.TestCase):
    """GitHub's `paths:` filter semantics, which the lane rows are written in."""

    def test_star_stays_in_a_directory_and_double_star_crosses(self):
        cases = {
            ("*.rs", "lib.rs"): True,
            ("*.rs", "src/lib.rs"): False,
            ("crates/*/Cargo.toml", "crates/xybrid-core/Cargo.toml"): True,
            ("crates/*/Cargo.toml", "crates/a/b/Cargo.toml"): False,
            ("crates/**", "crates/xybrid-core/src/lib.rs"): True,
            ("**.md", "README.md"): True,
            ("**.md", "crates/llama-cpp-sys/README.md"): True,
            ("**/BUILD.bazel", "BUILD.bazel"): True,
            ("**/BUILD.bazel", "crates/xybrid-core/BUILD.bazel"): True,
            ("**/BUILD.bazel", "crates/xybrid-core/BUILD.bazel.bak"): False,
            (".cargo/**", ".cargo/config.toml"): True,
            ("Cargo.lock", "crates/x/Cargo.lock"): False,
        }
        for (pattern, path), expected in cases.items():
            with self.subTest(pattern=pattern, path=path):
                self.assertEqual(bool(glob_regex(pattern).fullmatch(path)), expected)

    def test_a_directory_glob_does_not_match_a_submodule_gitlink(self):
        # A submodule bump changes one path, the gitlink itself.
        self.assertFalse(path_selected(["vendor/llama-cpp/**"], "vendor/llama-cpp"))
        self.assertTrue(path_selected(["vendor/llama-cpp"], "vendor/llama-cpp"))
        self.assertFalse(path_selected(["vendor/llama-cpp"], "vendor/llama-cpp-extra"))

    def test_the_last_matching_pattern_decides(self):
        patterns = ["crates/**", "!**.md", "crates/keep/**"]
        self.assertTrue(path_selected(patterns, "crates/a/lib.rs"))
        self.assertFalse(path_selected(patterns, "crates/a/README.md"))
        self.assertTrue(path_selected(patterns, "crates/keep/README.md"))
        self.assertFalse(path_selected(patterns, "docs/a.rs"))
        self.assertFalse(path_selected(["!**.md"], "a.rs"))

    def test_glob_characters_with_easily_misread_meanings_are_refused(self):
        for pattern in ("a?.rs", "a+.rs", "[ab].rs"):
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                glob_regex(pattern)

    def test_every_registry_pattern_compiles(self):
        for lane, patterns in {**LANE_PATHS, "pipeline": PIPELINE_PATHS}.items():
            for pattern in patterns:
                with self.subTest(lane=lane, pattern=pattern):
                    glob_regex(pattern.removeprefix("!"))


class ChangeSelectionTests(unittest.TestCase):
    def test_docs_only_changes_skip_the_matrix(self):
        outputs = classify("pull_request", pr_files(
            "README.md", "docs/guide.json", "examples/hello.rs",
            ".github/ISSUE_TEMPLATE/bug.yml", "LICENSE", "tools/scripts/README.md",
        ))
        self.assertTrue(all(value == "false" for value in outputs.values()), outputs)

    def test_mixed_code_and_changelog_requires_full_ci(self):
        outputs = classify("pull_request", pr_files("CHANGELOG.md", "vendor/llama-cpp"))
        self.assertEqual(outputs["full_ci"], "true")

    def test_pagination_includes_later_code_changes(self):
        outputs = classify("pull_request", pr_files("README.md") + pr_files("Cargo.toml"))
        self.assertEqual(outputs["full_ci"], "true")

    def test_renaming_code_to_docs_requires_full_ci_and_its_lanes(self):
        outputs = classify("pull_request", [[{
            "filename": "docs/old-wrapper.md", "previous_filename": "crates/llama-cpp-sys/wrapper.cpp",
        }]])
        self.assertEqual(outputs["full_ci"], "true")
        self.assertEqual(outputs["llamacpp"], "true")

    def test_push_uses_the_same_docs_rule(self):
        self.assertEqual(classify("push", {"files": pr_files("README.md")[0]})["full_ci"], "false")
        self.assertEqual(classify("push", {"files": pr_files("Cargo.lock")[0]})["full_ci"], "true")

    def test_js_release_surfaces_run_full_ci(self):
        for path in (
            "bindings/web/scripts/check-pack.mjs", "bindings/web/package.json",
            "bindings/observability-js/src/version.ts",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["full_ci"], "true")

    def test_only_the_staging_script_runs_the_onnx_runtime_legs(self):
        self.assertEqual(
            classify("pull_request", pr_files("tools/scripts/stage_unity_desktop_ort.py"))["ort_staging"], "true"
        )
        for path in ("tools/scripts/tests/test_stage_unity_desktop_ort.py", "tools/scripts/gen_python_bolt.py"):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["ort_staging"], "false")

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

    def test_observability_scenario_inputs_run_the_python_suite(self):
        # The Python suite runs the shared scenario end to end over the
        # harness, the SDK and the observability crate.
        for path in (
            "tools/scripts/observability_capture/check_capture.py",
            "crates/xybrid-observability/src/exporter.rs",
            "crates/xybrid-sdk/src/observability.rs",
            "bindings/python/tests/observability_scenario.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["python_sdk"], "true")

    def test_unity_drift_inputs_select_both_unity_jobs(self):
        for path in (
            "bindings/unity/Runtime/Bolt/XybridError.cs",
            "bindings/unity/Runtime/Api/Observation.cs",
            "tools/scripts/gen_unity_bolt_csharp.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify("pull_request", pr_files(path))["unity_bolt"], "true")

    def test_the_pipeline_and_its_registry_run_every_job_and_lane(self):
        for path in (
            ".github/workflows/ci.yml", "tools/scripts/ci_gate.py",
            ".github/actions/setup-rust/action.yml", ".github/actions/new-action/action.yml",
        ):
            with self.subTest(path=path):
                outputs = classify("pull_request", pr_files(path))
                self.assertTrue(all(value == "true" for value in outputs.values()), outputs)
        # A README beside an action is still documentation.
        self.assertTrue(all(
            value == "false" for value in classify("pull_request", pr_files(".github/actions/README.md")).values()
        ))

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


class LaneTests(unittest.TestCase):
    """Each row reproduces the `paths:` filter of the workflow it replaced."""

    def lanes(self, *paths: str) -> set:
        outputs = classify("pull_request", pr_files(*paths))
        return {lane for lane in LANE_PATHS if outputs[lane] == "true"}

    def test_each_lane_runs_on_its_own_inputs(self):
        cases = {
            "crates/xybrid-core/src/runtime_adapter/candle/whisper.rs": {"candle", "policy_routing"},
            "crates/xybrid-sdk/Cargo.toml": {"candle", "policy_routing"},
            "crates/whisper-cpp-sys/build.rs": {"whispercpp"},
            "vendor/whisper-cpp": {"whispercpp"},
            "crates/xybrid-core/src/runtime_adapter/whisper_cpp/mod.rs": {"whispercpp", "policy_routing"},
            "crates/xybrid-cli/src/main.rs": {"policy_routing"},
            ".cargo/config.toml": {"policy_routing"},
            "Cargo.toml": {"policy_routing"},
            "Cargo.lock": {"candle", "whispercpp", "policy_routing"},
            "integration-tests/tests/llamacpp_release.rs": {"candle", "whispercpp", "policy_routing", "llamacpp"},
            "crates/llama-cpp-sys/build.rs": {"whispercpp", "policy_routing", "llamacpp"},
            ".github/llamacpp-version.json": {"llamacpp"},
            "tools/scripts/llamacpp_update.py": {"llamacpp"},
            "crates/xybrid-core/src/runtime_adapter/streaming_postprocess.rs": {"policy_routing", "llamacpp"},
            "crates/xybrid-observability/src/lib.rs": set(),
            "bindings/kotlin/build.gradle.kts": set(),
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(self.lanes(path), expected)

    def test_markdown_skips_the_lanes_that_excluded_it(self):
        self.assertEqual(self.lanes("crates/xybrid-core/src/audio/README.md"), set())
        self.assertEqual(self.lanes("integration-tests/README.md"), set())
        # The llama.cpp lane never excluded Markdown.
        self.assertEqual(self.lanes("crates/llama-cpp-sys/README.md"), {"llamacpp"})

    def test_a_llama_cpp_submodule_bump_runs_every_llama_lane(self):
        # Policy routing listed `vendor/llama-cpp/**`, which no submodule bump matches.
        self.assertEqual(self.lanes("vendor/llama-cpp"), {"whispercpp", "policy_routing", "llamacpp"})

    def test_the_natives_manifest_push_runs_every_check_the_llama_cpp_release_waits_on(self):
        # build-natives.yml's manifest PR is the last merge before a llama.cpp
        # release: release_plan() then requires each polled check to have
        # succeeded on that HEAD, so the push must select every polled lane.
        outputs = classify("push", {"files": [{"filename": "crates/llama-cpp-sys/natives-manifest.txt"}]})
        jobs = ci_jobs()
        polled = [job for job in jobs.values() if job.check_name in llamacpp_update.REQUIRED_CHECKS]
        self.assertTrue(polled)
        for job in polled:
            with self.subTest(job=job.id):
                flag = OPTIONAL_JOBS.get(job.id, "full_ci")
                self.assertEqual(outputs[flag], "true")

    def test_dispatch_runs_every_job_and_the_named_lanes(self):
        everything = dispatch("")
        self.assertTrue(all(value == "true" for value in everything.values()))
        some = dispatch("candle, whispercpp")
        self.assertEqual({lane for lane in LANE_PATHS if some[lane] == "true"}, {"candle", "whispercpp"})
        self.assertTrue(all(some[key] == "true" for key in ("full_ci", *PATH_FILTERS)))
        with self.assertRaises(ValueError):
            dispatch("candel")

    def test_dispatch_on_master_runs_every_lane(self):
        # A skipped lane would put a newer, skipped check run on master's HEAD
        # under the name llamacpp_update.check_problems() requires to succeed.
        # Failing would skip every job instead, which is worse.
        for lanes in ("", "llamacpp", "candel"):
            with self.subTest(lanes=lanes), contextlib.redirect_stderr(io.StringIO()):
                self.assertTrue(all(value == "true" for value in dispatch(lanes, "refs/heads/master").values()))
        self.assertEqual(dispatch("llamacpp", "refs/heads/topic")["whispercpp"], "false")


class PipelineShapeTests(unittest.TestCase):
    """ci.yml and the registry describe the same jobs."""

    def test_every_job_is_always_on_gated_by_full_ci_or_gated_by_its_flag(self):
        for job in ci_jobs().values():
            if job.id == "ci-success":
                continue
            with self.subTest(job=job.id):
                if job.id in ALWAYS_JOBS:
                    self.assertIsNone(job.condition, "an always-on job has no path gate")
                else:
                    flag = OPTIONAL_JOBS.get(job.id, "full_ci")
                    self.assertEqual(job.condition, f"needs.tooling-changes.outputs.{flag} == 'true'")
                    self.assertIn("tooling-changes", job.needs)

    def test_ci_success_needs_exactly_the_non_advisory_jobs(self):
        jobs = ci_jobs()
        required = jobs["ci-success"].needs
        self.assertEqual(len(required), len(set(required)), "a job is listed twice")
        self.assertEqual(set(required), set(jobs) - {"ci-success"} - ADVISORY_JOBS)
        self.assertLessEqual(ADVISORY_JOBS, set(jobs), "an advisory job no longer exists")
        self.assertLessEqual(set(OPTIONAL_JOBS), set(jobs), "a gated job no longer exists")
        self.assertLessEqual(ALWAYS_JOBS, set(required))

    def test_the_detector_outputs_one_line_per_flag(self):
        workflow = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        for flag in ("full_ci", *PATH_FILTERS, *LANE_PATHS):
            with self.subTest(flag=flag):
                self.assertIn(f"      {flag}: ${{{{ steps.filter.outputs.{flag} }}}}\n", workflow)
        self.assertEqual(set(OPTIONAL_JOBS.values()) - set(PATH_FILTERS) - set(LANE_PATHS), set())

    def test_master_push_runs_are_never_cancelled(self):
        # A push selects lanes from its own diff; cancelling it would drop the
        # lanes and the cache saves of the commit before.
        workflow = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        group = re.search(r"^concurrency:\n  group: (.*)$", workflow, re.M)[1]
        self.assertIn("github.event_name == 'push' && github.sha", group)

    def test_repo_checks_run_every_tooling_test_and_the_index(self):
        workflow = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        start = workflow.index("  repo-checks:\n")
        block = workflow[start:workflow.index("\n  clippy:\n", start)]
        self.assertIn("python3 tools/scripts/gen_workflow_index.py --check", block)
        self.assertIn("bash tools/scripts/version-sync.sh --check", block)
        self.assertIn("python3 -m unittest discover -s tools/scripts/tests -p 'test_*.py'", block)

    def test_every_check_the_llama_cpp_release_waits_on_exists(self):
        # A misspelt or renamed job leaves the automation waiting forever.
        names = set()
        for workflow in gen_workflow_index.load(WORKFLOWS):
            if {"push", "pull_request"} <= set(workflow.events):
                names |= check_names(workflow.file)
        self.assertEqual(llamacpp_update.REQUIRED_CHECKS - names, set())


class RequiredGateTests(unittest.TestCase):
    def needs(self, *paths):
        return {
            "tooling-changes": {"result": "success", "outputs": classify("pull_request", pr_files(*paths))},
            "repo-checks": {"result": "success"},
            "fmt": {"result": "success"},
            "clippy": {"result": "success"},
            "tooling-tests": {"result": "skipped"},
        }

    def test_docs_only_accepts_skipped_matrix(self):
        needs = self.needs("README.md")
        needs["clippy"]["result"] = "skipped"
        check_jobs(needs)

    def test_code_accepts_irrelevant_skips(self):
        check_jobs(self.needs("crates/xybrid-core/src/lib.rs"))

    def test_mixed_change_cannot_mask_a_failed_required_check(self):
        for result in ("failure", "cancelled", "skipped"):
            needs = self.needs("CHANGELOG.md", "vendor/llama-cpp")
            needs["clippy"]["result"] = result
            with self.subTest(result=result), self.assertRaises(ValueError):
                check_jobs(needs)

    def test_always_on_jobs_cannot_fail_or_skip_even_on_docs_only_changes(self):
        # Docs-only PRs are where the install pins and agent skills get edited.
        for job in ALWAYS_JOBS - {"tooling-changes"}:
            for result in ("failure", "cancelled", "skipped"):
                needs = self.needs("README.md", "docs/content/docs/quickstart.mdx")
                needs[job] = {"result": result}
                with self.subTest(job=job, result=result), self.assertRaises(ValueError):
                    check_jobs(needs)

    def test_a_gated_job_is_optional_only_when_its_inputs_are_untouched(self):
        needs = self.needs("crates/xybrid-core/src/lib.rs")
        needs["unity-bolt-drift"] = {"result": "skipped"}
        check_jobs(needs)
        for path in ("crates/xybrid-bolt/src/lib.rs", "bindings/unity/Runtime/Bolt/XybridError.cs"):
            needs = self.needs(path)
            needs["unity-bolt-drift"] = {"result": "skipped"}
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_jobs(needs)

    def test_a_required_lane_skip_follows_its_own_flag_not_full_ci(self):
        # The llama.cpp lane runs on a Markdown-only change inside its paths.
        needs = self.needs("crates/llama-cpp-sys/README.md")
        needs["llamacpp-validate"] = {"result": "skipped"}
        with self.assertRaises(ValueError):
            check_jobs(needs)
        needs = self.needs("crates/xybrid-observability/src/lib.rs")
        needs["llamacpp-validate"] = {"result": "skipped"}
        check_jobs(needs)

    def test_the_onnx_runtime_legs_cannot_be_skipped_when_their_script_changes(self):
        with self.assertRaises(ValueError):
            check_jobs(self.needs("tools/scripts/stage_unity_desktop_ort.py"))

    def test_missing_or_failed_detection_cannot_pass(self):
        for detector in ({}, {"result": "failure"}, {"result": "skipped"}, {"result": "success"}):
            with self.subTest(detector=detector), self.assertRaises(ValueError):
                check_jobs({"tooling-changes": detector})


if __name__ == "__main__":
    unittest.main()
