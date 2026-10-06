"""Regression tests for docs-only selection and the required CI gate."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci_gate import PATH_FILTERS, check_jobs, classify


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

    def test_relevant_tooling_cannot_be_skipped(self):
        with self.assertRaises(ValueError):
            check_jobs(self.needs("tools/scripts/ci_gate.py"))

    def test_missing_or_failed_detection_cannot_pass(self):
        for detector in ({}, {"result": "failure"}, {"result": "skipped"}, {"result": "success"}):
            with self.subTest(detector=detector), self.assertRaises(ValueError):
                check_jobs({"tooling-changes": detector})


if __name__ == "__main__":
    unittest.main()
