"""Report validation and timeout handling for the RN consumer-smoke host runners."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rn_consumer_smoke.report import (
    REPORT_SCHEMA_VERSION,
    REQUIRED_STEP_IDS,
    ReportError,
    validate_report,
    validate_report_file,
    wait_for_report,
)


def ok_report(**overrides):
    report = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "runId": "run-test-1",
        "startedAt": "2026-11-16T10:00:00+00:00",
        "finishedAt": "2026-11-16T10:01:00+00:00",
        "success": True,
        "versions": {
            "npmPackage": "0.18.0",
            "nativeSdk": "0.18.0",
            "expo": "58.0.25",
            "reactNative": "0.88.1",
        },
        "results": [{"id": step_id, "ok": True} for step_id in REQUIRED_STEP_IDS],
    }
    report.update(overrides)
    return report


class ValidateReportTests(unittest.TestCase):
    def test_accepts_a_complete_successful_report(self):
        validate_report(ok_report())

    def test_rejects_missing_steps(self):
        report = ok_report()
        report["results"] = [r for r in report["results"] if r["id"] != "warmup"]
        with self.assertRaisesRegex(ReportError, "incomplete"):
            validate_report(report)

    def test_rejects_failed_steps(self):
        report = ok_report()
        for entry in report["results"]:
            if entry["id"] == "inference":
                entry["ok"] = False
        with self.assertRaisesRegex(ReportError, "failed steps: inference"):
            validate_report(report)

    def test_rejects_version_mismatch(self):
        report = ok_report()
        report["versions"]["nativeSdk"] = "0.19.0"
        with self.assertRaisesRegex(ReportError, "version mismatch"):
            validate_report(report)

    def test_rejects_success_without_release(self):
        report = ok_report()
        for entry in report["results"]:
            if entry["id"] == "release":
                entry["ok"] = False
        report["success"] = True
        with self.assertRaisesRegex(ReportError, "failed steps: release"):
            validate_report(report)

    def test_rejects_stale_run_id(self):
        with self.assertRaisesRegex(ReportError, "stale report"):
            validate_report(ok_report(), expected_run_id="other")

    def test_rejects_missing_file(self):
        with self.assertRaisesRegex(ReportError, "report missing"):
            validate_report_file(Path("/tmp/xybrid-no-such-report.json"))

    def test_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(ReportError, "not valid JSON"):
                validate_report_file(path)


class WaitForReportTests(unittest.TestCase):
    def test_returns_when_a_valid_report_appears(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "report.json"

            def pull():
                dest.write_text(json.dumps(ok_report()), encoding="utf-8")

            path = wait_for_report(pull, dest, timeout_s=5, poll_s=0.01)
            self.assertEqual(path, dest)

    def test_timeout_raises_report_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "report.json"

            def pull():
                return None

            with self.assertRaisesRegex(ReportError, "timed out"):
                wait_for_report(pull, dest, timeout_s=0.05, poll_s=0.01)

    def test_retries_empty_and_partially_written_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "report.json"
            complete = json.dumps(ok_report())
            snapshots = iter(("", complete[:-1], complete))

            def pull():
                dest.write_text(next(snapshots), encoding="utf-8")

            self.assertEqual(wait_for_report(pull, dest, timeout_s=5, poll_s=0.01), dest)

    def test_timeout_includes_last_parse_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "report.json"

            def pull():
                dest.write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ReportError, "timed out.*not valid JSON"):
                wait_for_report(pull, dest, timeout_s=0.05, poll_s=0.01)

    def test_finished_failure_stops_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "report.json"
            failed = ok_report(success=False)
            for entry in failed["results"]:
                if entry["id"] == "inference":
                    entry["ok"] = False
            pulls = iter((json.dumps(failed),))

            def pull():
                dest.write_text(next(pulls), encoding="utf-8")

            with self.assertRaisesRegex(ReportError, "^qualification failed: failed steps"):
                wait_for_report(pull, dest, timeout_s=5, poll_s=0.01)

    def test_stale_started_at_is_rejected_until_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "report.json"
            stale = ok_report(startedAt="2020-01-01T00:00:00+00:00", success=False)

            def pull():
                dest.write_text(json.dumps(stale), encoding="utf-8")

            with self.assertRaisesRegex(ReportError, "timed out.*stale"):
                wait_for_report(
                    pull,
                    dest,
                    timeout_s=0.05,
                    poll_s=0.01,
                    started_after=1_700_000_000,
                )


if __name__ == "__main__":
    unittest.main()
