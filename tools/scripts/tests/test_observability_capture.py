"""Tests for the observability runtime-scenario harness.

tools/scripts/observability_capture/ holds the OTLP receiver every native
binding lane exports to, the checker that compares a capture with the
scenario's expectations, and the driver tying them together. The checker is
exercised against a recorded passing run of the Python reference scenario
(fixtures/python) and against single-fault mutations of it: each mutation is a
regression a binding could ship, and the test pins which check reports it.
"""

import base64
import contextlib
import copy
import gzip
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from unittest import mock

HARNESS = Path(__file__).resolve().parents[1] / "observability_capture"
sys.path.insert(0, str(HARNESS))

import capture_server  # noqa: E402
import check_capture  # noqa: E402
import run_scenario  # noqa: E402

FIXTURE = HARNESS / "fixtures" / "python"
SPEC = check_capture.load_spec()
REPO = HARNESS.parents[2]

# Every lane's scenario source, with the public calls that configure the omit
# rule and record the content it drops (scenario.md, "API per binding").
LANE_SOURCES = {
    "python": ("bindings/python/tests/observability_scenario.py",
               r"\.omit_content_attribute\(", r"\.record_system_instructions\("),
    "swift": ("bindings/apple/Tests/XybridTests/ObservabilityRuntimeTests.swift",
              r"\.omitContent\(", r"\.recordSystemInstructions\("),
    "kotlin": ("bindings/kotlin/consumer-smoke/app/src/androidTest/kotlin/ai/xybrid/smoke/ObservabilityScenario.kt",
               r"\.omitContent\(", r"\.recordSystemInstructions\("),
    "unity": ("bindings/unity/Tests/Editor/ObservabilityRuntimeScenarioTests.cs",
              r"\.WithoutContentAttribute\(", r"\.RecordSystemInstructions\("),
    "flutter": ("bindings/flutter/test/observability_scenario_test.dart",
                r"FfiRedactionRule\.omitAttribute\(", r"\.recordSystemInstructions\("),
}


def load_fixture():
    entries = [json.loads(line) for line in (FIXTURE / "capture.jsonl").read_text().splitlines() if line]
    expectations = json.loads((FIXTURE / "expectations.json").read_text())
    return entries, expectations


def iter_spans(entries):
    for entry in entries:
        for resource in (entry.get("body") or {}).get("resourceSpans", []):
            for scope in resource["scopeSpans"]:
                yield from scope["spans"]


def span(entries, span_id):
    return next(item for item in iter_spans(entries) if item["spanId"] == span_id)


def set_attribute(item, key, value):
    remove_attribute(item, key)
    item.setdefault("attributes", []).append({"key": key, "value": {"stringValue": value}})


def remove_attribute(item, key):
    item["attributes"] = [attribute for attribute in item.get("attributes", []) if attribute["key"] != key]


def attribute(item, key):
    return next((a["value"]["stringValue"] for a in item.get("attributes", []) if a["key"] == key), None)


def run_span(entries, expectations):
    workflow = expectations["references"]["workflow"]["span_id"]
    taken = {ref["span_id"] for ref in expectations["references"].values() if ref}
    return next(item for item in iter_spans(entries)
                if item.get("parentSpanId") == workflow and item["spanId"] not in taken)


def remove_spans(entries, span_ids):
    for entry in entries:
        for resource in (entry.get("body") or {}).get("resourceSpans", []):
            for scope in resource["scopeSpans"]:
                scope["spans"] = [item for item in scope["spans"] if item["spanId"] not in span_ids]


def log_records(entries):
    for entry in entries:
        for resource in (entry.get("body") or {}).get("resourceLogs", []):
            for scope in resource["scopeLogs"]:
                yield scope["logRecords"]


def metric_points(entries, name):
    for entry in entries:
        for resource in (entry.get("body") or {}).get("resourceMetrics", []):
            for scope in resource["scopeMetrics"]:
                for metric in scope["metrics"]:
                    if metric["name"] == name:
                        yield metric["histogram"]["dataPoints"]


def is_cancelled(point):
    return any(a["key"] == "error.type" and a["value"].get("stringValue") == "cancelled"
               for a in point.get("attributes", []))


def results(entries, expectations, binding=None):
    checker = check_capture.Checker(check_capture.Capture.from_entries(entries), expectations, SPEC, binding)
    return {result.name: result for result in checker.run()}


class CheckerTestCase(unittest.TestCase):
    def setUp(self):
        self.entries, self.expected = load_fixture()
        self.refs = self.expected["references"]

    def assertOnlyFails(self, *names, binding=None):
        outcome = results(self.entries, self.expected, binding)
        failed = sorted(name for name, result in outcome.items() if not result.ok)
        self.assertEqual(failed, sorted(names), {name: outcome[name].failures for name in failed})
        return outcome

    def assertFails(self, *names):
        outcome = results(self.entries, self.expected)
        for name in names:
            self.assertFalse(outcome[name].ok, f"{name} should fail")
        return outcome


class PassingRunTests(CheckerTestCase):
    def test_recorded_python_run_passes_every_check(self):
        outcome = self.assertOnlyFails()
        self.assertEqual(len(outcome), len(check_capture.Checker.CHECKS))

    def test_a_load_rejected_at_load_time_is_an_accepted_path(self):
        run = run_span(self.entries, self.expected)
        descendants, frontier = {run["spanId"]}, {run["spanId"]}
        while frontier:
            frontier = {item["spanId"] for item in iter_spans(self.entries) if item.get("parentSpanId") in frontier}
            descendants |= frontier
        remove_spans(self.entries, descendants)
        for records in log_records(self.entries):
            records[:] = [record for record in records if record["spanId"] not in descendants]
        load = span(self.entries, self.refs["load"]["span_id"])
        set_attribute(load, "error.type", "load_error")
        load["status"] = {"code": 2, "message": "load_error"}
        self.refs["load"] = None
        self.expected["automatic"] = {"load": {"outcome": "error", "error": "LoadError: x"},
                                      "run": {"outcome": "skipped", "error": None}}
        for snapshot in ("after_flush", "after_shutdown"):
            counters = self.expected["stats"][snapshot]
            counters["traces"].update(accepted=6, exported=6)
            counters["logs"].update(accepted=1, exported=1)
        self.assertOnlyFails()

    def test_a_declared_sdk_binding_overrides_the_attribution_check(self):
        for entry in self.entries:
            for key in ("resourceSpans", "resourceLogs", "resourceMetrics"):
                for resource in (entry.get("body") or {}).get(key, []):
                    set_attribute(resource["resource"], "xybrid.sdk.binding", "rust")
        self.assertOnlyFails("resource.identity")
        self.expected["sdk_binding"] = "rust"
        self.assertOnlyFails()


class CaptureFaultTests(CheckerTestCase):
    def test_empty_capture_fails_without_crashing(self):
        self.entries = []
        outcome = self.assertFails("capture.decoded", "references.exported", "workflow.parent_in")
        self.assertIn("empty", outcome["capture.decoded"].failures[0])

    def test_protobuf_body_is_reported_as_misconfiguration(self):
        self.entries.append({"seq": 99, "path": "/v1/traces", "signal": "traces", "encoding": "protobuf",
                             "body": None, "body_base64": base64.b64encode(b"\x0a\x00").decode(),
                             "error": "protobuf body; configure the binding for OTLP/JSON"})
        outcome = self.assertOnlyFails("capture.decoded")
        self.assertIn("JSON protocol", outcome["capture.decoded"].failures[0])

    def test_wrong_signal_path_is_reported(self):
        self.entries[0]["signal"] = None
        self.entries[0]["path"] = "/v1/traces/v1/traces"
        self.assertFails("capture.decoded")


class ExpectationFaultTests(CheckerTestCase):
    def test_trace_id_of_another_binding_fails(self):
        self.expected["trace_id"], self.expected["parent_span_id"] = check_capture.scenario_ids("swift")
        outcome = self.assertFails("expectations.valid", "workflow.parent_in")
        self.assertIn("python derives", " ".join(outcome["expectations.valid"].failures))

    def test_scenario_error_fails(self):
        self.expected["scenario_error"] = "Traceback: boom"
        outcome = self.assertOnlyFails("expectations.valid")
        self.assertIn("boom", " ".join(outcome["expectations.valid"].failures))

    def test_missing_reference_section_fails(self):
        self.refs["tool"] = None
        self.assertFails("expectations.valid", "tool.content")

    def test_reference_naming_another_trace_fails(self):
        self.refs["tool"]["trace_id"] = "0" * 31 + "1"
        self.assertOnlyFails("references.exported")


class HierarchyFaultTests(CheckerTestCase):
    def test_unexported_reference_fails(self):
        remove_spans(self.entries, {self.refs["inference"]["span_id"]})
        self.assertFails("references.exported", "inference.content", "stats.consistent")

    def test_workflow_that_ignored_the_parent_fails(self):
        span(self.entries, self.refs["workflow"]["span_id"]).pop("parentSpanId")
        self.assertOnlyFails("workflow.parent_in", "trace.tree")

    def test_child_that_lost_the_workflow_context_fails(self):
        span(self.entries, self.refs["tool"]["span_id"])["parentSpanId"] = self.expected["parent_span_id"]
        self.assertOnlyFails("manual.children", "trace.tree")

    def test_orphaned_span_fails(self):
        run = run_span(self.entries, self.expected)
        child = next(item for item in iter_spans(self.entries) if item.get("parentSpanId") == run["spanId"])
        child["parentSpanId"] = "00000000000000aa"
        self.assertOnlyFails("trace.tree")


class ContentFaultTests(CheckerTestCase):
    def test_unredacted_secret_fails(self):
        tool = span(self.entries, self.refs["tool"]["span_id"])
        set_attribute(tool, "gen_ai.tool.call.result", SPEC["tool"]["result_json"])
        outcome = self.assertOnlyFails("content.redaction", "tool.content")
        self.assertIn("/v1/traces", outcome["content.redaction"].failures[0])

    def test_replacement_that_never_appears_fails(self):
        # A rule that dropped matches instead of replacing them leaks nothing
        # but is still not the configured rule.
        for entry in self.entries:
            entry["body"] = json.loads(json.dumps(entry["body"]).replace(SPEC["replacement"], ""))
        outcome = self.assertFails("content.redaction", "tool.content", "inference.content", "inference.failed")
        self.assertIn("never ran", outcome["content.redaction"].failures[0])

    def test_exported_omitted_attribute_fails(self):
        inference = span(self.entries, self.refs["inference"]["span_id"])
        set_attribute(inference, SPEC["omitted_attribute"], SPEC["inference"]["system_instructions_json"])
        self.assertOnlyFails("content.omit", "inference.content")

    def test_missing_device_id_fails(self):
        resource = self.entries[0]["body"]["resourceSpans"][0]["resource"]
        remove_attribute(resource, "device.id")
        self.assertOnlyFails("resource.identity")


class TerminalFaultTests(CheckerTestCase):
    def test_failed_inference_without_error_type_fails(self):
        remove_attribute(span(self.entries, self.refs["inference_failed"]["span_id"]), "error.type")
        self.assertOnlyFails("inference.failed")

    def test_failure_status_with_the_raw_message_fails(self):
        failed = span(self.entries, self.refs["inference_failed"]["span_id"])
        failed["status"]["message"] = SPEC["failed_inference"]["message"]
        self.assertOnlyFails("inference.failed", "content.redaction")

    def test_missing_exception_event_fails(self):
        failed_id = self.refs["inference_failed"]["span_id"]
        for records in log_records(self.entries):
            records[:] = [record for record in records if record["spanId"] != failed_id]
        self.assertOnlyFails("inference.failed", "stats.consistent")

    def test_release_that_completed_the_operation_fails(self):
        released = span(self.entries, self.refs["inference_released"]["span_id"])
        remove_attribute(released, "error.type")
        released["status"] = {"code": 0, "message": ""}
        self.assertOnlyFails("inference.released")

    def test_release_that_did_not_end_the_span_fails(self):
        stats = self.expected["stats"]
        stats["after_release"]["traces"] = dict(stats["before_release"]["traces"])
        self.assertOnlyFails("inference.released")

    def cancelled_durations(self):
        points = [point for points in metric_points(self.entries, "gen_ai.client.inference.duration")
                  for point in points if is_cancelled(point)]
        self.assertTrue(points, "the fixture records the release's cancelled duration")
        return points

    def test_release_without_a_cancelled_duration_point_fails(self):
        # The span says cancelled, but the metric never recorded the outcome.
        dropped = self.cancelled_durations()
        for points in metric_points(self.entries, "gen_ai.client.inference.duration"):
            points[:] = [point for point in points if point not in dropped]
        # Points never recorded were never exported either.
        final = self.expected["stats"]["after_shutdown"]["metrics"]
        final.update(accepted=final["accepted"] - len(dropped), exported=final["exported"] - len(dropped))
        outcome = self.assertOnlyFails("inference.released")
        self.assertIn("error.type 'cancelled'", " ".join(outcome["inference.released"].failures))

    def test_release_whose_duration_was_recorded_as_success_fails(self):
        for point in self.cancelled_durations():
            remove_attribute(point, "error.type")
        self.assertOnlyFails("inference.released")

    def test_release_recorded_twice_fails(self):
        # Every cumulative export after the release says the cancellation was measured twice.
        for point in self.cancelled_durations():
            point["count"] = "2"
        outcome = self.assertOnlyFails("inference.released")
        self.assertIn("holds 2 measurement(s)", " ".join(outcome["inference.released"].failures))

    def test_delta_exports_count_every_cancelled_point(self):
        # The fixture's two exports each carry the cancelled point; as deltas they are two measurements.
        for temporality in (1, "AGGREGATION_TEMPORALITY_DELTA"):
            with self.subTest(temporality=temporality):
                self.entries, self.expected = load_fixture()
                self.assertEqual(len(self.cancelled_durations()), 2)
                for entry in self.entries:
                    for resource in (entry.get("body") or {}).get("resourceMetrics", []):
                        for scope in resource["scopeMetrics"]:
                            for metric in scope["metrics"]:
                                if metric["name"] == "gen_ai.client.inference.duration":
                                    metric["histogram"]["aggregationTemporality"] = temporality
                outcome = self.assertOnlyFails("inference.released")
                self.assertIn("holds 2 measurement(s)", " ".join(outcome["inference.released"].failures))

    def test_cancellation_reported_as_exception_fails(self):
        records = next(log_records(self.entries))
        cancelled = dict(records[0], spanId=self.refs["inference_released"]["span_id"])
        records.append(cancelled)
        self.assertFails("inference.released")


class AutomaticPathFaultTests(CheckerTestCase):
    def test_run_without_conversation_id_fails(self):
        remove_attribute(run_span(self.entries, self.expected), "gen_ai.conversation.id")
        self.assertOnlyFails("automatic.run")

    def test_load_without_conversation_id_fails(self):
        remove_attribute(span(self.entries, self.refs["load"]["span_id"]), "gen_ai.conversation.id")
        self.assertOnlyFails("automatic.load")

    def test_successful_load_without_reference_fails(self):
        self.refs["load"] = None
        self.assertFails("automatic.load")

    def test_run_in_another_trace_under_the_workflow_id_fails(self):
        # Same parentSpanId as the workflow, but a different trace: not its child.
        run_span(self.entries, self.expected)["traceId"] = "f" * 32
        self.assertFails("automatic.run")

    def test_run_that_claims_success_fails(self):
        self.expected["automatic"]["run"] = {"outcome": "ok", "error": None}
        self.assertFails("automatic.run")

    def test_a_run_reference_naming_the_run_span_passes(self):
        # Reference-out from a run result: a binding that returns one must
        # point at the run span, which stays the one run span under the workflow.
        run = run_span(self.entries, self.expected)
        self.refs["run"] = {"trace_id": run["traceId"], "span_id": run["spanId"]}
        self.assertOnlyFails()

    def test_a_run_reference_naming_another_span_fails(self):
        self.refs["run"] = dict(self.refs["tool"])
        outcome = self.assertOnlyFails("automatic.run")
        self.assertIn("is not the run span", outcome["automatic.run"].failures[0])


class StatsAndLifecycleFaultTests(CheckerTestCase):
    def test_inconsistent_counters_fail(self):
        self.expected["stats"]["after_flush"]["traces"]["pending"] = 1
        self.assertOnlyFails("stats.consistent")

    def test_too_few_exported_spans_fail(self):
        self.expected["stats"]["after_flush"]["traces"].update(accepted=3, exported=3)
        self.assertOnlyFails("stats.consistent")

    def test_more_exported_than_received_fails(self):
        # The final counters after shutdown claim as much, so only the
        # receiver comparison fails.
        for snapshot in ("after_flush", "after_shutdown"):
            self.expected["stats"][snapshot]["traces"].update(accepted=12, exported=12)
        outcome = self.assertOnlyFails("stats.consistent")
        self.assertEqual(len(outcome["stats.consistent"].failures), 2)

    def test_non_idempotent_shutdown_fails(self):
        self.expected["lifecycle"]["shutdown"] = ["ok", "error: TelemetryDelivery: x"]
        self.assertOnlyFails("lifecycle.shutdown")

    def test_handle_live_after_shutdown_fails(self):
        self.expected["lifecycle"]["after_shutdown"]["workflow_ref"] = self.refs["workflow"]
        self.assertOnlyFails("lifecycle.shutdown")

    def test_stats_enabled_after_shutdown_fail(self):
        self.expected["stats"]["after_shutdown"]["enabled"] = True
        self.assertOnlyFails("lifecycle.shutdown")

    def test_stats_after_shutdown_without_counters_fail(self):
        # The final counters must not hold vacuously over counters never read.
        self.expected["stats"]["after_shutdown"] = {"enabled": False}
        outcome = self.assertOnlyFails("lifecycle.shutdown")
        self.assertEqual(len(outcome["lifecycle.shutdown"].failures), len(check_capture.SIGNAL_KEYS))
        self.expected["stats"]["after_shutdown"] = {
            "enabled": False, **{signal: {"accepted": 0} for signal in check_capture.SIGNAL_KEYS}}
        self.assertOnlyFails("lifecycle.shutdown")

    def test_stats_reset_by_shutdown_fail(self):
        # After shutdown the stats are the session's
        # final counters until the next init, not zeros (the old behaviour).
        self.expected["stats"]["after_shutdown"] = {
            "enabled": False,
            **{signal: dict.fromkeys(check_capture.COUNTER_KEYS, 0) for signal in check_capture.SIGNAL_KEYS}}
        outcome = self.assertOnlyFails("lifecycle.shutdown")
        failures = outcome["lifecycle.shutdown"].failures
        self.assertTrue(all("not a reset" in failure for failure in failures), failures)
        # Every signal's accepted and exported were above zero after the flush.
        self.assertEqual(len(failures), 2 * len(check_capture.SIGNAL_KEYS))

    def test_stats_after_shutdown_must_be_settled_and_lossless(self):
        final = self.expected["stats"]["after_shutdown"]
        cases = (
            ("traces", {"pending": 1, "accepted": 10}, "pending is 1"),
            ("logs", {"accepted": 3}, "accepted 3 != exported + dropped + failed"),
            ("metrics", {"failed": 1, "accepted": 17}, "metrics.failed is 1"),
            ("traces", {"dropped": 2, "accepted": 11}, "traces.dropped is 2"),
        )
        for signal, change, message in cases:
            with self.subTest(signal=signal, change=change):
                original = dict(final[signal])
                final[signal].update(change)
                outcome = self.assertOnlyFails("lifecycle.shutdown")
                failures = outcome["lifecycle.shutdown"].failures
                self.assertTrue(any(message in failure for failure in failures), failures)
                final[signal] = original

    def test_more_exported_after_shutdown_than_received_fails(self):
        # The final counters are held against the receiver like the flush's.
        self.expected["stats"]["after_shutdown"]["metrics"].update(accepted=24, exported=24)
        outcome = self.assertOnlyFails("stats.consistent")
        self.assertEqual(outcome["stats.consistent"].failures,
                         ["after shutdown, metrics.exported 24 but the receiver holds 16 data points"])

    def test_export_after_shutdown_fails(self):
        late = copy.deepcopy(self.entries[0])
        late["seq"] = 99
        scope = late["body"]["resourceSpans"][0]["scopeSpans"][0]
        workflow = copy.deepcopy(span(self.entries, self.refs["workflow"]["span_id"]))
        workflow.update(spanId="00000000000000bb", name=f"invoke_workflow {SPEC['after_shutdown_workflow_name']}")
        workflow.pop("parentSpanId")
        workflow["traceId"] = "0" * 31 + "2"
        scope["spans"] = [workflow]
        self.entries.append(late)
        self.assertOnlyFails("lifecycle.nothing_after_shutdown")


class DecodingTests(unittest.TestCase):
    def test_ids_are_hex_and_tolerate_base64(self):
        raw = bytes(range(16))
        self.assertEqual(check_capture.normalize_id(raw.hex().upper(), 16), raw.hex())
        self.assertEqual(check_capture.normalize_id(base64.b64encode(raw).decode(), 16), raw.hex())
        self.assertIsNone(check_capture.normalize_id("", 8))

    def test_any_value_shapes(self):
        decoded = check_capture.attributes([
            {"key": "s", "value": {"stringValue": "x"}},
            {"key": "i", "value": {"intValue": "7"}},
            {"key": "a", "value": {"arrayValue": {"values": [{"boolValue": True}]}}},
            {"key": "k", "value": {"kvlistValue": {"values": [{"key": "n", "value": {"doubleValue": 1.5}}]}}},
        ])
        self.assertEqual(decoded, {"s": "x", "i": 7, "a": [True], "k": {"n": 1.5}})

    def test_string_status_codes(self):
        self.assertEqual(check_capture.status_code({"code": "STATUS_CODE_ERROR"}), 2)
        self.assertEqual(check_capture.status_code(None), 0)


class IdentifierTests(unittest.TestCase):
    def test_ids_follow_the_documented_rule(self):
        self.assertEqual(check_capture.scenario_ids("python"),
                         ("78793a707974686f6e00000000000000", "707974686f6e0000"))
        self.assertEqual(check_capture.scenario_ids("react-native"),
                         ("78793a72656163742d6e617469766500", "72656163742d6e61"))

    def test_ids_are_unique_per_binding(self):
        ids = [check_capture.scenario_ids(binding) for binding in SPEC["bindings"]]
        self.assertEqual(len({trace for trace, _ in ids}), len(ids))
        self.assertEqual(len({parent for _, parent in ids}), len(ids))

    def test_invalid_binding_names_are_refused(self):
        for name in ("", "Python", "a-very-long-binding-name", "py thon"):
            with self.assertRaises(ValueError):
                check_capture.scenario_ids(name)


class DocumentationTests(unittest.TestCase):
    """scenario.md is what lanes implement from; it must state every constant."""

    def setUp(self):
        self.doc = (HARNESS / "scenario.md").read_text(encoding="utf-8")

    def test_every_binding_identifier_is_tabulated(self):
        for binding in SPEC["bindings"]:
            trace_id, parent_span_id = check_capture.scenario_ids(binding)
            self.assertIn(f"| `{binding}` | `{trace_id}` | `{parent_span_id}` |", self.doc)

    def test_every_constant_appears_verbatim(self):
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from strings(item)
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)

        skipped = {"schema", "bindings", "ids"}
        for key, value in SPEC.items():
            if key in skipped:
                continue
            for text in strings(value):
                self.assertIn(text, self.doc, f"scenario.md does not state {key}: {text!r}")
        self.assertIn(SPEC["schema"], self.doc)
        self.assertIn(str(SPEC["timeout_ms"]), self.doc)

    def test_every_check_is_described(self):
        for name, _ in check_capture.Checker.CHECKS:
            self.assertIn(f"`{name}`", self.doc)


class LaneSourceTests(unittest.TestCase):
    """content.omit can only show that nothing omitted reached the receiver.

    A rule-omitted value leaves no trace on the wire, so a lane that dropped
    either the omit rule or the content it removes would still pass. The Python
    control run (test_observability_scenario.py) shows the reference records
    the content; this pins both calls in every lane's source.
    """

    @staticmethod
    def code(path):
        # Comment lines would otherwise satisfy the patterns.
        return "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                         if not line.lstrip().startswith(("#", "//", "/*", "*")))

    def test_every_lane_configures_the_omit_rule_and_records_the_content_it_drops(self):
        for binding, (path, omit_rule, record) in LANE_SOURCES.items():
            with self.subTest(binding=binding):
                self.assertIn(binding, SPEC["bindings"])
                code = self.code(REPO / path)
                self.assertRegex(code, omit_rule, f"{path} no longer configures the omit rule")
                self.assertRegex(code, record, f"{path} no longer records the system instructions")

    def test_a_commented_out_call_does_not_count(self):
        _, omit_rule, record = LANE_SOURCES["kotlin"]
        with tempfile.TemporaryDirectory() as temp:
            lane = Path(temp) / "Lane.kt"
            lane.write_text("config\n    // .omitContent(OMITTED_ATTRIBUTE)\n"
                            "inference.recordSystemInstructions(SYSTEM_INSTRUCTIONS_JSON)\n", encoding="utf-8")
            code = self.code(lane)
        self.assertNotRegex(code, omit_rule)
        self.assertRegex(code, record)


class CliTests(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = check_capture.main(list(argv))
        return status, out.getvalue(), err.getvalue()

    def test_exit_codes(self):
        capture, expectations = str(FIXTURE / "capture.jsonl"), str(FIXTURE / "expectations.json")
        status, out, _ = self.run_main("--capture", capture, "--expectations", expectations)
        self.assertEqual(status, 0, out)
        self.assertIn("18/18 checks passed", out)
        status, out, _ = self.run_main("--capture", capture, "--expectations", expectations, "--binding", "unity")
        self.assertEqual(status, 1)
        self.assertIn("FAIL expectations.valid", out)
        status, _, err = self.run_main("--capture", "/nonexistent", "--expectations", expectations)
        self.assertEqual(status, 2)

    def test_report_file(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "report.json"
            self.run_main("--capture", str(FIXTURE / "capture.jsonl"),
                          "--expectations", str(FIXTURE / "expectations.json"), "--report", str(report))
            data = json.loads(report.read_text())
        self.assertTrue(data["ok"])
        self.assertEqual(data["capture"]["spans"], 9)

    def test_ids_and_constants(self):
        status, out, _ = self.run_main("--ids", "kotlin")
        self.assertEqual(status, 0)
        self.assertIn("traceparent=00-78793a6b6f746c696e00000000000000-6b6f746c696e0000-01", out)
        status, out, _ = self.run_main("--constants", "swift")
        constants = json.loads(out)
        self.assertEqual(constants["device_id"], "scenario-device-swift")
        self.assertEqual(constants["secret"], SPEC["secret"])
        self.assertEqual(self.run_main("--ids", "Not Valid")[0], 2)


def post(url, body, content_type="application/json", encoding=None):
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    if encoding:
        request.add_header("Content-Encoding", encoding)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class ServerProcess:
    def __init__(self, directory):
        self.capture = directory / "capture.jsonl"
        self.ready = directory / "ready.json"
        self.process = subprocess.Popen(
            [sys.executable, "-I", str(HARNESS / "capture_server.py"), "--host", "127.0.0.1", "--port", "0",
             "--out", str(self.capture), "--ready-file", str(self.ready)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 20
        while not self.ready.exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                self.process.kill()
                raise RuntimeError(f"receiver did not start: {self.process.communicate()}")
            time.sleep(0.02)
        self.info = json.loads(self.ready.read_text())
        self.url = self.info["endpoint"]

    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
        out, err = self.process.communicate(timeout=10)
        return self.process.returncode, out, err

    def lines(self):
        return [json.loads(line) for line in self.capture.read_text().splitlines() if line]


class BoundedDecompressTests(unittest.TestCase):
    @staticmethod
    def bomb(encoding, expanded):
        """A body that expands to ``expanded`` zero bytes, built in chunks."""
        wbits = 16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS
        encoder = zlib.compressobj(9, zlib.DEFLATED, wbits)
        chunk = b"\0" * (1 << 20)
        parts = [encoder.compress(chunk) for _ in range(expanded // len(chunk))]
        return b"".join(parts) + encoder.flush()

    def test_a_bomb_is_refused_without_expanding_it(self):
        import tracemalloc
        limit = capture_server.MAX_BODY_BYTES
        for encoding in ("gzip", "deflate"):
            body = self.bomb(encoding, 4 * limit)
            tracemalloc.start()
            try:
                with self.assertRaises(capture_server.DecodedTooLarge):
                    capture_server.bounded_decompress(body, encoding)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertLess(peak, 1.5 * limit, f"{encoding}: decoding allocated {peak} bytes")

    def test_bodies_within_the_limit_round_trip(self):
        payload = json.dumps({"resourceSpans": []}).encode()
        self.assertEqual(capture_server.bounded_decompress(gzip.compress(payload), "gzip"), payload)
        self.assertEqual(capture_server.bounded_decompress(zlib.compress(payload), "deflate"), payload)
        with self.assertRaises(zlib.error):
            capture_server.bounded_decompress(zlib.compress(payload)[:-4], "deflate")


@unittest.skipIf(os.name == "nt", "SIGTERM delivery is POSIX-only")
class CaptureServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.temp, ignore_errors=True))
        self.server = ServerProcess(self.temp)
        self.addCleanup(lambda: self.server.process.poll() is None and self.server.process.kill())

    def test_records_every_signal_and_exits_cleanly_on_sigterm(self):
        traces = json.dumps({"resourceSpans": []}).encode()
        self.assertEqual(post(self.server.url + "/v1/traces", traces), (200, b"{}"))
        self.assertEqual(post(self.server.url + "/v1/metrics", gzip.compress(b'{"resourceMetrics": []}'),
                              encoding="gzip")[0], 200)
        self.assertEqual(post(self.server.url + "/v1/logs", b"\x0a\x02\x08\x01", "application/x-protobuf")[0], 200)
        self.assertEqual(post(self.server.url + "/v1/other", b"{}")[0], 404)
        with urllib.request.urlopen(self.server.url + "/healthz", timeout=10) as response:
            self.assertEqual(response.status, 200)
        status, out, _ = self.server.stop()
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(out.splitlines()[0])["port"], self.server.info["port"])

        lines = self.server.lines()
        self.assertEqual([line["seq"] for line in lines], [1, 2, 3, 4])
        self.assertEqual([line["signal"] for line in lines], ["traces", "metrics", "logs", None])
        self.assertEqual(lines[0]["body"], {"resourceSpans": []})
        self.assertEqual(lines[1]["body"], {"resourceMetrics": []})
        self.assertEqual(lines[2]["encoding"], "protobuf")
        self.assertEqual(base64.b64decode(lines[2]["body_base64"]), b"\x0a\x02\x08\x01")
        self.assertIn("JSON", lines[2]["error"])
        self.assertEqual(lines[3]["path"], "/v1/other")

    def test_compressed_body_that_expands_past_the_limit_is_refused(self):
        bomb = gzip.compress(b"\0" * (capture_server.MAX_BODY_BYTES + 1), compresslevel=9)
        self.assertLess(len(bomb), capture_server.MAX_BODY_BYTES)
        self.assertEqual(post(self.server.url + "/v1/traces", bomb, encoding="gzip")[0], 413)
        deflated = zlib.compress(b"\0" * (capture_server.MAX_BODY_BYTES + 1), 9)
        self.assertEqual(post(self.server.url + "/v1/traces", deflated, encoding="deflate")[0], 413)
        ok = zlib.compress(json.dumps({"resourceSpans": []}).encode())
        self.assertEqual(post(self.server.url + "/v1/traces", ok, encoding="deflate")[0], 200)
        self.server.stop()
        lines = self.server.lines()
        self.assertEqual([line["encoding"] for line in lines[:2]], ["invalid", "invalid"])
        self.assertIn("exceeds", lines[0]["error"])
        self.assertEqual(lines[2]["body"], {"resourceSpans": []})

    def test_invalid_json_is_kept_for_diagnosis(self):
        self.assertEqual(post(self.server.url + "/v1/traces", b"not json")[0], 200)
        self.server.stop()
        line = self.server.lines()[0]
        self.assertEqual(line["encoding"], "invalid")
        self.assertEqual(base64.b64decode(line["body_base64"]), b"not json")

    def test_ready_file_reports_the_bound_port(self):
        self.assertEqual(self.server.info["host"], "127.0.0.1")
        self.assertGreater(self.server.info["port"], 0)
        self.assertEqual(self.server.info["pid"], self.server.process.pid)


REPLAY = r"""
import json, os, sys, shutil, urllib.request
fixture, endpoint, expectations = sys.argv[1], sys.argv[2], sys.argv[3]
skip = set(sys.argv[4:])
assert os.environ["XYBRID_SCENARIO_BINDING"] == "python"
assert os.environ["XYBRID_SCENARIO_ENDPOINT"] == endpoint
assert not [key for key in os.environ if key.startswith("OTEL_")], "the driver passes no OTEL_* variables"
for line in open(os.path.join(fixture, "capture.jsonl")):
    entry = json.loads(line)
    if entry["signal"] in skip:
        continue
    request = urllib.request.Request(endpoint + entry["path"], data=json.dumps(entry["body"]).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(request, timeout=10).read()
shutil.copyfile(os.path.join(fixture, "expectations.json"), expectations)
"""


@unittest.skipIf(os.name == "nt", "the driver stops the receiver with SIGTERM")
class DriverTests(unittest.TestCase):
    """run_scenario.py with a stand-in scenario that replays the recorded run."""

    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.temp, ignore_errors=True))
        self.replay = self.temp / "replay.py"
        self.replay.write_text(REPLAY)

    def drive(self, *extra):
        out = self.temp / "out"
        argv = ["--binding", "python", "--out-dir", str(out), "--grace", "0", "--",
                sys.executable, str(self.replay), str(FIXTURE), "{endpoint}", "{expectations}", *extra]
        inherited = {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT": "no_content"}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.dict(os.environ, inherited):
            status = run_scenario.main(argv)
        return status, out

    def test_a_timeout_stops_every_process_the_command_started(self):
        pidfile = self.temp / "grandchild.pid"
        # The command starts a long-lived grandchild and then waits on it.
        script = f"sleep 300 & echo $! > {pidfile}; wait"
        with contextlib.redirect_stderr(io.StringIO()):
            status = run_scenario.run_command(["sh", "-c", script], dict(os.environ), 1.0)
        self.assertEqual(status, 124)
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.kill(pid, 9)
            self.fail("the grandchild outlived the timed-out command")

    def test_replayed_run_passes_and_writes_the_report(self):
        status, out = self.drive()
        self.assertEqual(status, 0)
        report = json.loads((out / "report.json").read_text())
        self.assertTrue(report["ok"])
        self.assertEqual(len((out / "capture.jsonl").read_text().splitlines()), 4)

    def test_a_run_missing_a_signal_fails(self):
        status, out = self.drive("logs")
        self.assertEqual(status, 1)
        report = json.loads((out / "report.json").read_text())
        self.assertIn("inference.failed", [check["name"] for check in report["checks"] if not check["ok"]])

    def test_missing_expectations_fail(self):
        self.replay.write_text("import sys\n")
        self.assertEqual(self.drive()[0], 1)

    def test_failing_command_fails_even_when_checks_pass(self):
        self.replay.write_text(REPLAY + "\nsys.exit(3)\n")
        self.assertEqual(self.drive()[0], 1)


if __name__ == "__main__":
    unittest.main()
