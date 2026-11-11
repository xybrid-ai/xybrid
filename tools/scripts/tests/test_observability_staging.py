"""Unit tests for tools/scripts/observability_staging (the step 6 staging driver and checker).

A synthetic launch is built from one list of spans, rendered both as the
Collector's tee received them (OTLP/JSON) and as the staging API answers them,
so the passing case exercises every check, and each test then changes one fact
and requires exactly the check that guards it to fail.

    python3 -m unittest tools/scripts/tests/test_observability_staging.py
"""

from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent / "observability_staging"
sys.path.insert(0, str(STAGING))
sys.path.insert(0, str(HERE.parent / "observability_interop"))
sys.path.insert(0, str(HERE.parent / "observability_capture"))

import run_staging  # noqa: E402 -- tool directory, imported after the path is set
import staging_checks as sc  # noqa: E402

SPEC = sc.load_spec()
RUN = "s6-20261111t150000z-python-mac"
COMMIT = "00e617102bc77ca0602115669cf62a68e56a92bb"
MODEL = "lfm2.5-230m"
IDS = {
    "direct": ("0123456789abcdef0123456789abcdef", "a1a1a1a1a1a1a1a1"),
    "collector": ("fedcba9876543210fedcba9876543210", "b2b2b2b2b2b2b2b2"),
}


def otlp_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def otlp_attributes(values: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"key": key, "value": otlp_value(value)} for key, value in values.items()]


def attr_map(values: Dict[str, Any]) -> Dict[str, Any]:
    return {key: otlp_value(value) for key, value in values.items()}


class Session:
    """One session's spans, rendered for the tee and for the staging API."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.trace_id, self.parent = IDS[name]
        prefix = "1" if name == "direct" else "2"
        self.ids = {role: prefix + str(index) * 15 for index, role in
                    enumerate(("workflow", "load", "stopped", "run"), start=1)}
        conversation = SPEC["per_lane"]["conversation_id"].format(run=RUN, session=name)
        captured = name == "collector"
        replaced = SPEC["completed"]["prompt"].replace(SPEC["secret"], SPEC["replacement"])
        self.resource = {
            "service.name": "xybrid-staging-python", "service.version": "1.0.0",
            "device.id": "staging-device-python-mac", "xybrid.sdk.binding": "python", "xybrid.sdk.version": "0.17.0",
            "telemetry.sdk.name": "xybrid", "xybrid.acceptance.run": RUN, "xybrid.acceptance.lane": "python-mac",
            "xybrid.acceptance.commit": COMMIT, "xybrid.acceptance.session": name,
        }
        request = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "llamacpp",
                   "gen_ai.request.model": MODEL, "xybrid.genai.execution.location": "local",
                   "gen_ai.conversation.id": conversation, "gen_ai.request.stream": True}
        self.spans = [
            {"role": "workflow", "parent": self.parent, "name": "invoke_workflow staging-workflow",
             "attrs": {"gen_ai.operation.name": "invoke_workflow", "gen_ai.workflow.name": "staging-workflow"},
             "type": "genai.workflow", "counted": False},
            {"role": "load", "parent": "workflow", "name": f"load_model {MODEL}",
             "attrs": {"gen_ai.operation.name": "load_model", "gen_ai.request.model": MODEL,
                       "gen_ai.conversation.id": conversation},
             "type": "genai.load_model", "counted": False},
            {"role": "stopped", "parent": "workflow", "name": f"chat {MODEL}",
             "attrs": {**request, "error.type": "cancelled", CHUNK: 35, "gen_ai.response.time_to_first_chunk": 0.04},
             "type": "genai.request", "counted": True, "error": "cancelled"},
            {"role": "run", "parent": "workflow", "name": f"chat {MODEL}",
             "attrs": {**request, CHUNK: 9, "gen_ai.usage.input_tokens": 31, "gen_ai.usage.output_tokens": 10,
                       "gen_ai.response.time_to_first_chunk": 0.05, "gen_ai.response.model": MODEL,
                       "xybrid.genai.response.time_to_first_token": 0.03},
             "type": "genai.request", "counted": True,
             "content": {"gen_ai.input.messages": json.dumps(
                 [{"role": "user", "parts": [{"type": "text", "content": replaced}]}]),
                 "gen_ai.output.messages": json.dumps(
                     [{"role": "assistant", "parts": [{"type": "text", "content": "Blue."}],
                       "finish_reason": "stop"}])} if captured else None},
        ]

    def span_id(self, role_or_id: str) -> str:
        return self.ids.get(role_or_id, role_or_id)

    def tee_entry(self, seq: int = 1) -> Dict[str, Any]:
        spans = []
        for span in self.spans:
            attrs = dict(span["attrs"])
            attrs.update(span.get("content") or {})
            entry = {"traceId": self.trace_id, "spanId": self.span_id(span["role"]),
                     "parentSpanId": self.span_id(span["parent"]), "name": span["name"],
                     "attributes": otlp_attributes(attrs)}
            if span.get("error"):
                entry["status"] = {"code": 2, "message": span["error"]}
            spans.append(entry)
        return {"seq": seq, "signal": "traces", "encoding": "json", "path": "/v1/traces",
                "body": {"resourceSpans": [{"resource": {"attributes": otlp_attributes(self.resource)},
                                            "scopeSpans": [{"spans": spans}]}]}}

    def logs_entry(self, seq: int = 2) -> Dict[str, Any]:
        run = self.ids["run"]
        records = [
            {"traceId": self.trace_id, "spanId": run, "eventName": "gen_ai.evaluation.result",
             "attributes": otlp_attributes({"gen_ai.evaluation.name": "user_feedback",
                                            "gen_ai.evaluation.score.value": 1.0,
                                            "gen_ai.evaluation.score.label": "thumbs_up"})},
            {"traceId": self.trace_id, "spanId": run, "eventName": "gen_ai.evaluation.result",
             "attributes": otlp_attributes({"gen_ai.evaluation.name": SPEC["evaluation"]["name"],
                                            "gen_ai.evaluation.score.value": 0.75,
                                            "gen_ai.evaluation.score.label": "pass"})},
        ]
        return {"seq": seq, "signal": "logs", "encoding": "json", "path": "/v1/logs",
                "body": {"resourceLogs": [{"resource": {"attributes": otlp_attributes(self.resource)},
                                           "scopeLogs": [{"logRecords": records}]}]}}

    def metrics_entry(self, seq: int = 3, ok: int = 1, cancelled: int = 1) -> Dict[str, Any]:
        points = [{"attributes": otlp_attributes({"gen_ai.request.model": MODEL}), "count": str(ok)},
                  {"attributes": otlp_attributes({"gen_ai.request.model": MODEL, "error.type": "cancelled"}),
                   "count": str(cancelled)}]
        metric = {"name": sc.INFERENCE_DURATION, "histogram": {"aggregationTemporality": 2, "dataPoints": points}}
        return {"seq": seq, "signal": "metrics", "encoding": "json", "path": "/v1/metrics",
                "body": {"resourceMetrics": [{"resource": {"attributes": otlp_attributes(self.resource)},
                                              "scopeMetrics": [{"metrics": [metric]}]}]}}

    def staging_spans(self) -> Dict[str, Any]:
        out = []
        for span in self.spans:
            attrs = span["attrs"]
            content = span.get("content")
            out.append({
                "trace_id": self.trace_id, "span_id": self.span_id(span["role"]),
                "parent_span_id": self.span_id(span["parent"]), "name": span["name"],
                "span_type": span["type"], "counted_request": span["counted"], "is_request": span["counted"],
                "status": {"code": "ERROR" if span.get("error") else "UNSET", "message": span.get("error") or ""},
                "attributes": attr_map(attrs), "resource": {"attributes": attr_map(self.resource)},
                "content": None if content is None else {"bytes": 200, "keys": sorted(content), "storage": "inline",
                                                         "values": attr_map(content)},
                "genai": {"operation_name": attrs.get("gen_ai.operation.name"),
                          "request_model": attrs.get("gen_ai.request.model"),
                          "response_model": attrs.get("gen_ai.response.model"),
                          "provider_name": attrs.get("gen_ai.provider.name"),
                          "execution_location": attrs.get("xybrid.genai.execution.location"),
                          "input_tokens": attrs.get("gen_ai.usage.input_tokens"),
                          "output_tokens": attrs.get("gen_ai.usage.output_tokens"),
                          "error_type": attrs.get("error.type"),
                          "time_to_first_chunk_s": attrs.get("gen_ai.response.time_to_first_chunk"),
                          "time_to_first_token_s": attrs.get("xybrid.genai.response.time_to_first_token")},
            })
        return {"trace_id": self.trace_id, "spans": out, "truncated": False}

    def staging_evaluations(self) -> Dict[str, Any]:
        run = self.ids["run"]
        return {"trace_id": self.trace_id, "truncated": False, "items": [
            {"trace_id": self.trace_id, "span_id": run, "name": "user_feedback", "score_value": 1.0,
             "score_label": "thumbs_up", "source": "feedback",
             "comment": SPEC["feedback"]["comment"].replace(SPEC["secret"], SPEC["replacement"])},
            {"trace_id": self.trace_id, "span_id": run, "name": SPEC["evaluation"]["name"], "score_value": 0.75,
             "score_label": "pass", "source": "evaluation", "comment": None}]}

    def expectations(self) -> Dict[str, Any]:
        def ref(role: str) -> Dict[str, str]:
            return {"trace_id": self.trace_id, "span_id": self.ids[role]}

        captured = self.name == "collector"
        counters = {"accepted": 4, "exported": 4, "dropped": 0, "failed": 0, "pending": 0}
        logs = {"accepted": 2, "exported": 2, "dropped": 0, "failed": 0, "pending": 0} if captured else \
            dict.fromkeys(sc.COUNTERS, 0)
        metrics = {"accepted": 6, "exported": 6, "dropped": 0, "failed": 0, "pending": 0} if captured else \
            dict.fromkeys(sc.COUNTERS, 0)
        snapshot = {"enabled": True, "traces": counters, "logs": logs, "metrics": metrics}
        return {
            "trace_id": self.trace_id, "parent_span_id": self.parent, "init": "ok",
            "references": {role: ref(role) for role in ("workflow", "load", "run")},
            "stopped": {"outcome": "cancelled", "chunks_consumed": 3, "delivery": "pull"},
            "completed": {"outcome": "ok", "chunks": 9, "tokens_out": 10, "text_chars": 5, "finish_reason": "stop"},
            "quality": {"feedback": captured, "evaluation": captured},
            "lifecycle": {"flush": "ok", "shutdown": "ok"},
            "stats": {"after_flush": snapshot, "after_shutdown": {**copy.deepcopy(snapshot), "enabled": False}},
        }


CHUNK = sc.CHUNK_COUNT


def build_launch() -> sc.Launch:
    sessions = {name: Session(name) for name in ("direct", "collector")}
    collector = sessions["collector"]
    config = {"schema": sc.SCHEMA, "binding": "python", "lane": "python-mac", "run": RUN, "commit": COMMIT,
              "model_id": MODEL, "sessions": {name: {"trace_id": s.trace_id, "parent_span_id": s.parent}
                                              for name, s in sessions.items()}}
    expectations = {"schema": sc.SCHEMA, "binding": "python", "lane": "python-mac", "run": RUN,
                    "sdk_version": "0.17.0", "model_id": MODEL, "scenario_error": None,
                    "sessions": {name: s.expectations() for name, s in sessions.items()}}
    readback = {name: sc.SessionReadback(spans=s.staging_spans(),
                                         evaluations=s.staging_evaluations() if name == "collector"
                                         else {"items": [], "trace_id": s.trace_id, "truncated": False})
                for name, s in sessions.items()}
    deployments = {"GCP - CD (auth, API) / deploy (api)": "f1b8cbb4", "Tinybird - CD / deploy-staging": "f1b8cbb4"}
    return sc.Launch(
        config=config, expectations=expectations, readback=readback,
        ground_truth=[collector.tee_entry(1), collector.logs_entry(2), collector.metrics_entry(3)],
        sinks=[collector.metrics_entry(1)],
        stats_before={"total_executions_24h": 10, "executions_on_device": 10},
        stats_after={"total_executions_24h": 14, "executions_on_device": 14},
        deployments_before=dict(deployments), deployments_after=dict(deployments),
        lane_exit=0, secret_hits=[])


def failing(checks: List[sc.Check]) -> List[str]:
    return sorted(f"{check.workload}:{check.name}" for check in checks if check.status == "fail")


def span(launch: sc.Launch, session: str, role: str) -> Dict[str, Any]:
    span_id = launch.expectations["sessions"][session]["references"].get(role, {}).get("span_id") \
        if role in ("workflow", "load", "run") else Session(session).ids[role]
    return next(item for item in launch.readback[session].spans["spans"] if item["span_id"] == span_id)


class PassingLaunchTest(unittest.TestCase):
    def test_every_check_passes(self) -> None:
        checks = sc.check_launch(build_launch(), SPEC)
        self.assertEqual(failing(checks), [], [check.as_dict() for check in checks if check.failures])
        names = {check.name for check in checks}
        for name in ("expectations.valid", "refs.stored", "trace.rooted", "identity", "load.span", "run.mapping",
                     "stream.cancelled", "lifecycle.final", "direct.no_content", "direct.quality", "direct.delivery",
                     "ids.spans", "ids.remote_parent", "mapping.usage", "content", "content.redaction",
                     "evaluations", "quality.linked", "collector.metric_ledger", "content.redacted",
                     "aggregate.stats", "deployment.unchanged", "evidence.secrets"):
            self.assertIn(name, names)


class SingleFaultTest(unittest.TestCase):
    """Each mutation breaks one fact; the check that guards it must fail."""

    def assert_fails(self, mutate, expected: str) -> None:
        launch = build_launch()
        mutate(launch)
        found = failing(sc.check_launch(launch, SPEC))
        self.assertIn(expected, found)

    def test_scenario_error(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.expectations["scenario_error"] = "boom"
        self.assert_fails(mutate, "python-mac/direct:expectations.valid")

    def test_run_reference_not_stored(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.expectations["sessions"]["direct"]["references"]["run"]["span_id"] = "ffffffffffffffff"
        self.assert_fails(mutate, "python-mac/direct:refs.stored")

    def test_workflow_not_under_p(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            span(launch, "direct", "workflow")["parent_span_id"] = "cccccccccccccccc"
        self.assert_fails(mutate, "python-mac/direct:trace.rooted")

    def test_p_stored(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            caller = copy.deepcopy(span(launch, "direct", "load"))
            caller.update({"span_id": IDS["direct"][1], "parent_span_id": None})
            launch.readback["direct"].spans["spans"].append(caller)
        self.assert_fails(mutate, "python-mac/direct:trace.rooted")

    def test_wrong_binding(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            span(launch, "direct", "run")["resource"]["attributes"]["xybrid.sdk.binding"] = {"stringValue": "rust"}
        self.assert_fails(mutate, "python-mac/direct:identity")

    def test_wrong_acceptance_commit(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            for item in launch.readback["direct"].spans["spans"]:
                item["resource"]["attributes"]["xybrid.acceptance.commit"] = {"stringValue": "deadbeef"}
        self.assert_fails(mutate, "python-mac/direct:identity")

    def test_load_failed(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            span(launch, "direct", "load")["status"] = {"code": "ERROR", "message": "load_error"}
        self.assert_fails(mutate, "python-mac/direct:load.span")

    def test_output_tokens_differ(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.expectations["sessions"]["direct"]["completed"]["tokens_out"] = 11
        self.assert_fails(mutate, "python-mac/direct:run.mapping")

    def test_chunk_count_differs(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.expectations["sessions"]["direct"]["completed"]["chunks"] = 8
        self.assert_fails(mutate, "python-mac/direct:run.mapping")

    def test_no_first_chunk(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            span(launch, "direct", "run")["genai"]["time_to_first_chunk_s"] = None
        self.assert_fails(mutate, "python-mac/direct:run.mapping")

    def test_remote_execution(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            span(launch, "direct", "run")["genai"]["execution_location"] = "remote"
        self.assert_fails(mutate, "python-mac/direct:run.mapping")

    def test_stopped_stream_completed(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            stopped = span(launch, "direct", "stopped")
            stopped["status"] = {"code": "UNSET", "message": ""}
            stopped["genai"]["error_type"] = None
        self.assert_fails(mutate, "python-mac/direct:stream.cancelled")

    def test_stopped_stream_ran_to_the_limit(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            span(launch, "direct", "stopped")["attributes"][CHUNK] = {"intValue": "256"}
        self.assert_fails(mutate, "python-mac/direct:stream.cancelled")

    def test_callback_delivery_counts_exactly(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.expectations["sessions"]["direct"]["stopped"]["delivery"] = "callback"
        self.assert_fails(mutate, "python-mac/direct:stream.cancelled")

    def test_pending_after_shutdown(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            traces = launch.expectations["sessions"]["direct"]["stats"]["after_shutdown"]["traces"]
            traces.update({"pending": 1, "accepted": 5})
        self.assert_fails(mutate, "python-mac/direct:lifecycle.final")

    def test_shutdown_failed(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.expectations["sessions"]["direct"]["lifecycle"]["shutdown"] = "error: Timeout"
        self.assert_fails(mutate, "python-mac/direct:lifecycle.final")

    def test_direct_content_stored(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            span(launch, "direct", "run")["content"] = {"bytes": 10, "keys": ["gen_ai.input.messages"],
                                                        "storage": "inline", "values": None}
        self.assert_fails(mutate, "python-mac/direct:direct.no_content")

    def test_direct_feedback_returned_true(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.expectations["sessions"]["direct"]["quality"]["feedback"] = True
        self.assert_fails(mutate, "python-mac/direct:direct.quality")

    def test_direct_evaluation_stored(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.readback["direct"].evaluations = Session("direct").staging_evaluations()
        self.assert_fails(mutate, "python-mac/direct:direct.quality")

    def test_direct_exported_more_than_stored(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            traces = launch.expectations["sessions"]["direct"]["stats"]["after_shutdown"]["traces"]
            traces.update({"accepted": 5, "exported": 5})
        self.assert_fails(mutate, "python-mac/direct:direct.delivery")

    def test_collector_span_missing_at_staging(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.readback["collector"].spans["spans"] = [
                item for item in launch.readback["collector"].spans["spans"] if item["name"].startswith("invoke")
                or item["span_id"] != Session("collector").ids["load"]]
        self.assert_fails(mutate, "python-mac/collector:ids.spans")

    def test_secret_stored(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            values = span(launch, "collector", "run")["content"]["values"]
            values["gen_ai.input.messages"] = {"stringValue": SPEC["completed"]["prompt"]}
        found = []
        launch = build_launch()
        mutate(launch)
        found = failing(sc.check_launch(launch, SPEC))
        self.assertIn("python-mac/collector:content.redaction", found)
        self.assertIn("python-mac/collector:content.redacted", found)

    def test_evaluation_not_linked(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.readback["collector"].evaluations["items"].pop()
        self.assert_fails(mutate, "python-mac/collector:quality.linked")

    def test_evaluation_wrong_score(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.readback["collector"].evaluations["items"][1]["score_value"] = 0.5
        self.assert_fails(mutate, "python-mac/collector:quality.linked")

    def test_metric_ledger(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.sinks = [Session("collector").metrics_entry(1, ok=2)]
        self.assert_fails(mutate, "python-mac/collector:collector.metric_ledger")

    def test_aggregate_off_by_one(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.stats_after["total_executions_24h"] = 15
        self.assert_fails(mutate, "python-mac:aggregate.stats")

    def test_redeployed(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.deployments_after["Tinybird - CD / deploy-staging"] = "0badc0de"
        self.assert_fails(mutate, "python-mac:deployment.unchanged")

    def test_key_in_evidence(self) -> None:
        def mutate(launch: sc.Launch) -> None:
            launch.secret_hits = ["lane.log: the key"]
        self.assert_fails(mutate, "python-mac:evidence.secrets")


class DriverTest(unittest.TestCase):
    def test_staging_host_guard(self) -> None:
        good = {"XYBRID_STAGING_INGEST_URL": "https://staging.ingest.xybrid.dev",
                "XYBRID_STAGING_API_URL": "https://staging.api.xybrid.dev/"}
        self.assertEqual(run_staging.staging_urls(good),
                         ("https://staging.ingest.xybrid.dev", "https://staging.api.xybrid.dev"))
        for bad in ("https://ingest.xybrid.dev", "https://api.xybrid.dev", "https://staging.example.com",
                    "ftp://staging.api.xybrid.dev"):
            with self.subTest(bad=bad), self.assertRaises(run_staging.SetupError):
                run_staging.staging_urls({**good, "XYBRID_STAGING_API_URL": bad})

    def test_session_configs_keep_the_key_where_it_is_needed(self) -> None:
        names = sc.lane_sessions(SPEC, "rust-mac")
        sessions = run_staging.session_configs(names, "sk_test_key", "https://staging.ingest.xybrid.dev",
                                               "https://staging.api.xybrid.dev", "http://127.0.0.1:4318")
        self.assertEqual(list(sessions), names)
        with_key = sorted(name for name, values in sessions.items() if "api_key" in values)
        self.assertEqual(with_key, ["case-fallback", "direct"])
        ids = [values["trace_id"] for values in sessions.values()]
        self.assertEqual(len(set(ids)), len(ids))
        for values in sessions.values():
            self.assertRegex(values["trace_id"], "^[0-9a-f]{32}$")
            self.assertRegex(values["parent_span_id"], "^[0-9a-f]{16}$")
        public = sc.public_config({"sessions": sessions})
        self.assertNotIn("sk_test_key", json.dumps(public))

    def test_read_key_refuses_a_shared_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key"
            path.write_text("sk_test_key\n", encoding="utf-8")
            path.chmod(0o644)
            with self.assertRaises(run_staging.SetupError):
                run_staging.read_key(path)
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            self.assertEqual(run_staging.read_key(path), "sk_test_key")
            path.write_text("two keys\n", encoding="utf-8")
            with self.assertRaises(run_staging.SetupError):
                run_staging.read_key(path)

    def test_secret_scan_names_files_not_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clean.json").write_text("{}", encoding="utf-8")
            (root / "raw").mkdir()
            (root / "raw" / "leak.json").write_text('{"header": "Bearer sk_test_key"}', encoding="utf-8")
            hits = run_staging.scan_secrets(root, {"the key": "sk_test_key", "Bearer": "Bearer "})
            self.assertEqual(hits, ["raw/leak.json: the key", "raw/leak.json: Bearer"])
            self.assertNotIn("sk_test_key", json.dumps(hits))

    def test_lane_env_drops_otel_and_keys(self) -> None:
        env = run_staging.lane_env({"config": "/private/config.json"},
                                   {"PATH": "/bin", "OTEL_EXPORTER_OTLP_ENDPOINT": "x", "XYBRID_API_KEY": "k",
                                    "XYBRID_STAGING_API_KEY": "k", "XYBRID_STAGING_API_KEY_B": "k"})
        self.assertEqual(env, {"PATH": "/bin", "XYBRID_STAGING_CONFIG": "/private/config.json"})

    def test_private_config_is_owner_only_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "key"
            private = run_staging.private_dir(key_file, "s6-test")
            path = private / "config.json"
            run_staging.write_private(path, {"api_key": "sk_test_key"})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o700)
            run_staging.remove_private(private)
            self.assertFalse(private.exists())

    def test_every_lane_has_a_known_binding_and_sessions(self) -> None:
        for lane, values in SPEC["lanes"].items():
            with self.subTest(lane=lane):
                self.assertIn(values["binding"], ("python", "rust", "swift", "kotlin", "react-native", "web",
                                                  "unity", "flutter"))
                self.assertIn(values["delivery"], ("pull", "push", "callback"))
                for session in sc.lane_sessions(SPEC, lane):
                    sc.session_kind(session)

    def test_lanes_script(self) -> None:
        script = STAGING / "lanes.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        listed = subprocess.run(["bash", str(script), "list"], check=True, capture_output=True, text=True,
                                env={**os.environ, "XYBRID_MAC_LAN_IP": "127.0.0.1"}).stdout.split()
        for lane in listed:
            # The CLI lanes check --trace-export files (export-check), not a scenario.
            if lane not in ("cli-mac", "cli-legacy-mac"):
                self.assertIn(lane, SPEC["lanes"])
        self.assertIn("python-mac", listed)
        self.assertIn("cli-mac", listed)


if __name__ == "__main__":
    unittest.main()
