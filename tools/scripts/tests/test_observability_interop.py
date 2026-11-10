"""Tests for the observability interop harness (tools/scripts/observability_interop).

The harness drives real producers through the pinned Collector into Langfuse,
Phoenix and Xybrid and compares each backend's API answer with what the
Collector received. These tests pin everything that runs without a backend:

- the Collector pin, download verification and its self-metrics parsing;
- the guard that keeps the harness off production Xybrid;
- the per-workload split of the ground truth and the producers' reference lines;
- every destination normalizer and the table-driven checks, against fixture
  API responses (fixtures/api, shaped after each backend's API spec from a
  recorded ground truth), the recorded profile output (fixtures/wire) and
  single-fault mutations, each pinned to the check that must report it.
"""

import copy
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HARNESS = Path(__file__).resolve().parents[1] / "observability_interop"
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HARNESS.parent / "observability_capture"))

import check_capture  # noqa: E402
import destinations  # noqa: E402
import expectations  # noqa: E402
import otelcol  # noqa: E402
import run_interop  # noqa: E402
import workloads  # noqa: E402

FIXTURES = HARNESS / "fixtures"
REPO = HARNESS.parents[2]
PROFILES = REPO / "crates" / "xybrid-observability" / "collector"


def ground_truth(name, directory=FIXTURES / "ground-truth"):
    entries = [json.loads(line) for line in (directory / f"{name}.jsonl").read_text().splitlines() if line]
    refs = json.loads((FIXTURES / "ground-truth" / "refs.json").read_text()).get(name, {})
    return expectations.GroundTruth.build(workloads.WORKLOADS[name], entries, refs)


def api(destination, name):
    return json.loads((FIXTURES / "api" / destination / name).read_text())


def statuses(checks):
    return {check.name: check.status for check in checks}


def failed(checks):
    return sorted(check.name for check in checks if check.status == "fail")


class DestinationRun:
    """A destination's normalized answer for one workload, with helpers to mutate it."""

    def __init__(self, destination, workload):
        self.destination = destination
        self.truth = ground_truth(workload)
        self.workload = workload
        if destination == "langfuse":
            self.spans_payload = api("langfuse", f"observations-v2-{workload}.json")
            self.evaluations_payload = api("langfuse", "scores-v3-empty.json")
        elif destination == "phoenix":
            self.spans_payload = api("phoenix", f"spans-{workload}.json")
            self.evaluations_payload = api("phoenix", "span-annotations-empty.json")
        else:
            self.spans_payload = api("xybrid", f"spans-{workload}.json")
            self.evaluations_payload = api("xybrid", f"evaluations-{workload}.json")
            self.contents = {}
            for path in (FIXTURES / "api" / "xybrid").glob(f"content-{workload}-*.json"):
                content = json.loads(path.read_text())
                self.contents[content["span_id"]] = content

    def normalized(self):
        if self.destination == "langfuse":
            spans = destinations.langfuse_spans([self.spans_payload])
            evaluations = destinations.langfuse_scores([self.evaluations_payload])
        elif self.destination == "phoenix":
            spans = destinations.phoenix_spans([self.spans_payload])
            evaluations = destinations.phoenix_annotations([self.evaluations_payload], {})
        else:
            spans = destinations.xybrid_spans(self.spans_payload, self.contents)
            evaluations = destinations.xybrid_evaluations(self.evaluations_payload)
        return spans, evaluations

    def raw_text(self):
        parts = [self.spans_payload, self.evaluations_payload]
        parts += list(getattr(self, "contents", {}).values())
        return "\n".join(json.dumps(part, sort_keys=True) for part in parts)

    def checks(self, workload=None):
        spans, evaluations = self.normalized()
        truth = self.truth
        if workload is not None:
            truth = expectations.GroundTruth(workload, truth.entries, truth.capture, truth.refs)
        return expectations.destination_checks(self.destination, truth, spans, evaluations, self.raw_text())

    def items(self):
        if self.destination == "xybrid":
            return self.spans_payload["spans"]
        return self.spans_payload["data"]

    def item(self, span_id):
        for item in self.items():
            if self.destination == "phoenix":
                ident = item["context"]["span_id"]
            else:
                ident = item.get("span_id") or item.get("id")
            if ident == span_id:
                return item
        raise KeyError(span_id)

    def span_of_role(self, role):
        return next(span for span in self.truth.spans if expectations.role(span) == role)


# ---------------------------------------------------------------------------
# Collector pin and process helpers
# ---------------------------------------------------------------------------


class CollectorPinTest(unittest.TestCase):
    def test_lock_matches_collector_readme(self):
        lock = otelcol.load_lock()
        readme = (PROFILES / "README.md").read_text()
        self.assertIn(f"v{lock['version']}", readme)
        pinned = dict(re.findall(r"otelcol-contrib_[\d.]+_(\w+?)\.tar\.gz` \| sha256 `([0-9a-f]{64})`", readme))
        self.assertTrue(pinned, "collector/README.md lists no archive checksums")
        for platform_name, digest in pinned.items():
            self.assertEqual(lock["sha256"].get(platform_name), digest, platform_name)
        self.assertIn("darwin_arm64", lock["sha256"])

    def test_host_platform(self):
        self.assertEqual(otelcol.host_platform("Darwin", "arm64"), "darwin_arm64")
        self.assertEqual(otelcol.host_platform("Linux", "x86_64"), "linux_amd64")
        self.assertEqual(otelcol.host_platform("Linux", "aarch64"), "linux_arm64")
        with self.assertRaises(otelcol.CollectorError):
            otelcol.host_platform("Windows", "AMD64")

    def test_validation_sets_every_variable_a_profile_requires(self):
        # A ${env:X} without a default must be documented for its file, so a
        # traces-only setup never depends on an undeclared variable.
        for name, env in otelcol.PROFILE_ENV.items():
            text = (PROFILES / name).read_text()
            required = {match for match in re.findall(r"\$\{env:([A-Z0-9_]+)\}", text)}
            self.assertLessEqual(required, set(env), name)
        for combination in otelcol.PROFILE_COMBINATIONS:
            for name in combination:
                self.assertTrue((PROFILES / name).is_file(), name)

    def test_profiles_never_send_an_empty_bearer_or_go_to_production_from_the_harness(self):
        self.assertNotIn("Authorization", (PROFILES / "phoenix.yaml").read_text().split("exporters:", 1)[1]
                         .split("service:", 1)[0].replace("# No Authorization", ""))
        env = run_interop.destination_env("xybrid", run_interop.load_settings(environ={}),
                                          {"XYBRID_STAGING_API_KEY": "k"}, None)
        for key in ("XYBRID_TRACES_ENDPOINT", "XYBRID_LOGS_ENDPOINT"):
            self.assertTrue(env[key].startswith("https://staging.ingest.xybrid.dev/"), env[key])

    def test_harness_overlays_add_pipelines_without_touching_the_profile_ones(self):
        overlay = (HARNESS / "collector-harness.yaml").read_text()
        for pipeline in ("traces/interop_capture", "metrics/interop_capture", "logs/interop_capture"):
            self.assertIn(f"{pipeline}:", overlay)
        pipelines = overlay.split("\n  pipelines:\n", 1)[1]
        self.assertNotRegex(pipelines, r"\n    (traces|metrics|logs):\n")
        self.assertIn("127.0.0.1:${env:INTEROP_COLLECTOR_PORT}", overlay)
        configs = otelcol.run_configs("langfuse", sinks=True)
        self.assertEqual([path.name for path in configs],
                         ["langfuse.yaml", "sinks.yaml", "collector-sinks-json.yaml", "collector-harness.yaml"])
        self.assertIn("otlphttp/phoenix:", otelcol.wire_overlay("phoenix"))


class CollectorDownloadTest(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp)

    def archive(self, members):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name, data, mode in members:
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), mode
                tar.addfile(info, io.BytesIO(data))
        return buffer.getvalue()

    def lock_for(self, payload):
        import hashlib
        return {"version": "0.165.0", "url": "https://example.invalid/{version}/{platform}.tar.gz",
                "sha256": {"linux_amd64": hashlib.sha256(payload).hexdigest()}}

    def fake_binary(self):
        return b"#!/bin/sh\necho 'otelcol-contrib version 0.165.0'\n"

    def test_verified_archive_is_extracted_cached_and_deleted(self):
        payload = self.archive([("README.md", b"x", 0o644), ("otelcol-contrib", self.fake_binary(), 0o755)])
        lock = self.lock_for(payload)
        calls = []

        def download(url, destination):
            calls.append(url)
            destination.write_bytes(payload)

        binary = otelcol.ensure_collector(lock, self.temp, platform_name="linux_amd64", download=download,
                                          log=lambda message: None)
        self.assertEqual((binary.version, binary.source), ("0.165.0", "download"))
        self.assertEqual(calls, ["https://example.invalid/0.165.0/linux_amd64.tar.gz"])
        directory = self.temp / "0.165.0" / "linux_amd64"
        self.assertEqual(sorted(path.name for path in directory.iterdir()), ["archive.sha256", "otelcol-contrib"])
        again = otelcol.ensure_collector(lock, self.temp, platform_name="linux_amd64", download=download,
                                         log=lambda message: None)
        self.assertEqual(again.source, "cache")
        self.assertEqual(len(calls), 1)

    def test_checksum_mismatch_is_refused_and_leaves_nothing(self):
        payload = self.archive([("otelcol-contrib", self.fake_binary(), 0o755)])
        lock = self.lock_for(b"something else")
        with self.assertRaisesRegex(otelcol.CollectorError, "refusing"):
            otelcol.ensure_collector(lock, self.temp, platform_name="linux_amd64",
                                     download=lambda url, destination: destination.write_bytes(payload),
                                     log=lambda message: None)
        directory = self.temp / "0.165.0" / "linux_amd64"
        self.assertEqual(list(directory.iterdir()), [])

    def test_archive_without_a_regular_binary_is_refused(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            link = tarfile.TarInfo("otelcol-contrib")
            link.type, link.linkname = tarfile.SYMTYPE, "/bin/sh"
            tar.addfile(link)
        payload = buffer.getvalue()
        with self.assertRaisesRegex(otelcol.CollectorError, "no single regular"):
            otelcol.ensure_collector(self.lock_for(payload), self.temp, platform_name="linux_amd64",
                                     download=lambda url, destination: destination.write_bytes(payload),
                                     log=lambda message: None)

    def test_unpinned_platform_is_refused(self):
        with self.assertRaisesRegex(otelcol.CollectorError, "pins no"):
            otelcol.ensure_collector({"version": "0.165.0", "url": "x", "sha256": {}}, self.temp,
                                     platform_name="darwin_arm64", log=lambda message: None)


class CollectorMetricsTest(unittest.TestCase):
    SAMPLE = "\n".join([
        "# HELP otelcol_exporter_sent_spans_total Number of spans sent",
        '# TYPE otelcol_exporter_sent_spans_total counter',
        'otelcol_exporter_sent_spans_total{exporter="otlphttp/xybrid",service_name="otelcol-contrib"} 21',
        'otelcol_exporter_sent_spans_total{exporter="otlphttp/interop_capture"} 21',
        'otelcol_exporter_send_failed_spans_total{exporter="otlphttp/xybrid"} 2',
        'otelcol_exporter_sent_log_records_total{exporter="otlphttp/xybrid"} 8',
        'otelcol_receiver_accepted_spans_total{receiver="otlp",transport="http"} 21',
        'otelcol_process_uptime_total 3.5',
    ])

    def test_summary_per_exporter_and_receiver(self):
        summary = otelcol.summarize_metrics(otelcol.parse_prometheus(self.SAMPLE))
        self.assertEqual(summary["exporter:otlphttp/xybrid"],
                         {"sent_spans": 21.0, "failed_spans": 2.0, "sent_log_records": 8.0})
        self.assertEqual(summary["receiver:otlp"], {"accepted_spans": 21.0})

    def test_export_errors_are_found_in_the_log(self):
        log = ('2026-11-09T16:45:40Z\tinfo\tservice started\n'
               '2026-11-09T16:45:41Z\terror\tExporting failed. Dropping data.\t{"error": "401"}\n')
        self.assertEqual(len(otelcol.export_errors(log)), 1)

    def test_export_check_reads_the_destination_exporter(self):
        truths = {"rust": ground_truth("rust")}
        binary = otelcol.CollectorBinary(Path("x"), str(otelcol.load_lock()["version"]), "linux_amd64", None, "explicit")
        metrics = otelcol.summarize_metrics(otelcol.parse_prometheus(self.SAMPLE))
        checks = run_interop.collector_run_checks("xybrid", binary, truths, metrics, "", None)
        export = next(check for check in checks if check.name == "collector.export")
        self.assertEqual(export.status, "fail")
        self.assertTrue(any("failed_spans" in failure for failure in export.failures), export.failures)
        self.assertTrue(any("log records" in failure for failure in export.failures), export.failures)


# ---------------------------------------------------------------------------
# Guards, settings, workloads
# ---------------------------------------------------------------------------


class GuardTest(unittest.TestCase):
    def test_only_staging_or_loopback_xybrid_hosts(self):
        for url in ("https://staging.api.xybrid.dev", "https://staging.ingest.xybrid.dev/",
                    "http://127.0.0.1:8787", "http://localhost:8080"):
            self.assertEqual(run_interop.check_xybrid_target(url), url.rstrip("/"))
        for url in ("https://api.xybrid.dev", "https://ingest.xybrid.dev", "https://xybrid.dev",
                    "https://staging.example.com", "https://eu.api.xybrid.dev", "ftp://staging.api.xybrid.dev"):
            with self.assertRaises(run_interop.SetupError, msg=url):
                run_interop.check_xybrid_target(url)

    def test_production_settings_are_refused_before_any_request(self):
        settings = dict(run_interop.load_settings(environ={}), XYBRID_STAGING_INGEST_URL="https://ingest.xybrid.dev")
        with self.assertRaises(run_interop.SetupError):
            run_interop.destination_env("xybrid", settings, {"XYBRID_STAGING_API_KEY": "k"}, None)
        settings = dict(run_interop.load_settings(environ={}), XYBRID_STAGING_API_URL="https://api.xybrid.dev")
        with self.assertRaises(run_interop.SetupError):
            run_interop.make_client("xybrid", settings, run_interop.RawLog(), {"XYBRID_STAGING_API_KEY": "k"})

    def test_xybrid_needs_a_staging_key(self):
        settings = run_interop.load_settings(environ={})
        with self.assertRaisesRegex(run_interop.SetupError, "XYBRID_STAGING_API_KEY"):
            run_interop.destination_env("xybrid", settings, {}, None)
        wire = run_interop.destination_env("xybrid", settings, {}, "http://127.0.0.1:9")
        self.assertEqual(wire["XYBRID_LOGS_ENDPOINT"], "http://127.0.0.1:9/v1/logs")

    def test_settings_file_and_overrides(self):
        settings = run_interop.load_settings(environ={"PHOENIX_HOST_PORT": "26006", "UNRELATED": "x"})
        self.assertEqual(settings["PHOENIX_HOST_PORT"], "26006")
        self.assertEqual(settings["INTEROP_COLLECTOR_PORT"], "4318")
        self.assertEqual(settings["LANGFUSE_INIT_ORG_NAME"], "Xybrid interop")
        self.assertNotIn("UNRELATED", settings)
        langfuse = run_interop.destination_env("langfuse", settings, {}, None)
        self.assertEqual(langfuse["LANGFUSE_OTLP_ENDPOINT"],
                         f"http://127.0.0.1:{settings['LANGFUSE_HOST_PORT']}/api/public/otel")


class ComposeTest(unittest.TestCase):
    TEXT = (HARNESS / "compose.yaml").read_text()

    def test_every_image_is_pinned_by_digest(self):
        images = re.findall(r"image: (\S+)", self.TEXT)
        self.assertGreaterEqual(len(images), 7)
        for image in images:
            self.assertRegex(image, r"@sha256:[0-9a-f]{64}\}?$", image)
        for image in re.findall(r"LANGFUSE_(?:WEB|WORKER)_IMAGE=(\S+)", self.TEXT):
            self.assertRegex(image, r":3\.\d+\.\d+@sha256:[0-9a-f]{64}$")

    def test_published_ports_are_loopback_and_from_the_settings(self):
        settings = run_interop.load_settings(environ={})
        for port in re.findall(r"^\s+- (\S+:\S+:\d+)$", self.TEXT, re.M):
            self.assertTrue(port.startswith("127.0.0.1:"), port)
        variables = set(re.findall(r"\$\{([A-Z0-9_]+)\}", self.TEXT))
        self.assertLessEqual(variables, set(settings), variables - set(settings))

    def test_phoenix_runs_without_auth_and_langfuse_initializes_the_project(self):
        self.assertIn('PHOENIX_ENABLE_AUTH: "false"', self.TEXT)
        for key in ("ORG_ID", "PROJECT_ID", "PROJECT_PUBLIC_KEY", "PROJECT_SECRET_KEY", "USER_EMAIL",
                    "USER_PASSWORD"):
            self.assertIn(f"LANGFUSE_INIT_{key}: ${{LANGFUSE_INIT_{key}}}", self.TEXT)
        for name in run_interop.CONTAINERS["langfuse"] + run_interop.CONTAINERS["phoenix"]:
            self.assertIn(f"container_name: {name}", self.TEXT)

    @unittest.skipIf(shutil.which("bash") is None, "bash is not available")
    def test_scripts_parse_with_bash(self):
        for script in ("up.sh", "down.sh"):
            subprocess.run(["bash", "-n", str(HARNESS / script)], check=True)


class WorkloadTest(unittest.TestCase):
    def test_reference_lines(self):
        text = ("noise\nobservation refs: {\"workflow\": {\"trace_id\": \"AB\", \"span_id\": \"CD\"}}\n"
                "observation refs: {\"inference\": {\"trace_id\": \"aa\", \"span_id\": \"bb\"}, \"tool\": null, "
                "\"attempts\": [{\"trace_id\": \"aa\", \"span_id\": \"01\"}, null]}\n")
        self.assertEqual(workloads.parse_refs_line(text), {
            "inference": {"trace_id": "aa", "span_id": "bb"},
            "attempts.0": {"trace_id": "aa", "span_id": "01"},
        })
        self.assertEqual(workloads.parse_refs_line("observation refs: {not json"), {})
        self.assertEqual(workloads.parse_refs_line("nothing"), {})

    def test_examples_print_their_references(self):
        rust = (REPO / "crates/xybrid-observability/examples/standalone.rs").read_text()
        js = (REPO / "bindings/observability-js/examples/node-basic.ts").read_text()
        self.assertIn('println!("observation refs: {refs}")', rust)
        self.assertIn("observation refs: ${JSON.stringify(refs)}", js)
        quality = (HARNESS / "python_quality.py").read_text()
        self.assertIn('"observation refs: "', quality)

    def test_steps_carry_the_workload_attribute_and_endpoint(self):
        tools = workloads.Toolchain(python="python3", node="node", js_runtime="bun")
        for name, workload in workloads.WORKLOADS.items():
            steps = workloads.steps_for(workload, "http://127.0.0.1:4318", Path("/tmp/x"), tools)
            for step in steps:
                self.assertEqual(step.env["OTEL_RESOURCE_ATTRIBUTES"], f"xybrid.interop.workload={name}")
        metadata = workloads.steps_for(workloads.WORKLOADS["js-metadata"], "e", Path("/tmp/x"), tools)[0]
        self.assertEqual(metadata.env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT"], "no_content")
        python = workloads.steps_for(workloads.WORKLOADS["python"], "e", Path("/tmp/x"), tools)
        self.assertEqual(python[0].env["XYBRID_SCENARIO_ENDPOINT"], "e")
        self.assertEqual(len(python), 2)

    def test_split_by_attribute_then_service_then_time(self):
        def entry(received, resource, span_id):
            return {"seq": 1, "signal": "traces", "encoding": "json", "received_unix_nano": received, "body": {
                "resourceSpans": [{"resource": {"attributes": [
                    {"key": key, "value": {"stringValue": value}} for key, value in resource.items()]},
                    "scopeSpans": [{"spans": [{"traceId": "0" * 31 + "1", "spanId": span_id, "name": "s"}]}]}]}}

        entries = [
            entry(5, {"service.name": "observability-example", "xybrid.interop.workload": "js-metadata"}, "a" * 16),
            entry(15, {"service.name": "xybrid-observability-standalone"}, "b" * 16),
            entry(25, {"service.name": "observability-example"}, "c" * 16),
            entry(35, {"service.name": "observability-example"}, "d" * 16),
        ]
        windows = {"rust": (10, 19), "js": (20, 29), "js-metadata": (30, 39)}
        split = run_interop.assign_workloads(entries, ["rust", "js", "js-metadata"], windows)
        ids = {name: [span.span_id for span in check_capture.Capture.from_entries(found).spans]
               for name, found in split.items()}
        self.assertEqual(ids, {"rust": ["b" * 16], "js": ["c" * 16], "js-metadata": ["a" * 16, "d" * 16]})


# ---------------------------------------------------------------------------
# Ground truth and Collector-stage checks
# ---------------------------------------------------------------------------


class GroundTruthTest(unittest.TestCase):
    def test_recorded_workloads_pass_the_collector_checks(self):
        for name in workloads.DEFAULT_WORKLOADS:
            truth = ground_truth(name)
            checks = expectations.collector_checks(truth, [0])
            self.assertEqual(failed(checks), [], name)
            self.assertGreaterEqual(len(truth.evaluations()), 2, name)

    def test_roles_follow_the_platform_classification(self):
        roles = sorted(expectations.role(span) for span in ground_truth("rust").spans)
        self.assertEqual(roles, ["attempt", "attempt", "inference", "load", "tool", "workflow"])
        python = sorted(expectations.role(span) for span in ground_truth("python").spans)
        self.assertEqual(python.count("inference"), 3)
        self.assertIn("predict", python)

    def test_python_scenario_checks_hold_after_the_collector(self):
        truth = ground_truth("python")
        checks = run_interop.scenario_checks(truth, FIXTURES / "ground-truth" / "python-expectations.json")
        self.assertEqual(len(checks), len(check_capture.Checker.CHECKS))
        self.assertEqual(failed(checks), [])

    def test_a_secret_reaching_the_collector_fails(self):
        truth = ground_truth("rust")
        leaked = expectations.GroundTruth.build(
            workloads.WORKLOADS["rust"], json.loads(json.dumps(truth.entries).replace("[KEY]", "sk-demo-123")),
            truth.refs)
        self.assertEqual(failed(expectations.collector_checks(leaked, [0])), ["collector.redacted_at_source"])

    def test_unexported_reference_and_failed_producer(self):
        truth = ground_truth("js")
        truth.refs["inference"] = {"trace_id": truth.trace_ids()[0], "span_id": "f" * 16}
        self.assertEqual(failed(expectations.collector_checks(truth, [0, 1])),
                         ["producer.exit", "producer.refs_exported"])

    def test_sink_routing(self):
        truth = ground_truth("rust")
        evaluations_only = expectations.filter_entries(truth.entries, lambda resource: True)
        # A sink that saw exactly the Collector's metrics and logs passes for
        # Langfuse, which routes no log record to its destination...
        sink = expectations.GroundTruth.build(truth.workload, [
            {**entry, "body": {key: value for key, value in entry["body"].items() if key != "resourceSpans"}}
            for entry in evaluations_only if set(entry["body"]) - {"resourceSpans"}])
        self.assertEqual(failed(expectations.sink_checks(truth, sink, "langfuse")), [])
        # ...but for Xybrid the evaluation results must not be there.
        self.assertEqual(failed(expectations.sink_checks(truth, sink, "xybrid")), ["sinks.routing"])
        empty = expectations.GroundTruth.build(truth.workload, [])
        self.assertEqual(failed(expectations.sink_checks(truth, empty, "phoenix")), ["sinks.routing"])


# ---------------------------------------------------------------------------
# Destination normalizers and checks
# ---------------------------------------------------------------------------


class NormalizerTest(unittest.TestCase):
    def test_langfuse_observation(self):
        run = DestinationRun("langfuse", "rust")
        spans, _ = run.normalized()
        chat = next(span for span in spans if span.kind == "GENERATION")
        truth = run.span_of_role("inference")
        self.assertEqual((chat.trace_id, chat.span_id, chat.parent_span_id),
                         (truth.trace_id, truth.span_id, truth.parent_span_id))
        # Langfuse names the observation after the response model (request tiny-chat).
        self.assertEqual((chat.model, chat.provider), ("tiny-chat-q4", "fake-runtime"))
        self.assertEqual((chat.input_tokens, chat.output_tokens), (24.0, 7.0))
        # The profile's empty usage_details leaves attempts without usage.
        attempts = [span for span in spans if span.name and span.name.startswith("attempt")]
        self.assertTrue(attempts and all(span.input_tokens is None for span in attempts))
        # Metadata values are strings in Langfuse; the timing survives parsing.
        self.assertAlmostEqual(chat.first_chunk_s, truth.attributes[expectations.TTFC])
        self.assertAlmostEqual(chat.first_token_s, truth.attributes[expectations.TTFT])
        self.assertIsNone(chat.completion_start_offset_s)

    def test_lookup_handles_flat_nested_prefixed_and_json_string_forms(self):
        key = "gen_ai.response.time_to_first_chunk"
        for container in ({key: 1}, {"gen_ai": {"response": {"time_to_first_chunk": 1}}},
                          {"attributes." + key: 1}, {"attributes": json.dumps({key: 1})},
                          {"metadata": [{"attributes": {key: 1}}]}):
            self.assertEqual(destinations.lookup(container, key), 1, container)
        self.assertIsNone(destinations.lookup({"gen_ai": {"response": {}}}, key))

    def test_phoenix_nested_attributes(self):
        spans, _ = DestinationRun("phoenix", "rust").normalized()
        chat = next(span for span in spans if span.kind == "LLM")
        self.assertEqual((chat.model, chat.provider, chat.input_tokens), ("tiny-chat-q4", "fake-runtime", 24.0))
        attempts = [span for span in spans if span.name and span.name.startswith("attempt")]
        self.assertTrue(attempts and all(span.input_tokens is None for span in attempts))

    def test_xybrid_inline_and_object_content(self):
        run = DestinationRun("xybrid", "rust")
        spans, evaluations = run.normalized()
        tool = next(span for span in spans if span.kind == "genai.tool")
        self.assertIn("temperature_c", tool.output)
        self.assertEqual({evaluation.source for evaluation in evaluations}, {"evaluation", "feedback"})
        without = destinations.xybrid_spans(run.spans_payload, {})
        self.assertIsNone(next(span for span in without if span.kind == "genai.tool").output)

    def test_langfuse_v3_score_shapes(self):
        scores = destinations.langfuse_scores([{"data": [
            {"name": "relevance", "value": 1, "dataType": "NUMERIC", "source": "API",
             "subject": {"kind": "observation", "id": "AB", "traceId": "CD"}},
            {"name": "user_feedback", "value": "thumbs_up", "dataType": "CATEGORICAL",
             "traceId": "ef", "observationId": "01"}]}])
        self.assertEqual([(s.trace_id, s.span_id, s.score, s.label) for s in scores],
                         [("cd", "ab", 1.0, None), ("ef", "01", None, "thumbs_up")])


class DestinationChecksTest(unittest.TestCase):
    def assert_clean(self, run, skipped=()):
        checks = run.checks()
        self.assertEqual(failed(checks), [], [check.failures for check in checks if check.failures])
        self.assertEqual(sorted(check.name for check in checks if check.status == "skip"), sorted(skipped))
        return checks

    def test_fixtures_pass_on_every_destination(self):
        for destination in ("langfuse", "phoenix", "xybrid"):
            with self.subTest(destination=destination):
                self.assert_clean(DestinationRun(destination, "rust"), ["ids.remote_parent"])
                self.assert_clean(DestinationRun(destination, "python"),
                                  ["mapping.usage", "timing.first_chunk_vs_first_token"])

    def test_missing_span(self):
        run = DestinationRun("langfuse", "rust")
        run.spans_payload["data"].pop()
        # Every Rust span is also a reported reference.
        self.assertEqual(failed(run.checks()), ["ids.producer_refs", "ids.spans"])

    def test_changed_parent(self):
        run = DestinationRun("phoenix", "rust")
        attempt = run.span_of_role("attempt")
        run.item(attempt.span_id)["parent_id"] = run.span_of_role("workflow").span_id
        self.assertEqual(failed(run.checks()), ["ids.parents"])

    def test_dropped_remote_parent(self):
        run = DestinationRun("xybrid", "python")
        run.item(run.span_of_role("workflow").span_id)["parent_span_id"] = ""
        self.assertEqual(failed(run.checks()), ["ids.remote_parent"])

    def test_attempt_typed_as_generation_double_counts(self):
        run = DestinationRun("langfuse", "rust")
        run.item(run.span_of_role("attempt").span_id)["type"] = "GENERATION"
        self.assertEqual(failed(run.checks()), ["mapping.kind"])

    def test_attempt_with_llm_tokens_double_counts_in_phoenix(self):
        run = DestinationRun("phoenix", "rust")
        attempt = run.item(run.span_of_role("attempt").span_id)
        attempt["attributes"]["llm"] = {"token_count": {"prompt": 24, "completion": 7}}
        self.assertEqual(failed(run.checks()), ["mapping.usage"])

    def test_counted_attempt_in_xybrid(self):
        run = DestinationRun("xybrid", "rust")
        run.item(run.span_of_role("attempt").span_id)["counted_request"] = True
        self.assertEqual(failed(run.checks()), ["mapping.usage"])

    def test_a_missing_counted_flag_fails_where_the_table_promises_it(self):
        run = DestinationRun("xybrid", "rust")
        del run.item(run.span_of_role("inference").span_id)["counted_request"]
        self.assertEqual(failed(run.checks()), ["mapping.usage"])

    def test_a_missing_evaluation_source_fails_where_the_table_promises_it(self):
        run = DestinationRun("xybrid", "rust")
        del run.evaluations_payload["items"][0]["source"]
        self.assertEqual(failed(run.checks()), ["evaluations"])

    def test_model_and_provider(self):
        run = DestinationRun("phoenix", "rust")
        llm = run.item(run.span_of_role("inference").span_id)["attributes"]["llm"]
        llm["model_name"] = "tiny-chat"  # the request model: the profile prefers the response model
        llm["provider"] = "other"
        checks = run.checks()
        self.assertEqual(failed(checks), ["mapping.model_provider"])
        self.assertEqual(len(next(c for c in checks if c.name == "mapping.model_provider").failures), 2)

    def test_langfuse_stores_the_response_model(self):
        # Langfuse's extractModelName checks gen_ai.response.model before
        # gen_ai.request.model; a request-first observation fails.
        run = DestinationRun("langfuse", "rust")
        run.item(run.span_of_role("inference").span_id)["model"] = "tiny-chat"
        checks = run.checks()
        self.assertEqual(failed(checks), ["mapping.model_provider"])
        self.assertIn("expected 'tiny-chat-q4'",
                      next(c for c in checks if c.name == "mapping.model_provider").failures[0])

    def test_langfuse_provider_comes_from_the_profile_metadata(self):
        # gen_ai.provider.name is still under metadata.attributes: it must not
        # stand in for the profile's langfuse.observation.metadata.provider.
        run = DestinationRun("langfuse", "rust")
        item = run.item(run.span_of_role("inference").span_id)
        self.assertIn("gen_ai.provider.name", item["metadata"]["attributes"])
        del item["metadata"]["provider"]
        self.assertEqual(failed(run.checks()), ["mapping.model_provider"])

    def test_attempt_usage_double_counts_in_langfuse(self):
        # Langfuse sums usage over every observation of a trace, SPANs included.
        run = DestinationRun("langfuse", "rust")
        attempt = next(span for span in run.truth.spans if expectations.role(span) == "attempt"
                       and span.attributes.get("gen_ai.usage.input_tokens") is not None)
        run.item(attempt.span_id)["usageDetails"] = {"input": 24, "output": 7, "total": 31}
        checks = run.checks()
        self.assertEqual(failed(checks), ["mapping.usage"])
        self.assertIn("counted twice", next(c for c in checks if c.name == "mapping.usage").failures[0])
        # A zero count adds nothing to the totals.
        run.item(attempt.span_id)["usageDetails"] = {"input": 0, "output": 0, "total": 0}
        self.assertEqual(failed(run.checks()), [])

    def test_lost_error_status_and_wrong_error_type(self):
        run = DestinationRun("langfuse", "rust")
        failing = next(span for span in run.truth.spans if span.is_error)
        run.item(failing.span_id)["level"] = "DEFAULT"
        self.assertEqual(failed(run.checks()), ["mapping.errors"])
        run = DestinationRun("xybrid", "python")
        failing = next(span for span in run.truth.spans if span.is_error and span.attributes.get("error.type"))
        run.item(failing.span_id)["genai"]["error_type"] = "_OTHER"
        self.assertEqual(failed(run.checks()), ["mapping.errors"])

    def test_first_chunk_reported_as_first_token(self):
        run = DestinationRun("xybrid", "rust")
        genai = run.item(run.span_of_role("inference").span_id)["genai"]
        genai["time_to_first_token_s"] = genai["time_to_first_chunk_s"]
        self.assertEqual(failed(run.checks()), ["timing.first_chunk_vs_first_token"])

    def test_completion_start_derived_from_the_first_chunk(self):
        from datetime import timedelta
        run = DestinationRun("langfuse", "rust")
        truth = run.span_of_role("inference")
        item = run.item(truth.span_id)
        start = destinations.parse_time(item["startTime"])
        item["completionStartTime"] = (start + timedelta(seconds=truth.attributes[expectations.TTFC])
                                       ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        self.assertEqual(failed(run.checks()), ["timing.first_chunk_vs_first_token"])

    def test_missing_content_and_leaked_secret(self):
        run = DestinationRun("phoenix", "rust")
        del run.item(run.span_of_role("tool").span_id)["attributes"]["output"]
        self.assertEqual(failed(run.checks()), ["content"])
        run = DestinationRun("langfuse", "python")
        inference = run.item(run.span_of_role("inference").span_id)
        inference["output"] = inference["output"].replace("[scenario-redacted]", "xybrid-scenario-secret-4c1d")
        self.assertEqual(failed(run.checks()), ["content", "content.redaction"])

    def test_object_stored_content_is_read_through_the_content_route(self):
        run = DestinationRun("xybrid", "rust")
        run.contents = {}
        self.assertEqual(failed(run.checks()), ["content"])

    def test_metadata_only_workload(self):
        metadata = workloads.WORKLOADS["js-metadata"]
        rust_metadata = workloads.Workload(**{**metadata.__dict__, "name": "rust", "secrets": ("sk-demo-123",)})
        run = DestinationRun("xybrid", "rust")
        self.assertEqual(failed(run.checks(rust_metadata)), ["content"])
        for span in run.spans_payload["spans"]:
            span["content"] = None
        run.contents = {}
        for item in run.evaluations_payload["items"]:
            item["comment"] = None
        self.assertEqual(failed(run.checks(rust_metadata)), ["evaluations"])  # the producer did send a comment

    def test_evaluations_linked_or_dropped(self):
        run = DestinationRun("xybrid", "rust")
        run.evaluations_payload["items"].pop()
        self.assertEqual(failed(run.checks()), ["evaluations"])
        run = DestinationRun("xybrid", "python")
        run.evaluations_payload["items"][0]["source"] = "feedback" if run.evaluations_payload["items"][0][
            "name"] != "user_feedback" else "evaluation"
        self.assertEqual(failed(run.checks()), ["evaluations"])
        run = DestinationRun("langfuse", "rust")
        target = run.span_of_role("inference")
        run.evaluations_payload["data"].append({"name": "relevance", "value": 1, "observationId": target.span_id,
                                                "traceId": target.trace_id})
        self.assertEqual(failed(run.checks()), ["evaluations"])

    def test_other_runs_in_the_same_trace_are_ignored(self):
        run = DestinationRun("xybrid", "python")
        foreign = copy.deepcopy(run.spans_payload["spans"][0])
        foreign["span_id"], foreign["name"] = "0123456789abcdef", "from an earlier run"
        run.spans_payload["spans"].append(foreign)
        checks = run.checks()
        self.assertEqual(failed(checks), [])
        self.assertTrue(any("other spans" in note for note in checks[0].notes))

    def read_xybrid_with_an_earlier_run(self, run, carried):
        """Reads ``run`` back through the real Xybrid client, its trace also
        holding an earlier run's tool span (object-stored content) and quality
        result, each carrying ``carried``; returns the checks run_destination
        makes from that readback (destination_readback, its API branch)."""
        earlier_id = "0123456789abcdef"
        spans = copy.deepcopy(run.spans_payload)
        tool = next(span for span in spans["spans"] if (span.get("content") or {}).get("storage") == "object")
        earlier = copy.deepcopy(tool)
        earlier.update(span_id=earlier_id, name="from an earlier run",
                       status={"code": "ERROR", "message": f"rejected {carried}"})
        spans["spans"].append(earlier)
        contents = dict(run.contents)
        contents[earlier_id] = {**copy.deepcopy(contents[tool["span_id"]]), "span_id": earlier_id,
                                "values": {"gen_ai.tool.call.arguments": {"stringValue": carried}}}
        evaluations = copy.deepcopy(run.evaluations_payload)
        evaluations["items"].append({**evaluations["items"][-1], "span_id": earlier_id,
                                     "comment": f"earlier feedback {carried}"})
        raw = run_interop.RawLog()
        client = destinations.Xybrid("https://staging.api.xybrid.dev", "key", raw)

        def get(path, params=None, label="", raw_text=False):
            if path.endswith("/content"):
                value = contents[path.split("/")[-2]]
            elif path.endswith("/evaluations"):
                value = evaluations
            else:
                value = spans
            raw(label or path, path, value)  # what Client.get records for raw/
            return value

        client.client.get = get
        table = expectations.DESTINATIONS["xybrid"]
        from datetime import datetime, timezone
        with mock.patch.object(run_interop.time, "sleep"), mock.patch("sys.stdout", io.StringIO()):
            checks, read = run_interop.destination_readback("xybrid", run.truth, table, client, raw,
                                                            datetime.now(timezone.utc), 5)
        self.assertEqual(read["read_errors"], [])
        # The earlier run's span was read (and then ignored by the span checks).
        ids_spans = next(check for check in checks if check.name == "ids.spans")
        self.assertIn("1 other spans in the same trace(s) (earlier runs share a fixed trace id)", ids_spans.notes)
        self.assertEqual(read["spans_read"], len(run.truth.spans) + 1)
        return checks

    def test_an_earlier_runs_secret_in_the_same_trace_does_not_fail_this_run(self):
        # The Python scenario's trace id is fixed, so a persistent backend
        # (Xybrid staging) returns every earlier run's spans with this one's.
        run = DestinationRun("xybrid", "python")
        checks = self.read_xybrid_with_an_earlier_run(run, run.truth.workload.secrets[0])
        self.assertEqual(failed(checks), [], [check.failures for check in checks if check.failures])

    def test_an_earlier_runs_replacement_does_not_stand_in_for_this_runs(self):
        run = DestinationRun("xybrid", "python")
        replacement = run.truth.workload.replacements[0]

        def without_replacement(payload):
            return json.loads(json.dumps(payload).replace(replacement, "[lost]"))

        run.spans_payload = without_replacement(run.spans_payload)
        run.evaluations_payload = without_replacement(run.evaluations_payload)
        run.contents = {span_id: without_replacement(content) for span_id, content in run.contents.items()}
        checks = self.read_xybrid_with_an_earlier_run(run, replacement)
        content = next(check for check in checks if check.name == "content")
        self.assertIn(f"redaction replacements not found at the destination: {[replacement]}", content.failures)


class WireTest(unittest.TestCase):
    """The recorded profile output (what each backend receives) passes the same table."""

    def wire_checks(self, destination, wire_path, workload, truth=None):
        wire = [json.loads(line) for line in wire_path.read_text().splitlines() if line]
        truth = truth or ground_truth(workload)
        spans = [span for span in destinations.wire_spans(destination, wire) if span.trace_id in truth.trace_ids()]
        evaluations = destinations.wire_evaluations(wire) if destination == "xybrid" else []
        raw = "\n".join(json.dumps(span.raw, sort_keys=True) for span in spans)
        return expectations.destination_checks(destination, truth, spans, evaluations, raw)

    def test_langfuse_and_phoenix_profiles(self):
        for destination in ("langfuse", "phoenix"):
            with self.subTest(destination=destination):
                directory = FIXTURES / "wire" / f"{destination}-rust"
                entries = [json.loads(line) for line in (directory / "ground-truth.jsonl").read_text().splitlines()
                           if line]
                truth = expectations.GroundTruth.build(workloads.WORKLOADS["rust"], entries)
                checks = self.wire_checks(destination, directory / "wire.jsonl", "rust", truth)
                self.assertEqual(failed(checks), [], [check.failures for check in checks if check.failures])

    def test_xybrid_profile_sends_spans_and_only_evaluation_results(self):
        wire = FIXTURES / "wire" / "xybrid.jsonl"
        for name in workloads.DEFAULT_WORKLOADS:
            checks = self.wire_checks("xybrid", wire, name)
            self.assertEqual(failed(checks), [], name)
            self.assertEqual(statuses(checks)["evaluations"], "pass", name)
        entries = [json.loads(line) for line in wire.read_text().splitlines() if line]
        names = destinations.wire_log_event_names(entries)
        self.assertEqual(set(names), {expectations.EVALUATION_EVENT})
        self.assertEqual(len(names), sum(len(ground_truth(name).evaluations()) for name in workloads.DEFAULT_WORKLOADS))

    def test_langfuse_wire_follows_langfuse_model_and_usage_rules(self):
        attributes = {"gen_ai.request.model": "req", "gen_ai.response.model": "resp"}
        self.assertEqual(destinations.langfuse_wire_model(attributes), "resp")
        self.assertEqual(destinations.langfuse_wire_model({"gen_ai.request.model": "req"}), "req")
        self.assertEqual(destinations.langfuse_wire_model(
            {**attributes, "langfuse.observation.model.name": "explicit"}), "explicit")
        usage = {"gen_ai.usage.input_tokens": 3, "gen_ai.usage.output_tokens": 4}
        self.assertEqual(destinations.langfuse_wire_usage(usage), {"input": 3, "output": 4})
        self.assertEqual(destinations.langfuse_wire_usage({**usage, "langfuse.observation.usage_details": "{}"}), {})

    def test_langfuse_profile_without_the_attempt_usage_override_fails(self):
        directory = FIXTURES / "wire" / "langfuse-rust"
        text = (directory / "wire.jsonl").read_text()
        self.assertIn("langfuse.observation.usage_details", text)
        entries = [json.loads(line) for line in (directory / "ground-truth.jsonl").read_text().splitlines() if line]
        truth = expectations.GroundTruth.build(workloads.WORKLOADS["rust"], entries)
        with tempfile.TemporaryDirectory() as tmp:
            stripped = Path(tmp) / "wire.jsonl"
            wire = [json.loads(line) for line in text.splitlines() if line]
            for entry in wire:
                for resource in entry["body"].get("resourceSpans") or []:
                    for scope in resource["scopeSpans"]:
                        for span in scope["spans"]:
                            span["attributes"] = [item for item in span["attributes"]
                                                  if item["key"] != "langfuse.observation.usage_details"]
            stripped.write_text("".join(json.dumps(entry) + "\n" for entry in wire))
            self.assertEqual(failed(self.wire_checks("langfuse", stripped, "rust", truth)), ["mapping.usage"])

    def test_phoenix_wire_derives_openinference_attributes_like_phoenix(self):
        attributes = {"gen_ai.request.model": "req", "gen_ai.usage.input_tokens": 3, "gen_ai.usage.output_tokens": 4.0}
        stored = destinations.phoenix_wire_attributes(attributes)
        self.assertEqual((stored["llm.model_name"], stored["llm.token_count.prompt"],
                          stored["llm.token_count.completion"], stored["llm.token_count.total"]), ("req", 3, 4, 7))
        explicit = destinations.phoenix_wire_attributes({**attributes, "llm.token_count.prompt": 9,
                                                         "llm.model_name": "resp"})
        self.assertEqual((explicit["llm.token_count.prompt"], explicit["llm.model_name"]), (9, "resp"))
        embedding = destinations.phoenix_wire_attributes({"gen_ai.operation.name": "embeddings",
                                                          "gen_ai.request.model": "emb"})
        self.assertEqual(embedding.get("embedding.model_name"), "emb")
        self.assertNotIn("llm.model_name", embedding)
        self.assertEqual(destinations.phoenix_wire_attributes({"llm.token_count.prompt": 1}),
                         {"llm.token_count.prompt": 1})

    def test_phoenix_profile_keeping_attempt_usage_fails(self):
        # Phoenix derives llm.token_count.* from gen_ai.usage.* on any span
        # kind, so an attempt that reaches it with usage is counted twice.
        directory = FIXTURES / "wire" / "phoenix-rust"
        entries = [json.loads(line) for line in (directory / "ground-truth.jsonl").read_text().splitlines() if line]
        truth = expectations.GroundTruth.build(workloads.WORKLOADS["rust"], entries)
        wire = [json.loads(line) for line in (directory / "wire.jsonl").read_text().splitlines() if line]
        attempts = 0
        for entry in wire:
            for resource in entry["body"].get("resourceSpans") or []:
                for scope in resource["scopeSpans"]:
                    for span in scope["spans"]:
                        keys = {item["key"] for item in span["attributes"]}
                        if "xybrid.genai.attempt.index" in keys:
                            self.assertFalse(any(key.startswith("gen_ai.usage.") for key in keys), span["name"])
                            span["attributes"].append({"key": "gen_ai.usage.input_tokens", "value": {"intValue": "24"}})
                            attempts += 1
        self.assertGreater(attempts, 0)
        with tempfile.TemporaryDirectory() as tmp:
            kept = Path(tmp) / "wire.jsonl"
            kept.write_text("".join(json.dumps(entry) + "\n" for entry in wire))
            self.assertEqual(failed(self.wire_checks("phoenix", kept, "rust", truth)), ["mapping.usage"])

    def test_langfuse_wire_type_mapping(self):
        directory = FIXTURES / "wire" / "langfuse-rust"
        wire = [json.loads(line) for line in (directory / "wire.jsonl").read_text().splitlines() if line]
        kinds = sorted(span.kind for span in destinations.wire_spans("langfuse", wire))
        self.assertEqual(kinds, ["CHAIN", "GENERATION", "SPAN", "SPAN", "SPAN", "TOOL"])


class ClientTest(unittest.TestCase):
    def test_langfuse_falls_back_to_the_v1_api(self):
        client = destinations.Langfuse("http://127.0.0.1:1", "pk", "sk")
        observation = api("langfuse", "observations-v2-rust.json")["data"][0]
        calls = []

        def get(path, params=None, label="", raw_text=False):
            calls.append(path)
            if path == "/api/public/v2/observations":
                raise destinations.HttpError(404, path, "not found")
            return {"data": [observation], "meta": {"totalPages": 1}}

        client.client.get = get
        spans = client.spans(observation["traceId"], ("a", "b"))
        self.assertEqual([span.span_id for span in spans], [observation["id"]])
        self.assertEqual(calls, ["/api/public/v2/observations", "/api/public/observations"])
        self.assertEqual(client.api, "v1")

    def test_phoenix_keeps_only_the_asked_trace(self):
        client = destinations.Phoenix("http://127.0.0.1:1")
        payload = api("phoenix", "spans-rust.json")
        other = copy.deepcopy(payload["data"][0])
        other["context"]["trace_id"] = "f" * 32
        client.client.get = lambda path, params=None, label="", raw_text=False: {
            "data": payload["data"] + [other], "next_cursor": None}
        trace = payload["data"][0]["context"]["trace_id"]
        self.assertEqual(len(client.spans(trace, ("a", "b"))), len(payload["data"]))

    def test_xybrid_reads_object_content_and_treats_404_as_absent(self):
        client = destinations.Xybrid("https://staging.api.xybrid.dev", "key")
        payload = api("xybrid", "spans-rust.json")
        contents = {json.loads(path.read_text())["span_id"]: json.loads(path.read_text())
                    for path in (FIXTURES / "api" / "xybrid").glob("content-rust-*.json")}
        paths = []

        def get(path, params=None, label="", raw_text=False):
            paths.append(path)
            if path.endswith("/content"):
                return contents[path.split("/")[-2]]
            return payload

        client.client.get = get
        spans = client.spans(payload["trace_id"], ("a", "b"))
        self.assertTrue(any(path.endswith("/content") for path in paths))
        self.assertTrue(next(span for span in spans if span.kind == "genai.tool").output)

        def missing(path, params=None, label="", raw_text=False):
            raise destinations.HttpError(404, path, "")

        client.client.get = missing
        self.assertEqual(client.spans("a" * 32, ("a", "b")), [])
        self.assertEqual(client.evaluations("a" * 32, []), [])

    def test_langfuse_scores_fall_back_to_v2_and_fail_when_both_reads_do(self):
        client = destinations.Langfuse("http://127.0.0.1:1", "pk", "sk")
        calls = []

        def answering(statuses):
            def get(path, params=None, label="", raw_text=False):
                calls.append(path)
                status = statuses.get(path)
                if status:
                    raise destinations.HttpError(status, path, "")
                return {"data": [{"name": "relevance", "value": 1, "observationId": "ab", "traceId": "cd"}]}
            return get

        client.client.get = answering({"/api/public/v3/scores": 404})
        self.assertEqual([e.name for e in client.evaluations("cd", ["ab"])], ["relevance"])
        self.assertEqual(calls, ["/api/public/v3/scores", "/api/public/v2/scores"])
        # v3 refuses the `fields` query, v2 is gone: nothing was read.
        client.client.get = answering({"/api/public/v3/scores": 400, "/api/public/v2/scores": 404})
        with self.assertRaises(destinations.HttpError) as caught:
            client.evaluations("cd", ["ab"])
        self.assertEqual(caught.exception.status, 404)

    def test_a_redirect_is_a_failed_read_that_never_carries_the_key_elsewhere(self):
        import http.server
        import threading

        seen = []

        class Elsewhere(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(self.headers.get("Authorization"))
                body = b"{}"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        def serve(handler):
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(thread.join, 10)
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            return server

        elsewhere = serve(Elsewhere)
        target = f"http://127.0.0.1:{elsewhere.server_address[1]}/v1/telemetry/traces/x/spans"

        class Redirecting(Elsewhere):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", target)
                self.send_header("Content-Length", "0")
                self.end_headers()

        staging = serve(Redirecting)
        client = destinations.Xybrid(f"http://127.0.0.1:{staging.server_address[1]}", "staging-key")
        client.client.timeout = 10
        with self.assertRaises(destinations.HttpError) as caught:
            client.spans("x", ("a", "b"))
        self.assertEqual(caught.exception.status, 302)
        self.assertIn(f"redirect to {target} not followed", str(caught.exception))
        self.assertEqual(seen, [])


class ReadbackTest(unittest.TestCase):
    """A failed read never turns into a pass, and never crashes the run."""

    class FakeClient:
        def __init__(self, spans, span_failures=0, evaluations_fail=False):
            self.payload, self.span_failures, self.evaluations_fail = spans, span_failures, evaluations_fail

        def spans(self, trace_id, window):
            if self.span_failures:
                self.span_failures -= 1
                raise destinations.HttpError(0, "http://127.0.0.1:1/spans", "ConnectionResetError: reset")
            return [span for span in self.payload if span.trace_id == trace_id]

        def evaluations(self, trace_id, span_ids):
            if self.evaluations_fail:
                raise destinations.HttpError(503, "http://127.0.0.1:1/scores", "unavailable")
            return []

    def setUp(self):
        self.run = DestinationRun("langfuse", "rust")
        self.spans, _ = self.run.normalized()
        self.table = expectations.DESTINATIONS["langfuse"]
        from datetime import datetime, timezone
        self.started = datetime.now(timezone.utc)

    def read(self, client):
        with mock.patch.object(run_interop.time, "sleep"):
            return run_interop.read_back(client, self.run.truth, self.table, self.started, timeout=5)

    def test_a_failed_quality_read_is_not_read_as_dropped(self):
        result = self.read(self.FakeClient(self.spans, evaluations_fail=True))
        self.assertIsNone(result.evaluations)
        self.assertEqual(len(result.errors), 1)
        checks = expectations.destination_checks("langfuse", self.run.truth, result.spans, result.evaluations,
                                                 self.run.raw_text())
        self.assertEqual(statuses(checks)["evaluations"], "skip")
        # With a successful read the same answer is "dropped, as expected".
        result = self.read(self.FakeClient(self.spans))
        self.assertEqual((result.evaluations, result.errors), ([], []))

    def test_a_recovered_span_read_is_only_a_note(self):
        result = self.read(self.FakeClient(self.spans, span_failures=1))
        self.assertEqual(result.errors, [])
        self.assertTrue(any("ConnectionResetError" in note for note in result.notes))
        self.assertEqual(len(result.spans), len(self.spans))

    def test_a_langfuse_score_read_that_fails_on_both_endpoints_is_not_read_as_dropped(self):
        client = destinations.Langfuse("http://127.0.0.1:1", "pk", "sk")
        spans = self.spans

        def get(path, params=None, label="", raw_text=False):
            if path == "/api/public/v2/observations":
                trace = dict(params)["traceId"]
                return {"data": [span.raw for span in spans if span.trace_id == trace], "meta": {}}
            raise destinations.HttpError(400 if path == "/api/public/v3/scores" else 404, path, "")

        client.client.get = get
        result = self.read(client)
        self.assertIsNone(result.evaluations)
        self.assertEqual(len(result.errors), len(self.run.truth.trace_ids()))
        self.assertEqual(len(result.spans), len(self.spans))
        checks = expectations.destination_checks("langfuse", self.run.truth, result.spans, result.evaluations,
                                                 self.run.raw_text())
        self.assertEqual(statuses(checks)["evaluations"], "skip")

    def test_transport_errors_become_read_failures(self):
        client = destinations.Client("http://127.0.0.1:9", {}, timeout=2)
        with mock.patch.object(destinations.OPENER, "open",
                               side_effect=destinations.urllib.error.URLError(ConnectionRefusedError(61, "refused"))):
            with self.assertRaises(destinations.HttpError) as caught:
                client.get("/x")
        self.assertEqual(caught.exception.status, 0)
        for error in (TimeoutError("timed out"), ConnectionResetError(54, "reset"),
                      destinations.http.client.RemoteDisconnected("closed")):
            with mock.patch.object(destinations.OPENER, "open", side_effect=error):
                with self.assertRaises(destinations.HttpError):
                    client.get("/x")

        class Page(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch.object(destinations.OPENER, "open", return_value=Page(b"<html>proxy</html>")):
            with self.assertRaisesRegex(destinations.HttpError, "not JSON"):
                client.get("/x")

    def test_unreadable_collector_metrics_fail_the_xybrid_routing_proof(self):
        truths = {"rust": ground_truth("rust")}
        binary = otelcol.CollectorBinary(Path("x"), str(otelcol.load_lock()["version"]), "linux_amd64", None, "explicit")
        export = next(c for c in run_interop.collector_run_checks("xybrid", binary, truths, {}, "", None)
                      if c.name == "collector.export")
        self.assertEqual(export.status, "fail")
        export = next(c for c in run_interop.collector_run_checks("phoenix", binary, truths, {}, "", None)
                      if c.name == "collector.export")
        self.assertEqual(export.status, "skip")


class ProducerProcessTest(unittest.TestCase):
    def test_hosted_credentials_and_otel_variables_never_reach_a_producer(self):
        env = run_interop.producer_env({"OTEL_RESOURCE_ATTRIBUTES": "xybrid.interop.workload=rust"}, {
            "XYBRID_API_KEY": "k", "XYBRID_STAGING_API_KEY": "s", "XYBRID_PLATFORM_API_KEY": "p",
            "OTEL_EXPORTER_OTLP_HEADERS": "x", "PATH": "/bin"})
        self.assertEqual(env, {"OTEL_RESOURCE_ATTRIBUTES": "xybrid.interop.workload=rust", "PATH": "/bin"})

    @unittest.skipUnless(os.name == "posix", "process groups are POSIX")
    def test_a_timed_out_step_takes_its_children_with_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            # Like `cargo run`: a parent that waits on the program it started.
            script = ("import subprocess, sys, pathlib; "
                      "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                      f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); child.wait()")
            step = workloads.Step([sys.executable, "-c", script], Path(tmp), {}, "nested")
            code, _ = run_interop.run_step(step, Path(tmp) / "step.log", timeout=2)
            self.assertEqual(code, 124)
            child = int(pid_file.read_text())
            self.assertTrue(wait_until_gone(child), f"pid {child} outlived the timeout")


class CaptureServerStartupTest(unittest.TestCase):
    """A capture server that does not start never outlives the run, and
    neither do the ones started before it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.started = []

    def tracked_start(self, fail_name):
        original = run_interop.CaptureServer.start
        started = self.started

        def start(server, timeout=20.0):
            if server.out.stem == fail_name:
                raise run_interop.SetupError(f"capture server {server.out.name} did not start")
            started.append(server)
            self.addCleanup(server.stop)  # only matters when the harness leaked it
            return original(server, timeout)

        return start

    def test_a_server_past_its_deadline_is_stopped(self):
        # Alive, but never writes its ready file.
        stalled = self.dir / "stalled.py"
        stalled.write_text("import time\ntime.sleep(60)\n")
        server = run_interop.CaptureServer(self.dir, "stalled")
        self.addCleanup(server.stop)
        with mock.patch.object(run_interop, "CAPTURE_SERVER", stalled):
            with self.assertRaisesRegex(run_interop.SetupError, "did not start"):
                server.start(timeout=0.5)
        self.assertIsNotNone(server.process.poll(), "the stalled server is still running")

    def test_a_failed_start_stops_the_servers_already_running(self):
        args = run_interop.parse_args(["--destination", "phoenix", "--wire", "--workloads", "rust",
                                       "--out-dir", str(self.dir)])
        tools = workloads.Toolchain(python=sys.executable, node=None, js_runtime="node", cargo_release=False)
        with mock.patch.object(run_interop.CaptureServer, "start", self.tracked_start("sinks")), \
                mock.patch.object(run_interop.otelcol, "ensure_collector", return_value=mock.Mock()), \
                mock.patch.object(run_interop, "producer_provenance", return_value={}), \
                mock.patch.object(run_interop.workloads, "preflight", return_value=[]), \
                mock.patch.object(run_interop, "settle"), \
                mock.patch("sys.stdout", io.StringIO()):
            with self.assertRaisesRegex(run_interop.SetupError, "sinks.jsonl did not start"):
                run_interop.run_destination("phoenix", args, run_interop.load_settings(), tools)
        self.assertEqual([server.out.stem for server in self.started], ["ground-truth"])
        self.assertIsNotNone(self.started[0].process.poll(), "the ground-truth tee is still running")


def wait_until_gone(pid, limit=5.0):
    """True once ``pid`` no longer runs (gone or a zombie awaiting its reaper)."""
    import time
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split(") ")[-1].startswith("Z"):
            return True
        time.sleep(0.05)
    return False


class ReportTest(unittest.TestCase):
    def test_markdown_names_versions_and_every_check(self):
        checks = DestinationRun("phoenix", "rust").checks()
        report = {
            "destination": "phoenix", "mode": "api", "ok": True, "started_at": "s", "finished_at": "f",
            "summary": expectations.summarize(checks), "checks": [check.as_dict() for check in checks],
            "producers": {"git_sha": "abc123", "git_branch": "b", "runtimes": {"node": "v25"}},
            "collector": {"distribution": "otelcol-contrib", "version": "0.165.0", "configs": ["phoenix.yaml"]},
            "destination_info": {"product": "Arize Phoenix", "version": "21.0.0",
                                 "containers": {"xybrid-interop-phoenix": "img sha256:1"}},
            "workloads": {"rust": {"program": "p", "exit_codes": [0], "ground_truth": {}, "reported_refs": {}}},
        }
        text = run_interop.render_markdown(report)
        for needle in ("abc123", "otelcol-contrib 0.165.0", "Arize Phoenix 21.0.0", "xybrid-interop-phoenix",
                       "`timing.first_chunk_vs_first_token`", "What the destination keeps"):
            self.assertIn(needle, text)

    def test_every_destination_table_is_complete(self):
        keys = {"label", "profile", "kind", "model", "model_roles", "provider_roles", "usage", "counted", "error",
                "status_message", "error_type", "content_roles", "timing", "evaluations", "evaluation_source",
                "remote_parent", "maps", "drops"}
        for name, table in expectations.DESTINATIONS.items():
            self.assertEqual(set(table), keys, name)
            self.assertTrue((PROFILES / table["profile"]).is_file())
            self.assertIn(table["evaluations"], ("linked", "dropped"))
            # A destination that drops results cannot promise their source.
            self.assertFalse(table["evaluation_source"] and table["evaluations"] == "dropped", name)

    def test_cli_requires_a_destination_and_known_workloads(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            run_interop.parse_args([])
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            run_interop.parse_args(["--destination", "phoenix", "--workloads", "rust,cobol"])
        args = run_interop.parse_args(["--destination", "phoenix", "--workloads", "rust, python"])
        self.assertEqual(args.workloads, ["rust", "python"])
        self.assertTrue(args.sinks)
        self.assertIn("xybrid-interop", str(args.out_dir))


@unittest.skipUnless(os.environ.get("XYBRID_INTEROP_OTELCOL"), "set XYBRID_INTEROP_OTELCOL to a pinned otelcol-contrib")
class ValidateProfilesTest(unittest.TestCase):
    def test_every_profile_combination_validates(self):
        binary = otelcol.ensure_collector(explicit=Path(os.environ["XYBRID_INTEROP_OTELCOL"]))
        results = otelcol.validate_profiles(binary.path)
        self.assertTrue(all(result["ok"] for result in results), [r for r in results if not r["ok"]])


if __name__ == "__main__":
    unittest.main()
