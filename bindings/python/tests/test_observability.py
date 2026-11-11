"""Observation behaviour exported to a real OTLP/HTTP JSON receiver.

These run the compiled bridge end to end: the generated handles, the SDK
session the configuration installs, the exporter, and what lands on the wire.
`test_sdk.py` covers the Python surface; this file asserts the spans.
"""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

try:
    import xybrid
    import xybrid._bolt as bolt
except ImportError as exc:  # pragma: no cover - native artifacts not built
    pytest.skip(str(exc), allow_module_level=True)

TIMEOUT_MS = 5000
PRIVATE = "private-detail-do-not-export"


# The bridge holds the GIL while a native call such as `observability_flush`
# blocks, so the receiver runs in its own interpreter: an in-process server
# thread could not answer the export it is waiting for.
_RECEIVER = r"""
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

out = Path(sys.argv[1])
count = 0

# Doubles as an OpenAI-compatible gateway, so a cloud pipeline stage runs
# against loopback with no model and no external service.
COMPLETION = b'{"choices":[{"message":{"role":"assistant","content":"stage output"},"finish_reason":"stop"}]}'

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global count
        body = self.rfile.read(int(self.headers.get("content-length", "0")))
        count += 1
        signal = self.path.rstrip("/").rsplit("/", 1)[-1]
        (out / f"{count:04d}-{signal}.json").write_bytes(body)
        reply = COMPLETION if self.path.endswith("/chat/completions") else b"{}"
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
"""


class _Collector:
    """A local OTLP/HTTP receiver, in a child process, that keeps every body."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-c", _RECEIVER, str(directory)],
            stdout=subprocess.PIPE,
            text=True,
        )
        port = self.process.stdout.readline().strip()
        if not port:
            self.process.kill()
            raise RuntimeError("OTLP receiver did not start")
        self.endpoint = f"http://127.0.0.1:{port}"

    @property
    def raw(self) -> bytes:
        return b"".join(path.read_bytes() for path in sorted(self.directory.iterdir()))

    def spans(self) -> dict[str, dict]:
        bodies = [json.loads(path.read_bytes()) for path in sorted(self.directory.glob("*-traces.json"))]
        return {
            span["spanId"]: span
            for body in bodies
            for resource in body.get("resourceSpans", [])
            for scope in resource.get("scopeSpans", [])
            for span in scope.get("spans", [])
        }

    def close(self) -> None:
        self.process.kill()
        self.process.wait(timeout=10)
        self.process.stdout.close()


@pytest.fixture
def collector(tmp_path):
    exports = tmp_path / "otlp"
    exports.mkdir()
    receiver = _Collector(exports)
    try:
        yield receiver
    finally:
        # Leave the process-wide session off for the rest of the suite.
        bolt.observability_shutdown(TIMEOUT_MS)
        receiver.close()


def _start(collector: _Collector, configure=None) -> None:
    config = bolt.XybridObservabilityConfig.otlp(collector.endpoint)
    config.set_json_protocol(True)
    config.set_service("xybrid-python-test", None)
    if configure is not None:
        configure(config)
    config.init()


def _attribute(span: dict, key: str):
    for attribute in span.get("attributes", []):
        if attribute["key"] == key:
            value = attribute["value"]
            return next(iter(value.values())) if value else None
    return None


def _child_options(context: bolt.XybridObservationContext) -> bolt.XybridObservationOptions:
    return bolt.XybridObservationOptions(
        traceparent=context.traceparent,
        tracestate=context.tracestate,
        conversation_id=None,
        suppress_trace=False,
    )


def _tool(name: str, options=None) -> bolt.XybridToolObservation:
    return bolt.XybridToolObservation(
        bolt.XybridToolCallInfo(name, f"call-{name}", None, None, '{"query":1}'), options
    )


def _inference(options=None) -> bolt.XybridInferenceObservation:
    request = bolt.XybridInferenceRequest(
        "chat", "test-provider", "test-model", bolt.XybridExecutionLocation.LOCAL,
        None, None, False, None, None, None, None, None,
    )
    return bolt.XybridInferenceObservation(request, options)


def test_manual_handles_export_parented_spans_with_terminal_outcomes(collector):
    _start(collector)
    workflow = bolt.XybridWorkflowObservation("python-root", None)
    workflow_ref = workflow.observation_ref()
    context = workflow.context()
    assert workflow_ref is not None and context is not None
    assert context.traceparent.split("-")[1:3] == [workflow_ref.trace_id, workflow_ref.span_id]
    child = _child_options(context)

    with _tool("lookup", child) as tool:
        tool.record_result('{"count":0}')
        ok_ref = tool.observation_ref()

    failing = _tool("failing", child)
    failing_ref = failing.observation_ref()
    with pytest.raises(RuntimeError):
        with failing:
            raise RuntimeError(PRIVATE)

    interrupted = _inference(child)
    interrupted_ref = interrupted.observation_ref()
    with pytest.raises(KeyboardInterrupt):
        with interrupted:
            raise KeyboardInterrupt

    load = bolt.XybridModelLoadObservation(
        bolt.XybridModelLoadInfo("test-model", None, None, None, None), child
    )
    load_ref = load.observation_ref()
    load.record_cache_hit(True)
    load.close()  # released unfinished: cancels without flushing

    workflow.complete()
    workflow.fail(error_type="late_error", message=None)  # first terminal call wins
    workflow.close()

    bolt.observability_flush(TIMEOUT_MS)
    spans = collector.spans()

    root = spans[workflow_ref.span_id]
    assert root["name"] == "invoke_workflow python-root"
    assert _attribute(root, "error.type") is None
    for reference in (ok_ref, failing_ref, interrupted_ref, load_ref):
        span = spans[reference.span_id]
        assert span["traceId"] == workflow_ref.trace_id
        assert span["parentSpanId"] == workflow_ref.span_id

    ok = spans[ok_ref.span_id]
    assert ok["name"] == "execute_tool lookup"
    assert _attribute(ok, "error.type") is None
    assert _attribute(ok, "gen_ai.tool.call.result") == '{"count":0}'

    assert _attribute(spans[failing_ref.span_id], "error.type") == "application_error"
    assert PRIVATE.encode() not in collector.raw

    assert _attribute(spans[interrupted_ref.span_id], "error.type") == "cancelled"
    assert _attribute(spans[load_ref.span_id], "error.type") == "cancelled"

    stats = bolt.observability_stats()
    assert stats.enabled
    assert stats.traces.exported >= 5
    assert stats.traces.dropped == 0 and stats.traces.failed == 0


def test_explicit_parent_and_unsampled_parent_keep_references(collector):
    _start(collector)
    trace_id = "0af7651916cd43dd8448eb211c80319c"
    sampled = bolt.XybridObservationOptions(
        traceparent=f"00-{trace_id}-b7ad6b7169203331-01",
        tracestate=None, conversation_id="chat-42", suppress_trace=False,
    )
    with bolt.XybridWorkflowObservation("explicit", sampled) as workflow:
        sampled_ref = workflow.observation_ref()

    unsampled = bolt.XybridObservationOptions(
        traceparent=f"00-{trace_id}-b7ad6b7169203331-00",
        tracestate=None, conversation_id=None, suppress_trace=False,
    )
    with bolt.XybridWorkflowObservation("unsampled", unsampled) as hidden:
        hidden_ref = hidden.observation_ref()
        hidden_context = hidden.context()

    bolt.observability_flush(TIMEOUT_MS)
    spans = collector.spans()

    exported = spans[sampled_ref.span_id]
    assert exported["traceId"] == trace_id
    assert exported["parentSpanId"] == "b7ad6b7169203331"
    # An unsampled parent is honoured: a reference and child context still
    # exist, but nothing is exported for it.
    assert hidden_ref is not None and hidden_ref.trace_id == trace_id
    assert hidden_context is not None and hidden_context.traceparent.endswith("-00")
    assert hidden_ref.span_id not in spans


def test_content_rules_and_metadata_only_govern_exported_content(collector):
    def configure(config):
        config.add_literal_redaction("secret-value", "[redacted]")
        config.omit_content_attribute("gen_ai.tool.call.arguments")

    _start(collector, configure)
    with _tool("redacted") as tool:
        tool.record_result('{"token":"secret-value"}')
        redacted_ref = tool.observation_ref()
    bolt.observability_flush(TIMEOUT_MS)
    span = collector.spans()[redacted_ref.span_id]
    assert _attribute(span, "gen_ai.tool.call.result") == '{"token":"[redacted]"}'
    assert _attribute(span, "gen_ai.tool.call.arguments") is None
    assert b"secret-value" not in collector.raw

    # A later configuration replaces the session: metadata only drops content.
    _start(collector, lambda config: config.set_metadata_only(True))
    with _tool("metadata") as tool:
        tool.record_result('{"count":1}')
        metadata_ref = tool.observation_ref()
    bolt.observability_flush(TIMEOUT_MS)
    span = collector.spans()[metadata_ref.span_id]
    assert span["name"] == "execute_tool metadata"
    assert _attribute(span, "gen_ai.tool.call.result") is None


def test_handles_started_before_replacement_keep_their_session(collector):
    _start(collector)
    workflow = bolt.XybridWorkflowObservation("outlives-default", None)
    reference = workflow.observation_ref()
    # Replacing the default session must not orphan the outstanding handle:
    # the retired pipeline drains on a background worker once it is released.
    _start(collector, lambda config: config.set_service("replacement", None))
    workflow.complete()
    workflow.close()
    deadline = time.monotonic() + 10
    while reference.span_id not in collector.spans():
        assert time.monotonic() < deadline, "the retired session never exported the handle's span"
        time.sleep(0.05)
    assert collector.spans()[reference.span_id]["name"] == "invoke_workflow outlives-default"


def test_cancellation_errors_cancel_instead_of_failing(collector):
    import asyncio
    import concurrent.futures

    _start(collector)
    references = []
    for interruption in (asyncio.CancelledError, concurrent.futures.CancelledError, GeneratorExit):
        operation = bolt.XybridWorkflowObservation(interruption.__name__, None)
        references.append(operation.observation_ref())
        with pytest.raises(interruption):
            with operation:
                raise interruption()
    bolt.observability_flush(TIMEOUT_MS)
    spans = collector.spans()
    for reference in references:
        assert _attribute(spans[reference.span_id], "error.type") == "cancelled"


def test_delivery_loss_raises_telemetry_delivery_and_counts_per_signal():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    config = bolt.XybridObservabilityConfig.otlp(f"http://127.0.0.1:{closed_port}")
    config.set_json_protocol(True)
    config.init()
    try:
        with bolt.XybridWorkflowObservation("undeliverable", None):
            pass
        with pytest.raises(xybrid.TelemetryDelivery) as raised:
            bolt.observability_flush(TIMEOUT_MS)
        assert isinstance(raised.value, xybrid.XybridError)
        assert isinstance(raised.value.error, bolt.XybridErrorTelemetryDelivery)
        stats = bolt.observability_stats()
        assert stats.traces.accepted == 1
        assert stats.traces.failed == 1 and stats.traces.exported == 0
        assert stats.logs.accepted == 0
    finally:
        # The shutdown export fails too; only that it uninstalls matters here.
        with contextlib.suppress(xybrid.TelemetryDelivery):
            bolt.observability_shutdown(TIMEOUT_MS)
    final = bolt.observability_stats()
    assert not final.enabled
    # The final counters stay readable until the next init: the loss is
    # still visible after shutdown.
    assert (final.traces.accepted, final.traces.failed, final.traces.pending) == (1, 1, 0)


def _metadata_only_model(directory: Path) -> Path:
    """A model directory with metadata only: loads lazily, fails when run."""
    directory.mkdir()
    (directory / "model_metadata.json").write_text(json.dumps({
        "model_id": "python-metadata-only",
        "version": "1.0",
        "execution_template": {"type": "Onnx", "model_file": "model.onnx"},
        "files": [],
    }))
    return directory


def test_directory_load_and_run_take_explicit_parents(collector, tmp_path):
    _start(collector)
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    parent = bolt.XybridObservationOptions(
        traceparent=f"00-{trace_id}-00f067aa0ba902b7-01",
        tracestate=None, conversation_id="chat-load", suppress_trace=False,
    )
    model_dir = _metadata_only_model(tmp_path / "model")
    model = bolt.XybridModel.load(bolt.XybridModelSourceDirectory(str(model_dir)), parent)
    load_ref = model.load_observation_ref()
    assert load_ref is not None and load_ref.trace_id == trace_id

    run_parent = f"00-{trace_id}-1111111111111111-01"
    options = xybrid.XybridRunOptions(
        generation_config=None, abort_on=[], fallback_to_cloud=False, max_grace_tokens=0,
        correlation_id=None, traceparent=run_parent, tracestate=None,
        conversation_id="chat-run", suppress_trace=False, frame_session_id=None,
    )
    with pytest.raises(xybrid.XybridError):
        model.run(xybrid.XybridEnvelope.text("hello"), options)
    model.close()

    bolt.observability_flush(TIMEOUT_MS)
    spans = collector.spans()
    load = spans[load_ref.span_id]
    assert load["traceId"] == trace_id
    assert load["parentSpanId"] == "00f067aa0ba902b7"
    assert _attribute(load, "error.type") is None

    runs = [span for span in spans.values() if span.get("parentSpanId") == "1111111111111111"]
    assert len(runs) == 1, f"one logical run span under the explicit parent, got {[s['name'] for s in runs]}"
    run = runs[0]
    assert run["traceId"] == trace_id
    assert _attribute(run, "error.type") is not None
    assert _attribute(run, "gen_ai.conversation.id") == "chat-run"


def test_pipeline_workflow_and_stage_references_follow_the_explicit_parent(collector):
    _start(collector)
    pipeline = bolt.XybridPipeline.from_yaml(f"""
name: python-observed
stages:
  - id: draft
    model: loopback-model
    target: cloud
    provider: openai
    gateway_url: "{collector.endpoint}/v1"
    api_key: local-test-only
    timeout_ms: 5000
""")
    trace_id = "5bf92f3577b34da6a3ce929d0e0e4736"
    options = xybrid.XybridRunOptions(
        generation_config=None, abort_on=[], fallback_to_cloud=False, max_grace_tokens=0,
        correlation_id=None, traceparent=f"00-{trace_id}-2222222222222222-01",
        tracestate=None, conversation_id="chat-pipeline", suppress_trace=False,
        frame_session_id=None,
    )
    result = pipeline.run(xybrid.XybridEnvelope.text("original input"), options)
    assert result.text == "stage output"
    workflow_ref = result.observation
    stage_ref = result.stage("draft").observation
    assert workflow_ref is not None and stage_ref is not None
    assert workflow_ref.trace_id == stage_ref.trace_id == trace_id

    bolt.observability_flush(TIMEOUT_MS)
    spans = collector.spans()
    workflow = spans[workflow_ref.span_id]
    assert workflow["parentSpanId"] == "2222222222222222"
    stage = spans[stage_ref.span_id]
    assert stage["traceId"] == trace_id
    # The stage hangs under the workflow, possibly through a stage span.
    ancestors, current = [], stage
    while current.get("parentSpanId") in spans:
        current = spans[current["parentSpanId"]]
        ancestors.append(current["spanId"])
    assert workflow_ref.span_id in ancestors
    assert _attribute(stage, "error.type") is None
