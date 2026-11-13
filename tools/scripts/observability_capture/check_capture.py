#!/usr/bin/env python3
"""Checks one binding's observability scenario run against what was exported.

    python3 check_capture.py --capture capture.jsonl --expectations expectations.json

``capture.jsonl`` is what ``capture_server.py`` recorded; ``expectations.json``
is what the binding's scenario wrote (see ``scenario.md``). Every check prints
``PASS`` or ``FAIL`` with the reason; the exit status is 1 when any check
failed, 2 when an input cannot be read.

    python3 check_capture.py --ids python        # trace id T and parent span id P
    python3 check_capture.py --constants swift   # every per-binding value, as JSON

Runs on python3 >= 3.9 with no third-party packages.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "scenario.json"
SCHEMA = "xybrid-observability-scenario/1"
BINDING_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,12}$")
STATUS_ERROR = 2
SIGNAL_KEYS = ("traces", "logs", "metrics")
COUNTER_KEYS = ("accepted", "exported", "dropped", "failed", "pending")
MANUAL_REFERENCES = ("workflow", "tool", "inference", "inference_failed", "inference_released")
METRIC_KINDS = ("sum", "gauge", "histogram", "exponentialHistogram", "summary")
INFERENCE_DURATION = "gen_ai.client.inference.duration"
# OTLP AggregationTemporality DELTA, as a number or, in proto3 JSON, its enum name.
DELTA_TEMPORALITY = (1, "1", "AGGREGATION_TEMPORALITY_DELTA")


# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------


def load_spec(path: Path = SPEC_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_ids(binding: str) -> Tuple[str, str]:
    """The fixed trace id T and remote parent span id P of ``binding``.

    Hex of the UTF-8 bytes, right-padded with ``0``, so every language can
    derive them without a hash function.
    """
    if not BINDING_PATTERN.match(binding):
        raise ValueError(f"binding name {binding!r} must match {BINDING_PATTERN.pattern}")
    trace_id = ("xy:" + binding).encode("utf-8").hex().ljust(32, "0")[:32]
    parent_span_id = binding.encode("utf-8").hex().ljust(16, "0")[:16]
    return trace_id, parent_span_id


def binding_constants(spec: Dict[str, Any], binding: str) -> Dict[str, Any]:
    """Every value a binding's scenario uses, with the per-binding ones filled in."""
    trace_id, parent_span_id = scenario_ids(binding)
    values = {key: value.format(binding=binding) for key, value in spec["per_binding"].items()}
    return {
        "schema": spec["schema"],
        "binding": binding,
        "trace_id": trace_id,
        "parent_span_id": parent_span_id,
        "traceparent": f"00-{trace_id}-{parent_span_id}-01",
        **values,
        **{key: spec[key] for key in spec if key not in ("schema", "bindings", "ids", "per_binding")},
    }


def redacted(spec: Dict[str, Any], text: str) -> str:
    return text.replace(spec["secret"], spec["replacement"])


# ---------------------------------------------------------------------------
# Capture decoding
# ---------------------------------------------------------------------------


def normalize_id(value: Any, size: int) -> Optional[str]:
    """Lower-case hex of an OTLP/JSON id (hex per the spec; base64 tolerated)."""
    if not isinstance(value, str) or value == "":
        return None
    if len(value) == size * 2 and re.fullmatch(r"[0-9a-fA-F]+", value):
        return value.lower()
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return value.lower()
    return raw.hex() if len(raw) == size else value.lower()


def any_value(value: Any) -> Any:
    """Decodes an OTLP/JSON ``AnyValue``."""
    if not isinstance(value, dict) or not value:
        return None
    kind, inner = next(iter(value.items()))
    if kind == "intValue":
        return int(inner)
    if kind == "doubleValue":
        return float(inner)
    if kind == "arrayValue":
        return [any_value(item) for item in (inner or {}).get("values", [])]
    if kind == "kvlistValue":
        return attributes((inner or {}).get("values", []))
    return inner


def attributes(items: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, Any]:
    return {item["key"]: any_value(item.get("value")) for item in items or [] if "key" in item}


def status_code(status: Optional[Dict[str, Any]]) -> int:
    code = (status or {}).get("code", 0)
    if isinstance(code, str):
        return {"STATUS_CODE_UNSET": 0, "STATUS_CODE_OK": 1, "STATUS_CODE_ERROR": 2}.get(code, 0)
    return int(code or 0)


@dataclass
class Span:
    seq: int
    trace_id: Optional[str]
    span_id: Optional[str]
    parent_span_id: Optional[str]
    name: str
    attributes: Dict[str, Any]
    status_code: int
    status_message: str
    resource: Dict[str, Any]

    @property
    def is_error(self) -> bool:
        return self.status_code == STATUS_ERROR

    def label(self) -> str:
        return f"{self.name!r} ({self.span_id})"


@dataclass
class LogRecord:
    seq: int
    trace_id: Optional[str]
    span_id: Optional[str]
    event_name: Optional[str]
    attributes: Dict[str, Any]
    resource: Dict[str, Any]


@dataclass
class MetricPoint:
    """One OTLP/JSON metric data point: its attributes and, for histograms, its count."""

    seq: int
    name: str
    kind: str
    attributes: Dict[str, Any]
    count: Optional[int]
    delta: bool = False


def point_count(point: Dict[str, Any]) -> Optional[int]:
    """A histogram point's ``count`` (a uint64, so OTLP/JSON may quote it)."""
    try:
        return int(point["count"])
    except (KeyError, TypeError, ValueError):
        return None


def measurements(points: Iterable[MetricPoint]) -> int:
    """How many measurements histogram ``points`` hold across every export.

    A cumulative point (the Rust exporter's temporality) repeats every earlier
    measurement of its series, so only the series' latest export counts. Every
    delta point counts.
    """
    latest: Dict[Tuple[str, str], MetricPoint] = {}
    total = 0
    for point in points:
        if point.delta:
            total += point.count or 0
            continue
        series = (point.name, json.dumps(point.attributes, sort_keys=True, default=str))
        if series not in latest or point.seq >= latest[series].seq:
            latest[series] = point
    return total + sum(point.count or 0 for point in latest.values())


@dataclass
class Capture:
    entries: List[Dict[str, Any]] = field(default_factory=list)
    spans: List[Span] = field(default_factory=list)
    logs: List[LogRecord] = field(default_factory=list)
    metric_points: int = 0
    metric_data: List[MetricPoint] = field(default_factory=list)
    metric_names: List[str] = field(default_factory=list)
    resources: List[Tuple[str, int, Dict[str, Any]]] = field(default_factory=list)
    text: str = ""

    @classmethod
    def from_jsonl(cls, path: Path) -> "Capture":
        entries = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: not JSON ({error})") from None
        return cls.from_entries(entries)

    @classmethod
    def from_entries(cls, entries: List[Dict[str, Any]]) -> "Capture":
        capture = cls(entries=list(entries))
        texts = []
        for entry in capture.entries:
            seq = int(entry.get("seq", 0))
            body = entry.get("body")
            if body is not None:
                texts.append(json.dumps(body, ensure_ascii=False))
            if entry.get("body_base64"):
                texts.append(base64.b64decode(entry["body_base64"]).decode("utf-8", "replace"))
            if not isinstance(body, dict):
                continue
            for resource_spans in body.get("resourceSpans", []) or []:
                resource = attributes((resource_spans.get("resource") or {}).get("attributes"))
                capture.resources.append(("traces", seq, resource))
                for scope in resource_spans.get("scopeSpans", []) or []:
                    for span in scope.get("spans", []) or []:
                        status = span.get("status") or {}
                        capture.spans.append(Span(
                            seq=seq,
                            trace_id=normalize_id(span.get("traceId"), 16),
                            span_id=normalize_id(span.get("spanId"), 8),
                            parent_span_id=normalize_id(span.get("parentSpanId"), 8),
                            name=span.get("name", ""),
                            attributes=attributes(span.get("attributes")),
                            status_code=status_code(status),
                            status_message=status.get("message") or "",
                            resource=resource,
                        ))
            for resource_logs in body.get("resourceLogs", []) or []:
                resource = attributes((resource_logs.get("resource") or {}).get("attributes"))
                capture.resources.append(("logs", seq, resource))
                for scope in resource_logs.get("scopeLogs", []) or []:
                    for record in scope.get("logRecords", []) or []:
                        capture.logs.append(LogRecord(
                            seq=seq,
                            trace_id=normalize_id(record.get("traceId"), 16),
                            span_id=normalize_id(record.get("spanId"), 8),
                            event_name=record.get("eventName"),
                            attributes=attributes(record.get("attributes")),
                            resource=resource,
                        ))
            for resource_metrics in body.get("resourceMetrics", []) or []:
                resource = attributes((resource_metrics.get("resource") or {}).get("attributes"))
                capture.resources.append(("metrics", seq, resource))
                for scope in resource_metrics.get("scopeMetrics", []) or []:
                    for metric in scope.get("metrics", []) or []:
                        name = metric.get("name", "")
                        capture.metric_names.append(name)
                        for kind in METRIC_KINDS:
                            if kind in metric:
                                data = metric[kind] or {}
                                points = data.get("dataPoints", []) or []
                                delta = data.get("aggregationTemporality") in DELTA_TEMPORALITY
                                capture.metric_points += len(points)
                                capture.metric_data.extend(
                                    MetricPoint(seq=seq, name=name, kind=kind,
                                                attributes=attributes(point.get("attributes")),
                                                count=point_count(point), delta=delta)
                                    for point in points if isinstance(point, dict))
        capture.text = "\n".join(texts)
        return capture

    def spans_by_id(self) -> Dict[str, Span]:
        index: Dict[str, Span] = {}
        for span in self.spans:
            if span.span_id is not None:
                index.setdefault(span.span_id, span)
        return index


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    failures: List[str]
    notes: List[str]

    @property
    def ok(self) -> bool:
        return not self.failures


class Checker:
    """Runs every check; each returns failure messages, empty when it passes."""

    def __init__(self, capture: Capture, expectations: Dict[str, Any], spec: Dict[str, Any],
                 binding: Optional[str] = None) -> None:
        self.capture = capture
        self.expected = expectations
        self.spec = spec
        self.binding = binding or str(expectations.get("binding", ""))
        self.index = capture.spans_by_id()
        self.notes: List[str] = []
        try:
            self.constants = binding_constants(spec, self.binding)
        except ValueError:
            self.constants = None

    # -- helpers ----------------------------------------------------------

    def ref(self, name: str) -> Optional[Dict[str, Any]]:
        value = (self.expected.get("references") or {}).get(name)
        return value if isinstance(value, dict) else None

    def span(self, name: str) -> Optional[Span]:
        reference = self.ref(name)
        if reference is None:
            return None
        return self.index.get(normalize_id(reference.get("span_id"), 8) or "")

    def require_span(self, name: str) -> Span:
        span = self.span(name)
        if span is None:
            raise LookupError(f"the {name} reference has no exported span (see references.exported)")
        return span

    def stats(self, name: str) -> Dict[str, Any]:
        return (self.expected.get("stats") or {}).get(name) or {}

    def manual_span_ids(self) -> set:
        """Span ids of the manual operations; ``references.run`` is not one of them."""
        return {normalize_id(reference.get("span_id"), 8)
                for reference in map(self.ref, MANUAL_REFERENCES) if reference is not None}

    def workflow_children(self) -> List[Span]:
        workflow = self.span("workflow")
        if workflow is None:
            return []
        # A child must be in the workflow's own trace: a span that only names
        # the workflow as its parent from another trace is not its child.
        return [span for span in self.capture.spans
                if span.parent_span_id == workflow.span_id and span.trace_id == workflow.trace_id]

    def load_span(self) -> Optional[Span]:
        if self.ref("load") is not None:
            return self.span("load")
        name = f"load_model {self.spec['model']['model_id']}"
        workflow = self.span("workflow")
        matches = [span for span in self.capture.spans
                   if span.name == name and workflow is not None and span.parent_span_id == workflow.span_id
                   and span.trace_id == workflow.trace_id]
        return matches[0] if len(matches) == 1 else None

    def run_spans(self) -> List[Span]:
        """Workflow children other than the manual operations and the load.

        A run that returns a result reports its span as ``references.run``;
        ``check_automatic_run`` requires it to be the one span left here, so the
        run reference must not be excluded.
        """
        excluded = self.manual_span_ids()
        load = self.load_span()
        if load is not None:
            excluded.add(load.span_id)
        return [span for span in self.workflow_children() if span.span_id not in excluded]

    def asserted_span_count(self) -> int:
        count = len(MANUAL_REFERENCES) + 1  # the load
        if ((self.expected.get("automatic") or {}).get("run") or {}).get("outcome") == "error":
            count += 1
        return count

    # -- checks -----------------------------------------------------------

    def check_capture_decoded(self) -> List[str]:
        failures = []
        if not self.capture.entries:
            return ["the capture is empty: nothing reached the receiver (is the endpoint reachable "
                    "from the device? an Android emulator reaches the host as 10.0.2.2)"]
        for entry in self.capture.entries:
            if entry.get("signal") is None:
                failures.append(f"request {entry.get('seq')} went to {entry.get('path')!r}, not an OTLP signal path")
            elif entry.get("encoding") != "json":
                failures.append(f"request {entry.get('seq')} to {entry.get('path')} is {entry.get('encoding')}: "
                                f"{entry.get('error')} (the scenario sets the JSON protocol)")
        if not self.capture.spans:
            failures.append("no spans were exported")
        return failures

    def check_expectations(self) -> List[str]:
        failures = []
        if self.expected.get("schema") != SCHEMA:
            failures.append(f"schema is {self.expected.get('schema')!r}, expected {SCHEMA!r}")
        if self.constants is None:
            return failures + [f"binding {self.binding!r} must match {BINDING_PATTERN.pattern}"]
        if self.binding not in self.spec["bindings"]:
            self.notes.append(f"binding {self.binding!r} is not one of {self.spec['bindings']}")
        for key in ("trace_id", "parent_span_id"):
            if self.expected.get(key) != self.constants[key]:
                failures.append(f"{key} is {self.expected.get(key)!r}; {self.binding} derives "
                                f"{self.constants[key]!r} (scenario.md, Identifiers)")
        if self.expected.get("scenario_error"):
            failures.append(f"the scenario reported an error: {self.expected['scenario_error']}")
        required = {
            "references": MANUAL_REFERENCES,
            "automatic": ("load", "run"),
            "stats": ("before_release", "after_release", "after_flush", "after_shutdown"),
            "lifecycle": ("flush", "shutdown", "after_shutdown"),
        }
        for section, keys in required.items():
            value = self.expected.get(section)
            if not isinstance(value, dict):
                failures.append(f"expectations has no {section!r} object")
                continue
            for key in keys:
                if value.get(key) is None:
                    failures.append(f"expectations.{section}.{key} is missing")
        return failures

    def check_references_exported(self) -> List[str]:
        failures = []
        references = self.expected.get("references") or {}
        for name, value in sorted(references.items()):
            if value is None:
                continue
            if not isinstance(value, dict):
                failures.append(f"{name}: reference must be an object with trace_id and span_id")
                continue
            trace_id = normalize_id(value.get("trace_id"), 16)
            span_id = normalize_id(value.get("span_id"), 8)
            span = self.index.get(span_id or "")
            if span is None:
                failures.append(f"{name}: no exported span has span id {span_id}")
            elif span.trace_id != trace_id:
                failures.append(f"{name}: span {span_id} is in trace {span.trace_id}, the reference says {trace_id}")
        for name in MANUAL_REFERENCES:
            value = references.get(name)
            if isinstance(value, dict) and normalize_id(value.get("trace_id"), 16) != self.expected.get("trace_id"):
                failures.append(f"{name}: reference trace {value.get('trace_id')} is not T {self.expected.get('trace_id')}")
        return failures

    def check_workflow_parent_in(self) -> List[str]:
        span = self.require_span("workflow")
        failures = []
        if span.trace_id != self.expected.get("trace_id"):
            failures.append(f"workflow trace is {span.trace_id}, not T {self.expected.get('trace_id')}")
        if span.parent_span_id != self.expected.get("parent_span_id"):
            failures.append(f"workflow parent is {span.parent_span_id}, not P {self.expected.get('parent_span_id')}")
        expected_name = f"invoke_workflow {self.spec['workflow_name']}"
        if span.name != expected_name:
            failures.append(f"workflow span is named {span.name!r}, expected {expected_name!r}")
        if span.is_error or span.attributes.get("error.type") is not None:
            failures.append(f"workflow ended with an error: {span.attributes.get('error.type')!r} {span.status_message!r}")
        return failures

    def check_manual_children(self) -> List[str]:
        workflow = self.require_span("workflow")
        failures = []
        names = {
            "tool": f"execute_tool {self.spec['tool']['name']}",
            "inference": f"chat {self.spec['inference_request']['model']}",
            "inference_failed": f"chat {self.spec['inference_request']['model']}",
            "inference_released": f"chat {self.spec['inference_request']['model']}",
        }
        for name, expected_name in names.items():
            span = self.span(name)
            if span is None:
                failures.append(f"{name}: no exported span")
                continue
            if span.trace_id != workflow.trace_id:
                failures.append(f"{name}: trace {span.trace_id} differs from the workflow's {workflow.trace_id}")
            if span.parent_span_id != workflow.span_id:
                failures.append(f"{name}: parent {span.parent_span_id} is not the workflow {workflow.span_id}")
            if span.name != expected_name:
                failures.append(f"{name}: span is named {span.name!r}, expected {expected_name!r}")
        return failures

    def _json_attribute(self, span: Span, key: str, expected_text: str, failures: List[str], label: str) -> None:
        value = span.attributes.get(key)
        if value is None:
            failures.append(f"{label}: {key} is missing")
            return
        try:
            actual = json.loads(value)
        except (TypeError, ValueError):
            failures.append(f"{label}: {key} is not JSON: {value!r}")
            return
        expected = json.loads(redacted(self.spec, expected_text))
        if actual != expected:
            failures.append(f"{label}: {key} is {json.dumps(actual, sort_keys=True)}, "
                            f"expected {json.dumps(expected, sort_keys=True)}")

    def check_tool_content(self) -> List[str]:
        span = self.require_span("tool")
        failures = []
        tool = self.spec["tool"]
        self._json_attribute(span, "gen_ai.tool.call.arguments", tool["arguments_json"], failures, "tool")
        self._json_attribute(span, "gen_ai.tool.call.result", tool["result_json"], failures, "tool")
        if span.attributes.get("gen_ai.tool.call.id") != tool["call_id"]:
            failures.append(f"tool: gen_ai.tool.call.id is {span.attributes.get('gen_ai.tool.call.id')!r}")
        if span.is_error or span.attributes.get("error.type") is not None:
            failures.append(f"tool ended with an error: {span.attributes.get('error.type')!r}")
        return failures

    def check_inference_content(self) -> List[str]:
        span = self.require_span("inference")
        failures = []
        inference = self.spec["inference"]
        self._json_attribute(span, "gen_ai.input.messages", inference["input_messages_json"], failures, "inference")
        self._json_attribute(span, "gen_ai.output.messages", inference["output_messages_json"], failures, "inference")
        if self.spec["omitted_attribute"] in span.attributes:
            failures.append(f"inference: the omit rule did not drop {self.spec['omitted_attribute']}")
        if span.is_error or span.attributes.get("error.type") is not None:
            failures.append(f"inference ended with an error: {span.attributes.get('error.type')!r}")
        return failures

    def check_redaction(self) -> List[str]:
        failures = []
        secret, replacement = self.spec["secret"], self.spec["replacement"]
        if secret in self.capture.text:
            leaks = sorted({entry.get("path") for entry in self.capture.entries
                            if secret in json.dumps(entry.get("body"), ensure_ascii=False)
                            or secret in base64.b64decode(entry.get("body_base64") or "").decode("utf-8", "replace")})
            failures.append(f"the secret {secret!r} was exported unredacted (in {', '.join(map(str, leaks))})")
        if replacement not in self.capture.text:
            failures.append(f"the replacement {replacement!r} appears nowhere: the literal rule never ran")
        return failures

    def check_omit(self) -> List[str]:
        """The omitted attribute and its content reached the receiver nowhere.

        Absence alone cannot show that the omit rule removed them: CONTRACT.md
        lists a rule-omitted value nowhere, so a lane that never recorded the
        system instructions passes too. The rule is shown by the Python control
        run, which drops it and must export the marker
        (``bindings/python/tests/test_observability_scenario.py``), and by
        ``test_observability_capture.py``, which requires every lane's source to
        configure the rule and record the content.
        """
        failures = []
        key = self.spec["omitted_attribute"]
        carriers = [span.label() for span in self.capture.spans if key in span.attributes]
        carriers += [f"log {record.event_name}" for record in self.capture.logs if key in record.attributes]
        if carriers:
            failures.append(f"{key} was exported on {', '.join(carriers)}")
        if self.spec["omit_marker"] in self.capture.text:
            failures.append(f"the omitted content {self.spec['omit_marker']!r} was exported")
        return failures

    def check_resource_identity(self) -> List[str]:
        if self.constants is None:
            raise LookupError("no per-binding constants (see expectations.valid)")
        expected = {
            "device.id": self.constants["device_id"],
            "service.name": self.constants["service_name"],
            "service.version": self.constants["service_version"],
            "xybrid.sdk.binding": self.expected.get("sdk_binding") or self.binding,
        }
        failures = []
        if not self.capture.resources:
            return ["no resource was exported"]
        for signal, seq, resource in self.capture.resources:
            for key, value in expected.items():
                if resource.get(key) != value:
                    failures.append(f"{signal} request {seq}: resource {key} is {resource.get(key)!r}, expected {value!r}")
        return sorted(set(failures))

    def check_failed_inference(self) -> List[str]:
        span = self.require_span("inference_failed")
        failures = []
        failed = self.spec["failed_inference"]
        message = redacted(self.spec, failed["message"])
        if span.attributes.get("error.type") != failed["error_type"]:
            failures.append(f"error.type is {span.attributes.get('error.type')!r}, expected {failed['error_type']!r}")
        if not span.is_error:
            failures.append(f"status code is {span.status_code}, expected ERROR ({STATUS_ERROR})")
        if span.status_message != message:
            failures.append(f"status message is {span.status_message!r}, expected the redacted {message!r}")
        events = [record for record in self.capture.logs if record.span_id == span.span_id]
        if self.stats("after_flush").get("logs", {}).get("accepted", 0) == 0:
            failures.append("no log record was accepted: the OTLP preset exports the exception event as a log")
        elif not any(record.attributes.get("exception.type") == failed["error_type"] for record in events):
            failures.append(f"no gen_ai.client.operation.exception log carries exception.type {failed['error_type']!r} "
                            f"for span {span.span_id}")
        for record in events:
            if record.attributes.get("exception.message") not in (None, message):
                failures.append(f"exception.message is {record.attributes.get('exception.message')!r}, expected {message!r}")
        return failures

    def check_released_inference(self) -> List[str]:
        span = self.require_span("inference_released")
        failures = []
        if span.attributes.get("error.type") != "cancelled":
            failures.append(f"error.type is {span.attributes.get('error.type')!r}: releasing the last owner of "
                            f"unfinished work must end it as 'cancelled'")
        if not span.is_error:
            failures.append(f"status code is {span.status_code}, expected ERROR ({STATUS_ERROR}) for a cancellation")
        if any(record.span_id == span.span_id for record in self.capture.logs):
            failures.append("a cancellation emitted an exception log; CONTRACT.md: a cancellation is not an exception")
        # Metric points name no span, so this relies on the release being the scenario's only cancellation.
        cancelled = measurements(point for point in self.capture.metric_data
                                 if point.name == INFERENCE_DURATION
                                 and point.attributes.get("error.type") == "cancelled")
        if cancelled != 1:
            failures.append(f"{INFERENCE_DURATION} holds {cancelled} measurement(s) with error.type 'cancelled', "
                            f"expected 1: the release must record the operation's duration once, with the "
                            f"cancelled outcome as on the span, and nothing else in the scenario cancels")
        before = self.stats("before_release").get("traces", {}).get("accepted")
        after = self.stats("after_release").get("traces", {}).get("accepted")
        if before is None or after is None:
            failures.append("stats.before_release / after_release traces.accepted are missing")
        elif after != before + 1:
            failures.append(f"traces.accepted went {before} -> {after} across the release; releasing the last "
                            f"owner must end exactly that span, synchronously")
        pending = self.stats("after_release").get("traces", {}).get("pending")
        if pending is not None:
            self.notes.append(f"inference_released: {pending} span(s) pending right after the release "
                              f"(release ends the span; it does not flush)")
        return failures

    def check_automatic_load(self) -> List[str]:
        outcome = ((self.expected.get("automatic") or {}).get("load") or {}).get("outcome")
        workflow = self.require_span("workflow")
        failures = []
        if outcome not in ("ok", "error"):
            return [f"automatic.load.outcome is {outcome!r}, expected 'ok' or 'error'"]
        if outcome == "ok" and self.ref("load") is None:
            return ["the load succeeded but references.load is null: a loaded model reports load_observation_ref"]
        span = self.load_span()
        if span is None:
            return [f"no load_model {self.spec['model']['model_id']} span under the workflow"]
        if not span.name.startswith("load_model"):
            failures.append(f"load span is named {span.name!r}")
        if span.trace_id != workflow.trace_id or span.parent_span_id != workflow.span_id:
            failures.append(f"load span {span.span_id} is not a child of the workflow (parent {span.parent_span_id})")
        conversation = self.constants["conversation_id"] if self.constants else None
        if span.attributes.get("gen_ai.conversation.id") != conversation:
            failures.append(f"load gen_ai.conversation.id is {span.attributes.get('gen_ai.conversation.id')!r}, "
                            f"expected {conversation!r}")
        error_type = span.attributes.get("error.type")
        if outcome == "ok" and error_type is not None:
            failures.append(f"the load succeeded but its span has error.type {error_type!r}")
        if outcome == "error" and error_type is None:
            failures.append("the load failed but its span has no error.type")
        return failures

    def check_automatic_run(self) -> List[str]:
        automatic = self.expected.get("automatic") or {}
        load = (automatic.get("load") or {}).get("outcome")
        run = (automatic.get("run") or {}).get("outcome")
        runs = self.run_spans()
        if load == "error":
            if run != "skipped":
                return [f"the load failed, so automatic.run.outcome must be 'skipped', not {run!r}"]
            return [f"unexpected workflow children {[span.label() for span in runs]}"] if runs else []
        if run != "error":
            return [f"automatic.run.outcome is {run!r}: a run on the weightless model must fail"]
        if len(runs) != 1:
            return [f"expected exactly one run span under the workflow besides the manual operations "
                    f"and the load, got {[span.label() for span in runs]}"]
        span = runs[0]
        failures = []
        conversation = self.constants["conversation_id"] if self.constants else None
        if span.attributes.get("gen_ai.conversation.id") != conversation:
            failures.append(f"run {span.label()} gen_ai.conversation.id is "
                            f"{span.attributes.get('gen_ai.conversation.id')!r}, expected {conversation!r}")
        if span.attributes.get("gen_ai.request.model") != self.spec["model"]["model_id"]:
            failures.append(f"run gen_ai.request.model is {span.attributes.get('gen_ai.request.model')!r}")
        if not span.attributes.get("error.type"):
            failures.append(f"the run failed but {span.label()} has no error.type")
        if not span.is_error:
            failures.append(f"the run failed but {span.label()} status is not ERROR")
        if self.ref("run") is not None and normalize_id(self.ref("run").get("span_id"), 8) != span.span_id:
            failures.append(f"references.run {self.ref('run')} is not the run span {span.span_id}")
        return failures

    def check_trace_tree(self) -> List[str]:
        failures = []
        trace_id = self.expected.get("trace_id")
        workflow = self.span("workflow")
        foreign = sorted({span.name for span in self.capture.spans if span.trace_id != trace_id})
        if foreign:
            self.notes.append(f"spans outside T: {foreign}")
        for span in self.capture.spans:
            if span.trace_id != trace_id:
                continue
            if span.parent_span_id == self.expected.get("parent_span_id"):
                if workflow is None or span.span_id != workflow.span_id:
                    failures.append(f"{span.label()} hangs directly under P; only the workflow should")
            elif span.parent_span_id not in self.index:
                failures.append(f"{span.label()} has parent {span.parent_span_id}, which was never exported")
        duplicates = len(self.capture.spans) - len(self.index)
        if duplicates:
            self.notes.append(f"{duplicates} span(s) were received more than once (export retries)")
        return failures

    def check_stats(self) -> List[str]:
        failures = []
        for snapshot in ("before_release", "after_release", "after_flush"):
            stats = self.stats(snapshot)
            for signal in SIGNAL_KEYS:
                counters = stats.get(signal)
                if not isinstance(counters, dict) or any(key not in counters for key in COUNTER_KEYS):
                    failures.append(f"stats.{snapshot}.{signal} lacks {list(COUNTER_KEYS)}")
                    continue
                total = sum(counters[key] for key in COUNTER_KEYS[1:])
                if counters["accepted"] != total:
                    failures.append(f"stats.{snapshot}.{signal}: accepted {counters['accepted']} != exported + "
                                    f"dropped + failed + pending ({total})")
            if stats.get("enabled") is not True:
                failures.append(f"stats.{snapshot}.enabled is {stats.get('enabled')!r} while export is configured")
        flushed = self.stats("after_flush")
        traces = flushed.get("traces") or {}
        logs = flushed.get("logs") or {}
        metrics = flushed.get("metrics") or {}
        for signal, counters in (("traces", traces), ("logs", logs), ("metrics", metrics)):
            for key in ("dropped", "failed", "pending"):
                if counters.get(key, 0) != 0:
                    failures.append(f"after flush, {signal}.{key} is {counters.get(key)}; the receiver accepted everything")
        asserted = self.asserted_span_count()
        if traces.get("exported", 0) < asserted:
            failures.append(f"traces.exported {traces.get('exported')} < the {asserted} spans this scenario asserts")
        # The final counters (after shutdown) too: the receiver holds at least
        # what each snapshot reports exported.
        for snapshot in ("after_flush", "after_shutdown"):
            stats = self.stats(snapshot)
            held = (("traces", len(self.index), "spans"), ("logs", len(self.capture.logs), "records"),
                    ("metrics", self.capture.metric_points, "data points"))
            for signal, count, what in held:
                counters = stats.get(signal)
                exported = counters.get("exported", 0) if isinstance(counters, dict) else 0
                if count < exported:
                    prefix = "" if snapshot == "after_flush" else "after shutdown, "
                    failures.append(f"{prefix}{signal}.exported {exported} but the receiver holds {count} {what}")
        if metrics.get("exported", 0) == 0:
            failures.append("no metric data point was exported; flush collects metrics")
        return failures

    def check_lifecycle(self) -> List[str]:
        lifecycle = self.expected.get("lifecycle") or {}
        failures = []
        if lifecycle.get("flush") != "ok":
            failures.append(f"flush returned {lifecycle.get('flush')!r}")
        if lifecycle.get("shutdown") != ["ok", "ok"]:
            failures.append(f"shutdown twice returned {lifecycle.get('shutdown')!r}, expected ['ok', 'ok'] (idempotent)")
        after = lifecycle.get("after_shutdown") or {}
        if after.get("workflow_ref") is not None or after.get("workflow_context") is not None:
            failures.append(f"a workflow started after shutdown has a reference/context: {after}")
        if after.get("flush") != "ok":
            failures.append(f"flush after shutdown returned {after.get('flush')!r}; it must be a no-op")
        failures.extend(self.final_counter_failures())
        return failures

    def final_counter_failures(self) -> List[str]:
        """Stats after shutdown: export off, and the shut-down session's final
        counters, not a reset. Every item is settled in
        one terminal bucket, the healthy receiver lost nothing, and the
        counters only grew since the flush (`stats.consistent` holds them
        against what the receiver got)."""
        failures = []
        stats = self.stats("after_shutdown")
        flushed = self.stats("after_flush")
        if stats.get("enabled") is not False:
            failures.append(f"stats after shutdown report enabled={stats.get('enabled')!r}")
        for signal in SIGNAL_KEYS:
            counters = stats.get(signal)
            if not isinstance(counters, dict) or any(key not in counters for key in COUNTER_KEYS):
                failures.append(f"stats.after_shutdown.{signal} lacks {list(COUNTER_KEYS)}")
                continue
            settled = counters["exported"] + counters["dropped"] + counters["failed"]
            if counters["pending"] != 0:
                failures.append(f"after shutdown, {signal}.pending is {counters['pending']}; shutdown settles "
                                "every item")
            if counters["accepted"] != settled:
                failures.append(f"stats.after_shutdown.{signal}: accepted {counters['accepted']} != exported + "
                                f"dropped + failed ({settled})")
            for key in ("dropped", "failed"):
                if counters[key] != 0:
                    failures.append(f"after shutdown, {signal}.{key} is {counters[key]}; the receiver accepted "
                                    "everything")
            before = flushed.get(signal) if isinstance(flushed.get(signal), dict) else {}
            for key in ("accepted", "exported"):
                if counters[key] < before.get(key, 0):
                    failures.append(f"stats.after_shutdown.{signal}.{key} {counters[key]} < {before.get(key)} after "
                                    "the flush; they must be the shut-down session's final counters, not a reset")
        return failures

    def check_nothing_after_shutdown(self) -> List[str]:
        name = self.spec["after_shutdown_workflow_name"]
        leaked = [span.label() for span in self.capture.spans
                  if span.name == f"invoke_workflow {name}" or span.attributes.get("gen_ai.workflow.name") == name]
        return [f"the workflow started after shutdown was exported: {leaked}"] if leaked else []

    CHECKS: List[Tuple[str, Callable[["Checker"], List[str]]]] = [
        ("capture.decoded", check_capture_decoded),
        ("expectations.valid", check_expectations),
        ("references.exported", check_references_exported),
        ("workflow.parent_in", check_workflow_parent_in),
        ("manual.children", check_manual_children),
        ("tool.content", check_tool_content),
        ("inference.content", check_inference_content),
        ("content.redaction", check_redaction),
        ("content.omit", check_omit),
        ("resource.identity", check_resource_identity),
        ("inference.failed", check_failed_inference),
        ("inference.released", check_released_inference),
        ("automatic.load", check_automatic_load),
        ("automatic.run", check_automatic_run),
        ("trace.tree", check_trace_tree),
        ("stats.consistent", check_stats),
        ("lifecycle.shutdown", check_lifecycle),
        ("lifecycle.nothing_after_shutdown", check_nothing_after_shutdown),
    ]

    def run(self) -> List[Result]:
        results = []
        for name, check in self.CHECKS:
            self.notes = []
            try:
                failures = check(self)
            except (LookupError, KeyError, TypeError, ValueError, AttributeError) as error:
                failures = [f"could not run: {type(error).__name__}: {error}"]
            results.append(Result(name, failures, self.notes))
        return results


def report(results: List[Result], binding: str, capture: Capture) -> Dict[str, Any]:
    return {
        "binding": binding,
        "ok": all(result.ok for result in results),
        "checks": [{"name": r.name, "ok": r.ok, "failures": r.failures, "notes": r.notes} for r in results],
        "capture": {
            "requests": len(capture.entries),
            "spans": len(capture.spans),
            "log_records": len(capture.logs),
            "metric_points": capture.metric_points,
        },
    }


def print_results(results: List[Result], binding: str, capture: Capture, stream: Any = None) -> None:
    stream = stream or sys.stdout
    for result in results:
        print(f"{'PASS' if result.ok else 'FAIL'} {result.name}", file=stream)
        for failure in result.failures:
            print(f"     - {failure}", file=stream)
        for note in result.notes:
            print(f"     note: {note}", file=stream)
    failed = [result.name for result in results if not result.ok]
    print(f"{binding or '?'}: {len(results) - len(failed)}/{len(results)} checks passed; "
          f"{len(capture.entries)} requests, {len(capture.spans)} spans, {len(capture.logs)} log records, "
          f"{capture.metric_points} metric points", file=stream)
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=stream)


def check_files(capture_path: Path, expectations_path: Path, binding: Optional[str] = None,
                spec_path: Path = SPEC_PATH) -> Tuple[List[Result], Capture, str]:
    capture = Capture.from_jsonl(capture_path)
    expectations = json.loads(expectations_path.read_text(encoding="utf-8"))
    checker = Checker(capture, expectations, load_spec(spec_path), binding)
    return checker.run(), capture, checker.binding


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--capture", type=Path, help="capture_server.py JSONL output")
    parser.add_argument("--expectations", type=Path, help="the scenario's expectations.json")
    parser.add_argument("--binding", help="override the binding named in expectations.json")
    parser.add_argument("--report", type=Path, help="also write the results as JSON here")
    parser.add_argument("--spec", type=Path, default=SPEC_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--ids", metavar="BINDING", help="print T and P for BINDING and exit")
    parser.add_argument("--constants", metavar="BINDING", help="print BINDING's scenario values as JSON and exit")
    args = parser.parse_args(argv)

    if args.ids or args.constants:
        try:
            if args.ids:
                trace_id, parent_span_id = scenario_ids(args.ids)
                print(f"trace_id={trace_id} parent_span_id={parent_span_id} "
                      f"traceparent=00-{trace_id}-{parent_span_id}-01")
            else:
                print(json.dumps(binding_constants(load_spec(args.spec), args.constants), indent=2))
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        return 0
    if args.capture is None or args.expectations is None:
        parser.error("--capture and --expectations are required")
    try:
        results, capture, binding = check_files(args.capture, args.expectations, args.binding, args.spec)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print_results(results, binding, capture)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report(results, binding, capture), indent=2) + "\n", encoding="utf-8")
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
