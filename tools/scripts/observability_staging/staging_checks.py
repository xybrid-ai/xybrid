"""Checks of one staging lane launch, as pure functions of what was saved.

``run_staging.py`` collects the inputs (the lane's ``expectations.json``, the
public part of its config, every staging answer, the Collector's tee and sink
captures, the dashboard aggregate before and after, the platform deployments
before and after) and calls ``check_launch``; the unit tests call it on the
recorded fixture and on single-fault mutations of it. See ``README.md``,
"Checks", for what each check asserts.

Python 3.9+ standard library only.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
INTEROP = HERE.parent / "observability_interop"
CAPTURE = HERE.parent / "observability_capture"
for _path in (INTEROP, CAPTURE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import check_capture  # noqa: E402 -- sibling tool directories, imported after the path is set
import destinations  # noqa: E402
import expectations as interop  # noqa: E402
import workloads  # noqa: E402

SPEC_PATH = HERE / "staging.json"
SCHEMA = "xybrid-staging-scenario/1"
SESSIONS = ("direct", "collector")
EVALUATING_SESSIONS = ("collector", "direct-logs")
SIGNALS = ("traces", "logs", "metrics")
COUNTERS = ("accepted", "exported", "dropped", "failed", "pending")
REFERENCES = ("workflow", "load", "run")
REQUEST_TYPE = "genai.request"
INFERENCE_DURATION = "gen_ai.client.inference.duration"
CHUNK_COUNT = "xybrid.genai.response.chunk_count"
CONVERSATION_ID = "gen_ai.conversation.id"
EXECUTION_LOCATION = "xybrid.genai.execution.location"
CONTENT_KEYS = ("gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions",
                "gen_ai.tool.call.arguments", "gen_ai.tool.call.result", "gen_ai.tool.definitions",
                "xybrid.genai.feedback.comment")
ACCEPTANCE = ("run", "lane", "commit", "session")

Check = interop.Check


def load_spec(path: Path = SPEC_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def lane_sessions(spec: Mapping[str, Any], lane: str) -> List[str]:
    """The sessions a lane runs, in order (``staging.json`` lanes, else the default pair)."""
    return list((spec["lanes"].get(lane) or {}).get("sessions") or spec.get("sessions") or SESSIONS)


def session_kind(session: str) -> str:
    """``direct`` (preset, metadata only), ``collector``, ``direct-logs``
    (traces and logs straight to staging, content captured) or ``case``."""
    if session.startswith("case-"):
        return "case"
    if session in ("direct", "collector", "direct-logs"):
        return session
    raise ValueError(f"unknown session {session!r}")


def lane_constants(spec: Mapping[str, Any], lane: str, run: str) -> Dict[str, Any]:
    """The per-lane values a scenario stamps (README, Steps 1)."""
    if lane not in spec["lanes"]:
        raise ValueError(f"unknown lane {lane!r}; staging.json lists {sorted(spec['lanes'])}")
    binding = spec["lanes"][lane]["binding"]
    values = {"binding": binding, "lane": lane, "run": run}
    per_lane = {key: template.format(binding=binding, lane=lane, run=run, session="{session}")
                for key, template in spec["per_lane"].items()}
    return {**spec["lanes"][lane], **values, **per_lane}


# ---------------------------------------------------------------------------
# What staging answered for one session
# ---------------------------------------------------------------------------


@dataclass
class SessionReadback:
    """Raw answers for one trace: ``spans`` is the ``GET .../spans`` body
    (``None`` when staging answered 404), ``contents`` the ``.../content``
    body per span id, ``evaluations`` the ``GET .../evaluations`` body
    (``None`` on 404). ``errors`` are reads whose last attempt failed."""

    spans: Optional[Dict[str, Any]] = None
    contents: Dict[str, Any] = field(default_factory=dict)
    evaluations: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SessionReadback":
        return cls(spans=value.get("spans"), contents=dict(value.get("contents") or {}),
                   evaluations=value.get("evaluations"), errors=list(value.get("errors") or []))

    def as_dict(self) -> Dict[str, Any]:
        return {"spans": self.spans, "contents": self.contents, "evaluations": self.evaluations,
                "errors": self.errors}

    def stored(self) -> List[destinations.DestSpan]:
        if not self.spans:
            return []
        return destinations.xybrid_spans(self.spans, {key.lower(): value for key, value in self.contents.items()})

    def stored_evaluations(self) -> List[destinations.DestEvaluation]:
        return destinations.xybrid_evaluations(self.evaluations or {})

    def text(self) -> str:
        return "\n".join(json.dumps(part, sort_keys=True) for part in
                         [self.spans, self.evaluations, *self.contents.values()] if part is not None)


@dataclass
class Launch:
    """Everything one lane launch left behind."""

    config: Dict[str, Any]
    expectations: Dict[str, Any]
    readback: Dict[str, SessionReadback]
    ground_truth: List[Dict[str, Any]] = field(default_factory=list)
    sinks: List[Dict[str, Any]] = field(default_factory=list)
    stats_before: Optional[Dict[str, Any]] = None
    stats_after: Optional[Dict[str, Any]] = None
    deployments_before: Optional[Dict[str, Any]] = None
    deployments_after: Optional[Dict[str, Any]] = None
    lane_exit: int = 0
    secret_hits: Optional[List[str]] = None


def attrs(span: destinations.DestSpan) -> Dict[str, Any]:
    return destinations.attr_map((span.raw or {}).get("attributes"))


def resource(span: destinations.DestSpan) -> Dict[str, Any]:
    return destinations.attr_map(((span.raw or {}).get("resource") or {}).get("attributes"))


def genai(span: destinations.DestSpan) -> Dict[str, Any]:
    return dict((span.raw or {}).get("genai") or {})


def label(span: destinations.DestSpan) -> str:
    return f"{span.name!r} ({span.span_id})"


def number(value: Any) -> Optional[float]:
    return destinations.number(value)


def ref_ids(reference: Any) -> Optional[Dict[str, str]]:
    if not isinstance(reference, dict):
        return None
    trace_id = destinations.lower_hex(reference.get("trace_id"))
    span_id = destinations.lower_hex(reference.get("span_id"))
    if not trace_id or not span_id:
        return None
    return {"trace_id": trace_id, "span_id": span_id}


# ---------------------------------------------------------------------------
# Per-session checks
# ---------------------------------------------------------------------------


class SessionChecks:
    def __init__(self, launch: Launch, session: str, spec: Mapping[str, Any]) -> None:
        self.launch = launch
        self.session = session
        self.spec = spec
        self.name = f"{launch.config.get('lane')}/{session}"
        self.config = (launch.config.get("sessions") or {}).get(session) or {}
        self.expected = ((launch.expectations.get("sessions") or {}).get(session)) or {}
        self.readback = launch.readback.get(session) or SessionReadback()
        self.trace_id = destinations.lower_hex(self.config.get("trace_id")) or ""
        self.parent = destinations.lower_hex(self.config.get("parent_span_id")) or ""
        self.spans = [span for span in self.readback.stored() if span.trace_id == self.trace_id]
        self.by_id = {span.span_id: span for span in self.spans}
        self.constants = lane_constants(spec, str(launch.config.get("lane")), str(launch.config.get("run")))

    def new(self, name: str) -> Check:
        return Check("staging", self.name, name)

    def ref(self, name: str) -> Optional[Dict[str, str]]:
        return ref_ids((self.expected.get("references") or {}).get(name))

    def span(self, name: str) -> Optional[destinations.DestSpan]:
        reference = self.ref(name)
        return None if reference is None else self.by_id.get(reference["span_id"])

    def workflow_children(self) -> List[destinations.DestSpan]:
        workflow = self.span("workflow")
        if workflow is None:
            return []
        return [span for span in self.spans if span.parent_span_id == workflow.span_id]

    def requests(self) -> List[destinations.DestSpan]:
        return [span for span in self.workflow_children() if span.kind == REQUEST_TYPE]

    def conversation_id(self) -> str:
        return self.spec["per_lane"]["conversation_id"].format(run=self.launch.config.get("run"),
                                                               session=self.session)

    # -- checks ----------------------------------------------------------

    def expectations_valid(self) -> Check:
        check = self.new("expectations.valid")
        expected, config = self.launch.expectations, self.launch.config
        if expected.get("schema") != SCHEMA:
            check.failures.append(f"schema is {expected.get('schema')!r}, expected {SCHEMA!r}")
        for key in ("binding", "lane", "run"):
            if expected.get(key) != config.get(key):
                check.failures.append(f"{key} is {expected.get(key)!r}, the config says {config.get(key)!r}")
        if expected.get("scenario_error"):
            check.failures.append(f"the scenario reported an error: {str(expected['scenario_error'])[:600]}")
        if not self.expected:
            check.failures.append(f"expectations has no {self.session!r} session")
            return check
        for key, want in (("trace_id", self.trace_id), ("parent_span_id", self.parent)):
            if destinations.lower_hex(self.expected.get(key)) != want:
                check.failures.append(f"{key} is {self.expected.get(key)!r}, the config says {want!r}")
        if self.expected.get("init") != "ok":
            check.failures.append(f"init: {self.expected.get('init')!r}")
        for section in ("references", "stopped", "completed", "quality", "lifecycle", "stats"):
            if not isinstance(self.expected.get(section), dict):
                check.failures.append(f"sessions.{self.session}.{section} is missing")
        if self.readback.errors:
            check.failures.extend(f"read failed: {error}" for error in self.readback.errors)
        if self.readback.spans is None:
            check.failures.append(f"staging answered 404 for trace {self.trace_id}")
        check.notes.append(f"sdk {expected.get('sdk_version')}, model {expected.get('model_id')}, "
                           f"{len(self.spans)} spans stored in T {self.trace_id}")
        return check

    def refs_stored(self) -> Check:
        check = self.new("refs.stored")
        for name in REFERENCES:
            reference = self.ref(name)
            if reference is None:
                check.failures.append(f"{name}: no reference reported")
                continue
            if reference["trace_id"] != self.trace_id:
                check.failures.append(f"{name}: reference trace {reference['trace_id']} is not T {self.trace_id}")
            if reference["span_id"] not in self.by_id:
                check.failures.append(f"{name}: span {reference['span_id']} is not stored in T")
        return check

    def trace_rooted(self) -> Check:
        check = self.new("trace.rooted")
        workflow = self.span("workflow")
        if workflow is None:
            check.failures.append("no stored workflow span")
            return check
        if workflow.parent_span_id != self.parent:
            check.failures.append(f"workflow parent is {workflow.parent_span_id}, not P {self.parent}")
        if self.parent in self.by_id:
            check.failures.append(f"P {self.parent} is stored as a span; it is the caller's and never exported")
        under_parent = [span for span in self.spans if span.parent_span_id == self.parent]
        if [span.span_id for span in under_parent] != [workflow.span_id]:
            check.failures.append(f"spans under P: {[label(span) for span in under_parent]}, expected only "
                                  f"the workflow")
        for span in self.spans:
            if span is workflow:
                continue
            if span.parent_span_id not in self.by_id:
                check.failures.append(f"{label(span)}: parent {span.parent_span_id} is not a stored span of T")
        expected_name = f"invoke_workflow {self.spec['workflow_name']}"
        if workflow.name != expected_name:
            check.failures.append(f"workflow is named {workflow.name!r}, expected {expected_name!r}")
        if workflow.is_error:
            check.failures.append(f"workflow ended with an error: {workflow.error_type!r}")
        return check

    def identity(self) -> Check:
        check = self.new("identity")
        want = {
            "xybrid.sdk.binding": self.constants["binding"],
            "service.name": self.constants["service_name"],
            "service.version": self.constants["service_version"],
            "device.id": self.constants["device_id"],
            "xybrid.acceptance.run": self.launch.config.get("run"),
            "xybrid.acceptance.lane": self.launch.config.get("lane"),
            "xybrid.acceptance.commit": self.launch.config.get("commit"),
            "xybrid.acceptance.session": self.session,
        }
        if not self.spans:
            check.failures.append("no stored spans to check")
        wrong: Dict[str, set] = {}
        for span in self.spans:
            found = resource(span)
            for key, value in want.items():
                if found.get(key) != value:
                    wrong.setdefault(key, set()).add(repr(found.get(key)))
        for key, values in sorted(wrong.items()):
            check.failures.append(f"resource {key}: {sorted(values)} on some spans, expected {want[key]!r}")
        if self.spans:
            sample = resource(self.spans[0])
            check.notes.append(", ".join(f"{key}={sample.get(key)}" for key in (
                "xybrid.sdk.version", "telemetry.sdk.version", "os.type", "os.version", "host.arch")
                if sample.get(key) is not None))
        return check

    def load_span(self) -> Check:
        check = self.new("load.span")
        span, workflow = self.span("load"), self.span("workflow")
        if span is None or workflow is None:
            check.failures.append("no stored load or workflow span (see refs.stored)")
            return check
        if genai(span).get("operation_name") != "load_model" and attrs(span).get("gen_ai.operation.name") != "load_model":
            check.failures.append(f"{label(span)}: operation {genai(span).get('operation_name')!r}, expected load_model")
        if span.parent_span_id != workflow.span_id:
            check.failures.append(f"{label(span)}: parent {span.parent_span_id} is not the workflow")
        if span.is_error:
            check.failures.append(f"{label(span)}: ended with an error {span.error_type!r}")
        if span.kind != "genai.load_model":
            check.failures.append(f"{label(span)}: span type {span.kind!r}, expected genai.load_model")
        return check

    def run_mapping(self) -> Check:
        check = self.new("run.mapping")
        span, workflow = self.span("run"), self.span("workflow")
        completed = self.expected.get("completed") or {}
        if span is None or workflow is None:
            check.failures.append("no stored run or workflow span (see refs.stored)")
            return check
        found, typed = attrs(span), genai(span)
        if span.kind != REQUEST_TYPE or span.counted is not True:
            check.failures.append(f"{label(span)}: span type {span.kind!r}, counted {span.counted!r}; "
                                  f"expected a counted {REQUEST_TYPE}")
        if span.parent_span_id != workflow.span_id:
            check.failures.append(f"{label(span)}: parent {span.parent_span_id} is not the workflow")
        if typed.get("operation_name") not in interop.REQUEST_OPERATIONS:
            check.failures.append(f"operation {typed.get('operation_name')!r}")
        model = self.launch.expectations.get("model_id") or self.launch.config.get("model_id")
        if span.request_model != model:
            check.failures.append(f"request model {span.request_model!r}, expected {model!r}")
        if not span.provider:
            check.failures.append("no provider name")
        location = typed.get("execution_location") or found.get(EXECUTION_LOCATION)
        if location != "local":
            check.failures.append(f"execution location {location!r}, expected 'local'")
        if found.get(CONVERSATION_ID) != self.conversation_id():
            check.failures.append(f"conversation id {found.get(CONVERSATION_ID)!r}, expected {self.conversation_id()!r}")
        if span.is_error or span.error_type:
            check.failures.append(f"ended with an error: {span.error_type!r} {span.status_message!r}")
        if completed.get("outcome") != "ok":
            check.failures.append(f"the binding reported the completed stream as {completed.get('outcome')!r}")
        tokens = completed.get("tokens_out")
        if tokens is None:
            check.notes.append("the binding reports no output token count")
            if span.output_tokens is None:
                check.failures.append("no output token count stored")
        elif number(span.output_tokens) != number(tokens):
            check.failures.append(f"output tokens {span.output_tokens}, the binding reported {tokens}")
        chunks = number(found.get(CHUNK_COUNT))
        if chunks != number(completed.get("chunks")):
            check.failures.append(f"chunk count {found.get(CHUNK_COUNT)!r}, the binding received "
                                  f"{completed.get('chunks')!r}")
        first_chunk = span.first_chunk_s
        if first_chunk is None or first_chunk <= 0:
            check.failures.append(f"time to first chunk {first_chunk!r}, expected a positive value")
        check.notes.append(f"operation {typed.get('operation_name')}, provider {span.provider}, response model "
                           f"{span.response_model}, tokens {span.input_tokens}/{span.output_tokens}, chunks "
                           f"{found.get(CHUNK_COUNT)}, first chunk {first_chunk}s, first token "
                           f"{span.first_token_s}s, finish {completed.get('finish_reason')!r}")
        return check

    def stream_cancelled(self) -> Check:
        check = self.new("stream.cancelled")
        stopped = self.expected.get("stopped") or {}
        run = self.ref("run")
        others = [span for span in self.requests() if run is None or span.span_id != run["span_id"]]
        if len(others) != 1:
            check.failures.append(f"{len(others)} other requests under the workflow "
                                  f"({[label(span) for span in others]}), expected the stopped stream only")
            if not others:
                return check
        span = others[0]
        found = attrs(span)
        if span.error_type != "cancelled" or span.is_error is not True:
            check.failures.append(f"{label(span)}: error type {span.error_type!r}, error status {span.is_error!r}; "
                                  f"expected cancelled with status ERROR")
        if found.get(CONVERSATION_ID) != self.conversation_id():
            check.failures.append(f"conversation id {found.get(CONVERSATION_ID)!r}")
        if stopped.get("outcome") != "cancelled":
            check.failures.append(f"the binding reported the stopped stream as {stopped.get('outcome')!r}")
        consumed = stopped.get("chunks_consumed")
        limit = self.spec["stopped"]["max_tokens"]
        recorded = number(found.get(CHUNK_COUNT)) or 0
        if consumed != self.spec["stopped"]["stop_after_chunks"]:
            check.failures.append(f"the binding consumed {consumed!r} chunks, the scenario stops after "
                                  f"{self.spec['stopped']['stop_after_chunks']}")
        delivery = stopped.get("delivery")
        if delivery == "callback":
            if recorded != number(consumed):
                check.failures.append(f"chunk count {recorded:g}, the callback accepted {consumed}")
        elif delivery in ("pull", "push"):
            if consumed is None or not number(consumed) <= recorded < limit:
                check.failures.append(f"chunk count {recorded:g}, expected at least the {consumed} consumed and "
                                      f"fewer than max_tokens {limit}")
        else:
            check.failures.append(f"delivery {delivery!r}, expected callback, pull or push")
        check.notes.append(f"{delivery}: consumed {consumed}, span counts {recorded:g} chunks")
        return check

    def lifecycle_final(self) -> Check:
        check = self.new("lifecycle.final")
        lifecycle = self.expected.get("lifecycle") or {}
        for step in ("flush", "shutdown"):
            if lifecycle.get(step) != "ok":
                check.failures.append(f"{step}: {lifecycle.get(step)!r}")
        stats = (self.expected.get("stats") or {})
        for snapshot in ("after_flush", "after_shutdown"):
            values = stats.get(snapshot)
            if not isinstance(values, dict):
                check.failures.append(f"stats.{snapshot} is missing")
                continue
            for signal in SIGNALS:
                counters = values.get(signal) or {}
                if any(not isinstance(counters.get(key), int) for key in COUNTERS):
                    check.failures.append(f"{snapshot}.{signal}: counters {counters}")
                    continue
                if counters["accepted"] != sum(counters[key] for key in COUNTERS[1:]):
                    check.failures.append(f"{snapshot}.{signal}: accepted != exported + dropped + failed + pending "
                                          f"({counters})")
                if snapshot == "after_shutdown" and (counters["pending"] or counters["dropped"]
                                                     or counters["failed"]):
                    check.failures.append(f"after_shutdown.{signal}: {counters}")
        return check

    # -- direct --------------------------------------------------------------

    def direct_no_content(self) -> Check:
        check = self.new("direct.no_content")
        for span in self.spans:
            content = (span.raw or {}).get("content")
            if content and (content.get("values") or content.get("keys") or content.get("bytes")):
                check.failures.append(f"{label(span)}: content stored ({content.get('keys')}, "
                                      f"{content.get('bytes')} bytes)")
            present = [key for key in CONTENT_KEYS if key in attrs(span)]
            if present:
                check.failures.append(f"{label(span)}: content attributes {present}")
        text = self.readback.text()
        markers = [self.spec["secret"], self.spec["stopped"]["prompt"][:40], self.spec["completed"]["prompt"][-40:],
                   self.spec["feedback"]["comment"]]
        found = [marker for marker in markers if marker in text]
        if found:
            check.failures.append(f"content found in staging's answers for T: {found}")
        return check

    def direct_quality(self) -> Check:
        check = self.new("direct.quality")
        quality = self.expected.get("quality") or {}
        for call in ("feedback", "evaluation"):
            if quality.get(call) is not False:
                check.failures.append(f"{call} returned {quality.get(call)!r}; the Xybrid preset exports "
                                      f"traces only, so it must return false")
        stored = self.readback.stored_evaluations()
        if stored:
            check.failures.append(f"{len(stored)} evaluations stored for T: "
                                  f"{[item.name for item in stored]}")
        return check

    def direct_delivery(self) -> Check:
        check = self.new("direct.delivery")
        after = ((self.expected.get("stats") or {}).get("after_shutdown")) or {}
        for signal in ("logs", "metrics"):
            counters = after.get(signal) or {}
            if counters.get("exported"):
                check.failures.append(f"{signal}: {counters.get('exported')} exported; the preset sends traces only")
        traces = after.get("traces") or {}
        if traces.get("exported") != len(self.spans):
            check.failures.append(f"{traces.get('exported')} spans exported, {len(self.spans)} stored in T")
        check.notes.append(f"traces {traces}")
        return check

    # -- collector -----------------------------------------------------------

    def truth(self) -> interop.GroundTruth:
        run = self.launch.config.get("run")

        def mine(found: Mapping[str, Any]) -> bool:
            return (found.get("xybrid.acceptance.run") == run
                    and found.get("xybrid.acceptance.session") == self.session)

        workload = workloads.Workload(
            name=self.name, language=str(self.constants["binding"]), program=str(self.launch.config.get("lane")),
            service_name=str(self.constants["service_name"]), content="captured",
            secrets=(self.spec["secret"],), replacements=(self.spec["replacement"],),
            evaluations=2, streaming_timing=True)
        refs = {name: ref for name in REFERENCES if (ref := self.ref(name)) is not None}
        return interop.GroundTruth.build(workload, interop.filter_entries(self.launch.ground_truth, mine), refs)

    def sink(self, workload: workloads.Workload) -> interop.GroundTruth:
        run = self.launch.config.get("run")
        entries = interop.filter_entries(
            self.launch.sinks, lambda found: found.get("xybrid.acceptance.run") == run
            and found.get("xybrid.acceptance.session") == self.session)
        return interop.GroundTruth.build(workload, entries)

    def interop_checks(self) -> List[Check]:
        truth = self.truth()
        sink = self.sink(truth.workload)
        dest_spans = [span for span in self.readback.stored() if span.trace_id == self.trace_id]
        evaluations = self.readback.stored_evaluations() if self.readback.evaluations is not None else []
        checks = interop.collector_checks(truth, [self.launch.lane_exit])
        checks += interop.sink_checks(truth, sink, "xybrid")
        checks += interop.destination_checks("xybrid", truth, dest_spans, evaluations, self.readback.text())
        for check in checks:
            check.stage = f"interop:{check.stage}"
        return checks

    def collector_quality(self) -> Check:
        check = self.new("quality.linked")
        quality = self.expected.get("quality") or {}
        for call in ("feedback", "evaluation"):
            if quality.get(call) is not True:
                check.failures.append(f"{call} returned {quality.get(call)!r}, expected true")
        run = self.ref("run")
        stored = [item for item in self.readback.stored_evaluations()
                  if run is not None and item.span_id == run["span_id"] and item.trace_id == run["trace_id"]]
        others = [item for item in self.readback.stored_evaluations() if item not in stored]
        if others:
            check.failures.append(f"{len(others)} evaluations not linked to the run span: "
                                  f"{[(item.name, item.span_id) for item in others]}")
        wanted = {
            "user_feedback": (self.spec["feedback"]["score"], self.spec["feedback"]["label"], "feedback"),
            self.spec["evaluation"]["name"]: (self.spec["evaluation"]["score_value"],
                                              self.spec["evaluation"]["score_label"], "evaluation"),
        }
        names = sorted(str(item.name) for item in stored)
        if names != sorted(wanted):
            check.failures.append(f"linked to the run span: {names}, expected {sorted(wanted)}")
        for item in stored:
            if item.name not in wanted:
                continue
            score, score_label, source = wanted[item.name]
            if number(item.score) != number(score) or item.label != score_label:
                check.failures.append(f"{item.name}: score {item.score!r} label {item.label!r}, expected "
                                      f"{score} {score_label!r}")
            if item.source != source:
                check.failures.append(f"{item.name}: source {item.source!r}, expected {source!r}")
            if item.comment is not None and self.spec["secret"] in item.comment:
                check.failures.append(f"{item.name}: the comment carries the secret")
            if item.name == "user_feedback":
                check.notes.append(f"feedback comment stored as {item.comment!r}")
        return check

    def collector_metric_ledger(self) -> Check:
        check = self.new("collector.metric_ledger")
        truth = self.truth()
        sink = self.sink(truth.workload)
        points = [point for point in sink.capture.metric_data if point.name == INFERENCE_DURATION]
        total = check_capture.measurements(points)
        cancelled = check_capture.measurements(point for point in points
                                               if point.attributes.get("error.type") == "cancelled")
        runs = 2
        if total != runs:
            check.failures.append(f"{total} {INFERENCE_DURATION} measurements in the sink, expected {runs} "
                                  f"(one per run)")
        if cancelled != 1:
            check.failures.append(f"{cancelled} cancelled measurements, expected 1 (the stopped stream)")
        check.notes.append(f"{len(points)} points, {total} measurements, {cancelled} cancelled")
        return check

    def captured_content(self) -> Check:
        """The run span's stored content: the prompt with the secret replaced."""
        check = self.new("content.redacted")
        span = self.span("run")
        if span is None:
            check.failures.append("no stored run span (see refs.stored)")
            return check
        stored = span.input or ""
        if self.spec["replacement"] not in stored:
            check.failures.append(f"the run's stored input does not carry the replacement: {stored[:200]!r}")
        if self.spec["secret"] in self.readback.text():
            check.failures.append("the secret is stored in staging's answers for T")
        if "clear daytime sky" not in stored:
            check.failures.append("the run's stored input is not the completed prompt")
        if not span.output:
            check.failures.append("the run's output messages are not stored")
        return check

    def direct_logs_delivery(self) -> Check:
        check = self.new("direct_logs.delivery")
        after = ((self.expected.get("stats") or {}).get("after_shutdown")) or {}
        traces, logs = after.get("traces") or {}, after.get("logs") or {}
        if traces.get("exported") != len(self.spans):
            check.failures.append(f"{traces.get('exported')} spans exported, {len(self.spans)} stored in T")
        if not isinstance(logs.get("exported"), int) or logs["exported"] < 2:
            check.failures.append(f"logs exported {logs.get('exported')!r}, expected at least the two quality records")
        check.notes.append(f"traces {traces}, logs {logs}")
        return check

    def run(self) -> List[Check]:
        checks = [self.expectations_valid()]
        if not self.expected:
            return checks
        checks += [self.refs_stored(), self.trace_rooted(), self.identity(), self.load_span(),
                   self.run_mapping(), self.stream_cancelled(), self.lifecycle_final()]
        kind = session_kind(self.session)
        if kind == "direct":
            checks += [self.direct_no_content(), self.direct_quality(), self.direct_delivery()]
        elif kind == "collector":
            checks += self.interop_checks() + [self.captured_content(), self.collector_quality(),
                                               self.collector_metric_ledger()]
        else:
            checks += [self.captured_content(), self.collector_quality(), self.direct_logs_delivery()]
        return checks


# ---------------------------------------------------------------------------
# Rust-only cases (one Collector session each)
# ---------------------------------------------------------------------------


TASK = "xybrid.genai.task"
ATTEMPT_INDEX = "xybrid.genai.attempt.index"
ATTEMPT_REASON = "xybrid.genai.attempt.reason"


class CaseChecks(SessionChecks):
    """A ``case-*`` session: the shared identity, rooting and lifecycle checks,
    then the case's own (README, "Rust-only cases")."""

    def __init__(self, launch: Launch, session: str, spec: Mapping[str, Any]) -> None:
        super().__init__(launch, session, spec)
        self.case = session[len("case-"):]
        self.cases = spec.get("cases") or {}

    def descendants(self, root: Optional[destinations.DestSpan]) -> List[destinations.DestSpan]:
        if root is None:
            return []
        out, frontier = [], {root.span_id}
        while frontier:
            found = [span for span in self.spans if span.parent_span_id in frontier and span not in out]
            out += found
            frontier = {span.span_id for span in found}
        return out

    def case_requests(self) -> List[destinations.DestSpan]:
        return [span for span in self.descendants(self.span("workflow")) if span.kind == REQUEST_TYPE]

    def case_valid(self) -> Check:
        check = self.new("case.valid")
        if self.launch.expectations.get("scenario_error"):
            check.notes.append(f"scenario error: {str(self.launch.expectations['scenario_error'])[:300]}")
        if not self.expected:
            check.failures.append(f"expectations has no {self.session!r} session")
            return check
        if self.expected.get("init") != "ok":
            check.failures.append(f"init: {self.expected.get('init')!r}")
        if self.expected.get("outcome") != "ok":
            check.failures.append(f"the case reported {str(self.expected.get('outcome'))[:400]!r}")
        if self.readback.errors:
            check.failures.extend(f"read failed: {error}" for error in self.readback.errors)
        check.notes.append(f"{len(self.spans)} spans stored in T {self.trace_id}")
        return check

    def pipeline(self) -> Check:
        check = self.new("case.pipeline_on_token_failure")
        record = self.expected.get("pipeline") or {}
        consumed = record.get("chunks_consumed")
        if not str(record.get("outcome", "")).startswith("error"):
            check.failures.append(f"the pipeline returned {record.get('outcome')!r}; its on_token failure must fail it")
        if consumed != self.cases.get("pipeline_stop_after_chunks"):
            check.failures.append(f"on_token accepted {consumed!r} chunks, the case stops after "
                                  f"{self.cases.get('pipeline_stop_after_chunks')}")
        requests = self.case_requests()
        if len(requests) != 1:
            check.failures.append(f"{len(requests)} requests under the workflow, expected the pipeline's one LLM stage")
        for span in requests:
            if span.error_type != "cancelled" or span.is_error is not True:
                check.failures.append(f"{label(span)}: error type {span.error_type!r}; a refused token ends the "
                                      f"stream cancelled")
            if number(attrs(span).get(CHUNK_COUNT)) != number(consumed):
                check.failures.append(f"{label(span)}: chunk count {attrs(span).get(CHUNK_COUNT)!r}, on_token "
                                      f"accepted {consumed}")
        workflows = [span for span in self.descendants(self.span("workflow"))
                     if genai(span).get("operation_name") == "invoke_workflow"]
        check.notes.append(f"pipeline workflow spans {[label(span) for span in workflows]}; "
                           f"outcome {str(record.get('outcome'))[:200]}")
        return check

    def tts(self) -> Check:
        check = self.new("case.tts_streaming")
        completed, stopped = self.expected.get("completed") or {}, self.expected.get("stopped") or {}
        done, cut = self.span("completed"), self.span("stopped")
        if done is None or cut is None:
            check.failures.append(f"stored references: completed {done is not None}, stopped {cut is not None}")
            return check
        for span, task in ((done, "text_to_speech"), (cut, "text_to_speech")):
            if span.kind != REQUEST_TYPE or attrs(span).get(TASK) != task:
                check.failures.append(f"{label(span)}: type {span.kind!r}, task {attrs(span).get(TASK)!r}")
        if completed.get("outcome") != "completed" or done.is_error:
            check.failures.append(f"completed: outcome {completed.get('outcome')!r}, error {done.error_type!r}")
        if number(attrs(done).get(CHUNK_COUNT)) != number(completed.get("chunks")):
            check.failures.append(f"completed: chunk count {attrs(done).get(CHUNK_COUNT)!r}, the callback "
                                  f"received {completed.get('chunks')}")
        if stopped.get("outcome") != "cancelled" or cut.error_type != "cancelled" or cut.is_error is not True:
            check.failures.append(f"stopped: outcome {stopped.get('outcome')!r}, error type {cut.error_type!r}")
        if number(attrs(cut).get(CHUNK_COUNT)) != number(stopped.get("chunks_consumed")):
            check.failures.append(f"stopped: chunk count {attrs(cut).get(CHUNK_COUNT)!r}, the callback received "
                                  f"{stopped.get('chunks_consumed')}")
        check.notes.append(f"completed {attrs(done).get(CHUNK_COUNT)} chunks, first chunk {done.first_chunk_s}s; "
                           f"stopped after {attrs(cut).get(CHUNK_COUNT)}")
        return check

    def asr(self) -> Check:
        check = self.new("case.asr_batch_and_live")
        batch, live = self.expected.get("batch") or {}, self.expected.get("live") or {}
        span = self.span("batch")
        if span is None:
            check.failures.append("the batch run's reference is not stored")
        elif span.kind != REQUEST_TYPE or attrs(span).get(TASK) != "speech_to_text" or span.is_error:
            check.failures.append(f"batch {label(span)}: type {span.kind!r}, task {attrs(span).get(TASK)!r}, "
                                  f"error {span.error_type!r}")
        if not batch.get("text_chars"):
            check.failures.append("the batch run transcribed nothing")
        sessions = [item for item in self.descendants(self.span("workflow"))
                    if item.name == "invoke_workflow asr_session"]
        if len(sessions) != 1:
            check.failures.append(f"{len(sessions)} asr_session workflows, expected the one live utterance")
        for session in sessions:
            decoders = [item for item in self.spans if item.parent_span_id == session.span_id
                        and item.kind == REQUEST_TYPE]
            if not decoders or session.is_error or any(item.is_error for item in decoders):
                check.failures.append(f"asr_session {session.span_id}: {len(decoders)} decoder runs, errors "
                                      f"{[item.error_type for item in decoders if item.is_error] or session.error_type}")
            check.notes.append(f"live: {live.get('feeds')} feeds, {len(decoders)} decoder runs, "
                               f"{live.get('text_chars')} characters")
        if not live.get("text_chars"):
            check.failures.append("the live stream transcribed nothing")
        return check

    def frames(self) -> Check:
        check = self.new("case.frame_session")
        record = self.expected.get("frames") or {}
        runs = record.get("runs")
        if runs != self.cases.get("frames"):
            check.failures.append(f"{runs!r} frames ran, the case runs {self.cases.get('frames')}")
        requests = self.case_requests()
        traced = [ref for ref in record.get("run_references") or [] if isinstance(ref, dict)]
        if len(requests) != 1:
            check.failures.append(f"{len(requests)} traced runs stored, expected 1 per live trace interval")
        if len(traced) != 1 or (requests and traced and traced[0].get("span_id") != requests[0].span_id):
            check.failures.append(f"the runs returned {len(traced)} references; the traced one must be the stored run")
        truth = self.truth()
        sink = self.sink(truth.workload)
        points = [point for point in sink.capture.metric_data if point.name == INFERENCE_DURATION]
        measured = check_capture.measurements(points)
        if measured != runs:
            check.failures.append(f"{measured} {INFERENCE_DURATION} measurements in the sink, expected one per frame "
                                  f"({runs})")
        check.notes.append(f"{runs} frames, {len(requests)} traced, {measured} measured")
        return check

    def fallback(self) -> Check:
        check = self.new("case.cloud_fallback")
        record = self.expected.get("fallback") or {}
        span = self.span("run")
        if span is None:
            check.failures.append(f"no stored run (outcome {str(record.get('outcome'))[:300]!r})")
            return check
        if span.kind != REQUEST_TYPE or span.counted is not True or span.is_error:
            check.failures.append(f"{label(span)}: type {span.kind!r}, counted {span.counted!r}, error {span.error_type!r}")
        attempts = sorted((item for item in self.spans if item.parent_span_id == span.span_id
                           and attrs(item).get(ATTEMPT_INDEX) is not None),
                          key=lambda item: number(attrs(item).get(ATTEMPT_INDEX)) or 0)
        if len(attempts) != 2:
            check.failures.append(f"{len(attempts)} attempts, expected the local one and the remote one")
            return check
        local, remote = attempts
        if attrs(local).get(EXECUTION_LOCATION) != "local" or local.error_type != "aborted_for_cloud_fallback":
            check.failures.append(f"attempt 0: location {attrs(local).get(EXECUTION_LOCATION)!r}, error type "
                                  f"{local.error_type!r}; expected local, aborted_for_cloud_fallback")
        if (attrs(remote).get(EXECUTION_LOCATION) != "remote" or attrs(remote).get(ATTEMPT_REASON) != "fallback"
                or remote.is_error):
            check.failures.append(f"attempt 1: location {attrs(remote).get(EXECUTION_LOCATION)!r}, reason "
                                  f"{attrs(remote).get(ATTEMPT_REASON)!r}, error {remote.error_type!r}")
        gateway = [item for item in self.spans if item.parent_span_id == remote.span_id]
        if not gateway:
            check.failures.append("no gateway span under the remote attempt")
        check.notes.append(f"seam {record.get('seam')}; gateway spans {[label(item) for item in gateway]}")
        return check

    def run(self) -> List[Check]:
        checks = [self.case_valid()]
        if not self.expected:
            return checks
        checks += [self.trace_rooted(), self.identity(), self.lifecycle_final()]
        checks.append({"pipeline": self.pipeline, "tts": self.tts, "asr": self.asr, "frames": self.frames,
                       "fallback": self.fallback}[self.case]())
        return checks


# ---------------------------------------------------------------------------
# Per-launch checks
# ---------------------------------------------------------------------------


def config_sessions(launch: Launch) -> List[str]:
    return list((launch.config.get("sessions") or {}).keys())


def counted_requests(launch: Launch) -> int:
    total = 0
    for session in config_sessions(launch):
        config = (launch.config.get("sessions") or {}).get(session) or {}
        trace_id = destinations.lower_hex(config.get("trace_id"))
        readback = launch.readback.get(session) or SessionReadback()
        total += sum(1 for span in readback.stored() if span.trace_id == trace_id and span.counted is True)
    return total


def stats_delta(before: Optional[Mapping[str, Any]], after: Optional[Mapping[str, Any]],
                key: str = "total_executions_24h") -> Optional[int]:
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    if not isinstance(before.get(key), int) or not isinstance(after.get(key), int):
        return None
    return after[key] - before[key]


def aggregate_check(launch: Launch) -> Check:
    check = Check("staging", str(launch.config.get("lane")), "aggregate.stats")
    expected = counted_requests(launch)
    delta = stats_delta(launch.stats_before, launch.stats_after)
    if delta is None:
        check.failures.append("the dashboard aggregate was not read before and after")
    elif delta != expected:
        check.failures.append(f"total_executions_24h changed by {delta}, the launch stored {expected} counted requests")
    device = stats_delta(launch.stats_before, launch.stats_after, "executions_on_device")
    check.notes.append(f"total_executions_24h +{delta}, executions_on_device +{device}, counted requests {expected}")
    return check


def deployment_check(launch: Launch) -> Check:
    check = Check("staging", str(launch.config.get("lane")), "deployment.unchanged")
    before, after = launch.deployments_before, launch.deployments_after
    if not before or not after:
        check.failures.append("the platform deployments were not recorded before and after")
        return check
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    if changed:
        check.failures.append(f"staging redeployed during the launch: {changed}")
    check.notes.append(", ".join(f"{key} {str(value)[:12]}" for key, value in sorted(before.items())))
    return check


def secrets_check(launch: Launch) -> Check:
    check = Check("staging", str(launch.config.get("lane")), "evidence.secrets")
    if launch.secret_hits is None:
        check.failures.append("the evidence was not scanned")
    elif launch.secret_hits:
        check.failures.append(f"key material found in: {launch.secret_hits}")
    return check


def check_launch(launch: Launch, spec: Optional[Mapping[str, Any]] = None) -> List[Check]:
    spec = spec or load_spec()
    checks: List[Check] = []
    for session in config_sessions(launch):
        if session_kind(session) == "case":
            checks += CaseChecks(launch, session, spec).run()
        else:
            checks += SessionChecks(launch, session, spec).run()
    checks += [aggregate_check(launch), deployment_check(launch), secrets_check(launch)]
    return checks


def summarize(checks: Iterable[Check]) -> Dict[str, int]:
    return interop.summarize(checks)


def public_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The config without the key, as it goes into the evidence."""
    out = json.loads(json.dumps(config))
    for session in (out.get("sessions") or {}).values():
        if isinstance(session, dict) and "api_key" in session:
            session["api_key"] = "<removed>"
    return out


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [f"# Staging acceptance: {report.get('lane')} ({report.get('run')})", ""]
    summary = report.get("summary") or {}
    lines.append(f"**{'PASS' if report.get('passed') else 'FAIL'}**: {summary.get('pass', 0)} passed, "
                 f"{summary.get('fail', 0)} failed, {summary.get('skip', 0)} skipped")
    lines.append("")
    header = report.get("header") or {}
    for key in ("sdk_commit", "binding", "host", "device", "model_id", "sdk_version", "collector", "platform"):
        if header.get(key) is not None:
            value = header[key] if not isinstance(header[key], (dict, list)) else json.dumps(header[key], sort_keys=True)
            lines.append(f"- {key}: {value}")
    for session, ids in (report.get("traces") or {}).items():
        lines.append(f"- {session}: T `{ids.get('trace_id')}`, P `{ids.get('parent_span_id')}`")
    lines += ["", "| Check | Session | Status | Detail |", "|---|---|---|---|"]
    for check in report.get("checks") or []:
        detail = "; ".join(check.get("failures") or []) or check.get("skipped") or "; ".join(check.get("notes") or [])
        detail = detail.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{check['name']}` | {check.get('workload')} | {check['status'].upper()} | {detail[:400]} |")
    return "\n".join(lines) + "\n"
