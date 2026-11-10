"""What each destination keeps, and the checks that compare it with the truth.

The ground truth is what the Collector received, copied before any mapping
(``collector-harness.yaml``). ``DESTINATIONS`` says, per backend, how each
canonical span role must look there and what the backend drops; every check
below reads its expectation from that table, so a backend that changes its
mapping fails one named row instead of a generic diff.

Roles (``role`` below; the order of Xybrid's own trace-ingest classification,
first match wins):

| Role | Canonical span |
|---|---|
| ``inference`` | ``gen_ai.operation.name`` = chat, text_completion, generate_content |
| ``embedding`` | ``gen_ai.operation.name`` = embeddings |
| ``predict`` | ``gen_ai.operation.name`` = predict (a Xybrid model run) |
| ``attempt`` | no request operation, ``xybrid.genai.attempt.index`` set |
| ``workflow`` / ``tool`` / ``load`` | invoke_workflow / execute_tool / load_model |
| ``other`` | anything else (runtime-internal spans) |

Runs on python3 >= 3.9 with no third-party packages.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "observability_capture"))

import check_capture  # noqa: E402 -- the shared OTLP/JSON decoding
from destinations import DestEvaluation, DestSpan  # noqa: E402
from workloads import Workload  # noqa: E402

REQUEST_OPERATIONS = ("chat", "text_completion", "generate_content")
TTFC = "gen_ai.response.time_to_first_chunk"
TTFT = "xybrid.genai.response.time_to_first_token"
EVALUATION_EVENT = "gen_ai.evaluation.result"

# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

DESTINATIONS: Dict[str, Dict[str, Any]] = {
    "langfuse": {
        "label": "Langfuse (self-hosted)",
        "profile": "langfuse.yaml",
        # Observation type per role (profile: transform/langfuse).
        "kind": {"inference": "GENERATION", "embedding": "EMBEDDING", "tool": "TOOL", "workflow": "CHAIN",
                 "attempt": "SPAN", "load": "SPAN", "predict": "SPAN"},
        # One model field. OtelIngestionProcessor.extractModelName (4.77.0,
        # same in 3.227.0) checks langfuse.observation.model.name,
        # litellm.provider.model, gen_ai.response.model, ai.model.id, then
        # gen_ai.request.model: the response model wins.
        "model": "response_first",
        "model_roles": ("inference",),
        # Provider is the profile's langfuse.observation.metadata.provider,
        # read back from metadata.provider only.
        "provider_roles": ("inference",),
        # usageDetails.input/output on the generation. Langfuse reads
        # gen_ai.usage.* on every observation type (shouldSkipModelUsageAndCost
        # skips only invoke_agent/agent_step) and its trace totals sum
        # usage_details over all observations, so attempts (SPAN) must carry
        # none: the profile sets langfuse.observation.usage_details to {}.
        "usage": {"inference": "equal", "attempt": "absent"},
        "counted": None,
        # ERROR status becomes level=ERROR; status.message -> statusMessage.
        "error": "level",
        "status_message": True,
        "error_type": False,
        "content_roles": ("inference", "tool"),
        # Both timing attributes are unmapped, so they stay apart under
        # metadata.attributes; completionStartTime is not set by the profile.
        "timing": "metadata",
        "evaluations": "dropped",
        "evaluation_source": False,
        "remote_parent": "kept",
        "maps": [
            "observation id = OTel span id, traceId = OTel trace id, parentObservationId = parent span id",
            "type: chat/text_completion/generate_content -> GENERATION, embeddings -> EMBEDDING, "
            "execute_tool -> TOOL, invoke_workflow -> CHAIN, load_model/predict/attempts -> SPAN",
            "input/output <- gen_ai.input.messages / gen_ai.output.messages (tool: arguments/result)",
            "model <- gen_ai.response.model (else gen_ai.request.model); usageDetails <- gen_ai.usage.* "
            "(attempts: usage_details {} so trace totals count each request once)",
            "level=ERROR <- span status ERROR; statusMessage <- status message",
            "metadata.provider / runtime / execution_location <- langfuse.observation.metadata.*",
            "every other attribute (first-chunk and first-token times included) -> metadata.attributes",
        ],
        "drops": [
            "gen_ai.evaluation.result log records (evaluations, feedback): Langfuse ingests no OTLP logs; "
            "scores come only from its own API (route the records elsewhere with sinks.yaml)",
            "metrics (nop unless sinks.yaml)",
            "span status code itself (only level/statusMessage remain); completionStartTime is not derived",
            "attempt usage as usageDetails (the canonical gen_ai.usage.* stay under metadata.attributes)",
        ],
    },
    "phoenix": {
        "label": "Arize Phoenix (self-hosted)",
        "profile": "phoenix.yaml",
        "kind": {"inference": "LLM", "embedding": "EMBEDDING", "tool": "TOOL", "workflow": "CHAIN",
                 "attempt": "CHAIN", "load": "CHAIN", "predict": "CHAIN", "other": "CHAIN"},
        # llm.model_name is set from the request model, then overwritten by
        # the response model when there is one.
        "model": "response_first",
        "model_roles": ("inference",),
        "provider_roles": ("inference",),
        # llm.token_count.* only on LLM spans: attempts carry none. Phoenix
        # derives token counts from gen_ai.usage.* on any span, so the profile
        # removes those from attempts.
        "usage": {"inference": "equal", "attempt": "absent"},
        "counted": None,
        "error": "status",
        "status_message": True,
        "error_type": False,
        "content_roles": ("inference", "tool"),
        "timing": "attributes",
        "evaluations": "dropped",
        "evaluation_source": False,
        "remote_parent": "kept",
        "maps": [
            "context.trace_id / span_id = OTel ids, parent_id = parent span id, status_code = OTel status",
            "span_kind (openinference.span.kind): chat/... -> LLM, embeddings -> EMBEDDING, "
            "execute_tool -> TOOL, everything else -> CHAIN",
            "llm.model_name <- response model (else request model); llm.provider <- gen_ai.provider.name",
            "llm.token_count.prompt/completion <- gen_ai.usage.* on LLM spans only",
            "input.value/output.value <- messages (tool: arguments/result)",
            "every other canonical attribute is kept as sent (first-chunk and first-token times included)",
        ],
        "drops": [
            "gen_ai.usage.* on attempt spans: Phoenix would derive llm.token_count.* from them and count the "
            "request's tokens twice",
            "gen_ai.evaluation.result log records: Phoenix ingests no OTLP logs; annotations come only from "
            "its own API (route the records elsewhere with sinks.yaml)",
            "metrics (nop unless sinks.yaml)",
        ],
    },
    "xybrid": {
        "label": "Xybrid platform (staging)",
        "profile": "xybrid.yaml",
        "kind": {"inference": "genai.request", "embedding": "genai.request", "predict": "genai.request",
                 "attempt": "genai.attempt", "tool": "genai.tool", "workflow": "genai.workflow",
                 "load": "genai.load_model", "other": "other"},
        "model": "both",
        "model_roles": ("inference", "attempt", "predict"),
        "provider_roles": ("inference", "attempt", "predict"),
        # Attempts keep their own usage; only the logical request is counted.
        "usage": {"inference": "equal", "attempt": "equal"},
        "counted": {"inference": True, "embedding": True, "predict": True, "attempt": False, "tool": False,
                    "workflow": False, "load": False, "other": False},
        "error": "status",
        "status_message": True,
        "error_type": True,
        "content_roles": ("inference", "tool"),
        "timing": "typed",
        "evaluations": "linked",
        # Each linked result says whether it is feedback or an evaluation.
        "evaluation_source": True,
        "remote_parent": "kept",
        "maps": [
            "trace_id / span_id / parent_span_id = OTel ids; status.code/message = OTel status",
            "span_type: request operations -> genai.request (counted once), attempts -> genai.attempt, "
            "genai.tool / genai.workflow / genai.load_model",
            "genai.request_model / response_model / provider_name / input_tokens / output_tokens / error_type",
            "genai.time_to_first_chunk_s and genai.time_to_first_token_s as separate typed columns",
            "content.values (or the /content route) <- governed content attributes",
            "gen_ai.evaluation.result -> /v1/logs -> GET .../evaluations, linked by trace + span id",
        ],
        "drops": [
            "log records other than gen_ai.evaluation.result (profile filter; sinks.yaml keeps them)",
            "metrics (nop unless sinks.yaml)",
        ],
    },
}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Check:
    stage: str  # "collector", "producer", "destination"
    workload: Optional[str]
    name: str
    failures: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    skipped: Optional[str] = None

    @property
    def status(self) -> str:
        if self.failures:
            return "fail"
        return "skip" if self.skipped else "pass"

    def as_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "workload": self.workload, "name": self.name, "status": self.status,
                "failures": self.failures, "notes": self.notes, "skipped": self.skipped}


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def role(span: check_capture.Span) -> str:
    operation = span.attributes.get("gen_ai.operation.name")
    if operation in REQUEST_OPERATIONS:
        return "inference"
    if operation == "embeddings":
        return "embedding"
    if operation == "predict":
        return "predict"
    if span.attributes.get("xybrid.genai.attempt.index") is not None:
        return "attempt"
    return {"invoke_workflow": "workflow", "execute_tool": "tool", "load_model": "load"}.get(operation, "other")


def filter_entries(entries: Sequence[Mapping[str, Any]], keep) -> List[Dict[str, Any]]:
    """Capture entries reduced to the resources ``keep(resource_attributes)`` accepts."""
    out = []
    for entry in entries:
        body = entry.get("body")
        if not isinstance(body, dict):
            out.append(dict(entry))
            continue
        filtered: Dict[str, Any] = {}
        for key in ("resourceSpans", "resourceLogs", "resourceMetrics"):
            kept = [resource for resource in body.get(key) or []
                    if keep(check_capture.attributes((resource.get("resource") or {}).get("attributes")))]
            if kept:
                filtered[key] = kept
        if filtered:
            out.append({**entry, "body": filtered})
    return out


@dataclass
class GroundTruth:
    """One workload's telemetry as the Collector received it."""

    workload: Workload
    entries: List[Dict[str, Any]]
    capture: check_capture.Capture
    refs: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @classmethod
    def build(cls, workload: Workload, entries: List[Dict[str, Any]],
              refs: Optional[Dict[str, Dict[str, str]]] = None) -> "GroundTruth":
        return cls(workload, entries, check_capture.Capture.from_entries(entries), dict(refs or {}))

    @property
    def spans(self) -> List[check_capture.Span]:
        return self.capture.spans

    def span_ids(self) -> Set[str]:
        return {span.span_id for span in self.spans if span.span_id}

    def trace_ids(self) -> List[str]:
        return sorted({span.trace_id for span in self.spans if span.trace_id})

    def evaluations(self) -> List[check_capture.LogRecord]:
        return [log for log in self.capture.logs if log.event_name == EVALUATION_EVENT]

    def text(self) -> str:
        return self.capture.text


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def same_number(a: Optional[float], b: Optional[float], tolerance: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def same_content(expected: Any, actual: Optional[str]) -> bool:
    """JSON-equal when both parse, else string-equal."""
    if actual is None:
        return False
    if not isinstance(expected, str):
        expected = json.dumps(expected)
    try:
        return json.loads(expected) == json.loads(actual)
    except ValueError:
        return expected == actual


def expected_model(rule: str, request: Optional[str], response: Optional[str]) -> Optional[str]:
    if rule == "request_first":
        return request or response
    return response or request


def _label(span: check_capture.Span) -> str:
    return f"{span.name!r} ({span.span_id})"


# ---------------------------------------------------------------------------
# Collector-stage checks (destination independent)
# ---------------------------------------------------------------------------


def collector_checks(truth: GroundTruth, exit_codes: Sequence[int]) -> List[Check]:
    name = truth.workload.name
    checks = []

    check = Check("producer", name, "producer.exit")
    if not exit_codes or any(code != 0 for code in exit_codes):
        check.failures.append(f"exit status {list(exit_codes)}")
    checks.append(check)

    check = Check("producer", name, "producer.refs_exported")
    index = truth.capture.spans_by_id()
    if not truth.refs:
        check.failures.append("the workload reported no observation references")
    for role_name, ref in sorted(truth.refs.items()):
        span = index.get(ref["span_id"])
        if span is None:
            check.failures.append(f"{role_name} {ref['span_id']}: no such span reached the Collector")
        elif span.trace_id != ref["trace_id"]:
            check.failures.append(f"{role_name}: span {ref['span_id']} is in trace {span.trace_id}, "
                                  f"reported {ref['trace_id']}")
    check.notes.append(f"{len(truth.refs)} references reported")
    checks.append(check)

    check = Check("collector", name, "collector.signals")
    evaluations = truth.evaluations()
    if not truth.spans:
        check.failures.append("no spans reached the Collector")
    if len(evaluations) < truth.workload.evaluations:
        check.failures.append(f"{len(evaluations)} {EVALUATION_EVENT} records, expected "
                              f"{truth.workload.evaluations}")
    if truth.capture.metric_points == 0:
        check.failures.append("no metric points reached the Collector")
    check.notes.append(f"{len(truth.spans)} spans, {len(truth.capture.logs)} log records "
                       f"({len(evaluations)} evaluation results), {truth.capture.metric_points} metric points")
    checks.append(check)

    check = Check("collector", name, "collector.redacted_at_source")
    leaked = [secret for secret in truth.workload.secrets if secret in truth.text()]
    if leaked:
        check.failures.append(f"redacted strings reached the Collector: {leaked}")
    checks.append(check)
    return checks


def sink_checks(truth: GroundTruth, sink: GroundTruth, destination: str) -> List[Check]:
    """``sinks.yaml``: every metric point, and the log records the profile does
    not route to its destination, reach the sink."""
    name = truth.workload.name
    check = Check("collector", name, "sinks.routing")
    if sink.capture.metric_points != truth.capture.metric_points:
        check.failures.append(f"sink received {sink.capture.metric_points} metric points, the Collector "
                              f"{truth.capture.metric_points}")
    expected_logs = [log for log in truth.capture.logs
                     if destination != "xybrid" or log.event_name != EVALUATION_EVENT]
    got = sorted(str(log.event_name) for log in sink.capture.logs)
    want = sorted(str(log.event_name) for log in expected_logs)
    if got != want:
        check.failures.append(f"sink log events {got}, expected {want}")
    if destination == "xybrid" and any(log.event_name == EVALUATION_EVENT for log in sink.capture.logs):
        check.failures.append("evaluation results reached the sink although xybrid.yaml routes them to Xybrid")
    if sink.spans:
        check.failures.append(f"{len(sink.spans)} spans reached the metrics/logs sink")
    check.notes.append(f"sink: {sink.capture.metric_points} metric points, {len(sink.capture.logs)} log records")
    return [check]


# ---------------------------------------------------------------------------
# Destination checks
# ---------------------------------------------------------------------------


def destination_checks(destination: str, truth: GroundTruth, spans: Sequence[DestSpan],
                       evaluations: Optional[Sequence[DestEvaluation]], raw_text: str,
                       table: Optional[Mapping[str, Any]] = None) -> List[Check]:
    """Every destination check for one workload, from the destination table."""
    table = table or DESTINATIONS[destination]
    name = truth.workload.name
    gt_ids = truth.span_ids()
    by_id: Dict[str, DestSpan] = {}
    for span in spans:
        if span.span_id in gt_ids and span.trace_id in set(truth.trace_ids()):
            by_id.setdefault(span.span_id, span)
    present = [gt for gt in truth.spans if gt.span_id in by_id]

    def new(check_name: str) -> Check:
        return Check("destination", name, check_name)

    checks: List[Check] = []

    # ids.spans: every ground-truth span is there, under the same ids.
    check = new("ids.spans")
    missing = [_label(gt) for gt in truth.spans if gt.span_id not in by_id]
    if missing:
        check.failures.append(f"{len(missing)}/{len(truth.spans)} spans missing: {missing}")
    foreign = [span for span in spans if span.span_id not in gt_ids]
    if foreign:
        check.notes.append(f"{len(foreign)} other spans in the same trace(s) (earlier runs share a fixed trace id)")
    checks.append(check)

    # ids.parents: parent links between exported spans are preserved.
    check = new("ids.parents")
    for gt in present:
        if gt.parent_span_id in gt_ids and by_id[gt.span_id].parent_span_id != gt.parent_span_id:
            check.failures.append(f"{_label(gt)}: parent {by_id[gt.span_id].parent_span_id}, "
                                  f"expected {gt.parent_span_id}")
        if gt.parent_span_id is None and by_id[gt.span_id].parent_span_id not in (None, ""):
            check.failures.append(f"{_label(gt)}: root span got parent {by_id[gt.span_id].parent_span_id}")
    checks.append(check)

    # ids.remote_parent: a caller's span that was never exported (parent-in).
    check = new("ids.remote_parent")
    remote = [gt for gt in present if gt.parent_span_id and gt.parent_span_id not in gt_ids]
    if not remote:
        check.skipped = "no span has a remote (not exported) parent"
    for gt in remote:
        kept = by_id[gt.span_id].parent_span_id
        if table["remote_parent"] == "kept" and kept != gt.parent_span_id:
            check.failures.append(f"{_label(gt)}: remote parent {gt.parent_span_id} became {kept!r}")
    checks.append(check)

    # ids.producer_refs: what the producer reported can be looked up.
    check = new("ids.producer_refs")
    for role_name, ref in sorted(truth.refs.items()):
        span = by_id.get(ref["span_id"])
        if span is None or span.trace_id != ref["trace_id"]:
            check.failures.append(f"{role_name} {ref['trace_id']}/{ref['span_id']} not found")
    if not truth.refs:
        check.skipped = "the workload reported no references"
    checks.append(check)

    # mapping.kind: the destination's span type per role.
    check = new("mapping.kind")
    for gt in present:
        expected = table["kind"].get(role(gt))
        actual = by_id[gt.span_id].kind
        if expected is not None and (actual or "").upper() != expected.upper():
            check.failures.append(f"{_label(gt)} ({role(gt)}): {actual!r}, expected {expected!r}")
    logical = [gt for gt in truth.spans if role(gt) == "inference"]
    expected_generation = table["kind"].get("inference")
    if expected_generation:
        typed = [gt for gt in present if (by_id[gt.span_id].kind or "").upper() == expected_generation.upper()]
        check.notes.append(f"{len(typed)} {expected_generation} spans for {len(logical)} logical inferences")
    checks.append(check)

    # mapping.model_provider
    check = new("mapping.model_provider")
    for gt in present:
        dest, gt_role = by_id[gt.span_id], role(gt)
        request = gt.attributes.get("gen_ai.request.model")
        response = gt.attributes.get("gen_ai.response.model")
        if gt_role in table["model_roles"] and (request or response):
            if table["model"] == "both":
                if request and dest.request_model != request:
                    check.failures.append(f"{_label(gt)}: request model {dest.request_model!r}, expected {request!r}")
                if response and dest.response_model != response:
                    check.failures.append(f"{_label(gt)}: response model {dest.response_model!r}, "
                                          f"expected {response!r}")
            else:
                want = expected_model(table["model"], request, response)
                if dest.model != want:
                    check.failures.append(f"{_label(gt)}: model {dest.model!r}, expected {want!r}")
        provider = gt.attributes.get("gen_ai.provider.name")
        if gt_role in table["provider_roles"] and provider and dest.provider != provider:
            check.failures.append(f"{_label(gt)}: provider {dest.provider!r}, expected {provider!r}")
    checks.append(check)

    # mapping.usage
    check = new("mapping.usage")
    asserted = 0
    for gt in present:
        rule = table["usage"].get(role(gt))
        dest = by_id[gt.span_id]
        want_in = gt.attributes.get("gen_ai.usage.input_tokens")
        want_out = gt.attributes.get("gen_ai.usage.output_tokens")
        if rule == "equal" and (want_in is not None or want_out is not None):
            asserted += 1
            if not same_number(dest.input_tokens, want_in) or not same_number(dest.output_tokens, want_out):
                check.failures.append(f"{_label(gt)}: tokens {dest.input_tokens}/{dest.output_tokens}, "
                                      f"expected {want_in}/{want_out}")
        elif rule == "absent":
            # A zero (or missing) count adds nothing to a backend's totals.
            if want_in is not None or want_out is not None:
                asserted += 1
            if dest.input_tokens or dest.output_tokens:
                check.failures.append(f"{_label(gt)} ({role(gt)}): carries tokens {dest.input_tokens}/"
                                      f"{dest.output_tokens}; they would be counted twice")
        counted_rule = (table.get("counted") or {}).get(role(gt))
        # A destination whose table promises the counted flag must expose it:
        # a missing flag is a mismatch, not a pass.
        if counted_rule is not None and dest.counted != counted_rule:
            check.failures.append(f"{_label(gt)} ({role(gt)}): counted_request={dest.counted}, "
                                  f"expected {counted_rule}")
    if asserted == 0 and not check.failures:
        check.skipped = "the workload reports no token usage"
    checks.append(check)

    # mapping.errors
    check = new("mapping.errors")
    for gt in present:
        dest = by_id[gt.span_id]
        if gt.is_error:
            if dest.is_error is not True:
                check.failures.append(f"{_label(gt)}: error status lost ({table['error']} = {dest.is_error})")
            if table["status_message"] and gt.status_message and dest.status_message != gt.status_message:
                check.failures.append(f"{_label(gt)}: status message {dest.status_message!r}, "
                                      f"expected {gt.status_message!r}")
            want_type = gt.attributes.get("error.type")
            if table["error_type"] and want_type and dest.error_type != want_type:
                check.failures.append(f"{_label(gt)}: error type {dest.error_type!r}, expected {want_type!r}")
        elif dest.is_error:
            check.failures.append(f"{_label(gt)}: marked as an error, the producer reported none")
    if not any(gt.is_error for gt in present):
        check.notes.append("no span of this workload failed")
    checks.append(check)

    # content: present (captured) or absent (metadata only), per role.
    check = new("content")
    roles = table["content_roles"]
    if truth.workload.content == "metadata_only":
        for gt in present:
            dest = by_id[gt.span_id]
            if dest.input is not None or dest.output is not None:
                check.failures.append(f"{_label(gt)}: content stored in metadata-only mode "
                                      f"(input={dest.input!r:.60}, output={dest.output!r:.60})")
        for evaluation in evaluations or []:
            if evaluation.span_id in gt_ids and evaluation.comment:
                check.failures.append(f"feedback comment stored in metadata-only mode: {evaluation.comment!r}")
    else:
        compared = 0
        for gt in present:
            dest, gt_role = by_id[gt.span_id], role(gt)
            if gt_role not in roles:
                continue
            pairs = (("input", gt.attributes.get("gen_ai.input.messages") or gt.attributes.get(
                "gen_ai.tool.call.arguments"), dest.input),
                     ("output", gt.attributes.get("gen_ai.output.messages") or gt.attributes.get(
                         "gen_ai.tool.call.result"), dest.output))
            for label, want, got in pairs:
                if want is None:
                    continue
                compared += 1
                if not same_content(want, got):
                    check.failures.append(f"{_label(gt)}: {label} {got!r:.120} differs from the producer's "
                                          f"{want!r:.120}")
        if compared == 0:
            check.skipped = "no inference or tool content in the ground truth"
        missing = [token for token in truth.workload.replacements if token not in raw_text]
        if missing and compared:
            check.failures.append(f"redaction replacements not found at the destination: {missing}")
    checks.append(check)

    # content.redaction: no redacted string reached the destination.
    check = new("content.redaction")
    leaked = [secret for secret in truth.workload.secrets if secret in raw_text]
    if leaked:
        check.failures.append(f"redacted or metadata-only strings found at the destination: {leaked}")
    checks.append(check)

    # timing.first_chunk_vs_first_token
    check = new("timing.first_chunk_vs_first_token")
    timed = [gt for gt in present if gt.attributes.get(TTFC) is not None or gt.attributes.get(TTFT) is not None]
    if not timed:
        check.skipped = "no span carries streaming timing"
    for gt in timed:
        dest = by_id[gt.span_id]
        chunk, token = gt.attributes.get(TTFC), gt.attributes.get(TTFT)
        if chunk is not None and not same_number(dest.first_chunk_s, chunk):
            check.failures.append(f"{_label(gt)}: first chunk {dest.first_chunk_s}, expected {chunk}")
        if token is not None and not same_number(dest.first_token_s, token):
            check.failures.append(f"{_label(gt)}: first token {dest.first_token_s}, expected {token}")
        if chunk is not None and token is not None:
            if same_number(chunk, token):
                check.notes.append(f"{_label(gt)}: the producer measured equal first chunk and first token")
            elif dest.completion_start_offset_s is not None and same_number(
                    dest.completion_start_offset_s, chunk, 1e-3) and not same_number(
                    dest.completion_start_offset_s, token, 1e-3):
                check.failures.append(f"{_label(gt)}: completion start is the first chunk ({chunk}s), "
                                      f"not the measured first token ({token}s)")
            check.notes.append(f"{_label(gt)}: first chunk {chunk}s, first token {token}s"
                               + (f", completion start offset {dest.completion_start_offset_s}s"
                                  if dest.completion_start_offset_s is not None else ""))
    checks.append(check)

    # evaluations: linked to the span, or dropped as the table says.
    check = new("evaluations")
    expected = truth.evaluations()
    linked = [evaluation for evaluation in evaluations or [] if evaluation.span_id in gt_ids]
    if evaluations is None:
        check.skipped = "not read"
    elif table["evaluations"] == "dropped":
        if linked:
            check.failures.append(f"{len(linked)} results found, but the table says this destination drops "
                                  f"them; update DESTINATIONS")
        else:
            check.notes.append(f"{len(expected)} results dropped, as expected")
    else:
        for record in expected:
            attrs = record.attributes
            match = [evaluation for evaluation in linked
                     if evaluation.span_id == record.span_id and evaluation.trace_id == record.trace_id
                     and evaluation.name == attrs.get("gen_ai.evaluation.name")]
            label = f"{attrs.get('gen_ai.evaluation.name')} on {record.span_id}"
            if not match:
                check.failures.append(f"{label}: not linked at the destination")
                continue
            found = match[0]
            if not same_number(found.score, attrs.get("gen_ai.evaluation.score.value")):
                check.failures.append(f"{label}: score {found.score}, expected "
                                      f"{attrs.get('gen_ai.evaluation.score.value')}")
            if attrs.get("gen_ai.evaluation.score.label") is not None and found.label != attrs.get(
                    "gen_ai.evaluation.score.label"):
                check.failures.append(f"{label}: label {found.label!r}, expected "
                                      f"{attrs.get('gen_ai.evaluation.score.label')!r}")
            want_source = "feedback" if attrs.get("gen_ai.evaluation.name") == "user_feedback" else "evaluation"
            # A destination that tells feedback from evaluations must say which
            # each result is; a missing source is a mismatch.
            if table.get("evaluation_source") and found.source != want_source:
                check.failures.append(f"{label}: source {found.source!r}, expected {want_source!r}")
            comment = attrs.get("xybrid.genai.feedback.comment")
            if comment is not None and found.comment != comment:
                check.failures.append(f"{label}: comment {found.comment!r}, expected {comment!r}")
        if not expected:
            check.skipped = "the workload recorded no evaluation result"
        else:
            check.notes.append(f"{len(linked)} of {len(expected)} results linked")
    checks.append(check)
    return checks


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def table_markdown(destination: str) -> str:
    table = DESTINATIONS[destination]
    lines = [f"### {table['label']} (`{table['profile']}`)", "", "Maps:", ""]
    lines += [f"- {item}" for item in table["maps"]]
    lines += ["", "Drops:", ""]
    lines += [f"- {item}" for item in table["drops"]]
    lines += ["", "| Role | Expected kind | Usage | Counted |", "|---|---|---|---|"]
    for role_name in ("inference", "embedding", "predict", "attempt", "tool", "workflow", "load", "other"):
        lines.append(f"| {role_name} | {table['kind'].get(role_name, '(not asserted)')} | "
                     f"{table['usage'].get(role_name, '-')} | "
                     f"{(table.get('counted') or {}).get(role_name, '-')} |")
    return "\n".join(lines)


def summarize(checks: Iterable[Check]) -> Dict[str, int]:
    summary = {"pass": 0, "fail": 0, "skip": 0}
    for check in checks:
        summary[check.status] += 1
    return summary
