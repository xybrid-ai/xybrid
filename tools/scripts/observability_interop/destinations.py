"""Destination readback: API clients and the normalizers the checks run on.

Every destination's answer is reduced to the same two shapes, ``DestSpan``
and ``DestEvaluation``, so ``expectations.py`` can compare any backend with the
Collector's ground truth through one table. The normalizers are pure functions
of the API responses and are unit-tested with fixture responses
(``fixtures/``); the clients only fetch.

| Destination | Spans | Quality results | Version |
|---|---|---|---|
| Langfuse | ``GET /api/public/v2/observations?traceId=`` (v4; v1 ``/api/public/observations`` on v3) | ``GET /api/public/v3/scores?traceId=`` (v2 on v3) | ``GET /api/public/health`` |
| Phoenix | ``GET /v1/projects/{project}/spans?trace_id=`` | ``GET /v1/projects/{project}/span_annotations?span_ids=`` | ``GET /arize_phoenix_version`` |
| Xybrid | ``GET /v1/telemetry/traces/{id}/spans`` (+ ``.../spans/{span}/content`` for object-stored content) | ``GET /v1/telemetry/traces/{id}/evaluations`` | ``GET /health`` |

``--wire`` mode reads no API: ``wire_spans``/``wire_evaluations`` normalize
what the profile's exporter sent to the harness's capture server, applying
the backend's documented field mapping (Langfuse) or none (OpenInference
attributes and Xybrid's OTLP are read as sent).
"""

from __future__ import annotations

import base64
import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

TTFC = "gen_ai.response.time_to_first_chunk"
TTFT = "xybrid.genai.response.time_to_first_token"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
TOOL_RESULT = "gen_ai.tool.call.result"
FEEDBACK_COMMENT = "xybrid.genai.feedback.comment"


@dataclass
class DestSpan:
    """One span as a destination stores it. ``None`` means "not exposed"."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    name: Optional[str] = None
    kind: Optional[str] = None
    is_error: Optional[bool] = None
    status_message: Optional[str] = None
    error_type: Optional[str] = None
    model: Optional[str] = None
    request_model: Optional[str] = None
    response_model: Optional[str] = None
    provider: Optional[str] = None
    input_tokens: Optional[float] = None
    output_tokens: Optional[float] = None
    counted: Optional[bool] = None
    input: Optional[str] = None
    output: Optional[str] = None
    first_chunk_s: Optional[float] = None
    first_token_s: Optional[float] = None
    completion_start_offset_s: Optional[float] = None
    raw: Any = field(default=None, repr=False)


@dataclass
class DestEvaluation:
    trace_id: Optional[str]
    span_id: Optional[str]
    name: Optional[str]
    score: Optional[float] = None
    label: Optional[str] = None
    comment: Optional[str] = None
    source: Optional[str] = None
    raw: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def lower_hex(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    return value.lower()


def number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def text(value: Any) -> Optional[str]:
    """A content value as text: strings as they are, structures as JSON."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def any_value(value: Any) -> Any:
    """Decodes an OTLP/JSON AnyValue (and the Xybrid AttrMap form)."""
    if not isinstance(value, dict) or not value:
        return None
    kind, inner = next(iter(value.items()))
    if kind == "intValue":
        try:
            return int(inner)
        except (TypeError, ValueError):
            return inner
    if kind == "doubleValue":
        return number(inner)
    if kind == "arrayValue":
        return [any_value(item) for item in (inner or {}).get("values", [])]
    if kind == "kvlistValue":
        return {item.get("key"): any_value(item.get("value")) for item in (inner or {}).get("values", [])}
    return inner


def attr_map(value: Any) -> Dict[str, Any]:
    """An AttrMap (``{key: AnyValue}``) or OTLP attribute list as plain values."""
    if isinstance(value, list):
        return {item.get("key"): any_value(item.get("value")) for item in value if isinstance(item, dict)}
    if isinstance(value, dict):
        return {key: any_value(inner) if isinstance(inner, dict) else inner for key, inner in value.items()}
    return {}


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def lookup(obj: Any, dotted: str, depth: int = 0) -> Any:
    """Finds attribute ``dotted`` anywhere in ``obj``, whatever the nesting.

    Backends store attributes flat (``{"a.b": 1}``), nested (``{"a": {"b":
    1}}``), under a prefix (``{"attributes.a.b": 1}``) or as JSON strings
    inside metadata. Returns the first match, ``None`` when absent.
    """
    if depth > 8:
        return None
    obj = _maybe_json(obj)
    if isinstance(obj, dict):
        if dotted in obj:
            return obj[dotted]
        for key, value in obj.items():
            if not isinstance(key, str):
                continue
            if key.endswith("." + dotted):
                return value
            if dotted.startswith(key + "."):
                found = lookup(value, dotted[len(key) + 1:], depth + 1)
                if found is not None:
                    return found
        for value in obj.values():
            if isinstance(value, (dict, list)) or (isinstance(value, str) and value[:1] in ("{", "[")):
                found = lookup(value, dotted, depth + 1)
                if found is not None:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            found = lookup(item, dotted, depth + 1)
            if found is not None:
                return found
    return None


def parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    # Python 3.9's fromisoformat takes at most 6 fractional digits.
    if "." in cleaned:
        head, _, rest = cleaned.partition(".")
        digits = ""
        while rest and rest[0].isdigit():
            digits, rest = digits + rest[0], rest[1:]
        cleaned = f"{head}.{digits[:6].ljust(6, '0')}{rest}"
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------


def langfuse_span(observation: Mapping[str, Any]) -> DestSpan:
    """One Langfuse observation (v2 or v1 API) as a DestSpan.

    Langfuse keeps the OTel ids (observation id = span id). It has no status
    code: the profile maps ERROR status to ``level=ERROR``, and the status
    message to ``statusMessage``. Provider and execution identity are the
    profile's ``langfuse.observation.metadata.*`` keys, read from
    ``metadata.provider`` only: every attribute Langfuse does not map lands
    under ``metadata.attributes``, and reading ``gen_ai.provider.name`` from
    there would pass even if the profile's mapping had no effect.
    """
    metadata = _maybe_json(observation.get("metadata"))
    # usageDetails when the answer has it (an empty one means "no usage"),
    # else v1's legacy ``usage``.
    usage = observation["usageDetails"] if "usageDetails" in observation else observation.get("usage")
    if isinstance(usage, str):
        usage = _maybe_json(usage)
    if not isinstance(usage, dict):
        usage = {}
    start = parse_time(observation.get("startTime"))
    completion = parse_time(observation.get("completionStartTime"))
    level = observation.get("level")
    return DestSpan(
        trace_id=lower_hex(observation.get("traceId")) or "",
        span_id=lower_hex(observation.get("id")) or "",
        parent_span_id=lower_hex(observation.get("parentObservationId")),
        name=observation.get("name"),
        kind=str(observation.get("type") or "").upper() or None,
        is_error=None if level is None else str(level).upper() == "ERROR",
        status_message=observation.get("statusMessage") or None,
        model=observation.get("model") or observation.get("providedModelName"),
        provider=text(metadata.get("provider")) if isinstance(metadata, dict) else None,
        input_tokens=number(usage.get("input")),
        output_tokens=number(usage.get("output")),
        input=text(observation.get("input")),
        output=text(observation.get("output")),
        first_chunk_s=number(lookup(metadata, TTFC)),
        first_token_s=number(lookup(metadata, TTFT)),
        completion_start_offset_s=(completion - start).total_seconds() if completion and start else None,
        raw=dict(observation),
    )


def langfuse_spans(pages: Iterable[Mapping[str, Any]]) -> List[DestSpan]:
    return [langfuse_span(item) for page in pages for item in (page.get("data") or [])]


def langfuse_scores(pages: Iterable[Mapping[str, Any]]) -> List[DestEvaluation]:
    """Scores (v3 ``subject`` form or the flat v1/v2 form) as evaluations."""
    out = []
    for page in pages:
        for score in page.get("data") or []:
            subject = score.get("subject") or {}
            trace_id = score.get("traceId") or subject.get("traceId") or (
                subject.get("id") if subject.get("kind") == "trace" else None)
            span_id = score.get("observationId") or (subject.get("id") if subject.get("kind") == "observation" else None)
            value = score.get("value")
            out.append(DestEvaluation(
                trace_id=lower_hex(trace_id), span_id=lower_hex(span_id), name=score.get("name"),
                score=number(value), label=score.get("stringValue") or (value if isinstance(value, str) else None),
                comment=score.get("comment"), source=score.get("source"), raw=dict(score)))
    return out


# ---------------------------------------------------------------------------
# Phoenix
# ---------------------------------------------------------------------------


def openinference_span(trace_id: str, span_id: str, parent: Optional[str], name: Optional[str],
                       kind: Optional[str], status: Optional[str], message: Optional[str],
                       attributes: Any, raw: Any) -> DestSpan:
    return DestSpan(
        trace_id=trace_id, span_id=span_id, parent_span_id=parent, name=name,
        kind=(kind or "").upper() or None,
        is_error=None if status is None else str(status).upper() in ("ERROR", "STATUS_CODE_ERROR", "2"),
        status_message=message or None,
        model=text(lookup(attributes, "llm.model_name")) or text(lookup(attributes, "embedding.model_name")),
        provider=text(lookup(attributes, "llm.provider")),
        input_tokens=number(lookup(attributes, "llm.token_count.prompt")),
        output_tokens=number(lookup(attributes, "llm.token_count.completion")),
        input=text(lookup(attributes, "input.value")),
        output=text(lookup(attributes, "output.value")),
        first_chunk_s=number(lookup(attributes, TTFC)),
        first_token_s=number(lookup(attributes, TTFT)),
        raw=raw,
    )


def phoenix_span(span: Mapping[str, Any]) -> DestSpan:
    """One span of Phoenix's REST ``/v1/projects/{project}/spans`` answer.

    ``span_kind`` is the OpenInference kind the profile set; ``context`` holds
    the OTel ids, ``parent_id`` the parent's OTel span id.
    """
    context = span.get("context") or {}
    return openinference_span(
        trace_id=lower_hex(context.get("trace_id")) or "", span_id=lower_hex(context.get("span_id")) or "",
        parent=lower_hex(span.get("parent_id")), name=span.get("name"), kind=span.get("span_kind"),
        status=span.get("status_code"), message=span.get("status_message"),
        attributes=span.get("attributes") or {}, raw=dict(span))


def phoenix_spans(pages: Iterable[Mapping[str, Any]]) -> List[DestSpan]:
    return [phoenix_span(item) for page in pages for item in (page.get("data") or [])]


def phoenix_annotations(pages: Iterable[Mapping[str, Any]], trace_of: Mapping[str, str]) -> List[DestEvaluation]:
    out = []
    for page in pages:
        for annotation in page.get("data") or []:
            result = annotation.get("result") or {}
            span_id = lower_hex(annotation.get("span_id"))
            out.append(DestEvaluation(
                trace_id=trace_of.get(span_id or ""), span_id=span_id, name=annotation.get("name"),
                score=number(result.get("score")), label=result.get("label"),
                comment=result.get("explanation"), source=annotation.get("annotator_kind"), raw=dict(annotation)))
    return out


# ---------------------------------------------------------------------------
# Xybrid
# ---------------------------------------------------------------------------


def xybrid_span(span: Mapping[str, Any], content: Optional[Mapping[str, Any]] = None) -> DestSpan:
    """One span of ``GET /v1/telemetry/traces/{id}/spans``.

    Typed GenAI columns are under ``genai``; captured content is
    ``content.values`` (an AttrMap) when stored inline, otherwise the
    ``.../content`` route's ``values`` (passed as ``content``).
    """
    genai = span.get("genai") or {}
    status = span.get("status") or {}
    stored = span.get("content") or {}
    values = attr_map((content or {}).get("values") if content is not None else stored.get("values"))
    return DestSpan(
        trace_id=lower_hex(span.get("trace_id")) or "",
        span_id=lower_hex(span.get("span_id")) or "",
        parent_span_id=lower_hex(span.get("parent_span_id")),
        name=span.get("name"),
        kind=span.get("span_type") or None,
        is_error=None if not status.get("code") else str(status.get("code")).upper() == "ERROR",
        status_message=status.get("message") or None,
        error_type=genai.get("error_type"),
        request_model=genai.get("request_model"),
        response_model=genai.get("response_model"),
        provider=genai.get("provider_name"),
        input_tokens=number(genai.get("input_tokens")),
        output_tokens=number(genai.get("output_tokens")),
        counted=span.get("counted_request"),
        input=text(values.get(INPUT_MESSAGES) or values.get(TOOL_ARGUMENTS)),
        output=text(values.get(OUTPUT_MESSAGES) or values.get(TOOL_RESULT)),
        first_chunk_s=number(genai.get("time_to_first_chunk_s")),
        first_token_s=number(genai.get("time_to_first_token_s")),
        raw=dict(span),
    )


def xybrid_spans(response: Mapping[str, Any], contents: Optional[Mapping[str, Mapping[str, Any]]] = None
                 ) -> List[DestSpan]:
    contents = contents or {}
    return [xybrid_span(span, contents.get(str(span.get("span_id", "")).lower()))
            for span in response.get("spans") or []]


def xybrid_evaluations(response: Mapping[str, Any]) -> List[DestEvaluation]:
    return [DestEvaluation(
        trace_id=lower_hex(item.get("trace_id")), span_id=lower_hex(item.get("span_id")), name=item.get("name"),
        score=number(item.get("score_value")), label=item.get("score_label"), comment=item.get("comment"),
        source=item.get("source"), raw=dict(item)) for item in response.get("items") or []]


# ---------------------------------------------------------------------------
# --wire: what the profile's exporter sent (capture_server.py entries)
# ---------------------------------------------------------------------------


def _wire_records(entries: Iterable[Mapping[str, Any]], signal: str) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    resource_key, scope_key, item_key = {
        "traces": ("resourceSpans", "scopeSpans", "spans"),
        "logs": ("resourceLogs", "scopeLogs", "logRecords"),
    }[signal]
    for entry in entries:
        body = entry.get("body")
        if entry.get("signal") != signal or not isinstance(body, dict):
            continue
        for resource in body.get(resource_key) or []:
            for scope in resource.get(scope_key) or []:
                for item in scope.get(item_key) or []:
                    yield item, attr_map(item.get("attributes"))


def _status(span: Mapping[str, Any]) -> Tuple[int, str]:
    status = span.get("status") or {}
    code = status.get("code", 0)
    if isinstance(code, str):
        code = {"STATUS_CODE_UNSET": 0, "STATUS_CODE_OK": 1, "STATUS_CODE_ERROR": 2}.get(code, 0)
    return int(code or 0), status.get("message") or ""


# The platform's span classification (ingest/core/src/otlp/normalize.rs,
# PLAN-OTLP-TRACES.md section 4.4): first match wins.
XYBRID_REQUEST_OPERATIONS = ("chat", "text_completion", "generate_content", "embeddings", "predict")


def xybrid_span_type(attributes: Mapping[str, Any]) -> Tuple[str, bool]:
    """``(span_type, is_request)`` the platform derives from a span's attributes."""
    operation = attributes.get("gen_ai.operation.name")
    if operation in XYBRID_REQUEST_OPERATIONS:
        return "genai.request", True
    if attributes.get("xybrid.genai.attempt.index") is not None:
        return "genai.attempt", False
    return {"invoke_workflow": "genai.workflow", "execute_tool": "genai.tool",
            "load_model": "genai.load_model"}.get(str(operation), "other"), False


# Langfuse's OtelIngestionProcessor.extractModelName order (v4.77.0 and
# v3.227.0): the first attribute with a value names the observation's model.
LANGFUSE_MODEL_KEYS = ("langfuse.observation.model.name", "litellm.provider.model", "gen_ai.response.model",
                       "ai.model.id", "gen_ai.request.model", "llm.response.model", "llm.model_name", "model")


def langfuse_wire_model(attributes: Mapping[str, Any]) -> Optional[str]:
    """The model Langfuse stores for a span with these attributes."""
    for key in LANGFUSE_MODEL_KEYS:
        if attributes.get(key):
            return text(attributes[key])
    return None


def langfuse_wire_usage(attributes: Mapping[str, Any]) -> Dict[str, Any]:
    """The usageDetails Langfuse stores: an explicit
    ``langfuse.observation.usage_details`` JSON object wins (extractUsageDetails
    returns it as is, ``{}`` included); otherwise ``gen_ai.usage.*_tokens``."""
    explicit = attributes.get("langfuse.observation.usage_details")
    if explicit:
        parsed = _maybe_json(explicit)
        return parsed if isinstance(parsed, dict) else {}
    usage = {"input": attributes.get("gen_ai.usage.input_tokens"),
             "output": attributes.get("gen_ai.usage.output_tokens")}
    return {key: value for key, value in usage.items() if value is not None}


def wire_spans(destination: str, entries: Sequence[Mapping[str, Any]]) -> List[DestSpan]:
    """The profile's output as the backend would receive it.

    Langfuse fields follow its OTel ingestion (type, input/output, level,
    ``langfuse_wire_model`` and ``langfuse_wire_usage``); Phoenix reads
    the OpenInference attributes as sent; Xybrid reads canonical attributes,
    with ``span_type`` and the counted flag derived by the platform's own rule.
    """
    out = []
    for span, attributes in _wire_records(entries, "traces"):
        trace_id = lower_hex(span.get("traceId")) or ""
        span_id = lower_hex(span.get("spanId")) or ""
        parent = lower_hex(span.get("parentSpanId"))
        code, message = _status(span)
        if destination == "langfuse":
            level = attributes.get("langfuse.observation.level")
            usage = langfuse_wire_usage(attributes)
            out.append(DestSpan(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent, name=span.get("name"),
                kind=(text(attributes.get("langfuse.observation.type")) or "").upper() or None,
                is_error=None if level is None else str(level).upper() == "ERROR",
                status_message=message or None,
                model=langfuse_wire_model(attributes),
                provider=text(attributes.get("langfuse.observation.metadata.provider")),
                input_tokens=number(usage.get("input")),
                output_tokens=number(usage.get("output")),
                input=text(attributes.get("langfuse.observation.input") or attributes.get(TOOL_ARGUMENTS)),
                output=text(attributes.get("langfuse.observation.output") or attributes.get(TOOL_RESULT)),
                first_chunk_s=number(attributes.get(TTFC)), first_token_s=number(attributes.get(TTFT)),
                raw={"span": span, "attributes": attributes}))
        elif destination == "phoenix":
            out.append(openinference_span(
                trace_id, span_id, parent, span.get("name"), text(attributes.get("openinference.span.kind")),
                "ERROR" if code == 2 else ("OK" if code == 1 else "UNSET"), message, attributes,
                {"span": span, "attributes": attributes}))
        else:
            span_type, is_request = xybrid_span_type(attributes)
            out.append(DestSpan(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent, name=span.get("name"),
                kind=span_type, counted=is_request, is_error=code == 2, status_message=message or None,
                error_type=text(attributes.get("error.type")),
                request_model=text(attributes.get("gen_ai.request.model")),
                response_model=text(attributes.get("gen_ai.response.model")),
                provider=text(attributes.get("gen_ai.provider.name")),
                input_tokens=number(attributes.get("gen_ai.usage.input_tokens")),
                output_tokens=number(attributes.get("gen_ai.usage.output_tokens")),
                input=text(attributes.get(INPUT_MESSAGES) or attributes.get(TOOL_ARGUMENTS)),
                output=text(attributes.get(OUTPUT_MESSAGES) or attributes.get(TOOL_RESULT)),
                first_chunk_s=number(attributes.get(TTFC)), first_token_s=number(attributes.get(TTFT)),
                raw={"span": span, "attributes": attributes}))
    return out


def wire_evaluations(entries: Sequence[Mapping[str, Any]]) -> List[DestEvaluation]:
    out = []
    for record, attributes in _wire_records(entries, "logs"):
        if record.get("eventName") != "gen_ai.evaluation.result":
            continue
        name = attributes.get("gen_ai.evaluation.name")
        out.append(DestEvaluation(
            trace_id=lower_hex(record.get("traceId")), span_id=lower_hex(record.get("spanId")), name=name,
            score=number(attributes.get("gen_ai.evaluation.score.value")),
            label=attributes.get("gen_ai.evaluation.score.label"), comment=attributes.get(FEEDBACK_COMMENT),
            source="feedback" if name == "user_feedback" else "evaluation", raw={"record": record}))
    return out


def wire_log_event_names(entries: Sequence[Mapping[str, Any]]) -> List[Optional[str]]:
    return [record.get("eventName") for record, _ in _wire_records(entries, "logs")]


# ---------------------------------------------------------------------------
# HTTP clients
# ---------------------------------------------------------------------------


class HttpError(RuntimeError):
    """A failed read: an HTTP error status, or ``status`` 0 when the request
    never got a usable answer (connection refused or reset, timeout, a body
    that is not JSON)."""

    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"HTTP {status} from {url}: {body[:300]}" if status else f"{url}: {body[:300]}")
        self.status = status
        self.url = url
        self.body = body


Recorder = Callable[[str, str, Any], None]


class Client:
    """A tiny JSON client; every response is handed to ``record`` (raw/)."""

    def __init__(self, base_url: str, headers: Mapping[str, str], record: Optional[Recorder] = None,
                 timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers)
        self.record = record
        self.timeout = timeout

    def get(self, path: str, params: Optional[Sequence[Tuple[str, Any]]] = None, label: str = "",
            raw_text: bool = False) -> Any:
        query = urllib.parse.urlencode([(key, value) for key, value in (params or []) if value is not None])
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        request = urllib.request.Request(url, headers={"Accept": "application/json", **self.headers})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                body = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            raise HttpError(error.code, url, error.read().decode("utf-8", "replace")) from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
            # Refused, reset, timed out (socket.timeout is an OSError) or a
            # broken response: a failed read, not a crash of the whole run.
            raise HttpError(0, url, f"{type(error).__name__}: {error}") from None
        try:
            value: Any = body if raw_text else json.loads(body or "null")
        except ValueError as error:
            raise HttpError(0, url, f"not JSON ({error}): {body[:200]}") from None
        if self.record is not None:
            self.record(label or path, url, value)
        return value


class Langfuse:
    name = "langfuse"

    def __init__(self, base_url: str, public_key: str, secret_key: str, record: Optional[Recorder] = None) -> None:
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.client = Client(base_url, {"Authorization": f"Basic {token}"}, record)
        self.api = "v2"

    def version(self) -> Dict[str, Any]:
        health = self.client.get("/api/public/health", label="health")
        return {"product": "Langfuse", "version": (health or {}).get("version"), "health": health}

    def spans(self, trace_id: str, window: Tuple[str, str]) -> List[DestSpan]:
        pages = []
        if self.api == "v2":
            cursor = None
            try:
                while True:
                    page = self.client.get("/api/public/v2/observations", [
                        ("traceId", trace_id), ("limit", 1000), ("cursor", cursor),
                        ("fromStartTime", window[0]), ("toStartTime", window[1]),
                        ("fields", "core,basic,time,io,metadata,model,usage,metrics"),
                        ("expandMetadata", "attributes,provider,runtime,execution_location"),
                    ], label=f"observations-v2-{trace_id}")
                    pages.append(page)
                    cursor = (page.get("meta") or {}).get("cursor")
                    if not cursor:
                        break
                return langfuse_spans(pages)
            except HttpError as error:
                if error.status != 404:
                    raise
                self.api = "v1"  # self-hosted Langfuse v3
        page_number = 1
        while True:
            page = self.client.get("/api/public/observations", [
                ("traceId", trace_id), ("limit", 100), ("page", page_number)], label=f"observations-v1-{trace_id}")
            pages.append(page)
            meta = page.get("meta") or {}
            if page_number >= int(meta.get("totalPages") or 1):
                break
            page_number += 1
        return langfuse_spans(pages)

    def evaluations(self, trace_id: str, span_ids: Sequence[str]) -> List[DestEvaluation]:
        for path in ("/api/public/v3/scores", "/api/public/v2/scores"):
            try:
                params: List[Tuple[str, Any]] = [("traceId", trace_id), ("limit", 100)]
                if path.endswith("v3/scores"):
                    params.append(("fields", "details,subject"))
                page = self.client.get(path, params, label=f"scores-{trace_id}")
                return langfuse_scores([page])
            except HttpError as error:
                if error.status not in (400, 404):
                    raise
        return []


class Phoenix:
    name = "phoenix"

    def __init__(self, base_url: str, project: str = "default", record: Optional[Recorder] = None,
                 api_key: Optional[str] = None) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = Client(base_url, headers, record)
        self.project = project

    def version(self) -> Dict[str, Any]:
        version = self.client.get("/arize_phoenix_version", label="version", raw_text=True)
        return {"product": "Arize Phoenix", "version": str(version).strip()}

    def spans(self, trace_id: str, window: Tuple[str, str]) -> List[DestSpan]:
        pages, cursor = [], None
        path = f"/v1/projects/{urllib.parse.quote(self.project, safe='')}/spans"
        while True:
            page = self.client.get(path, [("trace_id", trace_id), ("limit", 1000), ("cursor", cursor),
                                          ("start_time", window[0]), ("end_time", window[1])],
                                   label=f"spans-{trace_id}")
            pages.append(page)
            cursor = page.get("next_cursor")
            if not cursor:
                break
        # The filter is server-side; keep only the asked trace all the same.
        return [span for span in phoenix_spans(pages) if span.trace_id == trace_id]

    def evaluations(self, trace_id: str, span_ids: Sequence[str]) -> List[DestEvaluation]:
        if not span_ids:
            return []
        path = f"/v1/projects/{urllib.parse.quote(self.project, safe='')}/span_annotations"
        page = self.client.get(path, [("span_ids", span) for span in span_ids] + [("limit", 1000)],
                               label=f"span-annotations-{trace_id}")
        return phoenix_annotations([page], {span: trace_id for span in span_ids})


class Xybrid:
    name = "xybrid"

    def __init__(self, api_url: str, api_key: str, record: Optional[Recorder] = None,
                 ingest_url: Optional[str] = None) -> None:
        self.client = Client(api_url, {"Authorization": f"Bearer {api_key}"}, record)
        self.ingest_url = ingest_url

    def version(self) -> Dict[str, Any]:
        health = self.client.get("/health", label="health")
        info: Dict[str, Any] = {"product": "Xybrid platform (staging)", "version": (health or {}).get("version"),
                                "health": health, "api_url": self.client.base_url}
        if self.ingest_url:
            try:
                info["ingest_health"] = Client(self.ingest_url, {}).get("/health", label="ingest-health")
            except (HttpError, OSError, ValueError) as error:
                info["ingest_health"] = f"unavailable: {error}"
        return info

    def spans(self, trace_id: str, window: Tuple[str, str]) -> List[DestSpan]:
        try:
            response = self.client.get(f"/v1/telemetry/traces/{trace_id}/spans", label=f"spans-{trace_id}")
        except HttpError as error:
            if error.status == 404:
                return []
            raise
        contents = {}
        for span in response.get("spans") or []:
            stored = span.get("content") or {}
            if stored.get("storage") == "object" and stored.get("values") is None:
                span_id = str(span.get("span_id", "")).lower()
                contents[span_id] = self.client.get(
                    f"/v1/telemetry/traces/{trace_id}/spans/{span_id}/content", label=f"content-{span_id}")
        return xybrid_spans(response, contents)

    def evaluations(self, trace_id: str, span_ids: Sequence[str]) -> List[DestEvaluation]:
        try:
            response = self.client.get(f"/v1/telemetry/traces/{trace_id}/evaluations",
                                       label=f"evaluations-{trace_id}")
        except HttpError as error:
            if error.status == 404:
                return []
            raise
        return xybrid_evaluations(response)
