/**
 * Request encoding with catalog wire types, and conservative per-item size estimates.
 *
 * Encoding uses `@opentelemetry/otlp-transformer`'s serializers, then enforces catalog types for
 * the `xybrid-observability` scope: OTLP/JSON by rewriting the decoded request
 * ({@link enforceCatalogTypes}), OTLP/protobuf at the wire level. Structured null stays an
 * `AnyValue` without a value (`{}` in JSON) and an empty string stays `stringValue: ""`.
 */
import type { Attributes, AttributeValue } from "@opentelemetry/api";
import {
  JsonLogsSerializer,
  JsonMetricsSerializer,
  JsonTraceSerializer,
  ProtobufLogsSerializer,
  ProtobufMetricsSerializer,
  ProtobufTraceSerializer,
} from "@opentelemetry/otlp-transformer";
import type { ReadableLogRecord } from "@opentelemetry/sdk-logs";
import type { ResourceMetrics } from "@opentelemetry/sdk-metrics";
import type { ReadableSpan } from "@opentelemetry/sdk-trace";
import { WIRE } from "../catalog.ts";
import { enforceCatalogTypes } from "../wire.ts";
import type { OtlpProtocol } from "./config.ts";
import { enforceProtobufCatalogTypes } from "./protobuf-wire.ts";

type Serializer<T> = { serializeRequest(request: T): Uint8Array | undefined };

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const SCOPE_MARKER = JSON.stringify(WIRE.enforced_scope);

const jsonWithCatalogTypes = (body: Uint8Array): Uint8Array => {
  const text = decoder.decode(body);
  // Requests without the enforced scope are sent as serialized.
  if (!text.includes(SCOPE_MARKER)) return body;
  return encoder.encode(JSON.stringify(enforceCatalogTypes(JSON.parse(text))));
};

const serialize = <T>(serializer: Serializer<T>, request: T): Uint8Array => {
  const body = serializer.serializeRequest(request);
  if (body === undefined) throw new TypeError("the OTLP serializer returned no body");
  return body;
};

/** Encodes one request; throws when the items cannot be encoded. */
export type Encode<T> = (request: T) => Uint8Array;

const encoderFor = <T>(
  signal: "traces" | "logs" | "metrics",
  protocol: OtlpProtocol,
  serializers: { readonly protobuf: Serializer<T>; readonly json: Serializer<T> },
): Encode<T> =>
  protocol === "http/json"
    ? (request) => jsonWithCatalogTypes(serialize(serializers.json, request))
    : (request) => enforceProtobufCatalogTypes(signal, serialize(serializers.protobuf, request));

export const traceEncoder = (protocol: OtlpProtocol): Encode<ReadableSpan[]> =>
  encoderFor("traces", protocol, { protobuf: ProtobufTraceSerializer, json: JsonTraceSerializer });

export const logEncoder = (protocol: OtlpProtocol): Encode<ReadableLogRecord[]> =>
  encoderFor("logs", protocol, { protobuf: ProtobufLogsSerializer, json: JsonLogsSerializer });

export const metricEncoder = (protocol: OtlpProtocol): Encode<ResourceMetrics> =>
  encoderFor("metrics", protocol, {
    protobuf: ProtobufMetricsSerializer,
    json: JsonMetricsSerializer,
  });

type PartialSuccess = { partialSuccess?: Record<string, unknown> | undefined };
type Deserializer = { deserializeResponse(data: Uint8Array): unknown };

const REJECTED = {
  traces: ["rejectedSpans", "rejected_spans"],
  logs: ["rejectedLogRecords", "rejected_log_records"],
  metrics: ["rejectedDataPoints", "rejected_data_points"],
} as const;

const PROTOBUF_RESPONSES: Readonly<Record<"traces" | "logs" | "metrics", Deserializer>> = {
  traces: ProtobufTraceSerializer,
  logs: ProtobufLogsSerializer,
  metrics: ProtobufMetricsSerializer,
};

/**
 * Rejected item count of a 2xx OTLP response body, or `undefined` when the body is empty or does
 * not decode (the whole request then counts as exported).
 */
export const rejectedItems = (
  signal: "traces" | "logs" | "metrics",
  protocol: OtlpProtocol,
  body: Uint8Array,
): number | undefined => {
  if (body.length === 0) return undefined;
  let response: unknown;
  try {
    if (protocol === "http/json") {
      response = JSON.parse(decoder.decode(body));
    } else {
      response = PROTOBUF_RESPONSES[signal].deserializeResponse(body);
    }
  } catch {
    return undefined;
  }
  if (typeof response !== "object" || response === null) return undefined;
  const record = response as PartialSuccess & { partial_success?: Record<string, unknown> };
  const partial = record.partialSuccess ?? record.partial_success;
  if (typeof partial !== "object" || partial === null) return undefined;
  for (const key of REJECTED[signal]) {
    const value = partial[key];
    const count = typeof value === "string" && /^\d+$/.test(value) ? Number(value) : value;
    if (typeof count === "number" && Number.isSafeInteger(count) && count >= 0) return count;
  }
  return undefined;
};

/** Data points in one metrics collection. */
export const countDataPoints = (metrics: ResourceMetrics): number => {
  let points = 0;
  for (const scope of metrics.scopeMetrics) {
    for (const metric of scope.metrics) points += metric.dataPoints.length;
  }
  return points;
};

// Conservative estimates (CONTRACT.md "Lifecycle and delivery"). The constants bound the
// OTLP/JSON encoding (the larger of the two): JSON-escaped string lengths, field names, and
// decimal numbers; protobuf is always smaller. Every item is charged its own instrumentation
// scope (name, version, schema URL, attributes), as in the Rust estimator: distinct scopes can be
// as large as item attributes, and grouping happens only after admission. The item base covers
// fixed field overhead and ordinary resource overhead; shared application resource attributes
// are not charged per item.
const ITEM_BASE = 1024;
const EVENT_BASE = 128;
const LINK_BASE = 192;
const ATTRIBUTE_OVERHEAD = 48;
const SCALAR = 32;
const ARRAY_OVERHEAD = 32;
const ELEMENT_OVERHEAD = 24;
const NULL_VALUE = 8;
const MAX_DEPTH = 128;

/** Bytes of `value` as a JSON string literal (quotes and escapes included; >= UTF-8 length). */
export const jsonStringBytes = (value: string): number => {
  let bytes = 2;
  for (let index = 0; index < value.length; index += 1) {
    const unit = value.charCodeAt(index);
    if (unit === 0x22 || unit === 0x5c) bytes += 2;
    else if (unit < 0x20)
      bytes += unit === 8 || unit === 9 || unit === 10 || unit === 12 || unit === 13 ? 2 : 6;
    else if (unit < 0x80) bytes += 1;
    else if (unit < 0x800) bytes += 2;
    else if (unit >= 0xd800 && unit <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        bytes += 4;
        index += 1;
      } else bytes += 6;
    } else if (unit >= 0xdc00 && unit <= 0xdfff) bytes += 6;
    else bytes += 3;
  }
  return bytes;
};

const valueBytes = (value: unknown, depth: number, path: Set<object>): number => {
  if (typeof value === "string") return jsonStringBytes(value);
  if (typeof value === "number" || typeof value === "boolean" || typeof value === "bigint") {
    return SCALAR;
  }
  if (value === null || typeof value !== "object") return NULL_VALUE;
  if (depth >= MAX_DEPTH || path.has(value)) return Number.POSITIVE_INFINITY;
  if (value instanceof Uint8Array) return 2 * value.length + SCALAR;
  path.add(value);
  try {
    let bytes = ARRAY_OVERHEAD;
    if (Array.isArray(value)) {
      for (const element of value) bytes += ELEMENT_OVERHEAD + valueBytes(element, depth + 1, path);
    } else {
      for (const key of Object.keys(value)) {
        bytes +=
          ATTRIBUTE_OVERHEAD +
          jsonStringBytes(key) +
          valueBytes((value as Record<string, unknown>)[key], depth + 1, path);
      }
    }
    return bytes;
  } finally {
    path.delete(value);
  }
};

const attributesBytes = (attributes: Attributes | undefined): number => {
  if (attributes === undefined || attributes === null) return 0;
  let bytes = 0;
  const path = new Set<object>();
  for (const key of Object.keys(attributes)) {
    const value: AttributeValue | undefined = attributes[key];
    bytes += ATTRIBUTE_OVERHEAD + jsonStringBytes(key) + valueBytes(value, 1, path);
  }
  return bytes;
};

const optionalString = (value: string | undefined): number =>
  typeof value === "string" ? jsonStringBytes(value) : 0;

/** Scope fields the OTLP encoders write (`attributes` for any scope that carries them, e.g. logs). */
type ScopeLike = {
  readonly name: string;
  readonly version?: string | undefined;
  readonly schemaUrl?: string | undefined;
  readonly attributes?: Attributes | undefined;
};

/** Encoded size of an item's instrumentation scope, charged per item. */
const scopeBytes = (scope: ScopeLike | undefined): number =>
  scope === undefined
    ? 0
    : ATTRIBUTE_OVERHEAD +
      optionalString(scope.name) +
      optionalString(scope.version) +
      optionalString(scope.schemaUrl) +
      attributesBytes(scope.attributes);

/** Conservative encoded size of a span's share of a request (never below the encoded size). */
export const estimateSpanBytes = (span: ReadableSpan): number => {
  let bytes =
    ITEM_BASE +
    scopeBytes(span.instrumentationScope) +
    jsonStringBytes(span.name) +
    optionalString(span.status?.message) +
    optionalString(span.spanContext().traceState?.serialize()) +
    attributesBytes(span.attributes);
  for (const event of span.events) {
    bytes += EVENT_BASE + jsonStringBytes(event.name) + attributesBytes(event.attributes);
  }
  for (const link of span.links) {
    bytes +=
      LINK_BASE +
      optionalString(link.context.traceState?.serialize()) +
      attributesBytes(link.attributes);
  }
  return bytes;
};

/** Conservative encoded size of a log record's share of a request. */
export const estimateLogBytes = (record: ReadableLogRecord): number =>
  ITEM_BASE +
  scopeBytes(record.instrumentationScope) +
  (record.body === undefined ? 0 : valueBytes(record.body, 1, new Set())) +
  optionalString(record.severityText) +
  optionalString(record.eventName) +
  attributesBytes(record.attributes);
