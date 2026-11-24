/**
 * Catalog wire types for OTLP encoders (CONTRACT.md "Wire types").
 *
 * OpenTelemetry JS has a single `number` type, so `@opentelemetry/otlp-transformer` writes every
 * whole number as `intValue`, including catalog doubles such as `temperature = 0` or a time to
 * first chunk of exactly 1 s. The package's exporters apply these helpers to telemetry of the
 * `xybrid-observability` scope before serialization; other scopes are never rewritten. Pure
 * functions with no SDK dependency.
 */
import { type AttributeType, catalogAttributeType, METRICS, WIRE } from "./catalog.ts";

/** The OTLP numeric field a catalog-typed number is encoded with. */
export type NumberWireType = "int" | "double";

const MIN_INT64 = -(2 ** 63);
const MAX_INT64 = 2 ** 63;

/**
 * How a number attribute is encoded: catalog `double` keys always as a double, catalog `int`
 * keys as an integer when the value is integral, anything else by the transformer's own rule
 * (integral numbers in the int64 range as integers).
 */
export const numberWireType = (key: string, value: number): NumberWireType => {
  const type: AttributeType | undefined = catalogAttributeType(key);
  if (type === "double") return "double";
  const integral = Number.isInteger(value) && value >= MIN_INT64 && value < MAX_INT64;
  return integral ? "int" : "double";
};

type AnyValueJson = {
  intValue?: unknown;
  doubleValue?: unknown;
  [field: string]: unknown;
};
type KeyValueJson = { key?: unknown; value?: AnyValueJson | null };

const toNumber = (value: unknown): number | undefined => {
  if (typeof value === "number") return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "object" && value !== null) {
    // protobufjs Long-like { low, high, unsigned }.
    const { low, high, unsigned } = value as { low?: unknown; high?: unknown; unsigned?: unknown };
    if (typeof low === "number" && typeof high === "number") {
      return (unsigned === true ? high >>> 0 : high) * 2 ** 32 + (low >>> 0);
    }
  }
  return undefined;
};

/** Rewrites one OTLP/JSON `AnyValue` in place to the catalog type of `key`. */
const coerceValue = (key: string, value: AnyValueJson | null | undefined): void => {
  if (value === null || value === undefined || typeof value !== "object") return;
  const type = catalogAttributeType(key);
  if (type === "double" && value.intValue !== undefined && value.intValue !== null) {
    const number = toNumber(value.intValue);
    if (number === undefined) return;
    delete value.intValue;
    value.doubleValue = number;
  } else if (
    type === "int" &&
    typeof value.doubleValue === "number" &&
    Number.isInteger(value.doubleValue)
  ) {
    const number = value.doubleValue;
    delete value.doubleValue;
    value.intValue = number;
  }
};

const coerceAttributes = (attributes: unknown): void => {
  if (!Array.isArray(attributes)) return;
  for (const attribute of attributes as KeyValueJson[]) {
    if (attribute !== null && typeof attribute === "object" && typeof attribute.key === "string") {
      coerceValue(attribute.key, attribute.value);
    }
  }
};

const enforced = (scope: unknown): boolean =>
  typeof scope === "object" &&
  scope !== null &&
  (scope as { name?: unknown }).name === WIRE.enforced_scope;

const list = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);
const field = (value: unknown, name: string): unknown =>
  typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)[name]
    : undefined;

/** Number data points per the metric's catalog `value_type` (`asInt` / `asDouble`). */
const coercePoint = (point: unknown, valueType: "int" | "double"): void => {
  if (typeof point !== "object" || point === null) return;
  const record = point as Record<string, unknown>;
  if (valueType === "double" && record["asInt"] !== undefined && record["asInt"] !== null) {
    const number = toNumber(record["asInt"]);
    if (number === undefined) return;
    delete record["asInt"];
    record["asDouble"] = number;
  } else if (
    valueType === "int" &&
    typeof record["asDouble"] === "number" &&
    Number.isInteger(record["asDouble"])
  ) {
    const number = record["asDouble"];
    delete record["asDouble"];
    record["asInt"] = number;
  }
};

/**
 * Enforces catalog value types, in place, on an OTLP/JSON-shaped export request (traces, logs
 * or metrics, as `createExport*ServiceRequest` builds them) and returns it. Only telemetry of
 * the `xybrid-observability` scope is rewritten: span, span event and link attributes, top-level
 * log record attributes (structured `json` content is left as JSON numbers), metric data point
 * attributes, and sum/gauge point values per the metric's `value_type`.
 */
export const enforceCatalogTypes = <T>(request: T): T => {
  for (const resource of list(field(request, "resourceSpans"))) {
    for (const scope of list(field(resource, "scopeSpans"))) {
      if (!enforced(field(scope, "scope"))) continue;
      for (const span of list(field(scope, "spans"))) {
        coerceAttributes(field(span, "attributes"));
        for (const event of list(field(span, "events")))
          coerceAttributes(field(event, "attributes"));
        for (const link of list(field(span, "links"))) coerceAttributes(field(link, "attributes"));
      }
    }
  }
  for (const resource of list(field(request, "resourceLogs"))) {
    for (const scope of list(field(resource, "scopeLogs"))) {
      if (!enforced(field(scope, "scope"))) continue;
      for (const record of list(field(scope, "logRecords"))) {
        coerceAttributes(field(record, "attributes"));
      }
    }
  }
  for (const resource of list(field(request, "resourceMetrics"))) {
    for (const scope of list(field(resource, "scopeMetrics"))) {
      if (!enforced(field(scope, "scope"))) continue;
      for (const metric of list(field(scope, "metrics"))) {
        const name = field(metric, "name");
        const spec =
          typeof name === "string" && Object.hasOwn(METRICS, name)
            ? METRICS[name as keyof typeof METRICS]
            : undefined;
        for (const kind of ["sum", "gauge", "histogram", "exponentialHistogram", "summary"]) {
          for (const point of list(field(field(metric, kind), "dataPoints"))) {
            coerceAttributes(field(point, "attributes"));
            if (spec !== undefined && (kind === "sum" || kind === "gauge")) {
              coercePoint(point, spec.value_type);
            }
          }
        }
      }
    }
  }
  return request;
};
