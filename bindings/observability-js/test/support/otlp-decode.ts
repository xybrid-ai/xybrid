/**
 * Decodes captured OTLP request bodies into the OTLP/JSON shape, so wire tests assert the same
 * field names (`intValue`/`doubleValue`, `asInt`/`asDouble`, null as `{}`) for both encodings.
 * The protobuf decoder is schema-driven from the OTLP `.proto` field numbers and independent of
 * the package's own wire rewriter.
 */

type Kind =
  | "string"
  | "hex"
  | "base64"
  | "int64"
  | "uint"
  | "bool"
  | "fixed64"
  | "sfixed64"
  | "double"
  | "fixed32"
  | "packedFixed64"
  | "packedDouble";

type FieldSpec = { readonly name: string; readonly type: Kind | Schema; readonly repeated?: true };
type Schema = { readonly [field: number]: FieldSpec };

const KEY_VALUE: Schema = {};
const ANY_VALUE: Schema = {};
const ARRAY_VALUE: Schema = { 1: { name: "values", type: ANY_VALUE, repeated: true } };
const KEY_VALUE_LIST: Schema = { 1: { name: "values", type: KEY_VALUE, repeated: true } };
Object.assign(KEY_VALUE, {
  1: { name: "key", type: "string" },
  2: { name: "value", type: ANY_VALUE },
});
Object.assign(ANY_VALUE, {
  1: { name: "stringValue", type: "string" },
  2: { name: "boolValue", type: "bool" },
  3: { name: "intValue", type: "int64" },
  4: { name: "doubleValue", type: "double" },
  5: { name: "arrayValue", type: ARRAY_VALUE },
  6: { name: "kvlistValue", type: KEY_VALUE_LIST },
  7: { name: "bytesValue", type: "base64" },
});
const ATTRIBUTES = (field: number): Schema => ({
  [field]: { name: "attributes", type: KEY_VALUE, repeated: true },
});
const RESOURCE: Schema = { ...ATTRIBUTES(1), 2: { name: "droppedAttributesCount", type: "uint" } };
const SCOPE: Schema = {
  1: { name: "name", type: "string" },
  2: { name: "version", type: "string" },
  ...ATTRIBUTES(3),
  4: { name: "droppedAttributesCount", type: "uint" },
};
const EVENT: Schema = {
  1: { name: "timeUnixNano", type: "fixed64" },
  2: { name: "name", type: "string" },
  ...ATTRIBUTES(3),
  4: { name: "droppedAttributesCount", type: "uint" },
};
const LINK: Schema = {
  1: { name: "traceId", type: "hex" },
  2: { name: "spanId", type: "hex" },
  3: { name: "traceState", type: "string" },
  ...ATTRIBUTES(4),
  5: { name: "droppedAttributesCount", type: "uint" },
  6: { name: "flags", type: "fixed32" },
};
const SPAN: Schema = {
  1: { name: "traceId", type: "hex" },
  2: { name: "spanId", type: "hex" },
  3: { name: "traceState", type: "string" },
  4: { name: "parentSpanId", type: "hex" },
  5: { name: "name", type: "string" },
  6: { name: "kind", type: "uint" },
  7: { name: "startTimeUnixNano", type: "fixed64" },
  8: { name: "endTimeUnixNano", type: "fixed64" },
  ...ATTRIBUTES(9),
  10: { name: "droppedAttributesCount", type: "uint" },
  11: { name: "events", type: EVENT, repeated: true },
  12: { name: "droppedEventsCount", type: "uint" },
  13: { name: "links", type: LINK, repeated: true },
  14: { name: "droppedLinksCount", type: "uint" },
  15: {
    name: "status",
    type: { 2: { name: "message", type: "string" }, 3: { name: "code", type: "uint" } },
  },
  16: { name: "flags", type: "fixed32" },
};
const TRACES: Schema = {
  1: {
    name: "resourceSpans",
    repeated: true,
    type: {
      1: { name: "resource", type: RESOURCE },
      2: {
        name: "scopeSpans",
        repeated: true,
        type: {
          1: { name: "scope", type: SCOPE },
          2: { name: "spans", type: SPAN, repeated: true },
          3: { name: "schemaUrl", type: "string" },
        },
      },
      3: { name: "schemaUrl", type: "string" },
    },
  },
};
const LOG_RECORD: Schema = {
  1: { name: "timeUnixNano", type: "fixed64" },
  2: { name: "severityNumber", type: "uint" },
  3: { name: "severityText", type: "string" },
  5: { name: "body", type: ANY_VALUE },
  ...ATTRIBUTES(6),
  7: { name: "droppedAttributesCount", type: "uint" },
  8: { name: "flags", type: "fixed32" },
  9: { name: "traceId", type: "hex" },
  10: { name: "spanId", type: "hex" },
  11: { name: "observedTimeUnixNano", type: "fixed64" },
  12: { name: "eventName", type: "string" },
};
const LOGS: Schema = {
  1: {
    name: "resourceLogs",
    repeated: true,
    type: {
      1: { name: "resource", type: RESOURCE },
      2: {
        name: "scopeLogs",
        repeated: true,
        type: {
          1: { name: "scope", type: SCOPE },
          2: { name: "logRecords", type: LOG_RECORD, repeated: true },
          3: { name: "schemaUrl", type: "string" },
        },
      },
      3: { name: "schemaUrl", type: "string" },
    },
  },
};
const NUMBER_POINT: Schema = {
  2: { name: "startTimeUnixNano", type: "fixed64" },
  3: { name: "timeUnixNano", type: "fixed64" },
  4: { name: "asDouble", type: "double" },
  6: { name: "asInt", type: "sfixed64" },
  ...ATTRIBUTES(7),
  8: { name: "flags", type: "uint" },
};
const HISTOGRAM_POINT: Schema = {
  2: { name: "startTimeUnixNano", type: "fixed64" },
  3: { name: "timeUnixNano", type: "fixed64" },
  4: { name: "count", type: "fixed64" },
  5: { name: "sum", type: "double" },
  6: { name: "bucketCounts", type: "packedFixed64" },
  7: { name: "explicitBounds", type: "packedDouble" },
  ...ATTRIBUTES(9),
  10: { name: "flags", type: "uint" },
  11: { name: "min", type: "double" },
  12: { name: "max", type: "double" },
};
const METRIC: Schema = {
  1: { name: "name", type: "string" },
  2: { name: "description", type: "string" },
  3: { name: "unit", type: "string" },
  5: { name: "gauge", type: { 1: { name: "dataPoints", type: NUMBER_POINT, repeated: true } } },
  7: {
    name: "sum",
    type: {
      1: { name: "dataPoints", type: NUMBER_POINT, repeated: true },
      2: { name: "aggregationTemporality", type: "uint" },
      3: { name: "isMonotonic", type: "bool" },
    },
  },
  9: {
    name: "histogram",
    type: {
      1: { name: "dataPoints", type: HISTOGRAM_POINT, repeated: true },
      2: { name: "aggregationTemporality", type: "uint" },
    },
  },
};
const METRICS_REQUEST: Schema = {
  1: {
    name: "resourceMetrics",
    repeated: true,
    type: {
      1: { name: "resource", type: RESOURCE },
      2: {
        name: "scopeMetrics",
        repeated: true,
        type: {
          1: { name: "scope", type: SCOPE },
          2: { name: "metrics", type: METRIC, repeated: true },
          3: { name: "schemaUrl", type: "string" },
        },
      },
    },
  },
};

class Reader {
  position: number;
  constructor(
    readonly buffer: Uint8Array,
    start: number,
    readonly end: number,
  ) {
    this.position = start;
  }
  varint(): bigint {
    let result = 0n;
    let shift = 0n;
    for (;;) {
      if (this.position >= this.end) throw new Error("truncated varint");
      const byte = this.buffer[this.position++] as number;
      result |= BigInt(byte & 0x7f) << shift;
      if ((byte & 0x80) === 0) return result;
      shift += 7n;
    }
  }
  view(length: number): DataView {
    if (this.position + length > this.end) throw new Error("truncated field");
    const view = new DataView(this.buffer.buffer, this.buffer.byteOffset + this.position, length);
    this.position += length;
    return view;
  }
  bytes(): Uint8Array {
    const length = Number(this.varint());
    if (this.position + length > this.end) throw new Error("truncated bytes");
    const bytes = this.buffer.subarray(this.position, this.position + length);
    this.position += length;
    return bytes;
  }
}

const toNumber = (value: bigint): number | string =>
  value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER)
    ? Number(value)
    : value.toString();

const decodeMessage = (buffer: Uint8Array, start: number, end: number, schema: Schema) => {
  const reader = new Reader(buffer, start, end);
  const result: Record<string, unknown> = {};
  while (reader.position < end) {
    const tag = Number(reader.varint());
    const number = tag >>> 3;
    const wire = tag & 7;
    const spec = schema[number];
    let value: unknown;
    if (wire === 0) {
      const raw = reader.varint();
      if (spec === undefined) continue;
      value =
        spec.type === "bool"
          ? raw !== 0n
          : spec.type === "int64"
            ? toNumber(BigInt.asIntN(64, raw))
            : toNumber(raw);
    } else if (wire === 1) {
      const view = reader.view(8);
      if (spec === undefined) continue;
      value =
        spec.type === "double"
          ? view.getFloat64(0, true)
          : spec.type === "sfixed64"
            ? toNumber(view.getBigInt64(0, true))
            : view.getBigUint64(0, true).toString();
    } else if (wire === 5) {
      const view = reader.view(4);
      if (spec === undefined) continue;
      value = view.getUint32(0, true);
    } else if (wire === 2) {
      const bytes = reader.bytes();
      if (spec === undefined) continue;
      if (spec.type === "string") value = new TextDecoder().decode(bytes);
      else if (spec.type === "hex") value = Buffer.from(bytes).toString("hex");
      else if (spec.type === "base64") value = Buffer.from(bytes).toString("base64");
      else if (spec.type === "packedFixed64" || spec.type === "packedDouble") {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
        const values: (number | string)[] = [];
        for (let offset = 0; offset < bytes.length; offset += 8) {
          values.push(
            spec.type === "packedDouble"
              ? view.getFloat64(offset, true)
              : view.getBigUint64(offset, true).toString(),
          );
        }
        value = values;
      } else if (typeof spec.type === "object") {
        value = decodeMessage(bytes, 0, bytes.length, spec.type);
      } else {
        throw new Error(`unexpected length-delimited ${spec.name}`);
      }
    } else {
      throw new Error(`unsupported wire type ${wire}`);
    }
    if (spec.repeated === true) {
      const list = (result[spec.name] as unknown[] | undefined) ?? [];
      list.push(value);
      result[spec.name] = list;
    } else {
      result[spec.name] = value;
    }
  }
  return result;
};

const SCHEMAS = { traces: TRACES, logs: LOGS, metrics: METRICS_REQUEST } as const;

export type Signal = keyof typeof SCHEMAS;

/** Decodes a captured request body (protobuf or JSON) into the OTLP/JSON shape. */
export const decodeRequest = (
  signal: Signal,
  contentType: string | undefined,
  body: Uint8Array,
  // biome-ignore lint/suspicious/noExplicitAny: decoded wire data is inspected dynamically in tests.
): any =>
  contentType?.includes("json") === true
    ? JSON.parse(new TextDecoder().decode(body))
    : decodeMessage(body, 0, body.length, SCHEMAS[signal]);

type KeyValueJson = { key: string; value: Record<string, unknown> };

/** The `AnyValue` of `key` in an OTLP/JSON attribute list. */
export const attribute = (
  attributes: readonly KeyValueJson[] | undefined,
  key: string,
): Record<string, unknown> | undefined => attributes?.find((entry) => entry.key === key)?.value;
