/**
 * Catalog wire types for OTLP/protobuf request bodies (CONTRACT.md "Wire types").
 *
 * `@opentelemetry/otlp-transformer` writes protobuf straight from SDK objects and encodes every
 * whole JavaScript number as `int_value`, including catalog doubles (`temperature = 0`). This
 * module rewrites a serialized `Export{Trace,Logs,Metrics}ServiceRequest` at the wire level,
 * for the `xybrid-observability` instrumentation scope only: catalog `double` attributes become
 * `double_value`, integral catalog `int` attributes `int_value`, and sum/gauge points follow
 * the metric's catalog `value_type`. Everything else is copied byte for byte; a request
 * without changes is returned unchanged (same array).
 */
import { catalogAttributeType, METRICS, WIRE } from "../catalog.ts";

const VARINT = 0;
const FIXED64 = 1;
const LENGTH = 2;
const FIXED32 = 5;

type Field = {
  readonly number: number;
  readonly wire: number;
  /** Start of the tag. */
  readonly start: number;
  /** Start of the payload (after the length prefix for length-delimited fields). */
  readonly valueStart: number;
  /** End of the field. */
  readonly end: number;
};

/** Reads a varint as a JS number; only used for tags and lengths (< 2^53). */
const readVarint = (buffer: Uint8Array, start: number): [number, number] => {
  let result = 0;
  let scale = 1;
  let position = start;
  for (;;) {
    if (position >= buffer.length) throw new RangeError("truncated protobuf varint");
    const byte = buffer[position] as number;
    position += 1;
    result += (byte & 0x7f) * scale;
    if ((byte & 0x80) === 0) return [result, position];
    scale *= 128;
    if (scale > 2 ** 63) throw new RangeError("protobuf varint too long");
  }
};

const readVarintBig = (buffer: Uint8Array, start: number): bigint => {
  let result = 0n;
  let shift = 0n;
  let position = start;
  for (;;) {
    if (position >= buffer.length || shift > 63n) throw new RangeError("bad protobuf varint");
    const byte = buffer[position] as number;
    position += 1;
    result |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return BigInt.asIntN(64, result);
    shift += 7n;
  }
};

/** Splits `buffer[start, end)` into fields; throws on malformed input. */
const parseFields = (buffer: Uint8Array, start: number, end: number): Field[] => {
  const fields: Field[] = [];
  let position = start;
  while (position < end) {
    const fieldStart = position;
    const [tag, afterTag] = readVarint(buffer, position);
    const number = Math.floor(tag / 8);
    const wire = tag % 8;
    if (number === 0) throw new RangeError("protobuf field number 0");
    let valueStart = afterTag;
    let fieldEnd: number;
    switch (wire) {
      case VARINT:
        fieldEnd = readVarint(buffer, afterTag)[1];
        break;
      case FIXED64:
        fieldEnd = afterTag + 8;
        break;
      case LENGTH: {
        const [length, afterLength] = readVarint(buffer, afterTag);
        valueStart = afterLength;
        fieldEnd = afterLength + length;
        break;
      }
      case FIXED32:
        fieldEnd = afterTag + 4;
        break;
      default:
        throw new RangeError(`unsupported protobuf wire type ${wire}`);
    }
    if (fieldEnd > end) throw new RangeError("truncated protobuf field");
    fields.push({ number, wire, start: fieldStart, valueStart, end: fieldEnd });
    position = fieldEnd;
  }
  return fields;
};

const encodeVarint = (value: number | bigint): Uint8Array => {
  let remaining = BigInt.asUintN(64, BigInt(value));
  const bytes: number[] = [];
  do {
    const low = Number(remaining & 0x7fn);
    remaining >>= 7n;
    bytes.push(remaining === 0n ? low : low | 0x80);
  } while (remaining !== 0n);
  return Uint8Array.from(bytes);
};

const tag = (number: number, wire: number): Uint8Array => encodeVarint(number * 8 + wire);

const encodeDouble = (value: number): Uint8Array => {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value, true);
  return bytes;
};

const encodeSfixed64 = (value: bigint): Uint8Array => {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setBigInt64(0, value, true);
  return bytes;
};

const readDouble = (buffer: Uint8Array, start: number): number =>
  new DataView(buffer.buffer, buffer.byteOffset + start, 8).getFloat64(0, true);

const readSfixed64 = (buffer: Uint8Array, start: number): bigint =>
  new DataView(buffer.buffer, buffer.byteOffset + start, 8).getBigInt64(0, true);

const concat = (parts: readonly Uint8Array[]): Uint8Array => {
  let length = 0;
  for (const part of parts) length += part.length;
  const result = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
};

const MIN_INT64 = -(2 ** 63);
const MAX_INT64 = 2 ** 63;
const decoder = new TextDecoder();

/** Rewrites a message payload; `undefined` means unchanged. */
type Rewriter = (buffer: Uint8Array, start: number, end: number) => Uint8Array | undefined;

/**
 * A message whose length-delimited `children` fields are rewritten recursively. `scalar` may
 * replace a non-length-delimited field (returning the complete new field, tag included).
 */
const message =
  (
    children: Readonly<Record<number, Rewriter>>,
    scalar?: (buffer: Uint8Array, field: Field) => Uint8Array | undefined,
  ): Rewriter =>
  (buffer, start, end) => {
    const fields = parseFields(buffer, start, end);
    let changed = false;
    const parts: Uint8Array[] = [];
    for (const field of fields) {
      let replacement: Uint8Array | undefined;
      if (field.wire === LENGTH) {
        const child = children[field.number];
        const rewritten = child?.(buffer, field.valueStart, field.end);
        if (rewritten !== undefined) {
          replacement = concat([
            tag(field.number, LENGTH),
            encodeVarint(rewritten.length),
            rewritten,
          ]);
        }
      } else if (scalar !== undefined) {
        replacement = scalar(buffer, field);
      }
      if (replacement === undefined) {
        parts.push(buffer.subarray(field.start, field.end));
      } else {
        changed = true;
        parts.push(replacement);
      }
    }
    return changed ? concat(parts) : undefined;
  };

/** `KeyValue { key = 1; value = 2 }` with the value coerced to the key's catalog type. */
const keyValue: Rewriter = (buffer, start, end) => {
  const fields = parseFields(buffer, start, end);
  let key: string | undefined;
  for (const field of fields) {
    if (field.number === 1 && field.wire === LENGTH) {
      key = decoder.decode(buffer.subarray(field.valueStart, field.end));
    }
  }
  if (key === undefined) return undefined;
  const type = catalogAttributeType(key);
  if (type !== "double" && type !== "int") return undefined;
  return message({
    2: (valueBuffer, valueStart, valueEnd) => {
      const valueFields = parseFields(valueBuffer, valueStart, valueEnd);
      if (valueFields.length !== 1) return undefined;
      const [only] = valueFields as [Field];
      if (type === "double" && only.number === 3 && only.wire === VARINT) {
        // AnyValue.int_value (3, varint) -> AnyValue.double_value (4, fixed64).
        const integer = readVarintBig(valueBuffer, only.valueStart);
        return concat([tag(4, FIXED64), encodeDouble(Number(integer))]);
      }
      if (type === "int" && only.number === 4 && only.wire === FIXED64) {
        const number = readDouble(valueBuffer, only.valueStart);
        if (Number.isInteger(number) && number >= MIN_INT64 && number < MAX_INT64) {
          return concat([tag(3, VARINT), encodeVarint(BigInt(number))]);
        }
      }
      return undefined;
    },
  })(buffer, start, end);
};

/** A `Scope*` message: rewritten only when its scope (field 1) is the enforced scope. */
const scoped =
  (children: Readonly<Record<number, Rewriter>>): Rewriter =>
  (buffer, start, end) => {
    const fields = parseFields(buffer, start, end);
    const scope = fields.find((field) => field.number === 1 && field.wire === LENGTH);
    if (scope === undefined) return undefined;
    const name = parseFields(buffer, scope.valueStart, scope.end).find(
      (field) => field.number === 1 && field.wire === LENGTH,
    );
    if (name === undefined) return undefined;
    if (decoder.decode(buffer.subarray(name.valueStart, name.end)) !== WIRE.enforced_scope) {
      return undefined;
    }
    return message(children)(buffer, start, end);
  };

const attributes = (field: number): Readonly<Record<number, Rewriter>> => ({ [field]: keyValue });

// Traces: Request.resource_spans(1) > ResourceSpans.scope_spans(2) > ScopeSpans.spans(2) >
// Span.attributes(9), Span.events(11) > Event.attributes(3), Span.links(13) > Link.attributes(4).
const span = message({ 9: keyValue, 11: message(attributes(3)), 13: message(attributes(4)) });
const traceRequest = message({ 1: message({ 2: scoped({ 2: span }) }) });

// Logs: Request.resource_logs(1) > ResourceLogs.scope_logs(2) > ScopeLogs.log_records(2) >
// LogRecord.attributes(6). The body and nested structured values are JSON content: untouched.
const logRequest = message({ 1: message({ 2: scoped({ 2: message(attributes(6)) }) }) });

/** `NumberDataPoint`: attributes(7); as_double(4) / as_int(6) per the metric value type. */
const numberPoint = (valueType: "int" | "double" | undefined): Rewriter =>
  message(attributes(7), (buffer, field) => {
    if (valueType === "double" && field.number === 6 && field.wire === FIXED64) {
      return concat([
        tag(4, FIXED64),
        encodeDouble(Number(readSfixed64(buffer, field.valueStart))),
      ]);
    }
    if (valueType === "int" && field.number === 4 && field.wire === FIXED64) {
      const number = readDouble(buffer, field.valueStart);
      if (Number.isInteger(number) && number >= MIN_INT64 && number < MAX_INT64) {
        return concat([tag(6, FIXED64), encodeSfixed64(BigInt(number))]);
      }
    }
    return undefined;
  });

const metric: Rewriter = (buffer, start, end) => {
  const name = parseFields(buffer, start, end).find(
    (field) => field.number === 1 && field.wire === LENGTH,
  );
  const metricName =
    name === undefined ? undefined : decoder.decode(buffer.subarray(name.valueStart, name.end));
  const valueType =
    metricName !== undefined && Object.hasOwn(METRICS, metricName)
      ? METRICS[metricName as keyof typeof METRICS].value_type
      : undefined;
  const points = numberPoint(valueType);
  // Metric.gauge(5), sum(7) > data_points(1): NumberDataPoint; histogram(9) >
  // HistogramDataPoint.attributes(9); exponential_histogram(10) > attributes(1);
  // summary(11) > SummaryDataPoint.attributes(7).
  return message({
    5: message({ 1: points }),
    7: message({ 1: points }),
    9: message({ 1: message(attributes(9)) }),
    10: message({ 1: message(attributes(1)) }),
    11: message({ 1: message(attributes(7)) }),
  })(buffer, start, end);
};

const metricRequest = message({ 1: message({ 2: scoped({ 2: metric }) }) });

const REQUESTS = { traces: traceRequest, logs: logRequest, metrics: metricRequest } as const;

/**
 * Enforces catalog wire types on a serialized OTLP/protobuf export request of `signal`.
 * Returns `body` itself when nothing changed. Throws `RangeError` on malformed protobuf.
 */
export const enforceProtobufCatalogTypes = (
  signal: keyof typeof REQUESTS,
  body: Uint8Array,
): Uint8Array => REQUESTS[signal](body, 0, body.length) ?? body;
