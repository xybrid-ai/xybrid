import {
  type Context,
  context as contextApi,
  createTraceState,
  ROOT_CONTEXT,
  type SpanContext,
  trace,
} from "@opentelemetry/api";

/** Identifies one recorded span: lowercase hex trace and span ids. */
export type ObservationRef = {
  readonly traceId: string;
  readonly spanId: string;
};

/** Structured-clone-safe form of an {@link ObservationContext}, for `postMessage`. */
export type SerializedObservationContext = {
  readonly traceparent: string;
  readonly tracestate?: string;
};

const TRACE_ID = /^[0-9a-f]{32}$/;
const SPAN_ID = /^[0-9a-f]{16}$/;
const HEX_BYTE = /^[0-9a-f]{2}$/;
const INVALID_TRACE_ID = "0".repeat(32);
const INVALID_SPAN_ID = "0".repeat(16);
const MAX_TRACESTATE_MEMBERS = 32;
// W3C Trace Context level 1, section 3.3.1.3: simple key or tenant@system key; printable
// ASCII value without ',' or '=' that does not end in a space.
const TRACESTATE_KEY =
  /^(?:[a-z][a-z0-9_\-*/]{0,255}|[a-z0-9][a-z0-9_\-*/]{0,240}@[a-z][a-z0-9_\-*/]{0,13})$/;
const TRACESTATE_VALUE = /^[\x20-\x2b\x2d-\x3c\x3e-\x7e]{0,255}[\x21-\x2b\x2d-\x3c\x3e-\x7e]$/;
const SAMPLED = 0x01;

/**
 * Normalizes a W3C `tracestate` header: trims optional whitespace and empty members.
 * Returns `undefined` when the header is absent, empty, or invalid; an invalid tracestate is
 * discarded as a whole while the traceparent stays usable.
 */
export const normalizeTracestate = (value: string | undefined | null): string | undefined => {
  if (value === undefined || value === null) return undefined;
  const members: string[] = [];
  const keys = new Set<string>();
  for (const raw of value.split(",")) {
    const member = raw.trim();
    if (member.length === 0) continue;
    const equals = member.indexOf("=");
    if (equals <= 0) return undefined;
    const key = member.slice(0, equals);
    const memberValue = member.slice(equals + 1);
    if (!TRACESTATE_KEY.test(key) || !TRACESTATE_VALUE.test(memberValue)) return undefined;
    if (keys.has(key)) return undefined;
    keys.add(key);
    members.push(`${key}=${memberValue}`);
  }
  if (members.length === 0 || members.length > MAX_TRACESTATE_MEMBERS) return undefined;
  return members.join(",");
};

/**
 * Portable W3C trace context: 16-byte trace id, 8-byte span id, trace flags and an optional
 * `tracestate`. Pass it (or its `traceparent`) explicitly across workers, threads, FFI and HTTP.
 */
export class ObservationContext {
  readonly traceId: string;
  readonly spanId: string;
  readonly traceFlags: number;
  readonly tracestate: string | undefined;
  /** Whether the context came from another process (header, worker message, FFI). */
  readonly isRemote: boolean;

  private constructor(
    traceId: string,
    spanId: string,
    traceFlags: number,
    tracestate: string | undefined,
    isRemote: boolean,
  ) {
    this.traceId = traceId;
    this.spanId = spanId;
    this.traceFlags = traceFlags;
    this.tracestate = tracestate;
    this.isRemote = isRemote;
  }

  /**
   * Parses a W3C `traceparent` (and optional `tracestate`). Returns `undefined` when the
   * traceparent is invalid: wrong field lengths, non-lowercase hex, version `ff`, all-zero ids,
   * or extra fields on version `00`. Higher versions are parsed as version `00` and may carry
   * extra `-`-separated fields, as W3C Trace Context level 1 requires.
   */
  static fromTraceparent(
    traceparent: string,
    tracestate?: string | null,
  ): ObservationContext | undefined {
    if (typeof traceparent !== "string") return undefined;
    const value = traceparent.trim();
    if (value.length < 55) return undefined;
    const version = value.slice(0, 2);
    if (!HEX_BYTE.test(version) || version === "ff") return undefined;
    if (version === "00" ? value.length !== 55 : value.length > 55 && value[55] !== "-") {
      return undefined;
    }
    if (value[2] !== "-" || value[35] !== "-" || value[52] !== "-") return undefined;
    const traceId = value.slice(3, 35);
    const spanId = value.slice(36, 52);
    const flags = value.slice(53, 55);
    if (!TRACE_ID.test(traceId) || traceId === INVALID_TRACE_ID) return undefined;
    if (!SPAN_ID.test(spanId) || spanId === INVALID_SPAN_ID) return undefined;
    if (!HEX_BYTE.test(flags)) return undefined;
    return new ObservationContext(
      traceId,
      spanId,
      Number.parseInt(flags, 16),
      normalizeTracestate(tracestate),
      true,
    );
  }

  /** Builds a context from an OpenTelemetry span context; `undefined` when it is invalid. */
  static fromSpanContext(spanContext: SpanContext): ObservationContext | undefined {
    if (!TRACE_ID.test(spanContext.traceId) || spanContext.traceId === INVALID_TRACE_ID) {
      return undefined;
    }
    if (!SPAN_ID.test(spanContext.spanId) || spanContext.spanId === INVALID_SPAN_ID) {
      return undefined;
    }
    return new ObservationContext(
      spanContext.traceId,
      spanContext.spanId,
      spanContext.traceFlags & 0xff,
      normalizeTracestate(spanContext.traceState?.serialize()),
      spanContext.isRemote === true,
    );
  }

  /** Reads the active span of an OpenTelemetry context (the active one by default). */
  static fromOtelContext(ctx: Context = contextApi.active()): ObservationContext | undefined {
    const spanContext = trace.getSpanContext(ctx);
    return spanContext === undefined ? undefined : ObservationContext.fromSpanContext(spanContext);
  }

  get sampled(): boolean {
    return (this.traceFlags & SAMPLED) === SAMPLED;
  }

  /** Formats the version `00` traceparent. */
  toTraceparent(): string {
    return `00-${this.traceId}-${this.spanId}-${this.traceFlags.toString(16).padStart(2, "0")}`;
  }

  toRef(): ObservationRef {
    return { traceId: this.traceId, spanId: this.spanId };
  }

  toSpanContext(): SpanContext {
    return {
      traceId: this.traceId,
      spanId: this.spanId,
      traceFlags: this.traceFlags,
      isRemote: this.isRemote,
      ...(this.tracestate === undefined ? {} : { traceState: createTraceState(this.tracestate) }),
    };
  }

  /** An OpenTelemetry context whose parent span is this context (`ROOT_CONTEXT` by default). */
  toOtelContext(base: Context = ROOT_CONTEXT): Context {
    return trace.setSpanContext(base, this.toSpanContext());
  }
}

/** Accepts an {@link ObservationContext} or a `traceparent` string. */
export type ParentInput = ObservationContext | string;

export const resolveParent = (
  parent: ParentInput | undefined,
  tracestate: string | undefined,
): ObservationContext | undefined => {
  if (parent === undefined) return undefined;
  if (typeof parent === "string") return ObservationContext.fromTraceparent(parent, tracestate);
  return parent;
};

type HeaderBag = Headers | Record<string, string | readonly string[] | undefined>;

const readHeader = (headers: HeaderBag, name: string): string | undefined => {
  if (typeof (headers as Headers).get === "function") {
    return (headers as Headers).get(name) ?? undefined;
  }
  for (const [key, value] of Object.entries(headers as Record<string, unknown>)) {
    if (key.toLowerCase() !== name) continue;
    if (typeof value === "string") return value;
    // Repeated tracestate headers combine with commas (RFC 9110 field lines).
    if (Array.isArray(value)) return value.join(",");
  }
  return undefined;
};

/**
 * Writes `traceparent` (and `tracestate` when present) into `headers` and returns it. Works with
 * `Headers` and plain header objects; does nothing when `ctx` is undefined.
 */
export const injectTraceHeaders = <T extends Headers | Record<string, string>>(
  ctx: ObservationContext | undefined,
  headers: T,
): T => {
  if (ctx === undefined) return headers;
  if (typeof (headers as Headers).set === "function") {
    (headers as Headers).set("traceparent", ctx.toTraceparent());
    if (ctx.tracestate !== undefined) (headers as Headers).set("tracestate", ctx.tracestate);
    return headers;
  }
  const record = headers as Record<string, string>;
  record["traceparent"] = ctx.toTraceparent();
  if (ctx.tracestate !== undefined) record["tracestate"] = ctx.tracestate;
  return headers;
};

/** Reads W3C trace headers (case-insensitive); `undefined` when absent or invalid. */
export const extractTraceContext = (headers: HeaderBag): ObservationContext | undefined => {
  const traceparent = readHeader(headers, "traceparent");
  if (traceparent === undefined) return undefined;
  return ObservationContext.fromTraceparent(traceparent, readHeader(headers, "tracestate"));
};

/** Plain-object form that survives `postMessage` / structured clone. */
export const serializeContext = (
  ctx: ObservationContext | undefined,
): SerializedObservationContext | undefined => {
  if (ctx === undefined) return undefined;
  return ctx.tracestate === undefined
    ? { traceparent: ctx.toTraceparent() }
    : { traceparent: ctx.toTraceparent(), tracestate: ctx.tracestate };
};

/** Inverse of {@link serializeContext}; `undefined` for missing or invalid input. */
export const deserializeContext = (value: unknown): ObservationContext | undefined => {
  if (typeof value !== "object" || value === null) return undefined;
  const { traceparent, tracestate } = value as Record<string, unknown>;
  if (typeof traceparent !== "string") return undefined;
  return ObservationContext.fromTraceparent(
    traceparent,
    typeof tracestate === "string" ? tracestate : undefined,
  );
};
