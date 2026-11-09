import {
  type Context,
  context as contextApi,
  createTraceState,
  ROOT_CONTEXT,
  type SpanContext,
  trace,
} from "@opentelemetry/api";
import { DEFAULTS } from "./catalog.ts";

/**
 * Identifies one span for results, evaluations and feedback: lowercase hex ids, neither all
 * zeros. The wire shape shared with the Rust crate and every binding.
 */
export type ObservationRef = {
  readonly trace_id: string;
  readonly span_id: string;
};

/** Structured-clone-safe form of an {@link ObservationContext}, for `postMessage`. */
export type SerializedObservationContext = {
  readonly traceparent: string;
  readonly tracestate?: string | undefined;
};

/**
 * The fields of an {@link ObservationContext}: what `structuredClone` of an instance, or an
 * instance from another copy of this package, looks like.
 */
export type ObservationContextLike = {
  readonly traceId: string;
  readonly spanId: string;
  readonly traceFlags: number;
  readonly tracestate?: string | null | undefined;
  readonly isRemote?: boolean | undefined;
};

const TRACE_ID = /^[0-9a-f]{32}$/;
const SPAN_ID = /^[0-9a-f]{16}$/;
const HEX_BYTE = /^[0-9a-f]{2}$/;
const INVALID_TRACE_ID = "0".repeat(32);
const INVALID_SPAN_ID = "0".repeat(16);
// W3C Trace Context level 1, section 3.3.1.3: simple key or tenant@system key; printable
// ASCII value without ',' or '=' that does not end in a space.
const TRACESTATE_KEY =
  /^(?:[a-z][a-z0-9_\-*/]{0,255}|[a-z0-9][a-z0-9_\-*/]{0,240}@[a-z][a-z0-9_\-*/]{0,13})$/;
const TRACESTATE_VALUE = /^[\x20-\x2b\x2d-\x3c\x3e-\x7e]{0,255}[\x21-\x2b\x2d-\x3c\x3e-\x7e]$/;
const SAMPLED = 0x01;

/** Trims W3C optional whitespace (spaces and tabs) only. */
const trimOws = (value: string): string => {
  let start = 0;
  let end = value.length;
  while (start < end && (value[start] === " " || value[start] === "\t")) start += 1;
  while (end > start && (value[end - 1] === " " || value[end - 1] === "\t")) end -= 1;
  return value.slice(start, end);
};

const validTraceId = (value: unknown): value is string =>
  typeof value === "string" && TRACE_ID.test(value) && value !== INVALID_TRACE_ID;
const validSpanId = (value: unknown): value is string =>
  typeof value === "string" && SPAN_ID.test(value) && value !== INVALID_SPAN_ID;

/**
 * Normalizes a W3C `tracestate` header: trims spaces and tabs around members and skips empty
 * ones. Returns `undefined` when the header is absent, empty, or invalid (a bad member, a
 * duplicate key, more than 32 members, or more than 512 characters once normalized); an
 * invalid tracestate is discarded as a whole while the traceparent stays usable.
 */
export const normalizeTracestate = (value: string | undefined | null): string | undefined => {
  if (typeof value !== "string") return undefined;
  const members: string[] = [];
  const keys = new Set<string>();
  for (const raw of value.split(",")) {
    const member = trimOws(raw);
    if (member.length === 0) continue;
    const equals = member.indexOf("=");
    if (equals <= 0) return undefined;
    const key = member.slice(0, equals);
    const memberValue = member.slice(equals + 1);
    if (!TRACESTATE_KEY.test(key) || !TRACESTATE_VALUE.test(memberValue)) return undefined;
    if (keys.has(key)) return undefined;
    keys.add(key);
    members.push(`${key}=${memberValue}`);
    if (members.length > DEFAULTS.max_tracestate_members) return undefined;
  }
  if (members.length === 0) return undefined;
  const normalized = members.join(",");
  return normalized.length > DEFAULTS.max_tracestate_chars ? undefined : normalized;
};

/**
 * Portable W3C trace context: 16-byte trace id, 8-byte span id, trace flags (sampled bit only)
 * and an optional normalized `tracestate`. Pass it (or its `traceparent`) explicitly across
 * workers, threads, FFI and HTTP.
 */
export class ObservationContext {
  readonly traceId: string;
  readonly spanId: string;
  /** `0` or `1`: only the sampled bit is kept (W3C Trace Context level 1). */
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
    this.traceFlags = traceFlags & SAMPLED;
    this.tracestate = tracestate;
    this.isRemote = isRemote;
  }

  /**
   * Parses a W3C `traceparent` (and optional `tracestate`). Returns `undefined` when the
   * traceparent is invalid: wrong field lengths or separators, non-lowercase hex, version `ff`,
   * all-zero ids, or extra fields on version `00`. Higher versions are parsed as version `00`
   * and may carry extra `-`-separated fields. Only spaces and tabs around the value are trimmed;
   * flags keep only the sampled bit.
   */
  static fromTraceparent(
    traceparent: string,
    tracestate?: string | null,
  ): ObservationContext | undefined {
    if (typeof traceparent !== "string") return undefined;
    const value = trimOws(traceparent);
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
    if (!validTraceId(traceId) || !validSpanId(spanId) || !HEX_BYTE.test(flags)) {
      return undefined;
    }
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
    if (!validTraceId(spanContext.traceId) || !validSpanId(spanContext.spanId)) return undefined;
    return new ObservationContext(
      spanContext.traceId,
      spanContext.spanId,
      spanContext.traceFlags,
      normalizeTracestate(spanContext.traceState?.serialize()),
      spanContext.isRemote === true,
    );
  }

  /** Reads the active span of an OpenTelemetry context (the active one by default). */
  static fromOtelContext(ctx: Context = contextApi.active()): ObservationContext | undefined {
    const spanContext = trace.getSpanContext(ctx);
    return spanContext === undefined ? undefined : ObservationContext.fromSpanContext(spanContext);
  }

  /** @internal Validated plain fields (structured clone, another package copy). */
  static fromFields(value: ObservationContextLike): ObservationContext | undefined {
    const { traceId, spanId, traceFlags, tracestate, isRemote } = value;
    if (!validTraceId(traceId) || !validSpanId(spanId)) return undefined;
    if (typeof traceFlags !== "number" || !Number.isInteger(traceFlags)) return undefined;
    if (traceFlags < 0 || traceFlags > 0xff) return undefined;
    if (tracestate !== undefined && tracestate !== null && typeof tracestate !== "string") {
      return undefined;
    }
    if (isRemote !== undefined && typeof isRemote !== "boolean") return undefined;
    return new ObservationContext(
      traceId,
      spanId,
      traceFlags,
      normalizeTracestate(tracestate),
      isRemote ?? true,
    );
  }

  get sampled(): boolean {
    return (this.traceFlags & SAMPLED) === SAMPLED;
  }

  /** Formats the version `00` traceparent (flags `00` or `01`). */
  toTraceparent(): string {
    return `00-${this.traceId}-${this.spanId}-0${this.traceFlags & SAMPLED}`;
  }

  /** The reference `{trace_id, span_id}` of this context (a fresh frozen object). */
  toRef(): ObservationRef {
    return Object.freeze({ trace_id: this.traceId, span_id: this.spanId });
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

/**
 * An explicit parent: an {@link ObservationContext}, a `traceparent` string (with
 * `StartOptions.tracestate`), the serialized `{traceparent, tracestate?}` form, or the plain
 * fields of a context (a structured clone, or an instance from another copy of the package).
 * `undefined` and `null` mean no explicit parent.
 */
export type ParentInput =
  | ObservationContext
  | string
  | SerializedObservationContext
  | ObservationContextLike
  | null
  | undefined;

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const fromSerialized = (value: Record<string, unknown>): ObservationContext | undefined => {
  const { traceparent, tracestate } = value;
  if (typeof traceparent !== "string") return undefined;
  if (tracestate !== undefined && tracestate !== null && typeof tracestate !== "string") {
    return undefined;
  }
  return ObservationContext.fromTraceparent(traceparent, tracestate);
};

/**
 * Converts an explicit parent object to a context: an {@link ObservationContext} instance, the
 * serialized `{traceparent, tracestate?}` form, or `{traceId, spanId, traceFlags, tracestate?,
 * isRemote?}` (a structured clone or an instance from another package copy). A string is read
 * as a `traceparent`. Returns `undefined` for anything else or for invalid values.
 */
export const toObservationContext = (value: unknown): ObservationContext | undefined => {
  if (value instanceof ObservationContext) return value;
  if (typeof value === "string") return ObservationContext.fromTraceparent(value);
  if (!isObject(value) || Array.isArray(value)) return undefined;
  try {
    if ("traceparent" in value) return fromSerialized(value);
    return ObservationContext.fromFields(value as ObservationContextLike);
  } catch {
    // A throwing getter or proxy trap: not a usable context.
    return undefined;
  }
};

/**
 * Resolves `StartOptions.parent`. `undefined`/`null` mean "no explicit parent"; anything else
 * must be a valid context or the call throws a `TypeError` (never naming the value).
 * @internal
 */
export const resolveParent = (
  parent: unknown,
  tracestate: unknown,
): ObservationContext | undefined => {
  if (parent === undefined || parent === null) return undefined;
  const resolved =
    typeof parent === "string"
      ? ObservationContext.fromTraceparent(
          parent,
          typeof tracestate === "string" ? tracestate : undefined,
        )
      : toObservationContext(parent);
  if (resolved === undefined) {
    throw new TypeError(
      "StartOptions.parent must be an ObservationContext, a valid traceparent string, a " +
        "serialized {traceparent, tracestate} context, or the fields of a context.",
    );
  }
  return resolved;
};

/**
 * Validates a reference: an object whose own `trace_id` and `span_id` are lowercase hex strings
 * of 32 and 16 characters, neither all zeros. Extra members are ignored; `traceId`/`spanId`
 * aliases, uppercase and trimming are rejected. Returns a fresh frozen reference.
 */
export const toObservationRef = (value: unknown): ObservationRef | undefined => {
  if (!isObject(value) || Array.isArray(value)) return undefined;
  try {
    if (!Object.hasOwn(value, "trace_id") || !Object.hasOwn(value, "span_id")) return undefined;
    const traceId = value["trace_id"];
    const spanId = value["span_id"];
    if (!validTraceId(traceId) || !validSpanId(spanId)) return undefined;
    return Object.freeze({ trace_id: traceId, span_id: spanId });
  } catch {
    return undefined;
  }
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
 * Writes `traceparent` (and `tracestate` when present) into `headers` and returns it, replacing
 * any existing trace headers whatever their case (a stale `tracestate` is removed when the
 * context has none). Works with `Headers` and plain header objects; does nothing when `ctx` is
 * undefined.
 */
export const injectTraceHeaders = <T extends Headers | Record<string, string>>(
  ctx: ObservationContext | undefined,
  headers: T,
): T => {
  if (ctx === undefined) return headers;
  if (typeof (headers as Headers).set === "function") {
    const bag = headers as Headers;
    bag.set("traceparent", ctx.toTraceparent());
    if (ctx.tracestate === undefined) bag.delete("tracestate");
    else bag.set("tracestate", ctx.tracestate);
    return headers;
  }
  const record = headers as Record<string, string>;
  for (const key of Object.keys(record)) {
    const lower = key.toLowerCase();
    if (lower === "traceparent" || lower === "tracestate") delete record[key];
  }
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
  if (!isObject(value) || Array.isArray(value)) return undefined;
  try {
    return fromSerialized(value);
  } catch {
    return undefined;
  }
};
