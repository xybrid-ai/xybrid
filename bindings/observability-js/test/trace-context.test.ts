import { describe, expect, test } from "bun:test";
import {
  deserializeContext,
  extractTraceContext,
  injectTraceHeaders,
  normalizeTracestate,
  ObservationContext,
  serializeContext,
  toObservationContext,
  toObservationRef,
} from "../src/index.ts";

const TRACE = "4bf92f3577b34da6a3ce929d0e0e4736";
const SPAN = "00f067aa0ba902b7";
const VALID = `00-${TRACE}-${SPAN}-01`;

describe("traceparent", () => {
  test("parses and formats a valid version 00 header", () => {
    const ctx = ObservationContext.fromTraceparent(VALID, "vendor=abc");
    expect(ctx?.traceId).toBe(TRACE);
    expect(ctx?.spanId).toBe(SPAN);
    expect(ctx?.traceFlags).toBe(1);
    expect(ctx?.sampled).toBe(true);
    expect(ctx?.tracestate).toBe("vendor=abc");
    expect(ctx?.isRemote).toBe(true);
    expect(ctx?.toTraceparent()).toBe(VALID);
  });

  test("unsampled flags round-trip", () => {
    const ctx = ObservationContext.fromTraceparent(`00-${TRACE}-${SPAN}-00`);
    expect(ctx?.sampled).toBe(false);
    expect(ctx?.toTraceparent()).toBe(`00-${TRACE}-${SPAN}-00`);
  });

  test.each([
    ["version ff", `ff-${TRACE}-${SPAN}-01`],
    ["non-hex version", `0g-${TRACE}-${SPAN}-01`],
    ["all-zero trace id", `00-${"0".repeat(32)}-${SPAN}-01`],
    ["all-zero span id", `00-${TRACE}-${"0".repeat(16)}-01`],
    ["uppercase trace id", `00-${TRACE.toUpperCase()}-${SPAN}-01`],
    ["uppercase span id", `00-${TRACE}-${SPAN.toUpperCase()}-01`],
    ["uppercase flags", `00-${TRACE}-${SPAN}-0A`],
    ["extra field on version 00", `${VALID}-extra`],
    ["short trace id", `00-${TRACE.slice(1)}-${SPAN}-01`],
    ["short span id", `00-${TRACE}-${SPAN.slice(1)}-01`],
    ["missing flags", `00-${TRACE}-${SPAN}`],
    ["wrong separators", `00_${TRACE}_${SPAN}_01`],
    ["empty", ""],
    ["higher version without dash after flags", `01-${TRACE}-${SPAN}-01x`],
  ])("rejects %s", (_name, header) => {
    expect(ObservationContext.fromTraceparent(header)).toBeUndefined();
  });

  test("accepts a higher version with extra fields, formatting as 00", () => {
    const ctx = ObservationContext.fromTraceparent(`01-${TRACE}-${SPAN}-01-what-the-future-holds`);
    expect(ctx?.toTraceparent()).toBe(VALID);
  });

  test("an invalid tracestate is dropped while the traceparent stays valid", () => {
    const ctx = ObservationContext.fromTraceparent(VALID, "Invalid Key=1");
    expect(ctx?.traceId).toBe(TRACE);
    expect(ctx?.tracestate).toBeUndefined();
  });
});

describe("tracestate", () => {
  test("normalizes whitespace and empty members", () => {
    expect(normalizeTracestate(" a=1 ,, b@vendor=x y ,")).toBe("a=1,b@vendor=x y");
  });
  test.each([
    ["duplicate keys", "a=1,a=2"],
    ["uppercase key", "A=1"],
    ["missing value", "a="],
    ["comma in value", "a=1=2"],
    ["control character in value", "a=1\u0000"],
  ])("rejects %s", (_name, value) => {
    expect(normalizeTracestate(value)).toBeUndefined();
  });
  test("rejects more than 32 members", () => {
    const members = Array.from({ length: 33 }, (_, index) => `k${index}=v`).join(",");
    expect(normalizeTracestate(members)).toBeUndefined();
    expect(normalizeTracestate(members.split(",").slice(0, 32).join(","))).toBeDefined();
  });
});

describe("headers and worker messages", () => {
  test("inject and extract with plain objects and Headers", () => {
    const ctx = ObservationContext.fromTraceparent(VALID, "vendor=abc");
    const plain = injectTraceHeaders(ctx, {} as Record<string, string>);
    expect(plain).toEqual({ traceparent: VALID, tracestate: "vendor=abc" });
    expect(extractTraceContext({ TraceParent: VALID })?.toTraceparent()).toBe(VALID);
    const headers = injectTraceHeaders(ctx, new Headers());
    expect(extractTraceContext(headers)?.tracestate).toBe("vendor=abc");
    expect(extractTraceContext({ traceparent: "garbage" })).toBeUndefined();
    expect(injectTraceHeaders(undefined, { a: "b" })).toEqual({ a: "b" });
  });

  test("inject replaces stale trace headers whatever their case", () => {
    const ctx = ObservationContext.fromTraceparent(VALID);
    const stale = `00-${"1".repeat(32)}-${"2".repeat(16)}-01`;
    const plain = injectTraceHeaders(ctx, {
      Traceparent: stale,
      TraceState: "old=1",
      accept: "x",
    } as Record<string, string>);
    expect(plain).toEqual({ accept: "x", traceparent: VALID });
    const bag = injectTraceHeaders(ctx, new Headers({ tracestate: "stale=1" }));
    expect(bag.get("tracestate")).toBeNull();
    expect(extractTraceContext(new Headers(plain))?.toTraceparent()).toBe(VALID);
  });

  test("serialize survives structured clone", () => {
    const ctx = ObservationContext.fromTraceparent(VALID, "vendor=abc");
    const cloned = structuredClone(serializeContext(ctx));
    const restored = deserializeContext(cloned);
    expect(restored?.traceId).toBe(TRACE);
    expect(restored?.spanId).toBe(SPAN);
    expect(restored?.tracestate).toBe("vendor=abc");
    expect(deserializeContext({ traceparent: 5 })).toBeUndefined();
    expect(deserializeContext(null)).toBeUndefined();
  });
});

describe("W3C normalization (contract v2, R-CTX-1)", () => {
  test("only spaces and tabs are trimmed around a traceparent", () => {
    expect(ObservationContext.fromTraceparent(` \t${VALID}\t `)?.toTraceparent()).toBe(VALID);
    expect(ObservationContext.fromTraceparent(`${VALID}\n`)).toBeUndefined();
    expect(ObservationContext.fromTraceparent(` ${VALID}`)).toBeUndefined();
  });

  test("flags keep only the sampled bit", () => {
    for (const [flags, expected] of [
      ["00", "00"],
      ["01", "01"],
      ["02", "00"],
      ["03", "01"],
      ["ff", "01"],
    ] as const) {
      const ctx = ObservationContext.fromTraceparent(`00-${TRACE}-${SPAN}-${flags}`);
      expect(ctx?.traceFlags).toBe(Number.parseInt(expected, 16));
      expect(ctx?.toTraceparent()).toBe(`00-${TRACE}-${SPAN}-${expected}`);
    }
    const fromSpan = ObservationContext.fromSpanContext({
      traceId: TRACE,
      spanId: SPAN,
      traceFlags: 0x83,
    });
    expect(fromSpan?.traceFlags).toBe(1);
    expect(fromSpan?.toTraceparent()).toBe(VALID);
  });

  test("tracestate: 512 normalized characters, OWS only, 14-character system ids", () => {
    const member = (key: string) => `${key}=${"x".repeat(250)}`;
    expect(normalizeTracestate([member("a"), member("b")].join(","))?.length).toBe(505);
    expect(normalizeTracestate([member("a"), member("b"), "c=1234"].join(","))?.length).toBe(512);
    expect(normalizeTracestate([member("a"), member("b"), "c=12345"].join(","))).toBeUndefined();
    // Whitespace around members does not count; only the normalized form is bounded.
    expect(normalizeTracestate(` ${member("a")} , ${member("b")} `)).toBeDefined();
    expect(normalizeTracestate("a=1\n")).toBeUndefined();
    expect(normalizeTracestate("t@abcdefghijklmn=1")).toBe("t@abcdefghijklmn=1");
    expect(normalizeTracestate("t@abcdefghijklmno=1")).toBeUndefined();
    expect(normalizeTracestate("")).toBeUndefined();
    expect(normalizeTracestate(" , ")).toBeUndefined();
  });
});

describe("references and explicit parent forms", () => {
  test("toObservationRef accepts exactly {trace_id, span_id}", () => {
    const ref = toObservationRef({ trace_id: TRACE, span_id: SPAN, extra: true });
    expect(ref).toEqual({ trace_id: TRACE, span_id: SPAN });
    expect(Object.isFrozen(ref)).toBe(true);
    for (const value of [
      { traceId: TRACE, spanId: SPAN },
      { trace_id: TRACE.toUpperCase(), span_id: SPAN },
      { trace_id: ` ${TRACE}`, span_id: SPAN },
      { trace_id: "0".repeat(32), span_id: SPAN },
      { trace_id: TRACE, span_id: "0".repeat(16) },
      Object.create({ trace_id: TRACE, span_id: SPAN }),
      [TRACE, SPAN],
      `${TRACE}-${SPAN}`,
      null,
      undefined,
    ]) {
      expect(toObservationRef(value)).toBeUndefined();
    }
  });

  test("context refs are fresh frozen snake_case objects", () => {
    const ctx = ObservationContext.fromTraceparent(VALID);
    const first = ctx?.toRef();
    expect(first).toEqual({ trace_id: TRACE, span_id: SPAN });
    expect(Object.isFrozen(first)).toBe(true);
    expect(ctx?.toRef()).not.toBe(first);
  });

  test("toObservationContext reads instances, clones, serialized forms and foreign copies", () => {
    const ctx = ObservationContext.fromTraceparent(`00-${TRACE}-${SPAN}-03`, "a=1, b=2");
    if (ctx === undefined) throw new Error("expected a context");
    // A copy of the class from another package instance: same fields, different prototype.
    class ForeignContext {
      constructor(
        readonly traceId: string,
        readonly spanId: string,
        readonly traceFlags: number,
        readonly tracestate: string | undefined,
        readonly isRemote: boolean,
      ) {}
      toTraceparent(): string {
        return "ignored";
      }
    }
    for (const form of [
      ctx,
      structuredClone(ctx),
      serializeContext(ctx),
      new ForeignContext(TRACE, SPAN, 0x81, " a=1 ,b=2", false),
      { traceparent: VALID, tracestate: "a=1,b=2" },
    ]) {
      const parsed = toObservationContext(form);
      expect(parsed?.toTraceparent()).toBe(VALID);
      expect(parsed?.tracestate).toBe("a=1,b=2");
    }
    for (const invalid of [
      { traceId: TRACE, spanId: SPAN },
      { traceId: TRACE, spanId: SPAN, traceFlags: 1.5 },
      { traceId: TRACE, spanId: SPAN, traceFlags: 256 },
      { traceId: TRACE.toUpperCase(), spanId: SPAN, traceFlags: 1 },
      { traceId: TRACE, spanId: SPAN, traceFlags: 1, tracestate: 5 },
      { traceId: TRACE, spanId: SPAN, traceFlags: 1, isRemote: "yes" },
      { traceparent: 5 },
      { traceparent: VALID, tracestate: 7 },
      [VALID],
      42,
      "garbage",
    ]) {
      expect(toObservationContext(invalid)).toBeUndefined();
    }
    // An invalid tracestate string is dropped while the parent stays valid.
    expect(
      toObservationContext({ traceId: TRACE, spanId: SPAN, traceFlags: 1, tracestate: "a=1,a=2" })
        ?.tracestate,
    ).toBeUndefined();
  });
});
