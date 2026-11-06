import { describe, expect, test } from "bun:test";
import {
  deserializeContext,
  extractTraceContext,
  injectTraceHeaders,
  normalizeTracestate,
  ObservationContext,
  serializeContext,
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
