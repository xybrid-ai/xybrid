// Content pipeline regressions (contract-decisions D16-D22; R-CONTENT-1/3/4, R-BUF-3; content-2,
// content-9, content-10, content-11).
import { afterEach, describe, expect, test } from "bun:test";
import { DiagLogLevel, diag } from "@opentelemetry/api";
import {
  AlwaysOffSampler,
  AlwaysOnSampler,
  ParentBasedSampler,
  type Sampler,
} from "@opentelemetry/sdk-trace";
import { applyReplacements, ContentProcessor, jsonCost } from "../src/content.ts";
import {
  boundJson,
  DEFAULTS,
  type JsonValue,
  type ObservationContext,
  Observer,
  type RedactionRule,
  type StartOptions,
  spanContentCeiling,
  toRedactionRules,
  utf8Length,
  utf8Prefix,
} from "../src/index.ts";
import { createHarness, type TracerKind } from "./support/harness.ts";

const request = { operation: "chat", provider: "llama.cpp", model: "m" } as const;
const MARKER = DEFAULTS.truncation_marker;

/** The v1 implementation of CONTRACT.md "Truncation" (quadratic array step), as a reference. */
const referenceBound = (original: JsonValue, budget: number): JsonValue | undefined => {
  const size = (value: JsonValue) => utf8Length(JSON.stringify(value));
  const maxString = (value: JsonValue): number =>
    typeof value === "string"
      ? utf8Length(value)
      : value !== null && typeof value === "object"
        ? Math.max(0, ...(Array.isArray(value) ? value : Object.values(value)).map(maxString))
        : 0;
  const cap = (value: JsonValue, limit: number): JsonValue => {
    if (typeof value === "string") {
      return utf8Length(value) > limit ? `${utf8Prefix(value, limit)}${MARKER}` : value;
    }
    if (value === null || typeof value !== "object") return value;
    if (Array.isArray(value)) return value.map((item: JsonValue) => cap(item, limit));
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, cap(item, limit)]));
  };
  if (size(original) <= budget) return original;
  let value = original;
  let limit = maxString(original);
  while (size(value) > budget && limit > 32) {
    limit = Math.floor(limit / 2);
    value = cap(original, limit);
  }
  if (size(value) > budget && Array.isArray(value)) {
    const items = [...(value as JsonValue[])];
    while (size(items) > budget && items.length > 1) items.shift();
    value = items;
  }
  return size(value) > budget ? undefined : value;
};

const message = (index: number): JsonValue => ({
  role: index % 2 === 0 ? "user" : "assistant",
  parts: [{ type: "text", content: `message ${index} ${"é".repeat(index % 7)}` }],
});

describe("truncation cost (D20, R-CONTENT-1, content-2)", () => {
  test("equals the reference implementation", () => {
    const inputs: JsonValue[] = [
      Array.from({ length: 300 }, (_, index) => message(index)),
      Array.from({ length: 40 }, (_, index) => ({ text: "x".repeat(200 + index) })),
      [{ a: "y".repeat(900) }, "short", 1, null, ["nested", "z".repeat(400)]],
      { only: "o".repeat(5000) },
    ];
    for (const input of inputs) {
      for (const budget of [64, 150, 700, 2048, 1 << 20]) {
        const bounded = boundJson(input, budget);
        const expected = referenceBound(input, budget);
        if (expected === undefined) expect(bounded.kind).toBe("omitted");
        else {
          expect(bounded.kind).toBe("kept");
          if (bounded.kind === "kept") {
            expect(bounded.value).toEqual(expected);
            expect(bounded.json).toBe(JSON.stringify(expected));
          }
        }
      }
    }
  });

  test("20 000 messages bound in linear time", () => {
    const messages = Array.from({ length: 20_000 }, (_, index) => ({
      role: "user",
      parts: [{ type: "text", content: `${index}`.padEnd(40, "x") }],
    }));
    const { observer, spans } = createHarness({ content: { maxTraversalBytes: 64 << 20 } });
    const started = performance.now();
    const op = observer.startInference(request);
    op.recordInputMessages(messages);
    op.end();
    expect(performance.now() - started).toBeLessThan(1000);
    const recorded = JSON.parse(
      String(spans.getFinishedSpans()[0]?.attributes["gen_ai.input.messages"]),
    );
    expect(recorded.at(-1).parts[0].content.startsWith("19999")).toBe(true);
    expect(utf8Length(JSON.stringify(recorded))).toBeLessThanOrEqual(16384);
  });
});

describe("preflight, clone and callback validation (D17, D18, R-CONTENT-3, content-9)", () => {
  test("message parts exhaust the preflight budget before cloning any messages", () => {
    let roleReads = 0;
    const processor = new ContentProcessor({ maxTraversalBytes: 256 });
    const outcome = processor.process(
      "gen_ai.input.messages",
      [
        {
          get role() {
            roleReads += 1;
            return "user";
          },
          parts: [{ type: "text", content: "x".repeat(512) }],
        },
      ],
      "input_messages",
    );
    expect(outcome).toEqual({ kind: "failed", reason: "traversal" });
    expect(roleReads).toBe(1);
  });

  test("toJSON values follow JSON.stringify; __proto__ stays an ordinary key", () => {
    const { observer, spans } = createHarness();
    const tool = observer.startTool({
      name: "t",
      arguments: {
        when: new Date(Date.UTC(2026, 0, 2, 3, 4, 5)),
        url: new URL("https://example.com/a?b=1"),
        skip: undefined,
        fn: () => 1,
        list: [undefined, () => 1, Number.NaN, Number.POSITIVE_INFINITY, 2],
        boxed: [Object(1), Object("s"), Object(true)],
        own: JSON.parse('{"__proto__": {"polluted": true}, "ok": 1}'),
        [Symbol("hidden")]: 1,
      },
    });
    tool.end();
    const recorded = String(spans.getFinishedSpans()[0]?.attributes["gen_ai.tool.call.arguments"]);
    expect(recorded).toBe(
      '{"when":"2026-01-02T03:04:05.000Z","url":"https://example.com/a?b=1",' +
        '"list":[null,null,null,null,2],"boxed":[1,"s",true],' +
        '"own":{"__proto__":{"polluted":true},"ok":1}}',
    );
    expect(({} as Record<string, unknown>)["polluted"]).toBeUndefined();
  });

  test("unrepresentable inputs fail closed before the callback runs", () => {
    const cycle: Record<string, unknown> = { a: 1 };
    cycle["self"] = cycle;
    const throwing = {
      get secret(): string {
        throw new Error("private getter");
      },
    };
    const inputs: unknown[] = [
      cycle,
      { big: 10n },
      new ArrayBuffer(4),
      new Uint8Array([1, 2]),
      new DataView(new ArrayBuffer(2)),
      new Blob(["private"]),
      throwing,
      {
        toJSON() {
          throw new Error("private toJSON");
        },
      },
    ];
    for (const input of inputs) {
      const calls: string[] = [];
      const { observer, spans } = createHarness({
        content: {
          redact: (kind, value) => {
            calls.push(kind);
            return value;
          },
        },
      });
      const tool = observer.startTool({ name: "t" });
      tool.recordResult(input);
      tool.end();
      const attributes = spans.getFinishedSpans()[0]?.attributes ?? {};
      expect(attributes["gen_ai.tool.call.result"]).toBeUndefined();
      expect(attributes["xybrid.genai.content.truncated_attributes"]).toEqual([
        "gen_ai.tool.call.result",
      ]);
      expect(calls).toEqual([]);
    }
  });

  test("depth 128 is kept and 129 fails; the traversal budget bounds the walk", () => {
    const nested = (depth: number): unknown => {
      let value: unknown = "x";
      for (let index = 0; index < depth; index += 1) value = [value];
      return value;
    };
    const { observer, spans } = createHarness({ content: { maxTraversalBytes: 1 << 20 } });
    const kept = observer.startTool({ name: "t" });
    kept.recordResult(nested(128));
    kept.end();
    const deep = observer.startTool({ name: "t" });
    deep.recordResult(nested(129));
    deep.end();
    const small = createHarness({ content: { maxTraversalBytes: 100 } });
    const big = small.observer.startTool({ name: "t" });
    big.recordResult({ text: "a".repeat(84) });
    big.end();
    const fits = small.observer.startTool({ name: "t" });
    // 16 (object) + 16 + 4 ("text") + 16 + 48 = 100.
    fits.recordResult({ text: "a".repeat(48) });
    fits.end();
    const [first, second] = spans.getFinishedSpans();
    expect(first?.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify(nested(128)));
    expect(second?.attributes["gen_ai.tool.call.result"]).toBeUndefined();
    expect(second?.attributes["xybrid.genai.content.truncated_attributes"]).toEqual([
      "gen_ai.tool.call.result",
    ]);
    const [over, under] = small.spans.getFinishedSpans();
    expect(over?.attributes["gen_ai.tool.call.result"]).toBeUndefined();
    expect(under?.attributes["gen_ai.tool.call.result"]).toBe(
      JSON.stringify({ text: "a".repeat(48) }),
    );
  });

  test("callback outputs must be JSON data; undefined drops; null is null for JSON kinds", () => {
    const cycle: unknown[] = [];
    cycle.push(cycle);
    const outputs: [unknown, "dropped" | "failed" | string][] = [
      [Promise.resolve("x"), "failed"],
      [
        Object.defineProperty({}, ["th", "en"].join(""), {
          value: () => undefined,
          enumerable: true,
        }),
        "failed",
      ],
      [() => "x", "failed"],
      [10n, "failed"],
      [cycle, "failed"],
      [new Date(0), "failed"],
      [Number.NaN, "failed"],
      [[1, undefined], "failed"],
      [{ a: 1, b: undefined }, '{"a":1}'],
      [undefined, "dropped"],
      [null, "null"],
      [Object.assign(Object.create(null), { k: "v" }), '{"k":"v"}'],
    ];
    for (const [output, expected] of outputs) {
      const { observer, spans } = createHarness({ content: { redact: () => output as never } });
      const tool = observer.startTool({ name: "t" });
      tool.recordResult({ value: 1 });
      tool.end();
      const attributes = spans.getFinishedSpans()[0]?.attributes ?? {};
      const listed = attributes["xybrid.genai.content.truncated_attributes"];
      if (expected === "failed") {
        expect(attributes["gen_ai.tool.call.result"]).toBeUndefined();
        expect(listed).toEqual(["gen_ai.tool.call.result"]);
      } else if (expected === "dropped") {
        expect(attributes["gen_ai.tool.call.result"]).toBeUndefined();
        expect(listed).toBeUndefined();
      } else {
        expect(attributes["gen_ai.tool.call.result"]).toBe(expected);
        expect(listed).toBeUndefined();
      }
    }
  });

  test("callback outputs whose inspection throws fail closed instead of throwing", () => {
    const throwingThen = Object.defineProperty({}, ["th", "en"].join(""), {
      get() {
        throw new Error("private-then-getter");
      },
    });
    const revoked = Proxy.revocable({}, {});
    revoked.revoke();
    const promise = Promise.resolve("private");
    Object.defineProperty(promise, "constructor", {
      get() {
        throw new Error("private-constructor-getter");
      },
    });
    for (const output of [throwingThen, revoked.proxy, promise]) {
      const { observer, spans } = createHarness({ content: { redact: () => output as never } });
      const tool = observer.startTool({ name: "t" });
      expect(() => tool.recordResult({ value: 1 })).not.toThrow();
      tool.end();
      const processor = new ContentProcessor({
        maxTraversalBytes: 1 << 20,
        redactor: () => output as never,
      });
      expect(processor.process("gen_ai.tool.call.result", { value: 1 }, "json")).toEqual({
        kind: "failed",
        reason: "redactor_output",
      });
      const attributes = spans.getFinishedSpans()[0]?.attributes ?? {};
      expect(attributes["gen_ai.tool.call.result"]).toBeUndefined();
      expect(attributes["xybrid.genai.content.truncated_attributes"]).toEqual([
        "gen_ai.tool.call.result",
      ]);
    }
  });

  test("the redactor sees a canonical copy, never the caller's object", () => {
    const input = { nested: { secret: "private" } };
    const { observer, spans } = createHarness({
      content: {
        redact: (_kind, value) => {
          (value as { nested: { secret: string } }).nested.secret = "mutated";
          return value;
        },
      },
    });
    const tool = observer.startTool({ name: "t" });
    tool.recordResult(input);
    input.nested.secret = "changed later";
    tool.end();
    expect(input.nested.secret).toBe("changed later");
    expect(spans.getFinishedSpans()[0]?.attributes["gen_ai.tool.call.result"]).toBe(
      '{"nested":{"secret":"mutated"}}',
    );
  });
});

describe("declarative rules (D19, R-CONTENT-4)", () => {
  test("invalid rules are rejected with TypeError", () => {
    const invalid: unknown[] = [
      "rules",
      [{ type: "replace_literal", find: "", replace: "x" }],
      [{ type: "replace_literal", find: "a" }],
      [{ type: "replace_literal", find: 1, replace: "x" }],
      [{ type: "omit_attribute", key: "" }],
      [{ type: "omit_attribute" }],
      [{ type: "regex", pattern: "a" }],
      [null],
    ];
    for (const rules of invalid) {
      expect(() => toRedactionRules(rules)).toThrow(TypeError);
      expect(() => new Observer({ content: { rules: rules as never } })).toThrow(TypeError);
    }
  });

  test("rules round-trip through JSON in canonical form", () => {
    const text =
      '[{"type":"replace_literal","find":"sk-","replace":"[KEY]"},' +
      '{"type":"omit_attribute","key":"gen_ai.system_instructions"}]';
    const rules = toRedactionRules(
      JSON.parse(
        '[{"replace":"[KEY]","find":"sk-","type":"replace_literal","note":1},' +
          '{"key":"gen_ai.system_instructions","type":"omit_attribute"}]',
      ),
    );
    expect(JSON.stringify(rules)).toBe(text);
    expect(Object.isFrozen(rules)).toBe(true);
    expect(Object.isFrozen(rules[0])).toBe(true);
  });

  test("rules apply after the callback to content, error messages and feedback comments", () => {
    const rules: RedactionRule[] = [
      { type: "replace_literal", find: "aa", replace: "b" },
      { type: "omit_attribute", key: "xybrid.genai.feedback.comment" },
    ];
    const { observer, spans, logs } = createHarness({
      content: { redact: (_kind, value) => value, rules },
    });
    const op = observer.startInference(request);
    // Non-overlapping, left to right: "aaa" -> "ba"; keys and numbers untouched.
    op.recordToolDefinitions([{ aa: "aaa", n: 11, flag: true }]);
    op.endWithError({ errorType: "e", message: "aaaa" });
    const ref = op.observationRef();
    if (ref === undefined) throw new Error("expected a reference");
    expect(observer.recordFeedback(ref, { comment: "aa", score: 1 })).toBe(true);
    const [span] = spans.getFinishedSpans();
    expect(span?.attributes["gen_ai.tool.definitions"]).toBe('[{"aa":"ba","n":11,"flag":true}]');
    expect(span?.status.message).toBe("bb");
    const feedback = logs.getFinishedLogRecords().at(-1);
    expect(feedback?.attributes["xybrid.genai.feedback.comment"]).toBeUndefined();
    expect(feedback?.attributes["gen_ai.evaluation.score.value"]).toBe(1);
  });
});

describe("budgets and re-recording (D20, D22, R-BUF-3, content-11)", () => {
  test("defaults equal the catalog and invalid budgets throw RangeError", () => {
    // The default retained budget is the span content ceiling of the default request bound.
    expect(DEFAULTS.max_retained_operation_bytes as number).toBe(208 * 1024);
    expect(spanContentCeiling(DEFAULTS.max_request_body_bytes)).toBe(
      DEFAULTS.max_retained_operation_bytes,
    );
    expect(DEFAULTS.max_traversal_bytes as number).toBe(480 * 1024);
    expect(DEFAULTS.max_json_depth as number).toBe(128);
    expect(new Observer().contentBudgets).toEqual({
      maxAttributeBytes: DEFAULTS.max_content_attribute_bytes,
      maxRetainedOperationBytes: DEFAULTS.max_retained_operation_bytes,
      maxRequestBodyBytes: DEFAULTS.max_request_body_bytes,
    });
    for (const value of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, "10"]) {
      for (const key of [
        "maxAttributeBytes",
        "maxRetainedOperationBytes",
        "maxTraversalBytes",
        "maxRequestBodyBytes",
      ]) {
        expect(() => new Observer({ content: { [key]: value as never } })).toThrow(RangeError);
      }
    }
    expect(() => new Observer({ content: { maxAttributeBytes: 1 } })).not.toThrow();
  });

  test("re-recording replaces the value; the retained budget omits and lists", () => {
    const { observer, spans } = createHarness({
      content: { capture: "span_only", maxRetainedOperationBytes: 120 },
    });
    const tool = observer.startTool({ name: "t" });
    tool.recordResult({ v: "first" });
    tool.recordResult({ v: "second" });
    tool.end();
    const limited = observer.startTool({ name: "t", arguments: { a: "x".repeat(60) } });
    // 16 + 23 ("gen_ai.tool.call.result") + 48 bytes of JSON no longer fit after the arguments.
    limited.recordResult({ r: "y".repeat(40) });
    limited.end();
    const [replaced, budgeted] = spans.getFinishedSpans();
    expect(replaced?.attributes["gen_ai.tool.call.result"]).toBe('{"v":"second"}');
    expect(budgeted?.attributes["gen_ai.tool.call.arguments"]).toBe(
      JSON.stringify({ a: "x".repeat(60) }),
    );
    expect(budgeted?.attributes["gen_ai.tool.call.result"]).toBeUndefined();
    expect(budgeted?.attributes["xybrid.genai.content.truncated_attributes"]).toEqual([
      "gen_ai.tool.call.result",
    ]);
  });

  test("prompt variables are processed in code point order", () => {
    const { observer, spans } = createHarness();
    const names = ["b", "😀", "￿", "a"];
    const op = observer.startInference({
      ...request,
      prompt: { name: "p", variables: Object.fromEntries(names.map((name) => [name, name])) },
    });
    op.end();
    const keys = Object.keys(spans.getFinishedSpans()[0]?.attributes ?? {}).filter((key) =>
      key.startsWith("gen_ai.prompt.variable."),
    );
    expect(keys.map((key) => key.slice("gen_ai.prompt.variable.".length))).toEqual([
      "a",
      "b",
      "￿",
      "😀",
    ]);
  });

  test("content goes on the span once, at the end, after metadata", () => {
    const { observer, spans } = createHarness();
    const op = observer.startInference(request);
    op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: "hi" }] }]);
    op.recordUsage({ inputTokens: 1 });
    op.end();
    const keys = Object.keys(spans.getFinishedSpans()[0]?.attributes ?? {});
    expect(keys.at(-1)).toBe("gen_ai.input.messages");
    expect(keys.indexOf("gen_ai.usage.input_tokens")).toBeLessThan(keys.length - 1);
  });
});

describe("diagnostics never carry content", () => {
  afterEach(() => diag.disable());

  test("fail-closed paths log the key and reason only", () => {
    const lines: string[] = [];
    const sink = (...args: unknown[]) => lines.push(args.map(String).join(" "));
    diag.setLogger(
      { error: sink, warn: sink, info: sink, debug: sink, verbose: sink },
      DiagLogLevel.ALL,
    );
    const { observer } = createHarness({
      content: {
        maxTraversalBytes: 200,
        redact: (kind) => {
          if (kind === "gen_ai.output.messages") throw new Error("private-error");
          return "private-output" as never;
        },
      },
    });
    const op = observer.startInference({
      ...request,
      prompt: { name: "p", variables: { v: "private-variable" } },
    });
    op.recordInputMessages([
      { role: "user", parts: [{ type: "text", content: `private-${"z".repeat(300)}` }] },
    ]);
    op.recordOutputMessages([
      {
        role: "assistant",
        parts: [{ type: "text", content: "private-out" }],
        finish_reason: "stop",
      },
    ]);
    op.endWithError({ errorType: "e", message: "private-message" });
    expect(lines.join("\n")).not.toContain("private");
    expect(lines.some((line) => line.includes("gen_ai.output.messages"))).toBe(true);
  });
});

test("the rules budget check is order-independent (amendment A1)", () => {
  // One rule expands the first string, a later one shrinks the second: the final document fits,
  // so the result must not depend on which string is visited first.
  const rules: RedactionRule[] = [
    { type: "replace_literal", find: "<e>", replace: "#".repeat(60) },
    { type: "replace_literal", find: "y".repeat(60), replace: "" },
  ];
  for (const order of [
    ["a", "b"],
    ["b", "a"],
  ] as const) {
    const values: Record<string, string> = { a: "<e>", b: "y".repeat(60) };
    const input = Object.fromEntries(order.map((key) => [key, values[key]]));
    // 16 + (16 + 1 + 19) + (16 + 1 + 76) = 145 before; 16 + (16 + 1 + 76) + (16 + 1 + 16) = 142 after.
    const { observer, spans } = createHarness({ content: { maxTraversalBytes: 145, rules } });
    const tool = observer.startTool({ name: "t" });
    tool.recordResult(input);
    tool.end();
    expect(
      JSON.parse(String(spans.getFinishedSpans()[0]?.attributes["gen_ai.tool.call.result"])),
    ).toEqual({
      a: "#".repeat(60),
      b: "",
    });
  }
  // A single string over the budget after any rule fails, even if a later rule would shrink it.
  const { observer, spans } = createHarness({
    content: {
      maxTraversalBytes: 100,
      rules: [
        { type: "replace_literal", find: "x", replace: "x".repeat(100) },
        { type: "replace_literal", find: "x".repeat(100), replace: "x" },
      ],
    },
  });
  const tool = observer.startTool({ name: "t" });
  tool.recordResult("x");
  tool.end();
  expect(spans.getFinishedSpans()[0]?.attributes["gen_ai.tool.call.result"]).toBeUndefined();
});

test("the traversal budget is the preflight cost of the canonical form, at the boundary", () => {
  const messages = [
    {
      role: "user",
      name: "ann",
      extra: "dropped by the message model",
      parts: [
        { type: "text", content: "héllo", ignored: true },
        { type: "tool_call", id: "c", name: "f", arguments: Number.NaN },
        { type: "tool_call_response", id: "c" },
        { type: "custom", z: [1, null, "s"], skip: undefined },
      ],
    },
  ];
  const canonical: JsonValue = [
    {
      role: "user",
      parts: [
        { type: "text", content: "héllo" },
        { type: "tool_call", id: "c", name: "f" },
        { type: "tool_call_response", id: "c", response: null },
        { type: "custom", z: [1, null, "s"] },
      ],
      name: "ann",
    },
  ];
  const cost = jsonCost(canonical);
  for (const [limit, kept] of [
    [cost, true],
    [cost - 1, false],
  ] as const) {
    const outcome = new ContentProcessor({ maxTraversalBytes: limit }).process(
      "gen_ai.input.messages",
      messages,
      "input_messages",
    );
    expect(outcome).toEqual(
      kept ? { kind: "value", value: canonical } : { kind: "failed", reason: "traversal" },
    );
  }
});

describe("replacement rules stop building once the document is provably over budget (A9)", () => {
  const expand = [{ type: "replace_literal", find: "a", replace: "b".repeat(100) }] as const;

  test("an expanding rule over many strings rebuilds only a few before failing", () => {
    // 1,000 strings of 10 bytes each fit (26,016 of 30,000), and each one alone still fits after
    // the rule (1,016), but every rewrite adds 990 bytes, so the document is over budget after a
    // handful. Without the bound every string would be rebuilt before the final check.
    const doc: JsonValue[] = Array.from({ length: 1000 }, () => "a".repeat(10));
    expect(() => applyReplacements(doc, expand, jsonCost(doc), 30_000)).toThrow();
    const rebuilt = doc.filter((item) => item !== "a".repeat(10)).length;
    expect(rebuilt).toBeGreaterThan(0);
    expect(rebuilt).toBeLessThanOrEqual(5);
  });

  test("outcomes are unchanged within budget and with shrinking rules", () => {
    const doc: JsonValue = { a: "aaa", b: ["xaxa", "y"], c: { d: "a" } };
    const out = applyReplacements(structuredClone(doc), expand, jsonCost(doc), 10_000);
    expect(out).toEqual({
      a: "b".repeat(300),
      b: [`x${"b".repeat(100)}x${"b".repeat(100)}`, "y"],
      c: { d: "b".repeat(100) },
    });
    const grow = { type: "replace_literal", find: "a", replace: "aaaa" } as const;
    const shrink = { type: "replace_literal", find: "aaaa", replace: "" } as const;
    const many: JsonValue[] = Array.from({ length: 50 }, () => "a".repeat(20));
    // Expands to 80 bytes then shrinks to nothing: the final cost fits, so nothing fails.
    const shrunk = applyReplacements(structuredClone(many), [grow, shrink], jsonCost(many), 2_000);
    expect(shrunk).toEqual(Array.from({ length: 50 }, () => ""));
  });
});

describe("content for spans that do not record", () => {
  // Every content key the operations below record, in recording order.
  const CONTENT_KEYS = [
    "gen_ai.prompt.variable.city",
    "gen_ai.system_instructions",
    "gen_ai.input.messages",
    "gen_ai.tool.definitions",
    "gen_ai.output.messages",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
  ];
  const sorted = (keys: readonly string[]): string[] => [...keys].sort();

  /** An inference and a tool, each recording every kind of content. */
  const recordEveryContentKind = (observer: Observer, options: StartOptions): void => {
    const op = observer.startInference(
      { ...request, prompt: { name: "p", variables: { city: "Lisbon" } } },
      options,
    );
    op.recordSystemInstructions([{ type: "text", content: "be brief" }]);
    op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: "hi" }] }]);
    op.recordToolDefinitions([{ name: "lookup" }]);
    op.recordOutputMessages([
      { role: "assistant", parts: [{ type: "text", content: "hello" }], finish_reason: "stop" },
    ]);
    op.end();
    const tool = observer.startTool({ name: "lookup", arguments: { q: "x" } }, options);
    tool.recordResult({ r: "y" });
    tool.end();
  };

  /** A redaction hook that records the kinds it was called for. */
  const counting = (): {
    calls: string[];
    redact: (kind: string, value: JsonValue) => JsonValue;
  } => {
    const calls: string[] = [];
    return {
      calls,
      redact: (kind, value) => {
        calls.push(kind);
        return value;
      },
    };
  };

  const unsampledParent = (): ObservationContext => {
    const { observer } = createHarness({ sampler: new AlwaysOffSampler() });
    const context = observer.startWorkflow("w").context();
    if (context === undefined || context.sampled) throw new Error("expected an unsampled context");
    return context;
  };
  type Case = { sampler: Sampler; options: StartOptions; tracer?: TracerKind };
  const CASES: readonly [string, () => Case][] = [
    ["sampler drop", () => ({ sampler: new AlwaysOffSampler(), options: {} })],
    [
      "unsampled parent",
      () => ({
        sampler: new ParentBasedSampler({ root: new AlwaysOnSampler() }),
        options: { parent: unsampledParent() },
      }),
    ],
    ["suppressed", () => ({ sampler: new AlwaysOnSampler(), options: { trace: "suppress" } })],
    // The SDK provider (always on) is not wired to the observer.
    ["no tracer", () => ({ sampler: new AlwaysOnSampler(), options: {}, tracer: "none" })],
    ["API-only tracer", () => ({ sampler: new AlwaysOnSampler(), options: {}, tracer: "noop" })],
  ];

  for (const [name, setup] of CASES) {
    test(`${name}: span_only never runs the pipeline; event capture still does`, () => {
      const hook = counting();
      const { sampler, options, tracer } = setup();
      const quiet = createHarness({
        sampler,
        ...(tracer === undefined ? {} : { tracer }),
        content: { capture: "span_only", redact: hook.redact },
      });
      recordEveryContentKind(quiet.observer, options);
      expect(hook.calls).toEqual([]);
      expect(quiet.spans.getFinishedSpans()).toHaveLength(0);
      expect(quiet.logs.getFinishedLogRecords()).toHaveLength(0);

      for (const capture of ["span_and_event", "event_only"] as const) {
        const events = counting();
        const { sampler: eventSampler, options: eventOptions, tracer: eventTracer } = setup();
        const harness = createHarness({
          sampler: eventSampler,
          ...(eventTracer === undefined ? {} : { tracer: eventTracer }),
          content: { capture, redact: events.redact },
        });
        recordEveryContentKind(harness.observer, eventOptions);
        expect(events.calls).toEqual(CONTENT_KEYS);
        const onEvents = harness.logs
          .getFinishedLogRecords()
          .flatMap((record) => Object.keys(record.attributes))
          .filter((key) => CONTENT_KEYS.includes(key));
        expect(sorted(onEvents)).toEqual(sorted(CONTENT_KEYS));
      }
    });

    test(`${name}: with no logger, event capture does not run the pipeline either`, () => {
      for (const capture of ["span_and_event", "event_only"] as const) {
        const hook = counting();
        const { sampler, options, tracer } = setup();
        const harness = createHarness({
          sampler,
          logs: false,
          ...(tracer === undefined ? {} : { tracer }),
          content: { capture, redact: hook.redact },
        });
        recordEveryContentKind(harness.observer, options);
        expect(hook.calls).toEqual([]);
        expect(harness.spans.getFinishedSpans()).toHaveLength(0);
        expect(harness.logs.getFinishedLogRecords()).toHaveLength(0);
      }
    });
  }

  test("a recording span still processes and records every value", () => {
    const hook = counting();
    const { observer, spans } = createHarness({
      content: { capture: "span_only", redact: hook.redact },
    });
    recordEveryContentKind(observer, {});
    expect(hook.calls).toEqual(CONTENT_KEYS);
    const onSpans = spans
      .getFinishedSpans()
      .flatMap((span) => Object.keys(span.attributes))
      .filter((key) => CONTENT_KEYS.includes(key));
    expect(sorted(onSpans)).toEqual(sorted(CONTENT_KEYS));
  });
});

describe("content budgets fit one span in the request bound", () => {
  const ceiling = (DEFAULTS.max_request_body_bytes - DEFAULTS.span_envelope_allowance_bytes) / 2;

  test("budgets are clamped, never rejected, and the effective values are exposed", () => {
    expect(ceiling).toBe(212_992);
    expect(spanContentCeiling(DEFAULTS.max_request_body_bytes)).toBe(ceiling);
    expect(spanContentCeiling(DEFAULTS.span_envelope_allowance_bytes)).toBe(1);
    expect(spanContentCeiling(1)).toBe(1);
    const raised = new Observer({
      content: { maxAttributeBytes: 400_000, maxRetainedOperationBytes: 1_000_000 },
    });
    expect(raised.contentBudgets).toEqual({
      maxAttributeBytes: ceiling,
      maxRetainedOperationBytes: ceiling,
      maxRequestBodyBytes: DEFAULTS.max_request_body_bytes,
    });
    // The attribute budget follows a lower retained budget.
    expect(
      new Observer({ content: { maxAttributeBytes: 5000, maxRetainedOperationBytes: 1000 } })
        .contentBudgets.maxAttributeBytes,
    ).toBe(1000);
    // A smaller request bound lowers both; values under it are kept.
    const bound = DEFAULTS.span_envelope_allowance_bytes + 2 * 10_000;
    expect(
      new Observer({
        content: {
          maxAttributeBytes: 400_000,
          maxRetainedOperationBytes: 1_000_000,
          maxRequestBodyBytes: bound,
        },
      }).contentBudgets,
    ).toEqual({
      maxAttributeBytes: 10_000,
      maxRetainedOperationBytes: 10_000,
      maxRequestBodyBytes: bound,
    });
    expect(
      new Observer({ content: { maxAttributeBytes: 64, maxRequestBodyBytes: bound } })
        .contentBudgets.maxAttributeBytes,
    ).toBe(64);
  });

  test("truncation uses the effective budget", () => {
    const bound = DEFAULTS.span_envelope_allowance_bytes + 2 * 10_000;
    const { observer, spans } = createHarness({
      content: { capture: "span_only", maxAttributeBytes: 400_000, maxRequestBodyBytes: bound },
    });
    const op = observer.startInference(request);
    op.recordOutputMessages([
      {
        role: "assistant",
        parts: [{ type: "text", content: "b".repeat(20_000) }],
        finish_reason: "stop",
      },
    ]);
    op.end();
    const attributes = spans.getFinishedSpans()[0]?.attributes ?? {};
    const output = String(attributes["gen_ai.output.messages"]);
    expect(utf8Length(output)).toBeLessThanOrEqual(10_000);
    expect(output.endsWith(`${MARKER}"}],"finish_reason":"stop"}]`)).toBe(true);
    expect(attributes["xybrid.genai.content.truncated_attributes"]).toEqual([
      "gen_ai.output.messages",
    ]);
  });

  test("plain text is charged at its JSON-escaped length against the retained budget", () => {
    // 16 + 24 ("gen_ai.prompt.variable.a") + 100 raw bytes fits 300 twice; written as JSON the
    // 100 quotes are 200 bytes, so only the first variable fits.
    for (const [value, kept] of [
      ["x".repeat(100), 2],
      ['"'.repeat(100), 1],
    ] as const) {
      const { observer, spans } = createHarness({
        content: { capture: "span_only", maxRetainedOperationBytes: 300 },
      });
      observer
        .startInference({ ...request, prompt: { name: "p", variables: { a: value, b: value } } })
        .end();
      const attributes = spans.getFinishedSpans()[0]?.attributes ?? {};
      const variables = Object.keys(attributes).filter((key) =>
        key.startsWith("gen_ai.prompt.variable."),
      );
      expect(variables).toHaveLength(kept);
    }
  });
});
