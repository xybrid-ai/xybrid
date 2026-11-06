import { describe, expect, test } from "bun:test";
import { canonicalPart } from "../src/content.ts";
import {
  boundJson,
  type ContentCapture,
  DEFAULTS,
  substringRedactor,
  utf8Length,
  utf8Prefix,
} from "../src/index.ts";
import { createHarness, normalizeMetrics } from "./support/harness.ts";

const MARKER = DEFAULTS.truncation_marker;

describe("UTF-8 accounting", () => {
  test("lengths match TextEncoder", () => {
    for (const text of ["", "abc", "é", "€", "😀", "a😀é€", "\ud800", "x\udc00y"]) {
      expect(utf8Length(text)).toBe(new TextEncoder().encode(text).byteLength);
    }
  });

  test("prefixes never split a code point", () => {
    expect(utf8Prefix("éé", 3)).toBe("é");
    expect(utf8Prefix("€x", 2)).toBe("");
    expect(utf8Prefix("a😀", 4)).toBe("a");
    expect(utf8Prefix("a😀", 5)).toBe("a😀");
    expect(utf8Prefix("😀😀", 7)).toBe("😀");
    expect(utf8Prefix("abc", 10)).toBe("abc");
  });
});

describe("boundJson", () => {
  test("keeps a document within budget untouched", () => {
    const value = [{ role: "user", parts: [{ type: "text", content: "hi" }] }];
    const result = boundJson(value, 1000);
    expect(result).toEqual({ kind: "kept", value, json: JSON.stringify(value), truncated: false });
  });

  test("halves the longest string budget and appends the marker", () => {
    const value = { a: "x".repeat(200), b: "short" };
    const result = boundJson(value, 120);
    // K: 200 -> 100 (still too long: 100 + marker + overhead > 120) -> 50.
    expect(result.kind).toBe("kept");
    if (result.kind !== "kept") return;
    expect(result.truncated).toBe(true);
    expect(result.value).toEqual({ a: `${"x".repeat(50)}${MARKER}`, b: "short" });
    expect(utf8Length(result.json)).toBeLessThanOrEqual(120);
  });

  test("cuts multi-byte characters at code point boundaries", () => {
    // 100 'é' are 200 bytes; K halves to 100 bytes = 50 'é', then 50 bytes = 25 'é'.
    const result = boundJson({ text: "é".repeat(100) }, 80);
    expect(result.kind).toBe("kept");
    if (result.kind !== "kept") return;
    expect(result.value).toEqual({ text: `${"é".repeat(25)}${MARKER}` });
  });

  test("cuts before a surrogate pair rather than inside it", () => {
    // 'a' + 20 emoji = 81 bytes; K = 40 keeps 'a' and 9 emoji (37 bytes).
    const result = boundJson([`a${"😀".repeat(20)}`], 60);
    expect(result.kind).toBe("kept");
    if (result.kind !== "kept") return;
    expect(result.value).toEqual([`a${"😀".repeat(9)}${MARKER}`]);
    expect(new TextDecoder().decode(new TextEncoder().encode(result.json))).toBe(result.json);
  });

  test("object keys are never truncated or counted as strings", () => {
    const key = "k".repeat(100);
    const result = boundJson({ [key]: "v" }, 50);
    expect(result).toEqual({ kind: "omitted" });
  });

  test("drops the oldest array elements once strings reach the floor", () => {
    const messages = Array.from({ length: 6 }, (_, index) => ({
      role: "user",
      parts: [{ type: "text", content: `${index}`.repeat(40) }],
    }));
    const result = boundJson(messages, 150);
    expect(result.kind).toBe("kept");
    if (result.kind !== "kept") return;
    const kept = result.value as typeof messages;
    expect(kept.length).toBeLessThan(6);
    // The newest message always survives longest.
    expect(kept.at(-1)?.parts[0]?.content.startsWith("5")).toBe(true);
    expect(utf8Length(result.json)).toBeLessThanOrEqual(150);
  });

  test("omits a document that cannot fit", () => {
    expect(boundJson({ nested: { deep: [1, 2, 3, 4, 5, 6, 7, 8, 9] } }, 10)).toEqual({
      kind: "omitted",
    });
    expect(boundJson(["x".repeat(500)], 10)).toEqual({ kind: "omitted" });
  });
});

describe("content attributes on operations", () => {
  const request = { operation: "chat", provider: "llama.cpp", model: "m" } as const;

  test("redaction runs before truncation and reaches error messages", () => {
    const { observer, spans } = createHarness({
      content: { maxAttributeBytes: 120, redact: substringRedactor(["hunter2"]) },
    });
    const op = observer.startInference(request);
    op.recordInputMessages([
      { role: "user", parts: [{ type: "text", content: `pw hunter2 ${"z".repeat(100)}` }] },
    ]);
    op.endWithError({ errorType: "bad", message: "leaked hunter2" });
    const [span] = spans.getFinishedSpans();
    const input = String(span?.attributes["gen_ai.input.messages"]);
    expect(input).not.toContain("hunter2");
    expect(input).toContain("[REDA");
    expect(span?.status.message).toBe("leaked [REDACTED]");
    expect(span?.attributes["xybrid.genai.content.truncated_attributes"]).toEqual([
      "gen_ai.input.messages",
    ]);
  });

  test("error messages are bounded to 256 code points", () => {
    const { observer, spans } = createHarness();
    observer.startInference(request).endWithError({ errorType: "x", message: "😀".repeat(300) });
    expect(Array.from(spans.getFinishedSpans()[0]?.status.message ?? "").length).toBe(256);
  });

  test("omitted attributes are still listed as truncated", () => {
    const { observer, spans } = createHarness({ content: { maxAttributeBytes: 16 } });
    const op = observer.startInference(request);
    op.recordToolDefinitions([{ type: "function", name: "lookup", description: "x".repeat(10) }]);
    op.end();
    const [span] = spans.getFinishedSpans();
    expect(span?.attributes["gen_ai.tool.definitions"]).toBeUndefined();
    expect(span?.attributes["xybrid.genai.content.truncated_attributes"]).toEqual([
      "gen_ai.tool.definitions",
    ]);
  });

  test("parts serialize with the contract key order", () => {
    const { observer, spans } = createHarness();
    const op = observer.startInference(request);
    op.recordOutputMessages([
      {
        finish_reason: "tool_call",
        parts: [
          { arguments: { a: 1 }, name: "f", type: "tool_call", id: "c1" },
          { height: 2, width: 3, mime_type: "image/png", modality: "image", type: "media" },
          { uri: "u", modality: "audio", type: "uri" },
        ],
        role: "assistant",
      },
    ]);
    op.end();
    expect(spans.getFinishedSpans()[0]?.attributes["gen_ai.output.messages"]).toBe(
      '[{"role":"assistant","parts":[{"type":"tool_call","id":"c1","name":"f","arguments":{"a":1}},' +
        '{"type":"media","modality":"image","mime_type":"image/png","width":3,"height":2},' +
        '{"type":"uri","modality":"audio","uri":"u"}],"finish_reason":"tool_call"}]',
    );
  });
});

describe("failing redactors", () => {
  const request = { operation: "chat", provider: "llama.cpp", model: "m" } as const;
  const contentKeys = [
    "gen_ai.system_instructions",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.tool.definitions",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "gen_ai.prompt.variable.secret",
    "xybrid.genai.feedback.comment",
    "exception.message",
  ];

  for (const capture of [
    "span_only",
    "event_only",
    "span_and_event",
    "no_content",
  ] satisfies ContentCapture[]) {
    test(`${capture}: content is omitted while operations, events and metrics finish`, async () => {
      const attempted: string[] = [];
      const { observer, spans, logs, collectMetrics } = createHarness({
        content: {
          capture,
          redact: (kind) => {
            attempted.push(kind);
            throw new Error("private-redactor-error");
          },
        },
      });
      // Prompt variables and tool arguments are recorded inside their constructors.
      const op = observer.startInference({
        ...request,
        prompt: { name: "template", variables: { secret: "private-prompt" } },
      });
      op.recordSystemInstructions([{ type: "text", content: "private-instructions" }]);
      op.recordInputMessages([
        { role: "user", parts: [{ type: "text", content: "private-input" }] },
      ]);
      op.recordToolDefinitions([{ name: "lookup", description: "private-schema" }]);
      op.recordOutputMessages([
        {
          role: "assistant",
          parts: [{ type: "text", content: "private-output" }],
          finish_reason: "stop",
        },
      ]);
      op.recordUsage({ inputTokens: 2, outputTokens: 3 });
      const tool = observer.startTool({ name: "lookup", arguments: { key: "private-arguments" } });
      tool.recordResult("private-result");
      tool.end();
      op.endWithError({ errorType: "failed", message: "private-message" });
      expect(op.ended).toBe(true);
      const target = op.observationRef();
      if (target === undefined) throw new Error("expected a recorded span");
      expect(observer.recordFeedback(target, { score: 1, comment: "private-comment" })).toBe(true);

      const finished = spans.getFinishedSpans();
      expect(finished).toHaveLength(2);
      expect(finished[1]?.status).toEqual({ code: 2, message: "failed" });
      const records = logs.getFinishedLogRecords();
      const exception = records.find(
        (record) => record.eventName === "gen_ai.client.operation.exception",
      );
      expect(exception?.attributes["exception.type"]).toBe("failed");
      const feedback = records.find((record) => record.eventName === "gen_ai.evaluation.result");
      expect(feedback?.attributes["gen_ai.evaluation.score.value"]).toBe(1);
      for (const attributes of [
        ...finished.map((span) => span.attributes),
        ...records.map((record) => record.attributes),
      ]) {
        for (const key of contentKeys) expect(attributes[key]).toBeUndefined();
      }
      const metrics = normalizeMetrics(await collectMetrics());
      expect(
        metrics.find((metric) => metric.name === "gen_ai.client.inference.duration")?.points,
      ).toHaveLength(1);
      expect(
        metrics.find((metric) => metric.name === "gen_ai.execute_tool.duration")?.points,
      ).toHaveLength(1);
      expect(
        JSON.stringify({
          spans: finished.map((span) => ({ attributes: span.attributes, status: span.status })),
          records: records.map((record) => record.attributes),
          metrics,
        }),
      ).not.toContain("private-");
      expect(attempted).toContain("gen_ai.tool.definitions");
      expect(attempted).toContain("error.message");
      if (capture !== "no_content") {
        for (const key of contentKeys.filter((key) => key !== "exception.message")) {
          expect(attempted).toContain(key);
        }
      }
    });
  }

  test("a failed callback does not disable later redaction or tracing", () => {
    let calls = 0;
    const { observer, spans } = createHarness({
      content: {
        redact: () => {
          calls += 1;
          if (calls === 1) throw "private-thrown-value";
          return "safe";
        },
      },
    });
    const first = observer.startInference({
      ...request,
      prompt: { name: "template", variables: { secret: "private" } },
    });
    first.endWithError({ errorType: "failed", message: "private" });
    const second = observer.startInference({
      ...request,
      prompt: { name: "template", variables: { secret: "private" } },
    });
    second.end();
    expect(
      spans.getFinishedSpans()[0]?.attributes["gen_ai.prompt.variable.secret"],
    ).toBeUndefined();
    expect(spans.getFinishedSpans()[0]?.status.message).toBe("safe");
    expect(spans.getFinishedSpans()[1]?.attributes["gen_ai.prompt.variable.secret"]).toBe("safe");
  });

  test("intentional null and undefined redactor results keep their existing JSON semantics", () => {
    for (const value of [null, undefined]) {
      const { observer, spans } = createHarness({ content: { redact: () => value } });
      const op = observer.startInference(request);
      op.recordToolDefinitions([{ name: "private" }]);
      op.end();
      expect(spans.getFinishedSpans()[0]?.attributes["gen_ai.tool.definitions"]).toBe("null");
    }
  });
});

test("message parts retain explicit null tool arguments and responses", () => {
  expect(canonicalPart({ type: "tool_call", arguments: null })).toEqual({
    type: "tool_call",
    arguments: null,
  });
  expect(canonicalPart({ type: "tool_call_response", response: null })).toEqual({
    type: "tool_call_response",
    response: null,
  });
});
