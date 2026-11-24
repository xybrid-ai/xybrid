import { expect, test } from "bun:test";
import {
  observeFoundationModelsStream,
  observeLiteRtEmbeddings,
  observeTransformersGeneration,
} from "../examples/runtime-readiness.ts";
import { createHarness } from "./support/harness.ts";

test("future runtime adapters preserve parents and missing usage without inventing token timing", async () => {
  const h = createHarness();
  const workflow = h.observer.startWorkflow("future runtimes");
  const options = { parent: workflow.context() };
  expect(
    await observeTransformersGeneration(
      h.observer,
      "tiny",
      "hello",
      async () => ({ text: "hi", outputTokens: 1 }),
      options,
    ),
  ).toEqual({ text: "hi", outputTokens: 1 });
  expect(
    await observeLiteRtEmbeddings(h.observer, "embed", async () => [0.1, 0.2], options),
  ).toEqual([0.1, 0.2]);
  const chunks = [];
  for await (const chunk of observeFoundationModelsStream(
    h.observer,
    "system",
    async function* () {
      yield "one";
      yield "two";
    },
    options,
  ))
    chunks.push(chunk);
  expect(chunks).toEqual(["one", "two"]);
  workflow.end();
  const spans = h.spans.getFinishedSpans();
  expect(spans).toHaveLength(4);
  const generated = spans.find(
    (span) => span.attributes["xybrid.genai.runtime.name"] === "transformers.js",
  );
  expect(generated?.attributes["gen_ai.usage.output_tokens"]).toBe(1);
  expect(generated?.attributes["gen_ai.usage.input_tokens"]).toBeUndefined();
  for (const span of spans.filter((span) => span !== spans.at(-1))) {
    expect(span.parentSpanContext?.spanId).toBe(workflow.observationRef()?.span_id);
    expect(span.attributes["xybrid.genai.response.time_to_first_token"]).toBeUndefined();
  }
  const streamed = spans.find(
    (span) => span.attributes["xybrid.genai.runtime.name"] === "foundation_models",
  );
  expect(streamed?.attributes["xybrid.genai.response.chunk_count"]).toBe(2);
  expect(streamed?.attributes["gen_ai.usage.output_tokens"]).toBeUndefined();
  await h.tracerProvider.shutdown();
});

test("a future stream abandoned by its caller cancels exactly once", async () => {
  const h = createHarness();
  for await (const _chunk of observeFoundationModelsStream(
    h.observer,
    "system",
    async function* () {
      yield "one";
      throw new Error("must never continue after abandonment");
    },
  ))
    break;
  expect(h.spans.getFinishedSpans()).toHaveLength(1);
  expect(h.spans.getFinishedSpans()[0]?.attributes["error.type"]).toBe("cancelled");
  await h.tracerProvider.shutdown();
});
