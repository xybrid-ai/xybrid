import { expect, spyOn, test } from "bun:test";
import type { ReadableSpan } from "@opentelemetry/sdk-trace";
import { OtlpHttpSender } from "../src/otlp/http.ts";
import { CountingBatchSpanProcessor } from "../src/otlp/span-processor.ts";
import { createHarness } from "./support/harness.ts";

const encoder = new TextEncoder();
const pause = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

for (const pending of ["request", "retry"] as const) {
  test(`a batch deadline cancels a pending ${pending}, skips split siblings and allows later batches`, async () => {
    const { spans, tracerProvider } = createHarness();
    const tracer = tracerProvider.getTracer("test");
    tracer.startSpan("first").end();
    tracer.startSpan("second").end();
    tracer.startSpan("third").end();
    tracer.startSpan("fourth").end();
    const finished = spans.getFinishedSpans();
    const serializer = {
      serializeRequest: (items: ReadableSpan[]) =>
        encoder.encode(JSON.stringify(items.map((span) => span.name))),
    };
    const sender = new OtlpHttpSender(
      "traces",
      { endpoint: "https://example.test/v1/traces", timeoutMs: 1000 },
      { json: serializer, protobuf: serializer },
      { maxRetries: 2, initialDelayMs: 200, maxDelayMs: 200 },
      12,
    );
    const processor = new CountingBatchSpanProcessor(sender, {
      maxExportBatchSize: 3,
      exportTimeoutMs: 30,
      scheduledDelayMs: 60_000,
    });
    const calls: string[] = [];
    let aborted = false;
    const mockedFetch = async (...[_url, init]: Parameters<typeof fetch>) => {
      const names = new TextDecoder().decode(init?.body as Uint8Array);
      calls.push(names);
      if (names.includes("second")) {
        if (pending === "retry") return new Response(null, { status: 503 });
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            "abort",
            () => {
              aborted = true;
              reject(new DOMException("cancelled", "AbortError"));
            },
            { once: true },
          );
        });
      }
      return new Response(null, { status: 200 });
    };
    const mocked = spyOn(globalThis, "fetch").mockImplementation(
      Object.assign(mockedFetch, globalThis.fetch),
    );
    try {
      for (const span of finished) processor.onEnd(span);
      await processor.forceFlush();
      expect(calls).toEqual(['["first"]', '["second"]', '["fourth"]']);
      if (pending === "request") expect(aborted).toBe(true);
      expect(processor.stats()).toEqual({
        spansQueued: 4,
        spansExported: 2,
        spansDropped: 2,
        exportFailures: 1,
      });
      await pause(240);
      expect(calls).toHaveLength(3);
      const cancelled = new AbortController();
      cancelled.abort();
      expect(await sender.sendItems(finished, (items) => items, cancelled.signal)).toEqual({
        exported: 0,
        failed: 4,
        oversized: 0,
      });
      expect(calls).toHaveLength(3);
    } finally {
      sender.abort();
      await processor.shutdown();
      mocked.mockRestore();
      await tracerProvider.shutdown();
    }
  });
}
