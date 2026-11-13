import { expect, test } from "bun:test";
import { deserializeContext, serializeContext } from "../src/index.ts";
import { createHarness } from "./support/harness.ts";

test("context crosses a worker boundary and parents the worker's span", async () => {
  const { observer } = createHarness();
  const op = observer.startInference({ operation: "chat", provider: "llama.cpp", model: "m" });
  const message = { context: serializeContext(op.context()) };
  // What postMessage does to the payload.
  expect(deserializeContext(structuredClone(message).context)?.spanId).toBe(
    op.observationRef()?.span_id ?? "",
  );
  const worker = new Worker(new URL("./support/worker.ts", import.meta.url).href);
  try {
    const reply = await new Promise<{ traceId: string; parentSpanId: string; startedAtMs: number }>(
      (resolve, reject) => {
        worker.onmessage = (event) => resolve(event.data);
        worker.onerror = (event) => reject(new Error(event.message));
        worker.postMessage(message);
      },
    );
    expect(reply.traceId).toBe(op.observationRef()?.trace_id ?? "");
    expect(reply.parentSpanId).toBe(op.observationRef()?.span_id ?? "");
    // Workers report wall-clock epoch milliseconds, comparable with the main thread's.
    expect(Math.abs(reply.startedAtMs - Date.now())).toBeLessThan(60_000);
  } finally {
    worker.terminate();
    op.end();
  }
});
