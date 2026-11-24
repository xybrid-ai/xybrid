// A Web Worker that receives a serialized context, records a child span with its own observer,
// and posts back what it recorded. Mirrors how a browser runtime worker joins the caller's trace.
import { deserializeContext, type SerializedObservationContext } from "../../src/index.ts";
import { createHarness } from "./harness.ts";

declare const self: Worker;

self.onmessage = (event: MessageEvent<{ context: SerializedObservationContext }>) => {
  const parent = deserializeContext(event.data.context);
  const { observer, spans } = createHarness();
  const started = performance.timeOrigin + performance.now();
  observer.startTool({ name: "worker_step" }, { parent }).end();
  const [span] = spans.getFinishedSpans();
  self.postMessage({
    traceId: span?.spanContext().traceId,
    parentSpanId: span?.parentSpanContext?.spanId,
    startedAtMs: started,
  });
};
