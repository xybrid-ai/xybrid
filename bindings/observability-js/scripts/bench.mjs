// Overhead of @xybrid/observability in Node: one streaming inference operation with N chunks and a
// single recordChunk call, disabled vs enabled with in-memory exporters, and the content pipeline
// (preflight, redaction, truncation) for representative message sizes up to the 480 KiB budget.
// The same workloads as the Rust benches in crates/xybrid-observability/benches.
//
//   pnpm build && node --expose-gc scripts/bench.mjs [--quick] [--out FILE]
//
// Writes JSON (default test-results/bench/observability-js-bench.json) and prints a summary.
// Runs against the built package in dist/; no runtime dependency beyond the OpenTelemetry dev
// dependencies the tests already use.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import os from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  InMemoryLogRecordExporter,
  LoggerProvider,
  SimpleLogRecordProcessor,
} from "@opentelemetry/sdk-logs";
import { MeterProvider, MetricReader } from "@opentelemetry/sdk-metrics";
import {
  AlwaysOffSampler,
  InMemorySpanExporter,
  SimpleSpanProcessor,
  TracerProvider,
} from "@opentelemetry/sdk-trace";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const distEntry = join(root, "dist", "index.js");
if (!existsSync(distEntry)) {
  console.error("bench: dist/index.js not found; run `pnpm build` first");
  process.exit(2);
}
const { Observer, VERSION, substringRedactor } = await import(pathToFileURL(distEntry).href);

const args = process.argv.slice(2);
const quick = args.includes("--quick");
const outIndex = args.indexOf("--out");
const outPath =
  outIndex >= 0 && args[outIndex + 1]
    ? resolve(args[outIndex + 1])
    : join(root, "test-results", "bench", "observability-js-bench.json");

const INPUT_MESSAGES = "gen_ai.input.messages";
const TRUNCATED = "xybrid.genai.content.truncated_attributes";
const TRAVERSAL_BUDGET = 491_520;
const ATTRIBUTE_BUDGET = 16_384;
const CHUNK_COUNTS = [1, 32, 256];
const BYTES_PER_CHUNK = 4;

// ---------------------------------------------------------------------------
// Representative content (identical to benches/support/mod.rs)
// ---------------------------------------------------------------------------

const PROSE =
  "Lisbon sits on seven hills above the Tagus estuary, and its yellow trams climb past tiled " +
  "facades towards the castle. Contact user@example.com about order 4411 or call +1-555-0100 " +
  "before noon. The deploy key sk-live-0000 must never appear in logs. Pastelarias open early; " +
  "the miradouros fill at sunset. ";

const text = (bytes) => {
  let out = "";
  while (out.length < bytes) out += PROSE;
  return out.slice(0, bytes);
};
const message = (role, bytes) => ({ role, parts: [{ type: "text", content: text(bytes) }] });
const single = (bytes) => [message("user", bytes)];
const history = (n, bytes) =>
  Array.from({ length: n }, (_, i) => message(i % 2 === 0 ? "user" : "assistant", bytes));

/** Preflight cost of the canonical JSON form (CONTRACT.md, preflight_node_bytes = 16). */
const cost = (value) => {
  if (value === null || typeof value === "boolean" || typeof value === "number") return 16;
  if (typeof value === "string") return 16 + Buffer.byteLength(value, "utf8");
  if (Array.isArray(value)) return 16 + value.reduce((sum, v) => sum + cost(v), 0);
  let sum = 16;
  for (const [k, v] of Object.entries(value)) sum += 16 + Buffer.byteLength(k, "utf8") + cost(v);
  return sum;
};
const jsonBytes = (value) => Buffer.byteLength(JSON.stringify(value), "utf8");

/** The largest history of `bytes`-byte messages whose preflight cost stays within `budget`. */
const historyWithin = (budget, bytes) => {
  const one = cost(history(1, bytes)) - 16;
  let n = Math.floor((budget - 16) / one);
  while (n > 0 && cost(history(n, bytes)) > budget) n -= 1;
  while (cost(history(n + 1, bytes)) <= budget) n += 1;
  return history(n, bytes);
};

const workloads = () => {
  const near4k = historyWithin(TRAVERSAL_BUDGET, 4096);
  const justUnder = ATTRIBUTE_BUDGET - jsonBytes(single(0));
  return [
    { id: "msg_256b", messages: single(256), expect: "kept" },
    { id: "msg_1k", messages: single(1024), expect: "kept" },
    { id: "msg_4k", messages: single(4096), expect: "kept" },
    { id: "msg_15k", messages: single(justUnder), expect: "kept" },
    { id: "msg_17k", messages: single(17 * 1024), expect: "truncated" },
    { id: "msg_64k", messages: single(64 * 1024), expect: "truncated" },
    { id: "msg_256k", messages: single(256 * 1024), expect: "truncated" },
    { id: "hist_32x4k", messages: history(32, 4096), expect: "truncated" },
    { id: "hist_near_budget_4k", messages: near4k, expect: "truncated" },
    {
      id: "hist_near_budget_64b",
      messages: historyWithin(TRAVERSAL_BUDGET, 64),
      expect: "truncated",
    },
    {
      id: "hist_over_budget",
      messages: [...near4k, message("user", 4096)],
      expect: "failed_closed",
    },
  ];
};

const RULES = [
  { type: "replace_literal", find: "user@example.com", replace: "[EMAIL]" },
  { type: "replace_literal", find: "+1-555-0100", replace: "[PHONE]" },
  { type: "replace_literal", find: "sk-live-", replace: "[KEY]-" },
  { type: "omit_attribute", key: "gen_ai.tool.definitions" },
];
const POLICIES = [
  ["metadata_only", { capture: "no_content" }],
  ["default", { capture: "span_only" }],
  ["rules", { capture: "span_only", rules: RULES }],
  [
    "hook_and_rules",
    { capture: "span_only", redact: substringRedactor(["Lisbon", "Tagus"]), rules: RULES },
  ],
  // The default policy under a sampler that drops every span: the span is not recording, so the
  // capture gate closes its destination and no content work is done (about metadata_only's cost).
  ["unsampled", { capture: "span_only" }],
];

// ---------------------------------------------------------------------------
// Setups
// ---------------------------------------------------------------------------

class ManualReader extends MetricReader {
  async onForceFlush() {}
  async onShutdown() {}
}

const memorySetup = (
  name,
  content,
  { traces = true, metrics = true, logs = true, sampled = true } = {},
) => {
  const spans = new InMemorySpanExporter();
  const tracerProvider = new TracerProvider({
    spanProcessors: [new SimpleSpanProcessor({ exporter: spans })],
    ...(sampled ? {} : { sampler: new AlwaysOffSampler() }),
  });
  const meterProvider = new MeterProvider({ readers: [new ManualReader()] });
  const loggerProvider = new LoggerProvider({
    processors: [new SimpleLogRecordProcessor({ exporter: new InMemoryLogRecordExporter() })],
  });
  return {
    name,
    observer: new Observer({
      tracerProvider: traces ? tracerProvider : null,
      meterProvider: metrics ? meterProvider : null,
      loggerProvider: logs ? loggerProvider : null,
      content,
    }),
    spans,
    drain: () => spans.reset(),
    shutdown: () =>
      Promise.all([tracerProvider.shutdown(), meterProvider.shutdown(), loggerProvider.shutdown()]),
  };
};

const disabledSetup = () => ({
  name: "disabled",
  observer: new Observer({ tracerProvider: null, meterProvider: null, loggerProvider: null }),
  spans: null,
  drain: () => {},
  shutdown: async () => {},
});

// ---------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------

const now = () => process.hrtime.bigint();
const ns = (start) => Number(process.hrtime.bigint() - start);
const gc = () => globalThis.gc?.();

const distribution = (values) => {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (v.length === 0) return { n: 0 };
  const at = (q) => v[Math.min(Math.max(Math.ceil(q * v.length), 1), v.length) - 1];
  const mean = v.reduce((s, x) => s + x, 0) / v.length;
  return {
    n: v.length,
    min: v[0],
    p50: at(0.5),
    p90: at(0.9),
    p95: at(0.95),
    p99: at(0.99),
    p999: at(0.999),
    max: v[v.length - 1],
    mean,
  };
};

const request = {
  operation: "chat",
  provider: "llama.cpp",
  model: "qwen3.5-0.8b",
  location: "local",
  runtime: "llama.cpp",
  stream: true,
};

const work = {
  system: [{ type: "text", content: text(512) }],
  input: single(1024),
  output: new Map(CHUNK_COUNTS.map((n) => [n, text(n * BYTES_PER_CHUNK)])),
};

const streamOp = (observer, chunks) => {
  const op = observer.startInference(request);
  op.recordSystemInstructions(work.system);
  op.recordInputMessages(work.input);
  for (let i = 0; i < chunks; i++) {
    op.recordChunk();
    if (i === 0) op.recordFirstToken();
  }
  op.recordUsage({ inputTokens: 300, outputTokens: chunks });
  op.recordFinishReasons(["stop"]);
  op.recordOutputMessages([
    {
      role: "assistant",
      parts: [{ type: "text", content: work.output.get(chunks) }],
      finish_reason: "stop",
    },
  ]);
  op.end();
};

/** Per-operation time: `samples` samples, each the mean of a timed batch. */
const measureOps = (setup, chunks, samples, targetNs) => {
  // Calibrate a batch size so each sample takes roughly `targetNs`.
  let batch = 1;
  for (;;) {
    const start = now();
    for (let i = 0; i < batch; i++) streamOp(setup.observer, chunks);
    const took = ns(start);
    setup.drain();
    if (took >= targetNs / 4 || batch >= 1 << 16) {
      batch = Math.max(1, Math.round((batch * targetNs) / Math.max(took, 1)));
      break;
    }
    batch *= 2;
  }
  const perOp = [];
  for (let s = 0; s < samples; s++) {
    const start = now();
    for (let i = 0; i < batch; i++) streamOp(setup.observer, chunks);
    perOp.push(ns(start) / batch);
    setup.drain();
  }
  return { batch, perOp };
};

const measureChunks = (setup, calls) => {
  const op = setup.observer.startInference(request);
  op.recordChunk();
  // Batch mean first (no per-call timer), then individually timed calls.
  const start = now();
  for (let i = 0; i < calls; i++) op.recordChunk();
  const batchMean = ns(start) / calls;
  const perCall = new Array(calls);
  for (let i = 0; i < calls; i++) {
    const t = now();
    op.recordChunk();
    perCall[i] = ns(t);
  }
  op.end();
  setup.drain();
  return { batchMean, perCall };
};

const contentOutcome = (setup, messages) => {
  setup.drain();
  const op = setup.observer.startInference(request);
  op.recordInputMessages(messages);
  op.end();
  const span = setup.spans.getFinishedSpans().at(-1);
  setup.drain();
  const recorded = span?.attributes[INPUT_MESSAGES];
  const marked = (span?.attributes[TRUNCATED] ?? []).includes(INPUT_MESSAGES);
  if (recorded === undefined) {
    return { outcome: marked ? "failed_closed" : "skipped", recordedBytes: 0 };
  }
  return {
    outcome: marked ? "truncated" : "kept",
    recordedBytes: Buffer.byteLength(String(recorded), "utf8"),
  };
};

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

const samples = quick ? 10 : 60;
const sampleTargetNs = quick ? 5e6 : 20e6;
const chunkCalls = quick ? 20_000 : 200_000;
const contentSamples = quick ? 10 : 200;

const setups = [
  disabledSetup(),
  memorySetup("memory_metadata", { capture: "no_content" }),
  memorySetup("memory_content", { capture: "span_only" }),
];

const results = { stream_op: [], chunk: [], content: [] };
const memory = {};

for (const setup of setups) {
  gc();
  const before = process.memoryUsage();
  // Warm up the JIT on this setup.
  for (let i = 0; i < 2000; i++) streamOp(setup.observer, 32);
  setup.drain();
  for (const chunks of CHUNK_COUNTS) {
    const { batch, perOp } = measureOps(setup, chunks, samples, sampleTargetNs);
    results.stream_op.push({
      setup: setup.name,
      chunks,
      ops_per_sample: batch,
      per_op_ns: distribution(perOp),
    });
  }
  const { batchMean, perCall } = measureChunks(setup, chunkCalls);
  results.chunk.push({
    setup: setup.name,
    calls: chunkCalls,
    batch_mean_ns: batchMean,
    per_call_ns: distribution(perCall),
  });
  gc();
  const after = process.memoryUsage();
  memory[setup.name] = {
    rss_before: before.rss,
    rss_after: after.rss,
    heap_used_before: before.heapUsed,
    heap_used_after: after.heapUsed,
  };
}

// The outcome a workload must have under a policy: metadata_only skips content; the callback's
// "[REDACTED]" is longer than the words it replaces, so it pushes the near-budget histories over
// the traversal budget and they fail closed (CONTRACT.md step 6).
const expected = (policy, w) => {
  if (policy === "metadata_only" || policy === "unsampled") return "skipped";
  if (policy === "hook_and_rules" && w.id.startsWith("hist_near_budget")) return "failed_closed";
  return w.expect;
};

const catalog = workloads();
for (const [policy, content] of POLICIES) {
  const setup = memorySetup(`content_${policy}`, content, {
    metrics: false,
    logs: false,
    sampled: policy !== "unsampled",
  });
  for (const w of catalog) {
    const check = contentOutcome(setup, w.messages);
    if (check.outcome !== expected(policy, w)) {
      throw new Error(
        `workload ${w.id} under ${policy}: expected ${expected(policy, w)}, got ${check.outcome}`,
      );
    }
    for (let i = 0; i < 3; i++) {
      const op = setup.observer.startInference(request);
      op.recordInputMessages(w.messages);
      op.end();
    }
    setup.drain();
    const perCall = [];
    for (let s = 0; s < contentSamples; s++) {
      const op = setup.observer.startInference(request);
      const t = now();
      op.recordInputMessages(w.messages);
      perCall.push(ns(t));
      op.end();
      setup.drain();
    }
    results.content.push({
      policy,
      workload: w.id,
      messages: w.messages.length,
      json_bytes: jsonBytes(w.messages),
      preflight_cost: cost(w.messages),
      outcome: check.outcome,
      recorded_bytes: check.recordedBytes,
      per_call_ns: distribution(perCall),
    });
  }
  await setup.shutdown();
}
await Promise.all(setups.map((s) => s.shutdown()));

const report = {
  bench: "observability-js",
  environment: {
    package_version: VERSION,
    node: process.version,
    v8: process.versions.v8,
    platform: process.platform,
    arch: process.arch,
    cpu: os.cpus()[0]?.model,
    cpu_count: os.cpus().length,
    expose_gc: typeof globalThis.gc === "function",
    quick,
  },
  config: {
    chunk_counts: CHUNK_COUNTS,
    samples,
    sample_target_ns: sampleTargetNs,
    chunk_calls: chunkCalls,
    content_samples: contentSamples,
    timer: "process.hrtime.bigint() around each sample or call (includes timer overhead)",
  },
  results,
  memory,
};

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);

const row = (cells) => `| ${cells.join(" | ")} |`;
const f = (v) => (v === undefined || v === null ? "–" : Math.round(v).toLocaleString("en-US"));
console.log(row(["setup", "chunks", "per-op p50 ns", "per-op p95 ns"]));
for (const r of results.stream_op) {
  console.log(row([r.setup, r.chunks, f(r.per_op_ns.p50), f(r.per_op_ns.p95)]));
}
console.log(row(["setup", "recordChunk mean ns", "p50 ns", "p99 ns", "p99.9 ns"]));
for (const r of results.chunk) {
  console.log(
    row([
      r.setup,
      f(r.batch_mean_ns),
      f(r.per_call_ns.p50),
      f(r.per_call_ns.p99),
      f(r.per_call_ns.p999),
    ]),
  );
}
console.log(row(["policy", "workload", "outcome", "p50 ns", "p99 ns"]));
for (const r of results.content) {
  console.log(row([r.policy, r.workload, r.outcome, f(r.per_call_ns.p50), f(r.per_call_ns.p99)]));
}
console.log(`wrote ${outPath}`);
