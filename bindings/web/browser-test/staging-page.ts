// The step 6 staging scenario (tools/scripts/observability_staging/README.md) for the browser lane,
// `web-mac`, through the public @xybrid/web and @xybrid/observability APIs. It mirrors the Python
// reference, bindings/python/tests/observability_staging.py. staging.spec.ts bundles this module
// with the application side it needs (@xybrid/observability and its OpenTelemetry SDK) and runs it
// in the page with the SDK under test, imported from /sdk/index.js.
//
// The browser has no Collector: staging ingest allows any origin, so both sessions export straight
// to it with the bearer key. `direct` is the Xybrid preset, traces only, metadata only;
// `direct-logs` adds the logs signal, so feedback and evaluations reach staging, and captures
// content with one literal redaction.
import type { ObservationRef, Observer } from "@xybrid/observability";
import {
  createOtlpPipeline,
  type OtlpPipeline,
  type OtlpPipelineOptions,
  type SignalStats,
  xybridTracesPreset,
} from "@xybrid/observability/otlp";
import type { GenerateOptions, XybridLlm } from "../src/index.ts";

const SESSIONS = ["direct", "direct-logs"] as const;
export type Session = (typeof SESSIONS)[number];
type WebSdk = typeof import("../src/index.ts");
/** The pinned GGUF (example/model.ts) and the runtime, where `vite preview` serves them. */
const MODEL_URL = "/model.gguf";
const WASM_PATH = "/sdk/runtime";

type Destination = {
  readonly api_key: string;
  readonly traces_endpoint: string;
  readonly trace_id: string;
  readonly parent_span_id: string;
};

/** The lane's config.json, as far as the scenario reads it. */
export type StagingConfig = {
  readonly binding: string;
  readonly lane: string;
  readonly run: string;
  readonly commit: string;
  readonly model_id: string;
  readonly sessions: {
    readonly direct: Destination;
    readonly "direct-logs": Destination & { readonly logs_endpoint: string };
  };
  readonly constants: {
    readonly per_lane: Readonly<
      Record<"service_name" | "service_version" | "device_id" | "conversation_id", string>
    >;
    readonly workflow_name: string;
    readonly stopped: {
      readonly prompt: string;
      readonly max_tokens: number;
      readonly stop_after_chunks: number;
    };
    readonly completed: { readonly prompt: string; readonly max_tokens: number };
    readonly secret: string;
    readonly replacement: string;
    readonly feedback: { readonly score: number; readonly label: string; readonly comment: string };
    readonly evaluation: {
      readonly name: string;
      readonly score_value: number;
      readonly score_label: string;
      readonly explanation: string;
    };
    readonly timeout_ms: number;
  };
};

type Reference = { trace_id: string; span_id: string } | null;
type Counters = Pick<SignalStats, "accepted" | "exported" | "dropped" | "failed" | "pending">;
type Snapshot = { enabled: null; traces: Counters; logs: Counters; metrics: Counters };

/** One session of expectations.json (README). */
export type SessionRecord = Partial<{
  trace_id: string;
  parent_span_id: string;
  init: string;
  references: Record<"workflow" | "load" | "run", Reference>;
  stopped: { outcome: string; chunks_consumed: number; delivery: "pull" };
  completed: {
    outcome: string;
    chunks: number;
    tokens_out: number | null;
    text_chars: number;
    finish_reason: null;
  };
  quality: { feedback: boolean; evaluation: boolean };
  lifecycle: { flush: string; shutdown: string };
  stats: { after_flush: Snapshot; after_shutdown: Snapshot };
}>;

/** What the page reports; staging.spec.ts adds the rest of expectations.json. */
export type PageResult = {
  sdk_version: string | null;
  sessions: Partial<Record<Session, SessionRecord>>;
  scenario_error: string | null;
};

/** Python's `str.format` for the staging.json templates. */
const format = (template: string, values: Readonly<Record<string, string>>): string =>
  template.replace(/\{(\w+)\}/g, (match, key: string) => values[key] ?? match);

const describe = (error: unknown): string =>
  error instanceof Error ? `${error.name}: ${error.message}` : String(error);

const reference = (value: ObservationRef | undefined): Reference =>
  value === undefined ? null : { trace_id: value.trace_id, span_id: value.span_id };

const counters = ({ accepted, exported, dropped, failed, pending }: SignalStats): Counters => ({
  accepted,
  exported,
  dropped,
  failed,
  pending,
});

/** The README's stats shape. The JS pipeline has per-signal counters but no `enabled` flag. */
const snapshot = (pipeline: OtlpPipeline): Snapshot => {
  const { traces, logs, metrics } = pipeline.stats();
  return {
    enabled: null,
    traces: counters(traces),
    logs: counters(logs),
    metrics: counters(metrics),
  };
};

/** `"ok"`, or why a lifecycle call failed: the JS pipeline resolves `false` on a timeout or loss. */
const outcome = async (name: string, call: () => Promise<boolean>): Promise<string> => {
  try {
    return (await call()) ? "ok" : `error: ${name} resolved false`;
  } catch (error) {
    return `error: ${describe(error)}`;
  }
};

/** Step 1: the session's destinations, identity and content rules. */
const configure = (
  config: StagingConfig,
  session: Session,
): { pipeline: OtlpPipeline; observer: Observer } => {
  const { constants, binding, lane } = config;
  const values = config.sessions[session];
  // The Xybrid preset: http/protobuf and exactly `Authorization: Bearer <key>`.
  const preset = xybridTracesPreset(values.api_key, { endpoint: values.traces_endpoint });
  const signals: OtlpPipelineOptions =
    session === "direct"
      ? preset
      : {
          ...preset,
          logs: { ...preset.traces, endpoint: config.sessions["direct-logs"].logs_endpoint },
        };
  const pipeline = createOtlpPipeline({
    ...signals,
    resource: {
      serviceName: format(constants.per_lane.service_name, { binding }),
      serviceVersion: constants.per_lane.service_version,
      // The JS API has no device id option; `device.id` is the attribute the other SDKs set. Nor is
      // `xybrid.sdk.binding` set: CONTRACT.md leaves `xybrid.sdk.*` to the SDK adapter, and
      // neither @xybrid/web nor @xybrid/observability stamps it.
      attributes: {
        "device.id": format(constants.per_lane.device_id, { lane }),
        "xybrid.acceptance.run": config.run,
        "xybrid.acceptance.lane": lane,
        "xybrid.acceptance.commit": config.commit,
        "xybrid.acceptance.session": session,
      },
    },
    environment: null,
  });
  // `no_content` is the contract's metadata-only capture; `direct-logs` keeps the default.
  const observer = pipeline.observer(
    session === "direct"
      ? { capture: "no_content" }
      : {
          rules: [
            { type: "replace_literal", find: constants.secret, replace: constants.replacement },
          ],
        },
  );
  return { pipeline, observer };
};

/**
 * Steps 4 and 5: pulls deltas and counts the non-empty ones, which are the span's chunks (an
 * incomplete multi-byte character arrives empty). Leaving the loop after `stopAfter` chunks returns
 * the generator, the SDK's own stop: it cancels the run at its next token, and the span ends as
 * cancelled.
 */
const stream = async (
  model: XybridLlm,
  prompt: string,
  options: GenerateOptions,
  stopAfter?: number,
): Promise<{ outcome: string; chunks: number; text: string }> => {
  let chunks = 0;
  let text = "";
  for await (const delta of model.generateStream(prompt, options)) {
    text += delta;
    if (delta.length > 0) chunks += 1;
    if (stopAfter !== undefined && chunks >= stopAfter) {
      return { outcome: "cancelled", chunks, text };
    }
  }
  return { outcome: stopAfter === undefined ? "ok" : "completed", chunks, text };
};

/** One session, steps 1 to 7; returns the runtime version the loaded model reports. */
const runSession = async (
  sdk: WebSdk,
  config: StagingConfig,
  session: Session,
  record: SessionRecord,
): Promise<string> => {
  const { constants } = config;
  const values = config.sessions[session];
  const timeoutMs = constants.timeout_ms;
  const conversationId = format(constants.per_lane.conversation_id, { run: config.run, session });
  const references: Record<"workflow" | "load" | "run", Reference> = {
    workflow: null,
    load: null,
    run: null,
  };
  record.trace_id = values.trace_id;
  record.parent_span_id = values.parent_span_id;
  record.references = references;
  let configured: { pipeline: OtlpPipeline; observer: Observer };
  try {
    configured = configure(config, session);
  } catch (error) {
    record.init = `error: ${describe(error)}`;
    throw new Error(`${session} init failed: ${record.init}`);
  }
  record.init = "ok";
  const { pipeline, observer } = configured;
  try {
    // Step 2: the workflow continues the driver's trace T under span P.
    const workflow = observer.startWorkflow(constants.workflow_name, {
      parent: `00-${values.trace_id}-${values.parent_span_id}-01`,
    });
    references.workflow = reference(workflow.observationRef());
    const parent = workflow.context();
    const options = (maxOutputTokens: number): GenerateOptions => ({
      maxOutputTokens,
      observation: { parent, conversationId },
    });

    // Step 3: the pinned GGUF under the workflow (load observations take no conversation id).
    const model = await sdk.XybridLlm.fromUrl(MODEL_URL, {
      accelerator: "wasm",
      wasmPath: WASM_PATH,
      contextLength: 512,
      observer,
      observation: { parent },
    });
    let version: string;
    try {
      version = model.loaded.rustVersion;
      references.load = reference(model.loaded.observation);

      // Step 4: stopped after three chunks by leaving the loop.
      const { stopped, completed } = constants;
      let result = await stream(
        model,
        stopped.prompt,
        options(stopped.max_tokens),
        stopped.stop_after_chunks,
      );
      record.stopped = {
        outcome: result.outcome,
        chunks_consumed: result.chunks,
        delivery: "pull",
      };

      // Step 5: streamed to the end; the run's metrics carry its reference.
      result = await stream(model, completed.prompt, options(completed.max_tokens));
      const run = model.lastRun;
      record.completed = {
        outcome: result.outcome,
        chunks: result.chunks,
        tokens_out: run?.generatedTokens ?? null,
        text_chars: [...result.text].length,
        // The SDK reports no finish reason; it derives one for the span only.
        finish_reason: null,
      };
      const runRef = run?.observation;
      references.run = reference(runRef);

      // Step 6: quality results on the run reference.
      if (runRef === undefined) throw new Error("the completed run reported no observation");
      const { feedback, evaluation } = constants;
      record.quality = {
        feedback: observer.recordFeedback(runRef, feedback),
        evaluation: observer.recordEvaluation(runRef, {
          name: evaluation.name,
          scoreValue: evaluation.score_value,
          scoreLabel: evaluation.score_label,
          explanation: evaluation.explanation,
        }),
      };
    } finally {
      await model.dispose();
    }

    // Step 7: lifecycle.
    workflow.end();
    const flush = await outcome("forceFlush", () => pipeline.forceFlush(timeoutMs));
    const afterFlush = snapshot(pipeline);
    const shutdown = await outcome("shutdown", () => pipeline.shutdown(timeoutMs));
    record.lifecycle = { flush, shutdown };
    record.stats = { after_flush: afterFlush, after_shutdown: snapshot(pipeline) };
    return version;
  } catch (error) {
    // As the reference does after a failure: deliver or count what is still queued.
    await pipeline.shutdown(timeoutMs);
    throw error;
  }
};

/** Both sessions in order, in one page. Never throws, so the spec can always write the result. */
export const runStaging = async (sdk: WebSdk, config: StagingConfig): Promise<PageResult> => {
  const result: PageResult = { sdk_version: null, sessions: {}, scenario_error: null };
  try {
    for (const session of SESSIONS) {
      const record: SessionRecord = {};
      result.sessions[session] = record;
      const version = await runSession(sdk, config, session, record);
      result.sdk_version ??= version;
    }
  } catch (error) {
    result.scenario_error =
      error instanceof Error ? (error.stack ?? describe(error)) : String(error);
  }
  return result;
};
