// Adapter from the browser runtime to an application-owned @xybrid/observability Observer.
// Only types are imported: the caller's observer carries all behavior, so @xybrid/web bundles
// none of the package and an unobserved model pays nothing.
import type {
  InferenceOperation,
  InferenceRequest,
  ObservationContext,
  ObservationRef,
  Observer,
} from "@xybrid/observability";
import { RuntimeConfigurationError } from "../errors.ts";
import type {
  GenerateObservation,
  GenerateOptions,
  SelectedAccelerator,
  TraceOptions,
} from "../types.ts";
import { epochClock } from "./clock.ts";
import { isAbortError } from "./loading.ts";
import type { GenerateObservationContext, LoadMetrics, RunMetrics } from "./protocol.ts";
import type { GenerationTelemetry } from "./runtime.ts";

/** llama.cpp serves local GGUF generation: it is both the provider and the runtime. */
const RUNTIME = "llama.cpp";
const ACCELERATOR_ATTRIBUTE = "xybrid.genai.accelerator";

/** Application-owned telemetry must never interrupt runtime work or cleanup. */
export const observeSafely = <T>(action: () => T): T | undefined => {
  try {
    return action();
  } catch {
    return undefined;
  }
};

export type LoadObservation = {
  readonly observer: Observer;
  readonly trace: TraceOptions | undefined;
};

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const validateTraceOptions = (value: unknown, name: string): TraceOptions | undefined => {
  if (value === undefined) return undefined;
  if (!isObject(value)) throw new RuntimeConfigurationError(`${name} must be an object.`);
  const parent = value["parent"];
  if (
    parent !== undefined &&
    parent !== null &&
    typeof parent !== "string" &&
    !(
      isObject(parent) &&
      (typeof parent["traceparent"] === "string" ||
        (typeof parent["traceId"] === "string" &&
          typeof parent["spanId"] === "string" &&
          typeof parent["traceFlags"] === "number"))
    )
  ) {
    throw new RuntimeConfigurationError(
      `${name}.parent must be an ObservationContext, a traceparent string, a serialized context or context fields.`,
    );
  }
  const tracestate = value["tracestate"];
  if (tracestate !== undefined && typeof tracestate !== "string") {
    throw new RuntimeConfigurationError(`${name}.tracestate must be a string.`);
  }
  const trace = value["trace"];
  if (trace !== undefined && trace !== "default" && trace !== "suppress") {
    throw new RuntimeConfigurationError(`${name}.trace must be "default" or "suppress".`);
  }
  return value as TraceOptions;
};

/** Reads `observer` and `observation` from raw load options. */
export const loadObservation = (options: unknown): LoadObservation | undefined => {
  if (!isObject(options)) return undefined;
  const observer = options["observer"];
  const trace = validateTraceOptions(options["observation"], "observation");
  if (observer === undefined) return undefined;
  if (!isObject(observer)) {
    throw new RuntimeConfigurationError("observer must be an @xybrid/observability Observer.");
  }
  const valid = observeSafely(
    () =>
      typeof observer["startInference"] === "function" &&
      typeof observer["startModelLoad"] === "function",
  );
  // A provider accessor can itself throw. Treat that as unavailable telemetry.
  if (valid === undefined) return undefined;
  if (!valid) {
    throw new RuntimeConfigurationError("observer must be an @xybrid/observability Observer.");
  }
  return { observer: observer as unknown as Observer, trace };
};

export const validateGenerateObservation = (value: unknown): GenerateObservation | undefined => {
  const trace = validateTraceOptions(value, "observation");
  if (trace === undefined) return undefined;
  const values = value as Record<string, unknown>;
  const conversationId = values["conversationId"];
  if (conversationId !== undefined && typeof conversationId !== "string") {
    throw new RuntimeConfigurationError("observation.conversationId must be a string.");
  }
  const prompt = values["prompt"];
  if (prompt !== undefined && (!isObject(prompt) || typeof prompt["name"] !== "string")) {
    throw new RuntimeConfigurationError("observation.prompt must have a string name.");
  }
  return value as GenerateObservation;
};

const startOptions = (trace: TraceOptions | undefined, startTimeMs?: number) => ({
  ...(trace?.parent === undefined ? {} : { parent: trace.parent }),
  ...(trace?.tracestate === undefined ? {} : { tracestate: trace.tracestate }),
  ...(trace?.trace === undefined ? {} : { trace: trace.trace }),
  ...(startTimeMs === undefined ? {} : { startTimeMs }),
});

/** A low-cardinality `error.type`: the error class name. */
const errorType = (error: unknown): string =>
  error instanceof Error && /^[A-Za-z][\w.]{0,63}$/.test(error.name) ? error.name : "_OTHER";

const errorMessage = (error: unknown): string | undefined =>
  error instanceof Error ? error.message : undefined;

/**
 * Records a finished (or failed) model load as a `load_model` operation spanning the whole
 * call: metadata, download, verification, runtime initialization and engine creation.
 */
export const recordLoad = <
  T extends { readonly loaded: LoadMetrics; readonly accelerator: SelectedAccelerator },
>(
  observation: LoadObservation,
  model: string,
  startedAtMs: number,
  outcome: { readonly session: T } | { readonly error: unknown },
  completedAtMs: number,
): ObservationRef | undefined => {
  const session = "session" in outcome ? outcome.session : undefined;
  const operation = observeSafely(() =>
    observation.observer.startModelLoad(
      {
        model,
        provider: RUNTIME,
        runtime: RUNTIME,
        format: "gguf",
        ...(session === undefined ? {} : { sizeBytes: session.loaded.modelBytes }),
      },
      startOptions(observation.trace, startedAtMs),
    ),
  );
  if (operation === undefined) return undefined;
  if (session !== undefined) {
    observeSafely(() => operation.setAttribute(ACCELERATOR_ATTRIBUTE, session.accelerator));
    observeSafely(() =>
      operation.recordResource({ memoryUsedBytes: session.loaded.memory.allocatedBytes }),
    );
    observeSafely(() => operation.end(completedAtMs));
  } else if ("error" in outcome && isAbortError(outcome.error)) {
    observeSafely(() => operation.cancel(completedAtMs));
  } else if ("error" in outcome) {
    // Download errors can embed signed URLs, userinfo and other credentials. Keep
    // the original error for the caller, but never hand its message to telemetry.
    observeSafely(() =>
      operation.endWithError({ errorType: errorType(outcome.error) }, completedAtMs),
    );
  }
  return observeSafely(() => operation.observationRef());
};

const contextFields = (context: ObservationContext | undefined): GenerateObservationContext =>
  context === undefined
    ? {}
    : {
        traceparent: context.toTraceparent(),
        ...(context.tracestate === undefined ? {} : { tracestate: context.tracestate }),
      };

/**
 * One observed generation. The inference starts on the main thread; the worker reports token
 * and completion times as epoch milliseconds, recorded here with the explicit-time API.
 */
export class GenerationObservation {
  readonly #operation: InferenceOperation | undefined;
  readonly #captureText: boolean;
  readonly #maxTokens: number;
  readonly #engineTiming: boolean;
  readonly #clock = epochClock();
  #text = "";
  #completedAtMs: number | undefined;

  constructor(
    observer: Observer,
    model: string,
    accelerator: SelectedAccelerator,
    prompt: string,
    options: GenerateOptions,
    maxTokens: number,
    streaming: boolean,
    engineTiming: boolean,
  ) {
    const observation = options.observation;
    const request: InferenceRequest = {
      // The runtime applies the model's chat template to the prompt as one user turn.
      operation: "chat",
      provider: RUNTIME,
      model,
      location: "local",
      runtime: RUNTIME,
      accelerator,
      maxTokens,
      ...(streaming ? { stream: true } : {}),
      ...(observation?.conversationId === undefined
        ? {}
        : { conversationId: observation.conversationId }),
      ...(observation?.prompt === undefined ? {} : { prompt: observation.prompt }),
      inputModalities: ["text"],
      outputModalities: ["text"],
    };
    this.#operation = observeSafely(() =>
      observer.startInference(request, startOptions(observation)),
    );
    this.#captureText =
      this.#operation !== undefined &&
      observeSafely(() => observer.contentCapture !== "no_content") === true;
    this.#maxTokens = maxTokens;
    this.#engineTiming = engineTiming;
    observeSafely(() =>
      this.#operation?.recordInputMessages([
        { role: "user", parts: [{ type: "text", content: prompt }] },
      ]),
    );
  }

  /** Hooks handed to the engine; the worker calls them with its own timestamps. */
  telemetry(): GenerationTelemetry {
    return {
      context: observeSafely(() => contextFields(this.#operation?.context())) ?? {},
      onToken: (atMs, text) => {
        observeSafely(() => this.#operation?.recordFirstToken(atMs));
        if (text.length > 0) observeSafely(() => this.#operation?.recordChunk(atMs));
      },
      onComplete: (atMs) => {
        this.#completedAtMs = atMs;
      },
    };
  }

  /** Each delta the caller receives. Engines without their own timing are timed here. */
  delta(text: string): void {
    if (this.#captureText) this.#text += text;
    if (!this.#engineTiming && text.length > 0) {
      observeSafely(() => this.#operation?.recordChunk(this.#clock()));
    }
  }

  /**
   * Ends the inference. `run` is this generation's metrics when the engine reported them;
   * usage is never fabricated otherwise.
   */
  finish(
    outcome: "completed" | "abandoned" | { readonly error: unknown },
    run: RunMetrics | undefined,
  ): ObservationRef | undefined {
    const operation = this.#operation;
    if (operation === undefined) return undefined;
    const at = this.#completedAtMs ?? this.#clock();
    if (run !== undefined) {
      observeSafely(() =>
        operation.recordUsage({ inputTokens: run.promptTokens, outputTokens: run.generatedTokens }),
      );
      const decodeTokensPerSecond = run.decodeTokensPerSecond;
      if (decodeTokensPerSecond !== null && decodeTokensPerSecond > 0) {
        observeSafely(() =>
          operation.recordTokenTiming({ timePerOutputTokenS: 1 / decodeTokensPerSecond }),
        );
      }
    }
    if (typeof outcome === "object") {
      observeSafely(() =>
        operation.endWithError(
          isAbortError(outcome.error)
            ? { errorType: "cancelled" }
            : { errorType: errorType(outcome.error), message: errorMessage(outcome.error) },
          at,
        ),
      );
    } else if (outcome === "abandoned" || run?.cancelled === true) {
      observeSafely(() => operation.cancel(at));
    } else {
      const finishReason =
        run !== undefined && run.generatedTokens >= this.#maxTokens ? "length" : "stop";
      observeSafely(() =>
        operation.recordOutputMessages([
          {
            role: "assistant",
            parts: [{ type: "text", content: this.#text }],
            finish_reason: finishReason,
          },
        ]),
      );
      observeSafely(() => operation.recordFinishReasons([finishReason]));
      observeSafely(() => operation.end(at));
    }
    return observeSafely(() => operation.observationRef());
  }
}
