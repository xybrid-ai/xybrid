import type { DownloadProgress, SelectedAccelerator } from "../types.ts";
import type { GenerateObservationContext, LoadMetrics, Memory, RunMetrics } from "./protocol.ts";

export class AcceleratorUnavailableError extends Error {}

export type RuntimeInitConfig = {
  readonly wasmPath: string | URL;
  readonly threads: false;
  readonly jspi: false;
};

export type LlmGeneration = {
  readonly stream: AsyncGenerator<string, void, undefined>;
  cancel(): void;
  dispose(): Promise<void>;
};

/** Timing hooks of an observed generation; times are epoch milliseconds. */
export type GenerationTelemetry = {
  /** Propagated to the engine (the worker) with the request. */
  readonly context: GenerateObservationContext;
  /** One decoded token; `text` is empty while a multi-byte character is still incomplete. */
  onToken(atMs: number, text: string): void;
  onComplete(atMs: number): void;
};

export type EngineGenerateOptions = {
  readonly maxOutputTokens?: number;
  readonly telemetry?: GenerationTelemetry;
};

export type LlmEngine = {
  readonly loaded?: LoadMetrics | undefined;
  readonly lastRun?: RunMetrics | undefined;
  readonly releasedMemory?: Memory | undefined;
  /** Whether `generate` calls the telemetry hooks with engine-measured times. */
  readonly reportsTiming?: boolean;
  generate(prompt: string, options: EngineGenerateOptions): Promise<LlmGeneration>;
  delete(): Promise<void>;
};

export type LlmRuntime<Model> = {
  initialize(config: RuntimeInitConfig): Promise<void>;
  probeAccelerator(accelerator: SelectedAccelerator): Promise<void>;
  fetchModel(
    modelUrl: URL,
    onProgress: ((progress: DownloadProgress) => void) | undefined,
    signal?: AbortSignal,
  ): Promise<Model>;
  modelFromChunks(chunks: readonly Uint8Array[], downloadMs?: number): Promise<Model>;
  createEngine(
    model: Model,
    accelerator: SelectedAccelerator,
    contextLength: number | undefined,
    signal?: AbortSignal,
  ): Promise<LlmEngine>;
};
