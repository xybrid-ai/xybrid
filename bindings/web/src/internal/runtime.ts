import type { DownloadProgress, SelectedAccelerator } from "../types.ts";
import type { LoadMetrics, Memory, RunMetrics } from "./protocol.ts";

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

export type LlmEngine = {
  readonly loaded?: LoadMetrics | undefined;
  readonly lastRun?: RunMetrics | undefined;
  readonly releasedMemory?: Memory | undefined;
  generate(prompt: string, options: { readonly maxOutputTokens?: number }): Promise<LlmGeneration>;
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
