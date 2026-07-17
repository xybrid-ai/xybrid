import type { DownloadProgress, SelectedAccelerator, TensorDetail, TensorValue } from "../types.ts";

export class AcceleratorUnavailableError extends Error {}

export type RuntimeInitConfig = {
  readonly wasmPath: string | URL;
  readonly threads: false;
  readonly jspi: false;
};

export type RuntimeTensor = {
  readonly detail: TensorDetail;
  read(): Promise<TensorValue>;
  delete(): void;
};

export type RuntimeModel = {
  readonly inputs: readonly TensorDetail[];
  readonly outputs: readonly TensorDetail[];
  readonly isFullyAccelerated: boolean;
  invoke(inputs: readonly RuntimeTensor[]): Promise<readonly RuntimeTensor[]>;
  delete(): void;
};

export type BrowserRuntime = {
  initialize(config: RuntimeInitConfig): Promise<void>;
  probeAccelerator(accelerator: SelectedAccelerator): Promise<void>;
  compileBytes(bytes: Uint8Array, accelerator: SelectedAccelerator): Promise<RuntimeModel>;
  createTensor(detail: TensorDetail, data: TensorValue, shape: readonly number[]): RuntimeTensor;
  onDeviceLost(callback: () => void): () => void;
};

export type LlmGeneration = {
  readonly stream: AsyncGenerator<string, void, undefined>;
  cancel(): void;
  dispose(): Promise<void>;
};

export type LlmEngine = {
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
  modelFromChunks(chunks: readonly Uint8Array[]): Promise<Model>;
  createEngine(
    model: Model,
    accelerator: SelectedAccelerator,
    contextLength: number | undefined,
  ): Promise<LlmEngine>;
};
