import type { SelectedAccelerator, TensorDetail, TensorValue } from "../types.ts";

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
  compile(modelUrl: URL, accelerator: SelectedAccelerator): Promise<RuntimeModel>;
  createTensor(detail: TensorDetail, data: TensorValue, shape: readonly number[]): RuntimeTensor;
  onDeviceLost(callback: () => void): () => void;
};
