import { RuntimeConfigurationError, RuntimeInitializationError } from "../errors.ts";

import type { RuntimeInitConfig } from "./runtime.ts";

const equalConfig = (left: RuntimeInitConfig, right: RuntimeInitConfig): boolean =>
  left.wasmPath.toString() === right.wasmPath.toString() &&
  left.threads === right.threads &&
  left.jspi === right.jspi;

type InitializableRuntime = {
  initialize(config: RuntimeInitConfig): Promise<void>;
};

export class RuntimeInitializer {
  private config: RuntimeInitConfig | undefined;
  private promise: Promise<void> | undefined;

  initialize(runtime: InitializableRuntime, config: RuntimeInitConfig): Promise<void> {
    if (this.config !== undefined && !equalConfig(this.config, config)) {
      return Promise.reject(new RuntimeConfigurationError());
    }
    if (this.promise !== undefined) {
      return this.promise;
    }
    this.config = config;
    this.promise = runtime.initialize(config).catch((error: unknown) => {
      this.config = undefined;
      this.promise = undefined;
      throw new RuntimeInitializationError(error);
    });
    return this.promise;
  }
}

export const sharedInitializer = new RuntimeInitializer();

// LiteRT-LM has its own per-page wasm module, separate from LiteRT's.
export const sharedLlmInitializer = new RuntimeInitializer();
