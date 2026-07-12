import {
  ConcurrentRunError,
  DeviceLostError,
  DisposedError,
  InferenceError,
  InputValidationError,
  InvalidMetadataError,
  RuntimeConfigurationError,
  XybridError,
} from "./errors.ts";
import { type RuntimeInitializer, sharedInitializer } from "./internal/initialization.ts";
import { validateInputs } from "./internal/input.ts";
import { liteRtRuntime } from "./internal/litert-runtime.ts";
import {
  compileModel,
  loadMetadata,
  type MetadataLoader,
  validateAcceleratorPreference,
} from "./internal/loading.ts";
import type { BrowserRuntime, RuntimeModel, RuntimeTensor } from "./internal/runtime.ts";
import { resolveMetadataUrl, resolveWasmPath } from "./internal/url.ts";
import { type ParsedMetadata, resolveModelUrl, validateBrowserMetadata } from "./metadata.ts";
import type {
  LoadOptions,
  RunResult,
  SelectedAccelerator,
  TensorDetail,
  TensorInputs,
  TensorOutput,
} from "./types.ts";

const MAX_OUTPUT_BYTES = 256 * 1024 * 1024;
const MODEL_CONSTRUCTION_TOKEN = Symbol("XybridModel construction");

const normalizeLoadOptions = (options: unknown, base: string | undefined): LoadOptions => {
  if (typeof options !== "object" || options === null) {
    throw new RuntimeConfigurationError("load options must be an object.");
  }
  const values = options as Record<string, unknown>;
  validateAcceleratorPreference(values["accelerator"]);
  const wasmPath = values["wasmPath"];
  if (typeof wasmPath !== "string" && !(wasmPath instanceof URL)) {
    throw new RuntimeConfigurationError("wasmPath must be a string or URL.");
  }
  return {
    accelerator: values["accelerator"],
    wasmPath: resolveWasmPath(wasmPath, base),
  };
};

const validateOutputShape = (detail: TensorDetail): void => {
  const bytesPerElement = detail.dataType === "uint8" ? 1 : 4;
  let elements = 1;
  for (const dimension of detail.shape) {
    if (!Number.isSafeInteger(dimension) || dimension < 0) {
      throw new InferenceError(`LiteRT returned an invalid output shape for ${detail.name}.`);
    }
    if (dimension !== 0 && elements > Math.floor(MAX_OUTPUT_BYTES / bytesPerElement / dimension)) {
      throw new InferenceError(`Output ${detail.name} exceeds the 256 MiB browser limit.`);
    }
    elements *= dimension;
  }
};

const validateOutputValue = (value: TensorOutput["data"]): TensorOutput["data"] => {
  if (value.byteLength > MAX_OUTPUT_BYTES) {
    throw new InferenceError("Output exceeds the 256 MiB browser limit.");
  }
  return value;
};

const deleteTensors = (tensors: readonly RuntimeTensor[]): void => {
  for (const tensor of tensors) {
    tensor.delete();
  }
};

class ModelSession {
  readonly inputs: readonly TensorDetail[];
  readonly outputs: readonly TensorDetail[];
  readonly isFullyAccelerated: boolean;
  private running: Promise<RunResult> | undefined;
  private disposePromise: Promise<void> | undefined;
  private disposed = false;
  private deviceLost = false;
  private readonly unsubscribe: () => void;

  constructor(
    private readonly runtime: BrowserRuntime,
    private readonly model: RuntimeModel,
    readonly accelerator: SelectedAccelerator,
  ) {
    this.inputs = Object.freeze([...model.inputs]);
    this.outputs = Object.freeze([...model.outputs]);
    this.isFullyAccelerated = model.isFullyAccelerated;
    this.unsubscribe =
      accelerator === "webgpu"
        ? runtime.onDeviceLost(() => this.markDeviceLost())
        : () => undefined;
  }

  async run(input: TensorInputs): Promise<RunResult> {
    this.assertRunnable();
    if (this.running !== undefined) {
      throw new ConcurrentRunError();
    }
    this.running = this.execute(input);
    try {
      return await this.running;
    } finally {
      this.running = undefined;
    }
  }

  async dispose(): Promise<void> {
    if (this.disposePromise !== undefined) {
      return this.disposePromise;
    }
    this.disposePromise = this.finishDisposal();
    return this.disposePromise;
  }

  private async finishDisposal(): Promise<void> {
    if (this.running !== undefined) {
      await Promise.allSettled([this.running]);
    }
    if (!this.disposed) {
      this.unsubscribe();
      this.model.delete();
      this.disposed = true;
    }
  }

  private markDeviceLost(): void {
    if (!this.disposed) {
      this.deviceLost = true;
    }
  }

  private assertRunnable(): void {
    if (this.disposed || this.disposePromise !== undefined) {
      throw new DisposedError();
    }
    if (this.deviceLost) {
      throw new DeviceLostError();
    }
  }

  private async execute(input: TensorInputs): Promise<RunResult> {
    const validated = validateInputs(input, this.inputs);
    const inputTensors: RuntimeTensor[] = [];
    let outputTensors: readonly RuntimeTensor[] = [];
    try {
      for (const [index, value] of validated.entries()) {
        const detail = this.inputs[index];
        if (detail === undefined) {
          throw new InputValidationError("Model input metadata changed unexpectedly.");
        }
        inputTensors.push(this.runtime.createTensor(detail, value.data, value.shape));
      }
      outputTensors = await this.model.invoke(inputTensors);
      if (outputTensors.length !== this.outputs.length) {
        throw new InferenceError(
          `Runtime produced ${outputTensors.length} outputs; metadata declares ${this.outputs.length}.`,
        );
      }
      const outputs: TensorOutput[] = [];
      const namedOutputs: [string, TensorOutput][] = [];
      for (const [index, tensor] of outputTensors.entries()) {
        const detail = this.outputs[index];
        if (detail === undefined) {
          throw new InferenceError("Runtime produced more outputs than metadata declares.");
        }
        validateOutputShape(tensor.detail);
        const output = {
          name: detail.name,
          shape: [...tensor.detail.shape],
          data: validateOutputValue(await tensor.read()),
        };
        outputs.push(output);
        namedOutputs.push([output.name, output]);
      }
      return { outputs, byName: Object.fromEntries(namedOutputs) };
    } catch (error: unknown) {
      if (error instanceof XybridError) {
        throw error;
      }
      throw new InferenceError(error);
    } finally {
      deleteTensors(outputTensors);
      deleteTensors(inputTensors);
    }
  }
}

export class XybridModel {
  private constructor(
    private readonly session: ModelSession,
    token: typeof MODEL_CONSTRUCTION_TOKEN,
  ) {
    if (token !== MODEL_CONSTRUCTION_TOKEN) {
      throw new RuntimeConfigurationError("XybridModel instances must be created with load().");
    }
  }

  static async load(metadataUrl: string | URL, options: LoadOptions): Promise<XybridModel> {
    const base = typeof location === "undefined" ? undefined : location.href;
    const normalizedMetadataUrl = resolveMetadataUrl(metadataUrl, base);
    const normalizedOptions = normalizeLoadOptions(options, base);
    const session = await loadModel(
      normalizedMetadataUrl,
      normalizedOptions,
      liteRtRuntime,
      loadMetadata,
      sharedInitializer,
    );
    return new XybridModel(session, MODEL_CONSTRUCTION_TOKEN);
  }

  get inputs(): readonly TensorDetail[] {
    return this.session.inputs;
  }

  get outputs(): readonly TensorDetail[] {
    return this.session.outputs;
  }

  get accelerator(): SelectedAccelerator {
    return this.session.accelerator;
  }

  get isFullyAccelerated(): boolean {
    return this.session.isFullyAccelerated;
  }

  run(input: TensorInputs): Promise<RunResult> {
    return this.session.run(input);
  }

  dispose(): Promise<void> {
    return this.session.dispose();
  }
}

export const loadModel = async (
  metadataUrl: URL,
  options: LoadOptions,
  runtime: BrowserRuntime,
  getMetadata: MetadataLoader,
  initializer: RuntimeInitializer,
): Promise<ModelSession> => {
  let metadata: ParsedMetadata;
  try {
    metadata = await getMetadata(metadataUrl);
  } catch (error: unknown) {
    if (error instanceof XybridError) {
      throw error;
    }
    throw new InvalidMetadataError("Failed to load model metadata.", error);
  }
  const modelFile = validateBrowserMetadata(metadata);
  const modelUrl = resolveModelUrl(metadataUrl, modelFile, metadata.files);
  await initializer.initialize(runtime, {
    wasmPath: options.wasmPath,
    threads: false,
    jspi: false,
  });
  const compiled = await compileModel(runtime, modelUrl, options.accelerator);
  return new ModelSession(runtime, compiled.model, compiled.accelerator);
};
