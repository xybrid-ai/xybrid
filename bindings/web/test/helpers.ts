import { RuntimeInitializer } from "../src/internal/initialization.ts";
import type { loadModelBytes } from "../src/internal/model-download.ts";
import type { BrowserRuntime, RuntimeModel, RuntimeTensor } from "../src/internal/runtime.ts";
import { AcceleratorUnavailableError } from "../src/internal/runtime.ts";
import { type ParsedMetadata, parseMetadata } from "../src/metadata.ts";
import { loadModel } from "../src/model.ts";
import type { LoadOptions, TensorDetail, TensorValue } from "../src/types.ts";

type TestRuntimeTensor = RuntimeTensor & { isDeleted(): boolean; reads(): number };

const isParsedMetadata = (
  value: ParsedMetadata | Record<string, unknown>,
): value is ParsedMetadata => "modelId" in value && "template" in value && "files" in value;

export const loadWithDependencies = (
  metadataUrl: URL,
  options: LoadOptions,
  runtime: BrowserRuntime,
  metadata: () => Promise<ParsedMetadata | Record<string, unknown>>,
  runtimeInitializer: RuntimeInitializer = new RuntimeInitializer(),
  loadBytes: typeof loadModelBytes = async () => new Uint8Array([0]),
) =>
  loadModel(
    metadataUrl,
    options,
    runtime,
    async () => {
      const value = await metadata();
      return isParsedMetadata(value) ? value : parseMetadata(value);
    },
    runtimeInitializer,
    loadBytes,
  );

export const tfliteMetadata = (
  overrides: Record<string, unknown> = {},
): Record<string, unknown> => ({
  model_id: "add",
  version: "1",
  execution_template: { type: "TfLite", model_file: "model.tflite" },
  preprocessing: [],
  postprocessing: [],
  files: ["model.tflite"],
  ...overrides,
});

const firstInput: TensorDetail = { name: "a", shape: [1, 2], dataType: "float32" };
const secondInput: TensorDetail = { name: "b", shape: [1, -1], dataType: "float32" };
const output: TensorDetail = { name: "Identity", shape: [1, 2], dataType: "float32" };

export const details: {
  readonly inputs: readonly TensorDetail[];
  readonly outputs: readonly TensorDetail[];
} = {
  inputs: [firstInput, secondInput],
  outputs: [output],
};

export type RuntimeControl = {
  readonly runtime: BrowserRuntime;
  readonly tensors: TestRuntimeTensor[];
  readonly initialized: readonly string[];
  readonly compiled: readonly string[];
  readonly holdInitialization: () => void;
  readonly releaseInitialization: () => void;
  readonly deletedModels: () => number;
  readonly failWebGpu: () => void;
  readonly failWebGpuCompilation: () => void;
  readonly failWasmCompilation: () => void;
  readonly failRun: () => void;
  readonly failTensorCreationAt: (call: number) => void;
  readonly omitOutputs: () => void;
  readonly setOutputName: (name: string) => void;
  readonly setOutputShape: (shape: readonly number[]) => void;
  readonly holdRun: () => void;
  readonly releaseRun: () => void;
  readonly loseDevice: () => void;
};

export const createRuntime = (): RuntimeControl => {
  const tensors: TestRuntimeTensor[] = [];
  const initialized: string[] = [];
  const compiled: string[] = [];
  let shouldFailWebGpu = false;
  let shouldFailWebGpuCompilation = false;
  let shouldFailWasmCompilation = false;
  let shouldFailRun = false;
  let tensorCreationFailure: number | undefined;
  let tensorCreationCalls = 0;
  let shouldOmitOutputs = false;
  let outputName = output.name;
  let outputShape = output.shape;
  let runGate: Promise<void> | undefined;
  let releaseGate: (() => void) | undefined;
  let initializationGate: Promise<void> | undefined;
  let releaseInitializationGate: (() => void) | undefined;
  let deletedModels = 0;
  let onLost: (() => void) | undefined;

  const compile = async (accelerator: "wasm" | "webgpu"): Promise<RuntimeModel> => {
    compiled.push(accelerator);
    if (accelerator === "webgpu" && shouldFailWebGpu) {
      throw new AcceleratorUnavailableError("GPU unavailable");
    }
    if (accelerator === "webgpu" && shouldFailWebGpuCompilation) {
      throw new DOMException("model compilation failed", "OperationError");
    }
    if (accelerator === "wasm" && shouldFailWasmCompilation) {
      throw new DOMException("wasm compilation failed", "OperationError");
    }
    const currentOutput = { ...output, name: outputName, shape: outputShape };
    const model: RuntimeModel = {
      inputs: details.inputs,
      outputs: [currentOutput],
      isFullyAccelerated: accelerator === "webgpu",
      invoke: async (inputs) => {
        await runGate;
        if (shouldFailRun) {
          throw new DOMException("backend failure", "OperationError");
        }
        if (shouldOmitOutputs) {
          return [];
        }
        const first = inputs[0];
        if (first === undefined) {
          throw new DOMException("missing input", "OperationError");
        }
        return [runtime.createTensor(currentOutput, await first.read(), currentOutput.shape)];
      },
      delete: () => {
        deletedModels += 1;
      },
    };
    return model;
  };

  const runtime: BrowserRuntime = {
    initialize: async (config) => {
      initialized.push(config.wasmPath.toString());
      await initializationGate;
    },
    probeAccelerator: async (accelerator) => {
      if (accelerator === "webgpu" && shouldFailWebGpu) {
        throw new AcceleratorUnavailableError("GPU unavailable");
      }
    },
    compileBytes: async (_bytes, accelerator) => compile(accelerator),
    createTensor: (detail, data, shape) => {
      tensorCreationCalls += 1;
      if (tensorCreationCalls === tensorCreationFailure) {
        throw new DOMException("tensor allocation failed", "OperationError");
      }
      let deleted = false;
      let reads = 0;
      const tensor: TestRuntimeTensor = {
        detail,
        read: async () => {
          reads += 1;
          return data;
        },
        delete: () => {
          deleted = true;
        },
        isDeleted: () => deleted,
        reads: () => reads,
      };
      if (shape.length === 0) {
        throw new DOMException("missing shape", "OperationError");
      }
      tensors.push(tensor);
      return tensor;
    },
    onDeviceLost: (callback) => {
      onLost = callback;
      return () => {
        onLost = undefined;
      };
    },
  };

  return {
    runtime,
    tensors,
    initialized,
    compiled,
    holdInitialization: () => {
      initializationGate = new Promise((resolve) => {
        releaseInitializationGate = resolve;
      });
    },
    releaseInitialization: () => {
      releaseInitializationGate?.();
      initializationGate = undefined;
      releaseInitializationGate = undefined;
    },
    deletedModels: () => deletedModels,
    failWebGpu: () => {
      shouldFailWebGpu = true;
    },
    failWebGpuCompilation: () => {
      shouldFailWebGpuCompilation = true;
    },
    failWasmCompilation: () => {
      shouldFailWasmCompilation = true;
    },
    failRun: () => {
      shouldFailRun = true;
    },
    failTensorCreationAt: (call) => {
      tensorCreationFailure = call;
    },
    omitOutputs: () => {
      shouldOmitOutputs = true;
    },
    setOutputName: (name) => {
      outputName = name;
    },
    setOutputShape: (shape) => {
      outputShape = shape;
    },
    holdRun: () => {
      runGate = new Promise((resolve) => {
        releaseGate = resolve;
      });
    },
    releaseRun: () => {
      releaseGate?.();
      runGate = undefined;
      releaseGate = undefined;
    },
    loseDevice: () => {
      onLost?.();
    },
  };
};

export const tensor = (values: readonly number[]): TensorValue => new Float32Array(values);
