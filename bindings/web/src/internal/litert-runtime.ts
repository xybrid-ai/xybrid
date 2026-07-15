import {
  type CompiledModel,
  type DType,
  getWebGpuDevice,
  loadAndCompile,
  loadLiteRt,
  Tensor,
  type TensorDetails,
} from "@litertjs/core";

import {
  InferenceError,
  RuntimeInitializationError,
  UnsupportedTensorTypeError,
} from "../errors.ts";
import type { SelectedAccelerator, TensorDataType, TensorDetail, TensorValue } from "../types.ts";
import { loadModelBytes } from "./model-download.ts";
import {
  AcceleratorUnavailableError,
  type BrowserRuntime,
  type RuntimeInitConfig,
  type RuntimeModel,
  type RuntimeTensor,
} from "./runtime.ts";

type OwnedTensorValue =
  | Float32Array<ArrayBuffer>
  | Int32Array<ArrayBuffer>
  | Uint8Array<ArrayBuffer>;

const ownedTensorValue = (value: TensorValue): OwnedTensorValue => {
  if (value.buffer instanceof ArrayBuffer) {
    return value as OwnedTensorValue;
  }
  if (value instanceof Float32Array) {
    return new Float32Array(value);
  }
  if (value instanceof Int32Array) {
    return new Int32Array(value);
  }
  return new Uint8Array(value);
};

const dataType = (dtype: DType): TensorDataType => {
  switch (dtype) {
    case "float32":
    case "int32":
    case "uint8":
      return dtype;
    default:
      throw new UnsupportedTensorTypeError(dtype);
  }
};

const detail = (value: TensorDetails): TensorDetail => ({
  name: value.name,
  shape: Array.from(value.shape),
  dataType: dataType(value.dtype),
});

const wrapTensor = (tensor: Tensor, expectedDetail: TensorDetail): RuntimeTensor => ({
  detail: {
    name: expectedDetail.name,
    shape: Array.from(tensor.type.layout.dimensions),
    dataType: dataType(tensor.type.dtype),
  },
  read: () => tensor.data(),
  delete: () => tensor.delete(),
});

const wrapModel = (model: CompiledModel): RuntimeModel => {
  try {
    const inputs = model.getInputDetails().map(detail);
    const outputs = model.getOutputDetails().map(detail);
    return {
      inputs,
      outputs,
      isFullyAccelerated: model.isFullyAccelerated,
      invoke: async (inputsToRun) => {
        const tensors = inputsToRun.map((input) => {
          if (!(input instanceof CoreTensor)) {
            throw new RuntimeInitializationError("Unexpected runtime tensor.");
          }
          return input.tensor;
        });
        const outputTensors = await model.run(tensors);
        if (outputTensors.length !== outputs.length) {
          for (const output of outputTensors) {
            output.delete();
          }
          throw new InferenceError(
            `LiteRT returned ${outputTensors.length} outputs; metadata declares ${outputs.length}.`,
          );
        }
        try {
          return outputTensors.map((output, index) => {
            const outputDetail = outputs[index];
            if (outputDetail === undefined) {
              throw new InferenceError("LiteRT returned an unexpected output tensor.");
            }
            return wrapTensor(output, outputDetail);
          });
        } catch (error: unknown) {
          for (const output of outputTensors) {
            if (!output.deleted) {
              output.delete();
            }
          }
          throw error;
        }
      },
      delete: () => model.delete(),
    };
  } catch (error: unknown) {
    model.delete();
    throw error;
  }
};

class CoreTensor implements RuntimeTensor {
  constructor(
    readonly tensor: Tensor,
    readonly detail: TensorDetail,
  ) {}

  read(): Promise<TensorValue> {
    return this.tensor.data();
  }

  delete(): void {
    this.tensor.delete();
  }
}

export const liteRtRuntime: BrowserRuntime = {
  initialize: async (config: RuntimeInitConfig) => {
    await loadLiteRt(config.wasmPath.toString(), { threads: false, jspi: false });
  },
  compile: async (modelUrl: URL, accelerator: SelectedAccelerator) => {
    return liteRtRuntime.compileBytes(await loadModelBytes(modelUrl), accelerator);
  },
  compileBytes: async (bytes: Uint8Array, accelerator: SelectedAccelerator) => {
    if (accelerator === "webgpu" && getWebGpuDevice() === null) {
      throw new AcceleratorUnavailableError("WebGPU is unavailable.");
    }
    return wrapModel(await loadAndCompile(bytes, { accelerator }));
  },
  createTensor: (tensorDetail, value, shape) =>
    new CoreTensor(Tensor.fromTypedArray(ownedTensorValue(value), Array.from(shape)), tensorDetail),
  onDeviceLost: (callback) => {
    const device = getWebGpuDevice();
    if (device === null) {
      return () => undefined;
    }
    let handler: (() => void) | undefined = callback;
    void device.lost.then(() => handler?.());
    return () => {
      handler = undefined;
    };
  },
};
