import { InputValidationError } from "../errors.ts";
import type {
  TensorDataType,
  TensorDetail,
  TensorInput,
  TensorInputs,
  TensorValue,
} from "../types.ts";

export type ValidatedInput = { readonly data: TensorValue; readonly shape: readonly number[] };

const MAX_TENSOR_BYTES = 256 * 1024 * 1024;

const isPositional = (input: TensorInputs): input is readonly TensorInput[] => Array.isArray(input);

const isTensorValue = (value: unknown): value is TensorValue =>
  value instanceof Float32Array || value instanceof Int32Array || value instanceof Uint8Array;

const isNumberArray = (value: unknown): value is number[] =>
  Array.isArray(value) && value.every((dimension) => typeof dimension === "number");

type ExplicitInput = { readonly data: TensorValue; readonly shape: readonly number[] };

const isExplicitInput = (value: unknown): value is ExplicitInput =>
  typeof value === "object" &&
  value !== null &&
  "data" in value &&
  isTensorValue(value.data) &&
  "shape" in value &&
  isNumberArray(value.shape);

const dataType = (value: TensorValue): TensorDataType => {
  if (value instanceof Float32Array) {
    return "float32";
  }
  if (value instanceof Int32Array) {
    return "int32";
  }
  return "uint8";
};

const checkedProduct = (shape: readonly number[]): number => {
  let product = 1;
  for (const dimension of shape) {
    if (!Number.isSafeInteger(dimension) || dimension < 0) {
      throw new InputValidationError("Tensor shapes must use non-negative integer dimensions.");
    }
    product *= dimension;
  }
  return product;
};

const inferredShape = (detail: TensorDetail, data: TensorValue): readonly number[] => {
  const dynamic = detail.shape.filter((dimension) => dimension < 0);
  if (dynamic.length === 0) {
    return detail.shape;
  }
  if (dynamic.length !== 1) {
    throw new InputValidationError(
      `Input ${detail.name} needs an explicit shape for multiple dynamic dimensions.`,
    );
  }
  const fixedSize = checkedProduct(detail.shape.map((dimension) => Math.max(dimension, 1)));
  if (fixedSize === 0 || data.length % fixedSize !== 0) {
    throw new InputValidationError(
      `Input ${detail.name} data length cannot satisfy its dynamic shape.`,
    );
  }
  const dynamicSize = data.length / fixedSize;
  return detail.shape.map((dimension) => (dimension < 0 ? dynamicSize : dimension));
};

const validateInput = (detail: TensorDetail, value: unknown): ValidatedInput => {
  if (!isTensorValue(value) && !isExplicitInput(value)) {
    throw new InputValidationError(`Input ${detail.name} must be a supported typed tensor.`);
  }
  const data = isTensorValue(value) ? value : value.data;
  if (data.byteLength > MAX_TENSOR_BYTES) {
    throw new InputValidationError(`Input ${detail.name} exceeds the 256 MiB browser limit.`);
  }
  const shape = isTensorValue(value) ? inferredShape(detail, data) : value.shape;
  if (dataType(data) !== detail.dataType) {
    throw new InputValidationError(`Input ${detail.name} must use ${detail.dataType}.`);
  }
  if (shape.length !== detail.shape.length) {
    throw new InputValidationError(
      `Input ${detail.name} has rank ${shape.length}; expected ${detail.shape.length}.`,
    );
  }
  for (const [index, dimension] of shape.entries()) {
    const expected = detail.shape[index];
    if (expected === undefined || (expected >= 0 && dimension !== expected)) {
      throw new InputValidationError(`Input ${detail.name} shape does not match the model.`);
    }
  }
  if (checkedProduct(shape) !== data.length) {
    throw new InputValidationError(`Input ${detail.name} data length does not match its shape.`);
  }
  return { data, shape };
};

export const validateInputs = (
  input: TensorInputs,
  details: readonly TensorDetail[],
): readonly ValidatedInput[] => {
  if (typeof input !== "object" || input === null) {
    throw new InputValidationError("Tensor inputs must be an array or a name-keyed record.");
  }
  if (isPositional(input)) {
    if (input.length !== details.length) {
      throw new InputValidationError(
        `Expected ${details.length} positional inputs; received ${input.length}.`,
      );
    }
    return details.map((detail, index) => {
      const value = input[index];
      if (value === undefined) {
        throw new InputValidationError(`Missing positional input ${detail.name}.`);
      }
      return validateInput(detail, value);
    });
  }
  const keys = Object.keys(input);
  if (keys.length !== details.length) {
    throw new InputValidationError(
      `Expected ${details.length} named inputs; received ${keys.length}.`,
    );
  }
  return details.map((detail) => {
    const value = input[detail.name];
    if (value === undefined) {
      throw new InputValidationError(`Missing named input ${detail.name}.`);
    }
    return validateInput(detail, value);
  });
};
