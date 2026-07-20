export type XybridErrorCode =
  | "concurrent_run"
  | "device_lost"
  | "disposed"
  | "inference"
  | "input_validation"
  | "invalid_metadata"
  | "huggingface"
  | "integrity"
  | "registry"
  | "runtime_configuration"
  | "runtime_initialization"
  | "unsupported_feature"
  | "unsupported_template"
  | "unsupported_tensor_type";

export class XybridError extends Error {
  readonly causeValue: unknown;

  protected constructor(
    readonly code: XybridErrorCode,
    message: string,
    causeValue: unknown = undefined,
  ) {
    super(message);
    this.name = "XybridError";
    this.causeValue = causeValue;
  }
}

export class InvalidMetadataError extends XybridError {
  constructor(message: string, causeValue: unknown = undefined) {
    super("invalid_metadata", message, causeValue);
    this.name = "InvalidMetadataError";
  }
}

export class RegistryError extends XybridError {
  constructor(message: string, causeValue: unknown = undefined) {
    super("registry", message, causeValue);
    this.name = "RegistryError";
  }
}

export class HuggingFaceError extends XybridError {
  constructor(message: string, causeValue: unknown = undefined) {
    super("huggingface", message, causeValue);
    this.name = "HuggingFaceError";
  }
}

export class IntegrityError extends XybridError {
  constructor(message: string, causeValue: unknown = undefined) {
    super("integrity", message, causeValue);
    this.name = "IntegrityError";
  }
}

export class UnsupportedTemplateError extends XybridError {
  constructor(template: string, supported = "TfLite") {
    super(
      "unsupported_template",
      `Browser preview supports ${supported} metadata, received ${template}.`,
    );
    this.name = "UnsupportedTemplateError";
  }
}

export class UnsupportedFeatureError extends XybridError {
  constructor(feature: string) {
    super("unsupported_feature", `Browser preview does not support ${feature}.`);
    this.name = "UnsupportedFeatureError";
  }
}

export class RuntimeConfigurationError extends XybridError {
  constructor(message = "LiteRT is already initialized with a different configuration.") {
    super("runtime_configuration", message);
    this.name = "RuntimeConfigurationError";
  }
}

export class RuntimeInitializationError extends XybridError {
  constructor(causeValue: unknown) {
    super("runtime_initialization", "LiteRT initialization or compilation failed.", causeValue);
    this.name = "RuntimeInitializationError";
  }
}

export class InputValidationError extends XybridError {
  constructor(message: string) {
    super("input_validation", message);
    this.name = "InputValidationError";
  }
}

export class UnsupportedTensorTypeError extends XybridError {
  constructor(dataType: string) {
    super("unsupported_tensor_type", `Browser preview does not support tensor dtype ${dataType}.`);
    this.name = "UnsupportedTensorTypeError";
  }
}

export class InferenceError extends XybridError {
  constructor(causeValue: unknown) {
    super("inference", "LiteRT inference failed.", causeValue);
    this.name = "InferenceError";
  }
}

export class ConcurrentRunError extends XybridError {
  constructor() {
    super("concurrent_run", "This model already has an in-flight run.");
    this.name = "ConcurrentRunError";
  }
}

export class DeviceLostError extends XybridError {
  constructor() {
    super("device_lost", "The WebGPU device was lost. Load a new model to continue.");
    this.name = "DeviceLostError";
  }
}

export class DisposedError extends XybridError {
  constructor() {
    super("disposed", "This model has been disposed or disposal was requested.");
    this.name = "DisposedError";
  }
}
