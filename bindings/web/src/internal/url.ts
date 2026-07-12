import { InvalidMetadataError, RuntimeConfigurationError } from "../errors.ts";

export const resolveMetadataUrl = (value: string | URL, base: string | undefined): URL => {
  try {
    const url =
      value instanceof URL ? value : base === undefined ? new URL(value) : new URL(value, base);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      throw new InvalidMetadataError("metadataUrl must use HTTP(S).");
    }
    return url;
  } catch (error: unknown) {
    if (error instanceof InvalidMetadataError) {
      throw error;
    }
    throw new InvalidMetadataError(
      "metadataUrl must be a valid absolute or browser-relative URL.",
      error,
    );
  }
};

export const resolveWasmPath = (value: string | URL, base: string | undefined): URL => {
  if (base === undefined) {
    throw new RuntimeConfigurationError("wasmPath requires a browser location.");
  }
  try {
    const pageUrl = new URL(base);
    const wasmUrl = new URL(value, pageUrl);
    if (
      (wasmUrl.protocol !== "http:" && wasmUrl.protocol !== "https:") ||
      wasmUrl.origin !== pageUrl.origin
    ) {
      throw new RuntimeConfigurationError("wasmPath must use the page's HTTP(S) origin.");
    }
    return wasmUrl;
  } catch (error: unknown) {
    if (error instanceof RuntimeConfigurationError) {
      throw error;
    }
    throw new RuntimeConfigurationError("wasmPath is not a valid browser URL.");
  }
};
