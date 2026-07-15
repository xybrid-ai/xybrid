import { z } from "zod";

import {
  InvalidMetadataError,
  UnsupportedFeatureError,
  UnsupportedTemplateError,
} from "./errors.ts";

// Engine context capacity drives KV-cache allocation, so a remote metadata
// document must not be able to request multi-gigabyte caches in a tab.
const MAX_CONTEXT_LENGTH = 32_768;

const metadataSchema = z
  .object({
    model_id: z.string(),
    version: z.string(),
    execution_template: z
      .object({
        type: z.string(),
        model_file: z.string().optional(),
        context_length: z.unknown().optional(),
      })
      .loose(),
    files: z.array(z.string()),
    preprocessing: z.array(z.unknown()).default([]),
    postprocessing: z.array(z.unknown()).default([]),
    voices: z.unknown().optional(),
    vision_encoder: z.unknown().optional(),
  })
  .loose();

export type ParsedMetadata = {
  readonly modelId: string;
  readonly version: string;
  readonly template: {
    readonly type: string;
    readonly modelFile: string | undefined;
    readonly contextLength: unknown;
  };
  readonly files: readonly string[];
  readonly preprocessing: readonly unknown[];
  readonly postprocessing: readonly unknown[];
  readonly voices: unknown;
  readonly visionEncoder: unknown;
};

export const parseMetadata = (input: unknown): ParsedMetadata => {
  const parsed = metadataSchema.safeParse(input);
  if (!parsed.success) {
    throw new InvalidMetadataError(
      "Model metadata does not match the Xybrid metadata contract.",
      parsed.error,
    );
  }
  return {
    modelId: parsed.data.model_id,
    version: parsed.data.version,
    template: {
      type: parsed.data.execution_template.type,
      modelFile: parsed.data.execution_template.model_file,
      contextLength: parsed.data.execution_template.context_length,
    },
    files: parsed.data.files,
    preprocessing: parsed.data.preprocessing,
    postprocessing: parsed.data.postprocessing,
    voices: parsed.data.voices,
    visionEncoder: parsed.data.vision_encoder,
  };
};

const assertBrowserFeatureSubset = (metadata: ParsedMetadata, template: string): string => {
  if (metadata.preprocessing.length > 0) {
    throw new UnsupportedFeatureError("metadata preprocessing");
  }
  if (metadata.postprocessing.length > 0) {
    throw new UnsupportedFeatureError("metadata postprocessing");
  }
  if (metadata.voices !== undefined && metadata.voices !== null) {
    throw new UnsupportedFeatureError("metadata voices");
  }
  if (metadata.visionEncoder !== undefined && metadata.visionEncoder !== null) {
    throw new UnsupportedFeatureError("metadata vision_encoder");
  }
  if (metadata.template.modelFile === undefined) {
    throw new InvalidMetadataError(`${template} metadata requires execution_template.model_file.`);
  }
  return metadata.template.modelFile;
};

export const validateBrowserMetadata = (metadata: ParsedMetadata): string => {
  if (metadata.template.type !== "TfLite") {
    throw new UnsupportedTemplateError(metadata.template.type, "TfLite");
  }
  return assertBrowserFeatureSubset(metadata, "TfLite");
};

export type LlmBrowserMetadata = {
  readonly modelFile: string;
  readonly contextLength: number | undefined;
};

export const validateLlmBrowserMetadata = (metadata: ParsedMetadata): LlmBrowserMetadata => {
  if (metadata.template.type !== "LiteRtLm") {
    throw new UnsupportedTemplateError(metadata.template.type, "LiteRtLm");
  }
  const modelFile = assertBrowserFeatureSubset(metadata, "LiteRtLm");
  const rawContextLength = metadata.template.contextLength;
  if (rawContextLength === undefined) {
    return { modelFile, contextLength: undefined };
  }
  if (
    typeof rawContextLength !== "number" ||
    !Number.isSafeInteger(rawContextLength) ||
    rawContextLength <= 0 ||
    rawContextLength > MAX_CONTEXT_LENGTH
  ) {
    throw new InvalidMetadataError(
      `LiteRtLm metadata context_length must be a positive safe integer no greater than ${MAX_CONTEXT_LENGTH}.`,
    );
  }
  return {
    modelFile,
    contextLength: rawContextLength,
  };
};

export const resolveModelUrl = (
  metadataUrl: URL,
  modelFile: string,
  files: readonly string[],
): URL => {
  if (
    modelFile.length === 0 ||
    modelFile === "." ||
    modelFile === ".." ||
    modelFile.includes("/") ||
    modelFile.includes("\\") ||
    modelFile.includes(":") ||
    modelFile.includes("%") ||
    modelFile.includes("?") ||
    modelFile.includes("#") ||
    modelFile.trim() !== modelFile ||
    !/^[A-Za-z0-9._-]+$/.test(modelFile)
  ) {
    throw new InvalidMetadataError("model_file must be a bare filename in the metadata directory.");
  }
  if (!files.includes(modelFile)) {
    throw new InvalidMetadataError("model_file must be present in metadata files.");
  }
  try {
    return new URL(modelFile, metadataUrl);
  } catch (error: unknown) {
    throw new InvalidMetadataError("model_file could not be resolved beside the metadata.", error);
  }
};
