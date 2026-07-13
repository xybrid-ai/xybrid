import ky, { isHTTPError } from "ky";
import { z } from "zod";

import { HuggingFaceError } from "../errors.ts";
import {
  type ParsedMetadata,
  parseMetadata,
  validateBrowserMetadata,
  validateLlmBrowserMetadata,
} from "../metadata.ts";
import { loadMetadata } from "./loading.ts";
import type { ModelResolution } from "./registry.ts";

const HUGGINGFACE_API_URL = "https://huggingface.co";
const HUGGINGFACE_TIMEOUT_MS = 30_000;
const MAX_MODEL_BYTES = 512 * 1024 * 1024;
const REVISION_DEFAULT = "main";

const treeEntrySchema = z
  .object({
    type: z.enum(["file", "directory"]),
    oid: z.string().optional(),
    size: z.number().optional(),
    path: z.string(),
    lfs: z
      .object({
        oid: z.string(),
        size: z.number(),
        pointerSize: z.number(),
      })
      .loose()
      .optional(),
    xetHash: z.string().optional(),
  })
  .loose();

const treeSchema = z.array(treeEntrySchema);

type HuggingFaceTreeEntry = z.infer<typeof treeEntrySchema>;

export type HuggingFaceResolveOptions = {
  readonly revision?: string | undefined;
  readonly file?: string | undefined;
  readonly signal?: AbortSignal | undefined;
};

type HuggingFaceRepo = {
  readonly organization: string;
  readonly name: string;
};

const pathSegment = (value: string): string => encodeURIComponent(value);

export const validateHfRepo = (repo: unknown): HuggingFaceRepo => {
  if (typeof repo !== "string") {
    throw new HuggingFaceError("HuggingFace repo must be in the form org/name.");
  }
  const segments = repo.split("/");
  if (
    segments.length !== 2 ||
    segments.some(
      (segment) =>
        segment === "." || segment === ".." || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(segment),
    )
  ) {
    throw new HuggingFaceError(
      `Invalid HuggingFace repo ${repo}; expected a single org/name pair with safe segments.`,
    );
  }
  const organization = segments[0];
  const name = segments[1];
  if (organization === undefined || name === undefined) {
    throw new HuggingFaceError("HuggingFace repo must be in the form org/name.");
  }
  return { organization, name };
};

export const validateRevision = (revision: unknown = REVISION_DEFAULT): string => {
  if (
    typeof revision !== "string" ||
    revision.length === 0 ||
    revision === "." ||
    revision === ".." ||
    /[\\/?#%\s]/.test(revision)
  ) {
    throw new HuggingFaceError(
      "HuggingFace revision must be a non-empty, single path-safe segment.",
    );
  }
  return revision;
};

const treeUrl = (repo: HuggingFaceRepo, revision: string): URL =>
  new URL(
    `${HUGGINGFACE_API_URL}/api/models/${pathSegment(repo.organization)}/${pathSegment(repo.name)}/tree/${pathSegment(revision)}`,
  );

const resolveUrl = (repo: HuggingFaceRepo, revision: string, file: string): URL =>
  new URL(
    `${HUGGINGFACE_API_URL}/${pathSegment(repo.organization)}/${pathSegment(repo.name)}/resolve/${pathSegment(revision)}/${pathSegment(file)}`,
  );

const fetchTree = async (
  repo: string,
  validatedRepo: HuggingFaceRepo,
  revision: string,
  signal: AbortSignal | undefined,
): Promise<readonly HuggingFaceTreeEntry[]> => {
  let response: Response;
  try {
    response = await ky.get(treeUrl(validatedRepo, revision), {
      retry: 0,
      timeout: HUGGINGFACE_TIMEOUT_MS,
      ...(signal === undefined ? {} : { signal }),
    });
  } catch (error: unknown) {
    if (isHTTPError(error)) {
      if (error.response.status === 404) {
        throw new HuggingFaceError(
          `HuggingFace repo ${repo} was not found at revision ${revision}.`,
          error,
        );
      }
      throw new HuggingFaceError(
        `HuggingFace tree request failed with HTTP ${error.response.status} for repo ${repo} at revision ${revision}.`,
        error,
      );
    }
    throw error;
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch (error: unknown) {
    throw new HuggingFaceError("HuggingFace tree response was not valid JSON.", error);
  }
  const parsed = treeSchema.safeParse(body);
  if (!parsed.success) {
    throw new HuggingFaceError(
      "HuggingFace tree response did not contain valid file entries.",
      parsed.error,
    );
  }
  return parsed.data;
};

const topLevelFiles = (entries: readonly HuggingFaceTreeEntry[]): readonly HuggingFaceTreeEntry[] =>
  entries.filter((entry) => entry.type === "file" && !entry.path.includes("/"));

const selectModelFile = (
  repo: string,
  files: readonly HuggingFaceTreeEntry[],
  expectedExtension: ".litertlm" | ".tflite",
  requestedFile: string | undefined,
): string => {
  const candidates = files.filter((entry) => entry.path.endsWith(expectedExtension));
  if (requestedFile !== undefined) {
    if (!requestedFile.endsWith(expectedExtension)) {
      throw new HuggingFaceError(
        `HuggingFace options.file ${requestedFile} must end with ${expectedExtension}.`,
      );
    }
    if (!candidates.some((entry) => entry.path === requestedFile)) {
      throw new HuggingFaceError(
        `HuggingFace options.file ${requestedFile} was not found in repo ${repo}.`,
      );
    }
    return requestedFile;
  }
  if (candidates.length === 0) {
    throw new HuggingFaceError(
      `HuggingFace repo ${repo} has no ${expectedExtension} file in repo.`,
    );
  }
  if (candidates.length > 1) {
    throw new HuggingFaceError(
      `HuggingFace repo ${repo} contains multiple ${expectedExtension} files: ${candidates
        .map((entry) => entry.path)
        .join(", ")}. Pass options.file to select one.`,
    );
  }
  const selected = candidates[0];
  if (selected === undefined) {
    throw new HuggingFaceError(
      `HuggingFace repo ${repo} has no ${expectedExtension} file in repo.`,
    );
  }
  return selected.path;
};

const sanitizedModelId = (name: string): string => name.toLowerCase().replace(/[^a-z0-9.-]/g, "-");

const synthesizeMetadata = (
  repoName: string,
  revision: string,
  modelFile: string,
  expectedFormat: "litertlm" | "tflite",
): ParsedMetadata => {
  const metadata = parseMetadata({
    model_id: sanitizedModelId(repoName),
    version: revision,
    execution_template: {
      type: expectedFormat === "litertlm" ? "LiteRtLm" : "TfLite",
      model_file: modelFile,
    },
    files: [modelFile],
    preprocessing: [],
    postprocessing: [],
  });
  if (expectedFormat === "litertlm") {
    validateLlmBrowserMetadata(metadata);
  } else {
    validateBrowserMetadata(metadata);
  }
  return metadata;
};

const validateSurfaceMetadata = (
  metadata: ParsedMetadata,
  expectedFormat: "litertlm" | "tflite",
): string =>
  expectedFormat === "litertlm"
    ? validateLlmBrowserMetadata(metadata).modelFile
    : validateBrowserMetadata(metadata);

const validateModelSize = (
  repo: string,
  modelFile: string,
  entry: HuggingFaceTreeEntry,
): number => {
  const sizeBytes = entry.lfs?.size ?? entry.size;
  if (
    typeof sizeBytes !== "number" ||
    !Number.isSafeInteger(sizeBytes) ||
    sizeBytes <= 0 ||
    sizeBytes > MAX_MODEL_BYTES
  ) {
    throw new HuggingFaceError(
      `HuggingFace model ${modelFile} in repo ${repo} must be a positive safe integer no greater than ${MAX_MODEL_BYTES} bytes.`,
    );
  }
  return sizeBytes;
};

export const resolveHuggingFaceModel = async (
  repo: unknown,
  expectedFormat: "litertlm" | "tflite",
  options: HuggingFaceResolveOptions = {},
): Promise<ModelResolution> => {
  const validatedRepo = validateHfRepo(repo);
  const revision = validateRevision(options.revision);
  const validatedRepoString = `${validatedRepo.organization}/${validatedRepo.name}`;
  const files = topLevelFiles(
    await fetchTree(validatedRepoString, validatedRepo, revision, options.signal),
  );
  const expectedExtension = expectedFormat === "litertlm" ? ".litertlm" : ".tflite";
  if (options.file !== undefined && !options.file.endsWith(expectedExtension)) {
    throw new HuggingFaceError(
      `HuggingFace options.file ${options.file} must end with ${expectedExtension}.`,
    );
  }

  const metadataEntry = files.find((entry) => entry.path === "model_metadata.json");
  let metadata: ParsedMetadata;
  let modelFile: string;
  if (metadataEntry !== undefined) {
    metadata = await loadMetadata(resolveUrl(validatedRepo, revision, metadataEntry.path));
    modelFile = validateSurfaceMetadata(metadata, expectedFormat);
    if (!files.some((entry) => entry.path === modelFile)) {
      throw new HuggingFaceError(
        `HuggingFace metadata model_file ${modelFile} was not found in the repo ${validatedRepoString} tree.`,
      );
    }
    if (options.file !== undefined && options.file !== modelFile) {
      throw new HuggingFaceError(
        `HuggingFace options.file ${options.file} conflicts with metadata model_file ${modelFile}.`,
      );
    }
  } else {
    modelFile = selectModelFile(validatedRepoString, files, expectedExtension, options.file);
    metadata = synthesizeMetadata(validatedRepo.name, revision, modelFile, expectedFormat);
  }

  const modelEntry = files.find((entry) => entry.path === modelFile);
  if (modelEntry === undefined) {
    throw new HuggingFaceError(
      `HuggingFace model file ${modelFile} was not found in repo ${validatedRepoString}.`,
    );
  }
  const sizeBytes = validateModelSize(validatedRepoString, modelFile, modelEntry);
  const sha256 =
    modelEntry.lfs?.oid !== undefined && /^[0-9a-f]{64}$/.test(modelEntry.lfs.oid)
      ? modelEntry.lfs.oid
      : undefined;
  return {
    modelUrl: resolveUrl(validatedRepo, revision, modelFile),
    metadata,
    sizeBytes,
    sha256,
  };
};
