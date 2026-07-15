import {
  ConcurrentRunError,
  DisposedError,
  InferenceError,
  InputValidationError,
  RuntimeConfigurationError,
  XybridError,
} from "./errors.ts";
import { resolveHuggingFaceModel } from "./internal/huggingface.ts";
import { type RuntimeInitializer, sharedLlmInitializer } from "./internal/initialization.ts";
import { liteRtLmRuntime } from "./internal/litert-lm-runtime.ts";
import {
  loadMetadata,
  type MetadataLoader,
  type NormalizedHuggingFaceLoadOptions,
  type NormalizedRegistryLoadOptions,
  normalizeBaseLoadOptions,
  normalizeHuggingFaceLoadOptions,
  normalizeRegistryLoadOptions,
  selectAccelerated,
  startLoadPrelude,
} from "./internal/loading.ts";
import { type ModelResolution, resolveRegistryModel } from "./internal/registry.ts";
import type { LlmEngine, LlmGeneration, LlmRuntime } from "./internal/runtime.ts";
import { resolveMetadataUrl } from "./internal/url.ts";
import { downloadVerifiedModel } from "./internal/verified-download.ts";
import { type ParsedMetadata, resolveModelUrl, validateLlmBrowserMetadata } from "./metadata.ts";
import type {
  GenerateOptions,
  HuggingFaceLoadOptions,
  LlmLoadOptions,
  RegistryLoadOptions,
  SelectedAccelerator,
} from "./types.ts";

const LLM_CONSTRUCTION_TOKEN = Symbol("XybridLlm construction");
const DEFAULT_LLM_WASM_PATH = "/xybrid/llm-runtime";

const normalizeLlmLoadOptions = (options: unknown, base: string | undefined): LlmLoadOptions => {
  const normalizedBase = normalizeBaseLoadOptions(options, base, DEFAULT_LLM_WASM_PATH);
  const values = options as Record<string, unknown>;
  const onDownloadProgress = values["onDownloadProgress"];
  if (onDownloadProgress !== undefined && typeof onDownloadProgress !== "function") {
    throw new RuntimeConfigurationError("onDownloadProgress must be a function.");
  }
  const normalized: LlmLoadOptions = { ...normalizedBase };
  if (onDownloadProgress === undefined) {
    return normalized;
  }
  return {
    ...normalized,
    onDownloadProgress: onDownloadProgress as NonNullable<LlmLoadOptions["onDownloadProgress"]>,
  };
};

const normalizeRegistryOptions = (
  options: unknown,
  base: string | undefined,
): NormalizedRegistryLoadOptions =>
  normalizeRegistryLoadOptions(options, base, DEFAULT_LLM_WASM_PATH);

const normalizeHuggingFaceOptions = (
  options: unknown,
  base: string | undefined,
): NormalizedHuggingFaceLoadOptions =>
  normalizeHuggingFaceLoadOptions(options, base, DEFAULT_LLM_WASM_PATH);

const validatePrompt = (prompt: unknown): string => {
  if (typeof prompt !== "string" || prompt.length === 0) {
    throw new InputValidationError("prompt must be a non-empty string.");
  }
  return prompt;
};

const validateGenerateOptions = (options: unknown): GenerateOptions => {
  if (options === undefined) {
    return {};
  }
  if (typeof options !== "object" || options === null) {
    throw new InputValidationError("generate options must be an object.");
  }
  const maxOutputTokens = (options as Record<string, unknown>)["maxOutputTokens"];
  if (maxOutputTokens === undefined) {
    return {};
  }
  if (!Number.isSafeInteger(maxOutputTokens) || (maxOutputTokens as number) <= 0) {
    throw new InputValidationError("maxOutputTokens must be a positive integer.");
  }
  return { maxOutputTokens: maxOutputTokens as number };
};

const asInferenceError = (error: unknown): XybridError =>
  error instanceof XybridError ? error : new InferenceError(error);

class LlmSession {
  private running: Promise<void> | undefined;
  private activeGeneration: LlmGeneration | undefined;
  private disposePromise: Promise<void> | undefined;

  constructor(
    private readonly engine: LlmEngine,
    readonly accelerator: SelectedAccelerator,
  ) {}

  generateStream(prompt: string, options?: GenerateOptions): AsyncGenerator<string, void, void> {
    const validatedPrompt = validatePrompt(prompt);
    const validatedOptions = validateGenerateOptions(options);
    this.assertRunnable();
    if (this.running !== undefined) {
      throw new ConcurrentRunError();
    }
    const session = this;
    return (async function* () {
      session.assertRunnable();
      if (session.running !== undefined) {
        throw new ConcurrentRunError();
      }
      let release: () => void = () => undefined;
      session.running = new Promise<void>((resolve) => {
        release = resolve;
      });
      try {
        let generation: LlmGeneration;
        try {
          generation = await session.engine.generate(validatedPrompt, validatedOptions);
        } catch (error: unknown) {
          throw asInferenceError(error);
        }
        session.activeGeneration = generation;
        if (session.disposePromise !== undefined) {
          generation.cancel();
          await generation.dispose();
          throw new DisposedError();
        }
        try {
          yield* generation.stream;
        } catch (error: unknown) {
          throw asInferenceError(error);
        }
      } finally {
        // Abandoned iteration must stop decoding; on normal completion the
        // cancel is a no-op against an already-closed stream.
        session.activeGeneration?.cancel();
        session.activeGeneration = undefined;
        session.running = undefined;
        release();
      }
    })();
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<string> {
    let text = "";
    for await (const delta of this.generateStream(prompt, options)) {
      text += delta;
    }
    return text;
  }

  async dispose(): Promise<void> {
    if (this.disposePromise !== undefined) {
      return this.disposePromise;
    }
    this.disposePromise = this.finishDisposal();
    return this.disposePromise;
  }

  private async finishDisposal(): Promise<void> {
    const generation = this.activeGeneration;
    generation?.cancel();
    if (this.running !== undefined) {
      await Promise.allSettled([this.running]);
    }
    await generation?.dispose();
    await this.engine.delete();
  }

  private assertRunnable(): void {
    if (this.disposePromise !== undefined) {
      throw new DisposedError();
    }
  }
}

export class XybridLlm {
  private constructor(
    private readonly session: LlmSession,
    token: typeof LLM_CONSTRUCTION_TOKEN,
  ) {
    if (token !== LLM_CONSTRUCTION_TOKEN) {
      throw new RuntimeConfigurationError("XybridLlm instances must be created with load().");
    }
  }

  static async load(metadataUrl: string | URL, options: LlmLoadOptions): Promise<XybridLlm> {
    const base = typeof location === "undefined" ? undefined : location.href;
    const normalizedMetadataUrl = resolveMetadataUrl(metadataUrl, base);
    const normalizedOptions = normalizeLlmLoadOptions(options, base);
    const session = await loadLlm(
      normalizedMetadataUrl,
      normalizedOptions,
      liteRtLmRuntime,
      loadMetadata,
      sharedLlmInitializer,
    );
    return new XybridLlm(session, LLM_CONSTRUCTION_TOKEN);
  }

  static async fromRegistry(id: string, options?: RegistryLoadOptions): Promise<XybridLlm> {
    const base = typeof location === "undefined" ? undefined : location.href;
    const normalizedOptions = normalizeRegistryOptions(options, base);
    const resolution = await resolveRegistryModel(id, "litertlm", {
      registryUrl: normalizedOptions.registryUrl,
      signal: normalizedOptions.signal,
      version: normalizedOptions.version,
    });
    const session = await loadLlmFromResolution(
      resolution,
      normalizedOptions,
      liteRtLmRuntime,
      sharedLlmInitializer,
    );
    return new XybridLlm(session, LLM_CONSTRUCTION_TOKEN);
  }

  static async fromHuggingFace(repo: string, options?: HuggingFaceLoadOptions): Promise<XybridLlm> {
    const base = typeof location === "undefined" ? undefined : location.href;
    const normalizedOptions = normalizeHuggingFaceOptions(options, base);
    const resolution = await resolveHuggingFaceModel(repo, "litertlm", {
      file: normalizedOptions.file,
      revision: normalizedOptions.revision,
      signal: normalizedOptions.signal,
    });
    const session = await loadLlmFromResolution(
      resolution,
      normalizedOptions,
      liteRtLmRuntime,
      sharedLlmInitializer,
    );
    return new XybridLlm(session, LLM_CONSTRUCTION_TOKEN);
  }

  get accelerator(): SelectedAccelerator {
    return this.session.accelerator;
  }

  generate(prompt: string, options?: GenerateOptions): Promise<string> {
    return this.session.generate(prompt, options);
  }

  generateStream(prompt: string, options?: GenerateOptions): AsyncGenerator<string, void, void> {
    return this.session.generateStream(prompt, options);
  }

  dispose(): Promise<void> {
    return this.session.dispose();
  }
}

export const loadLlm = async <Model>(
  metadataUrl: URL,
  options: LlmLoadOptions,
  runtime: LlmRuntime<Model>,
  getMetadata: MetadataLoader,
  initializer: RuntimeInitializer,
): Promise<LlmSession> => {
  const wasmPath = options.wasmPath;
  if (wasmPath === undefined) {
    throw new RuntimeConfigurationError("wasmPath must be provided to the internal LLM loader.");
  }
  const prelude = startLoadPrelude(metadataUrl, wasmPath, runtime, getMetadata, initializer);
  const metadata: ParsedMetadata = await prelude.metadata;
  const { modelFile, contextLength } = validateLlmBrowserMetadata(metadata);
  const modelUrl = resolveModelUrl(metadataUrl, modelFile, metadata.files);
  const modelPromise = runtime.fetchModel(modelUrl, options.onDownloadProgress);
  const [, model] = await Promise.all([prelude.initialization, modelPromise]);
  const { value, accelerator } = await selectAccelerated(options.accelerator ?? "auto", (target) =>
    runtime.createEngine(model, target, contextLength),
  );
  return new LlmSession(value, accelerator);
};

export const loadLlmFromResolution = async <Model>(
  resolution: ModelResolution,
  options: NormalizedRegistryLoadOptions | NormalizedHuggingFaceLoadOptions,
  runtime: LlmRuntime<Model>,
  initializer: RuntimeInitializer,
): Promise<LlmSession> => {
  const prelude = startLoadPrelude(
    resolution.metadata,
    options.wasmPath,
    runtime,
    async () => resolution.metadata,
    initializer,
  );
  const modelPromise = downloadVerifiedModel(resolution.modelUrl, {
    onProgress: options.onDownloadProgress,
    sha256: resolution.sha256,
    signal: options.signal,
    sizeBytes: resolution.sizeBytes,
  });
  const [, chunks] = await Promise.all([prelude.initialization, modelPromise]);
  const model = await runtime.modelFromChunks(chunks);
  const { value, accelerator } = await selectAccelerated(options.accelerator, (target) =>
    runtime.createEngine(
      model,
      target,
      validateLlmBrowserMetadata(resolution.metadata).contextLength,
    ),
  );
  return new LlmSession(value, accelerator);
};
