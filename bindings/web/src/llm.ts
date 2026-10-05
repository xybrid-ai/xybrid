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
import {
  type LoadCancellation,
  loadAfterMetadata,
  loadMetadata,
  type MetadataLoader,
  type NormalizedHuggingFaceLoadOptions,
  type NormalizedRegistryLoadOptions,
  normalizeBaseLoadOptions,
  normalizeHuggingFaceLoadOptions,
  normalizeRegistryLoadOptions,
  runWithLoadCancellation,
  selectAccelerated,
  startLoadPrelude,
  startRuntimeInitialization,
  throwIfAborted,
} from "./internal/loading.ts";
import type { LoadMetrics, Memory, RunMetrics } from "./internal/protocol.ts";
import { type ModelResolution, resolveRegistryModel } from "./internal/registry.ts";
import type { LlmEngine, LlmGeneration, LlmRuntime } from "./internal/runtime.ts";
import { rustLlmRuntime } from "./internal/rust-llm-runtime.ts";
import { resolveMetadataUrl } from "./internal/url.ts";
import { downloadVerifiedModel } from "./internal/verified-download.ts";
import {
  type ParsedMetadata,
  parseMetadata,
  resolveModelUrl,
  validateLlmBrowserMetadata,
} from "./metadata.ts";
import type {
  GenerateOptions,
  GgufLoadOptions,
  HuggingFaceLoadOptions,
  LlmLoadOptions,
  RegistryLoadOptions,
  SelectedAccelerator,
} from "./types.ts";

const LLM_CONSTRUCTION_TOKEN = Symbol("XybridLlm construction");
const DEFAULT_LLM_WASM_PATH = "/xybrid/runtime";

const normalizeLlmLoadOptions = (options: unknown, base: string | undefined): LlmLoadOptions => {
  const normalizedBase = normalizeBaseLoadOptions(options, base, DEFAULT_LLM_WASM_PATH);
  const values = options as Record<string, unknown>;
  const onDownloadProgress = values["onDownloadProgress"];
  if (onDownloadProgress !== undefined && typeof onDownloadProgress !== "function") {
    throw new RuntimeConfigurationError("onDownloadProgress must be a function.");
  }
  const normalized: LlmLoadOptions = {
    accelerator: normalizedBase.accelerator,
    wasmPath: normalizedBase.wasmPath,
    ...(normalizedBase.signal === undefined ? {} : { signal: normalizedBase.signal }),
  };
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
  private startingGeneration: Promise<LlmGeneration> | undefined;
  private activeIterator: AsyncGenerator<string, void, void> | undefined;
  private disposePromise: Promise<void> | undefined;
  private readonly loadMetrics: LoadMetrics | undefined;
  private runMetrics: RunMetrics | undefined;
  releasedMemory: Memory | undefined;

  constructor(
    private engine: LlmEngine | undefined,
    readonly accelerator: SelectedAccelerator,
  ) {
    this.loadMetrics = engine?.loaded;
  }

  get loaded(): LoadMetrics {
    if (this.loadMetrics === undefined) {
      throw new RuntimeConfigurationError("The engine did not provide load metrics.");
    }
    return this.loadMetrics;
  }

  get lastRun(): RunMetrics | undefined {
    return this.engine?.lastRun ?? this.runMetrics;
  }

  async cancel(): Promise<void> {
    const iterator = this.activeIterator;
    const generation = this.activeGeneration ?? (await this.startingGeneration);
    if (this.activeIterator === iterator) generation?.cancel();
    // Closing the iterator awaits native disposal and releases the run lock.
    await iterator?.return(undefined);
  }

  generateStream(prompt: string, options?: GenerateOptions): AsyncGenerator<string, void, void> {
    const validatedPrompt = validatePrompt(prompt);
    const validatedOptions = validateGenerateOptions(options);
    this.assertRunnable();
    if (this.running !== undefined) {
      throw new ConcurrentRunError();
    }
    const session = this;
    let iterator!: AsyncGenerator<string, void, void>;
    iterator = (async function* () {
      session.assertRunnable();
      if (session.running !== undefined) {
        throw new ConcurrentRunError();
      }
      let release: () => void = () => undefined;
      session.running = new Promise<void>((resolve) => {
        release = resolve;
      });
      session.activeIterator = iterator;
      try {
        let generation: LlmGeneration;
        const engine = session.engine;
        if (engine === undefined) {
          throw new DisposedError();
        }
        try {
          session.startingGeneration = engine.generate(validatedPrompt, validatedOptions);
          generation = await session.startingGeneration;
        } catch (error: unknown) {
          throw asInferenceError(error);
        }
        session.activeGeneration = generation;
        session.startingGeneration = undefined;
        if (session.disposePromise !== undefined) {
          generation.cancel();
          await generation.dispose();
          session.activeGeneration = undefined;
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
        try {
          await session.activeGeneration?.dispose();
        } finally {
          session.runMetrics = session.engine?.lastRun;
          session.activeGeneration = undefined;
          session.startingGeneration = undefined;
          if (session.activeIterator === iterator) {
            session.activeIterator = undefined;
          }
          session.running = undefined;
          release();
        }
      }
    })();
    return iterator;
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
    const iterator = this.activeIterator;
    const running = this.running;
    if (iterator !== undefined && running !== undefined) {
      try {
        await iterator.return(undefined);
      } catch {}
    }
    if (running !== undefined) {
      await Promise.allSettled([running]);
    }
    const engine = this.engine;
    try {
      await engine?.delete();
    } finally {
      this.runMetrics = engine?.lastRun;
      this.releasedMemory = engine?.releasedMemory;
      this.engine = undefined;
    }
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

  static async load(metadataUrl: string | URL, options: LlmLoadOptions = {}): Promise<XybridLlm> {
    const base = typeof location === "undefined" ? undefined : location.href;
    const normalizedMetadataUrl = resolveMetadataUrl(metadataUrl, base);
    const normalizedOptions = normalizeLlmLoadOptions(options, base);
    return runWithLoadCancellation(normalizedOptions.signal, async (cancellation) => {
      const session = await loadLlm(
        normalizedMetadataUrl,
        normalizedOptions,
        rustLlmRuntime,
        loadMetadata,
        sharedLlmInitializer,
        cancellation,
      );
      return new XybridLlm(session, LLM_CONSTRUCTION_TOKEN);
    });
  }

  /** Load a GGUF directly, with optional size and checksum verification. */
  static async fromUrl(modelUrl: string | URL, options: GgufLoadOptions = {}): Promise<XybridLlm> {
    const base = typeof location === "undefined" ? undefined : location.href;
    const url = resolveMetadataUrl(modelUrl, base);
    const normalized = normalizeLlmLoadOptions(options, base);
    const metadata = parseMetadata({
      model_id: "gguf",
      version: "1",
      execution_template: {
        type: "Gguf",
        model_file: "model.gguf",
        context_length: options.contextLength,
      },
      files: ["model.gguf"],
    });
    const { contextLength } = validateLlmBrowserMetadata(metadata);
    if (
      options.sizeBytes !== undefined &&
      (!Number.isSafeInteger(options.sizeBytes) ||
        options.sizeBytes <= 0 ||
        options.sizeBytes > 512 * 1024 * 1024)
    ) {
      throw new RuntimeConfigurationError(
        "sizeBytes must be a positive integer no greater than 512 MiB.",
      );
    }
    if (
      options.sha256 !== undefined &&
      (typeof options.sha256 !== "string" ||
        options.sizeBytes === undefined ||
        !/^[0-9a-f]{64}$/.test(options.sha256))
    ) {
      throw new RuntimeConfigurationError(
        "sha256 requires sizeBytes and 64 lowercase hex characters.",
      );
    }
    return runWithLoadCancellation(normalized.signal, async (cancellation) => {
      if (options.sizeBytes !== undefined) {
        const session = await loadLlmFromResolution(
          { modelUrl: url, metadata, sizeBytes: options.sizeBytes, sha256: options.sha256 },
          normalizeRegistryOptions(options, base),
          rustLlmRuntime,
          sharedLlmInitializer,
          cancellation,
        );
        return new XybridLlm(session, LLM_CONSTRUCTION_TOKEN);
      }
      const initialization = startRuntimeInitialization(
        normalized.wasmPath ?? DEFAULT_LLM_WASM_PATH,
        rustLlmRuntime,
        sharedLlmInitializer,
      );
      const loaded = await loadAfterMetadata(
        normalized.accelerator,
        initialization,
        () => rustLlmRuntime.probeAccelerator("webgpu"),
        () => rustLlmRuntime.fetchModel(url, normalized.onDownloadProgress, cancellation.signal),
        cancellation,
      );
      const { value, accelerator } = await selectAccelerated(
        normalized.accelerator,
        (target) =>
          rustLlmRuntime.createEngine(loaded.value, target, contextLength, cancellation.signal),
        loaded.preflight,
        cancellation.signal,
        (engine) => engine.delete(),
      );
      return new XybridLlm(new LlmSession(value, accelerator), LLM_CONSTRUCTION_TOKEN);
    });
  }

  static async fromRegistry(id: string, options?: RegistryLoadOptions): Promise<XybridLlm> {
    const base = typeof location === "undefined" ? undefined : location.href;
    const normalizedOptions = normalizeRegistryOptions(options, base);
    return runWithLoadCancellation(normalizedOptions.signal, async (cancellation) => {
      const resolution = await resolveRegistryModel(id, "gguf", {
        registryUrl: normalizedOptions.registryUrl,
        signal: cancellation.signal,
        version: normalizedOptions.version,
      });
      const session = await loadLlmFromResolution(
        resolution,
        normalizedOptions,
        rustLlmRuntime,
        sharedLlmInitializer,
        cancellation,
      );
      return new XybridLlm(session, LLM_CONSTRUCTION_TOKEN);
    });
  }

  static async fromHuggingFace(repo: string, options?: HuggingFaceLoadOptions): Promise<XybridLlm> {
    const base = typeof location === "undefined" ? undefined : location.href;
    const normalizedOptions = normalizeHuggingFaceOptions(options, base);
    return runWithLoadCancellation(normalizedOptions.signal, async (cancellation) => {
      const resolution = await resolveHuggingFaceModel(repo, "gguf", {
        file: normalizedOptions.file,
        revision: normalizedOptions.revision,
        signal: cancellation.signal,
      });
      const session = await loadLlmFromResolution(
        resolution,
        normalizedOptions,
        rustLlmRuntime,
        sharedLlmInitializer,
        cancellation,
      );
      return new XybridLlm(session, LLM_CONSTRUCTION_TOKEN);
    });
  }

  get accelerator(): SelectedAccelerator {
    return this.session.accelerator;
  }

  get loaded(): LoadMetrics {
    return this.session.loaded;
  }

  get lastRun(): RunMetrics | undefined {
    return this.session.lastRun;
  }

  get releasedMemory(): Memory | undefined {
    return this.session.releasedMemory;
  }

  cancel(): Promise<void> {
    return this.session.cancel();
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
  cancellation?: LoadCancellation,
): Promise<LlmSession> => {
  if (cancellation === undefined) {
    return runWithLoadCancellation(options.signal, (ownedCancellation) =>
      loadLlm(metadataUrl, options, runtime, getMetadata, initializer, ownedCancellation),
    );
  }
  const wasmPath = options.wasmPath;
  if (wasmPath === undefined) {
    throw new RuntimeConfigurationError("wasmPath must be provided to the internal LLM loader.");
  }
  const prelude = startLoadPrelude(metadataUrl, getMetadata, cancellation.signal);
  const metadata: ParsedMetadata = await prelude.metadata;
  const { modelFile, contextLength } = validateLlmBrowserMetadata(metadata);
  const modelUrl = resolveModelUrl(metadataUrl, modelFile, metadata.files);
  const preference = options.accelerator ?? "auto";
  throwIfAborted(cancellation.signal);
  const initialization = startRuntimeInitialization(wasmPath, runtime, initializer);
  const loaded = await loadAfterMetadata(
    preference,
    initialization,
    () => runtime.probeAccelerator("webgpu"),
    () => runtime.fetchModel(modelUrl, options.onDownloadProgress, cancellation.signal),
    cancellation,
  );
  const { value, accelerator } = await selectAccelerated(
    preference,
    (target) => runtime.createEngine(loaded.value, target, contextLength, cancellation.signal),
    loaded.preflight,
    cancellation.signal,
    (engine) => engine.delete(),
  );
  return new LlmSession(value, accelerator);
};

export const loadLlmFromResolution = async <Model>(
  resolution: ModelResolution,
  options: NormalizedRegistryLoadOptions | NormalizedHuggingFaceLoadOptions,
  runtime: LlmRuntime<Model>,
  initializer: RuntimeInitializer,
  cancellation?: LoadCancellation,
): Promise<LlmSession> => {
  if (cancellation === undefined) {
    return runWithLoadCancellation(options.signal, (ownedCancellation) =>
      loadLlmFromResolution(resolution, options, runtime, initializer, ownedCancellation),
    );
  }
  const { contextLength } = validateLlmBrowserMetadata(resolution.metadata);
  throwIfAborted(cancellation.signal);
  const initialization = startRuntimeInitialization(options.wasmPath, runtime, initializer);
  let downloadMs = 0;
  const loaded = await loadAfterMetadata(
    options.accelerator,
    initialization,
    () => runtime.probeAccelerator("webgpu"),
    async () => {
      const started = performance.now();
      const chunks = await downloadVerifiedModel(resolution.modelUrl, {
        onProgress: options.onDownloadProgress,
        sha256: resolution.sha256,
        signal: cancellation.signal,
        sizeBytes: resolution.sizeBytes,
      });
      downloadMs = performance.now() - started;
      return chunks;
    },
    cancellation,
  );
  throwIfAborted(cancellation.signal);
  const model = await runtime.modelFromChunks(loaded.value, downloadMs);
  throwIfAborted(cancellation.signal);
  const { value, accelerator } = await selectAccelerated(
    options.accelerator,
    (target) => runtime.createEngine(model, target, contextLength, cancellation.signal),
    loaded.preflight,
    cancellation.signal,
    (engine) => engine.delete(),
  );
  return new LlmSession(value, accelerator);
};
