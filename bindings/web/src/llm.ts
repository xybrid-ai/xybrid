import type { ObservationRef, Observer } from "@xybrid/observability";
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
import {
  epochNow,
  GenerationObservation,
  type LoadObservation,
  loadObservation,
  observeSafely,
  recordLoad,
  validateGenerateObservation,
} from "./internal/observability.ts";
import type { LoadMetrics, Memory, RunMetrics } from "./internal/protocol.ts";
import { type ModelResolution, resolveRegistryModel } from "./internal/registry.ts";
import type { LlmEngine, LlmGeneration, LlmRuntime } from "./internal/runtime.ts";
import { DEFAULT_MAX_OUTPUT_TOKENS, rustLlmRuntime } from "./internal/rust-llm-runtime.ts";
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
    ...observationOptions(options),
  };
  if (onDownloadProgress === undefined) {
    return normalized;
  }
  return {
    ...normalized,
    onDownloadProgress: onDownloadProgress as NonNullable<LlmLoadOptions["onDownloadProgress"]>,
  };
};

/** Validated `observer` / `observation`, carried through option normalization. */
const observationOptions = (options: unknown): Pick<LlmLoadOptions, "observer" | "observation"> => {
  const observation = loadObservation(options);
  if (observation === undefined) return {};
  return {
    observer: observation.observer,
    ...(observation.trace === undefined ? {} : { observation: observation.trace }),
  };
};

/** A readable model name for telemetry, without query strings or credentials. */
const modelNameFromUrl = (url: URL): string => {
  const segments = url.pathname.split("/").filter((segment) => segment.length > 0);
  const last = segments.at(-1) ?? url.hostname;
  const name = last.endsWith(".json") ? (segments.at(-2) ?? last) : last;
  // WHATWG URLs accept literal '%' and malformed UTF-8 escapes in paths.
  // A telemetry label must not prevent those URLs from reaching the loader.
  return (observeSafely(() => decodeURIComponent(name)) ?? name).replace(/\.gguf$/i, "");
};

/** Runs a load and, with an observer, records it as `load_model` and observes the session. */
const observedLoad = async (
  observation: LoadObservation | undefined,
  modelName: string | (() => string),
  load: () => Promise<LlmSession>,
): Promise<LlmSession> => {
  if (observation === undefined) return load();
  const fallbackModel = typeof modelName === "string" ? modelName : modelName();
  const startedAtMs = epochNow();
  let session: LlmSession;
  try {
    session = await load();
  } catch (error: unknown) {
    observeSafely(() => recordLoad(observation, fallbackModel, startedAtMs, { error }));
    throw error;
  }
  const model = session.modelId ?? fallbackModel;
  session.observe(
    observation.observer,
    model,
    observeSafely(() => recordLoad(observation, model, startedAtMs, { session })),
  );
  return session;
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
  const values = options as Record<string, unknown>;
  const maxOutputTokens = values["maxOutputTokens"];
  let observation: GenerateOptions["observation"];
  try {
    observation = validateGenerateObservation(values["observation"]);
  } catch (error: unknown) {
    throw new InputValidationError(error instanceof Error ? error.message : String(error));
  }
  const validated: GenerateOptions = observation === undefined ? {} : { observation };
  if (maxOutputTokens === undefined) {
    return validated;
  }
  if (!Number.isSafeInteger(maxOutputTokens) || (maxOutputTokens as number) <= 0) {
    throw new InputValidationError("maxOutputTokens must be a positive integer.");
  }
  return { ...validated, maxOutputTokens: maxOutputTokens as number };
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
  #observer: Observer | undefined;
  #model: string | undefined;
  #loadObservation: ObservationRef | undefined;
  #runObservation: ObservationRef | undefined;

  constructor(
    private engine: LlmEngine | undefined,
    readonly accelerator: SelectedAccelerator,
    /** The model's own id, when its metadata names one. */
    readonly modelId?: string,
  ) {
    this.loadMetrics = engine?.loaded;
  }

  /** Records later generations with `observer`; `loadObservation` is the load span. */
  observe(observer: Observer, model: string, loadObservation: ObservationRef | undefined): void {
    this.#observer = observer;
    this.#model = model;
    this.#loadObservation = loadObservation;
  }

  get loaded(): LoadMetrics {
    if (this.loadMetrics === undefined) {
      throw new RuntimeConfigurationError("The engine did not provide load metrics.");
    }
    return this.#loadObservation === undefined
      ? this.loadMetrics
      : { ...this.loadMetrics, observation: this.#loadObservation };
  }

  get lastRun(): RunMetrics | undefined {
    const metrics = this.engine?.lastRun ?? this.runMetrics;
    return metrics === undefined || this.#runObservation === undefined
      ? metrics
      : { ...metrics, observation: this.#runObservation };
  }

  async cancel(): Promise<void> {
    const iterator = this.activeIterator;
    const generation = this.activeGeneration ?? (await this.startingGeneration);
    if (this.activeIterator === iterator) generation?.cancel();
    // Closing the iterator awaits native disposal and releases the run lock.
    await iterator?.return(undefined);
  }

  generateStream(
    prompt: string,
    options?: GenerateOptions,
    streaming = true,
  ): AsyncGenerator<string, void, void> {
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
      session.#runObservation = undefined;
      const engine = session.engine;
      const observer = session.#observer;
      // Only observed runs pay for telemetry: no hooks, timestamps or text accumulation otherwise.
      let observation: GenerationObservation | undefined;
      const previousRun = engine?.lastRun;
      let outcome: "completed" | "abandoned" | { readonly error: unknown } = "abandoned";
      try {
        observation =
          observer === undefined || engine === undefined
            ? undefined
            : observeSafely(
                () =>
                  new GenerationObservation(
                    observer,
                    session.#model ?? "gguf",
                    session.accelerator,
                    validatedPrompt,
                    validatedOptions,
                    validatedOptions.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS,
                    streaming,
                    engine.reportsTiming === true,
                  ),
              );
        let generation: LlmGeneration;
        if (engine === undefined) {
          throw new DisposedError();
        }
        try {
          session.startingGeneration = engine.generate(validatedPrompt, {
            ...(validatedOptions.maxOutputTokens === undefined
              ? {}
              : { maxOutputTokens: validatedOptions.maxOutputTokens }),
            ...(observation === undefined ? {} : { telemetry: observation.telemetry() }),
          });
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
          if (observation === undefined) {
            yield* generation.stream;
          } else {
            for await (const delta of generation.stream) {
              observation.delta(delta);
              yield delta;
            }
          }
        } catch (error: unknown) {
          throw asInferenceError(error);
        }
        outcome = "completed";
      } catch (error: unknown) {
        outcome = { error };
        throw error;
      } finally {
        // Abandoned iteration must stop decoding; on normal completion the
        // cancel is a no-op against an already-closed stream.
        try {
          session.activeGeneration?.cancel();
        } finally {
          try {
            await session.activeGeneration?.dispose();
          } finally {
            try {
              session.runMetrics = session.engine?.lastRun;
              if (observation !== undefined) {
                const run = engine?.lastRun;
                const completedObservation = observation;
                session.#runObservation = observeSafely(() =>
                  completedObservation.finish(outcome, run === previousRun ? undefined : run),
                );
              }
            } finally {
              session.activeGeneration = undefined;
              session.startingGeneration = undefined;
              if (session.activeIterator === iterator) {
                session.activeIterator = undefined;
              }
              session.running = undefined;
              release();
            }
          }
        }
      }
    })();
    return iterator;
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<string> {
    let text = "";
    for await (const delta of this.generateStream(prompt, options, false)) {
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
    const observation = loadObservation(options);
    return runWithLoadCancellation(normalized.signal, async (cancellation) => {
      const session = await observedLoad(
        observation,
        () => modelNameFromUrl(url),
        () => loadGguf(url, options, normalized, metadata, contextLength, base, cancellation),
      );
      return new XybridLlm(session, LLM_CONSTRUCTION_TOKEN);
    });
  }

  static async fromRegistry(id: string, options?: RegistryLoadOptions): Promise<XybridLlm> {
    const base = typeof location === "undefined" ? undefined : location.href;
    const normalizedOptions = normalizeRegistryOptions(options, base);
    const observation = loadObservation(options);
    return runWithLoadCancellation(normalizedOptions.signal, async (cancellation) => {
      const session = await observedLoad(observation, String(id), async () => {
        const resolution = await resolveRegistryModel(id, "gguf", {
          registryUrl: normalizedOptions.registryUrl,
          signal: cancellation.signal,
          version: normalizedOptions.version,
        });
        return loadLlmFromResolution(
          resolution,
          normalizedOptions,
          rustLlmRuntime,
          sharedLlmInitializer,
          cancellation,
        );
      });
      return new XybridLlm(session, LLM_CONSTRUCTION_TOKEN);
    });
  }

  static async fromHuggingFace(repo: string, options?: HuggingFaceLoadOptions): Promise<XybridLlm> {
    const base = typeof location === "undefined" ? undefined : location.href;
    const normalizedOptions = normalizeHuggingFaceOptions(options, base);
    const observation = loadObservation(options);
    return runWithLoadCancellation(normalizedOptions.signal, async (cancellation) => {
      const session = await observedLoad(observation, String(repo), async () => {
        const resolution = await resolveHuggingFaceModel(repo, "gguf", {
          file: normalizedOptions.file,
          revision: normalizedOptions.revision,
          signal: cancellation.signal,
        });
        return loadLlmFromResolution(
          resolution,
          normalizedOptions,
          rustLlmRuntime,
          sharedLlmInitializer,
          cancellation,
        );
      });
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

/** The body of `fromUrl`: a direct GGUF, verified when its size (and hash) are declared. */
const loadGguf = async (
  url: URL,
  options: GgufLoadOptions,
  normalized: LlmLoadOptions,
  metadata: ParsedMetadata,
  contextLength: number | undefined,
  base: string | undefined,
  cancellation: LoadCancellation,
): Promise<LlmSession> => {
  if (options.sizeBytes !== undefined) {
    return loadLlmFromResolution(
      { modelUrl: url, metadata, sizeBytes: options.sizeBytes, sha256: options.sha256 },
      normalizeRegistryOptions(options, base),
      rustLlmRuntime,
      sharedLlmInitializer,
      cancellation,
    );
  }
  const initialization = startRuntimeInitialization(
    normalized.wasmPath ?? DEFAULT_LLM_WASM_PATH,
    rustLlmRuntime,
    sharedLlmInitializer,
  );
  const loaded = await loadAfterMetadata(
    normalized.accelerator ?? "auto",
    initialization,
    () => rustLlmRuntime.probeAccelerator("webgpu"),
    () => rustLlmRuntime.fetchModel(url, normalized.onDownloadProgress, cancellation.signal),
    cancellation,
  );
  const { value, accelerator } = await selectAccelerated(
    normalized.accelerator ?? "auto",
    (target) =>
      rustLlmRuntime.createEngine(loaded.value, target, contextLength, cancellation.signal),
    loaded.preflight,
    cancellation.signal,
    (engine) => engine.delete(),
  );
  return new LlmSession(value, accelerator);
};

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
  return observedLoad(
    loadObservation(options),
    () => modelNameFromUrl(metadataUrl),
    () =>
      loadFromMetadata(
        metadataUrl,
        options,
        wasmPath,
        runtime,
        getMetadata,
        initializer,
        cancellation,
      ),
  );
};

const loadFromMetadata = async <Model>(
  metadataUrl: URL,
  options: LlmLoadOptions,
  wasmPath: string | URL,
  runtime: LlmRuntime<Model>,
  getMetadata: MetadataLoader,
  initializer: RuntimeInitializer,
  cancellation: LoadCancellation,
): Promise<LlmSession> => {
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
  return new LlmSession(value, accelerator, metadata.modelId);
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
