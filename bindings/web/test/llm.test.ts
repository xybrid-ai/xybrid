import { describe, expect, test } from "bun:test";

import {
  ConcurrentRunError,
  DisposedError,
  InferenceError,
  InputValidationError,
  InvalidMetadataError,
  RuntimeConfigurationError,
  RuntimeInitializationError,
  UnsupportedTemplateError,
} from "../src/errors.ts";
import { RuntimeInitializer } from "../src/internal/initialization.ts";
import { liteRtLmRuntime } from "../src/internal/litert-lm-runtime.ts";
import type { LlmEngine, LlmGeneration, LlmRuntime } from "../src/internal/runtime.ts";
import { AcceleratorUnavailableError } from "../src/internal/runtime.ts";
import { loadLlm, XybridLlm } from "../src/llm.ts";
import { parseMetadata } from "../src/metadata.ts";
import type { DownloadProgress, LlmLoadOptions } from "../src/types.ts";
import { createRuntime, loadWithDependencies, tfliteMetadata } from "./helpers.ts";

const metadataUrl = new URL("https://models.example/smollm2/model_metadata.json");
const options: LlmLoadOptions = { wasmPath: "/litert-lm", accelerator: "auto" };

const litertLmMetadata = (overrides: Record<string, unknown> = {}): Record<string, unknown> => ({
  model_id: "smollm2-135m-instruct",
  version: "1",
  execution_template: {
    type: "LiteRtLm",
    model_file: "model.litertlm",
    context_length: 2048,
  },
  preprocessing: [],
  postprocessing: [],
  files: ["model.litertlm"],
  ...overrides,
});

type LlmRuntimeControl = {
  readonly runtime: LlmRuntime<{ readonly bytes: number }>;
  readonly initialized: readonly string[];
  readonly fetches: readonly string[];
  readonly fetchSignals: readonly AbortSignal[];
  readonly created: readonly { accelerator: string; contextLength: number | undefined }[];
  readonly cancelled: () => number;
  readonly disposedGenerations: () => number;
  readonly deletedEngines: () => number;
  readonly holdInitialization: () => void;
  readonly releaseInitialization: () => void;
  readonly holdFetch: () => void;
  readonly releaseFetch: () => void;
  readonly failWebGpu: () => void;
  readonly failEngineCreation: () => void;
  readonly failGeneration: () => void;
  readonly setDeltas: (deltas: readonly string[]) => void;
  readonly holdGeneration: () => void;
  readonly releaseGeneration: () => void;
  readonly holdEngineGenerate: () => void;
  readonly releaseEngineGenerate: () => void;
};

const createLlmRuntime = (): LlmRuntimeControl => {
  const initialized: string[] = [];
  const fetches: string[] = [];
  const fetchSignals: AbortSignal[] = [];
  const created: { accelerator: string; contextLength: number | undefined }[] = [];
  let deltas: readonly string[] = ["Hello", ", world."];
  let cancelled = 0;
  let disposedGenerations = 0;
  let deletedEngines = 0;
  let shouldFailWebGpu = false;
  let shouldFailEngineCreation = false;
  let shouldFailGeneration = false;
  let generationGate: Promise<void> | undefined;
  let releaseGate: (() => void) | undefined;
  let engineGenerateGate: Promise<void> | undefined;
  let releaseEngineGenerateGate: (() => void) | undefined;
  let initializationGate: Promise<void> | undefined;
  let releaseInitializationGate: (() => void) | undefined;
  let fetchGate: Promise<void> | undefined;
  let releaseFetchGate: (() => void) | undefined;

  const waitForFetch = (signal: AbortSignal | undefined): Promise<void> => {
    const gate = fetchGate;
    if (gate === undefined) {
      return signal?.aborted ? Promise.reject(signal.reason) : Promise.resolve();
    }
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      const finish = (callback: () => void): void => {
        if (settled) {
          return;
        }
        settled = true;
        signal?.removeEventListener("abort", onAbort);
        callback();
      };
      const onAbort = (): void => {
        finish(() => reject(signal?.reason));
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      void gate.then(() => finish(resolve));
    });
  };

  const runtime: LlmRuntime<{ readonly bytes: number }> = {
    initialize: async (config) => {
      initialized.push(config.wasmPath.toString());
      await initializationGate;
    },
    probeAccelerator: async (accelerator) => {
      if (accelerator === "webgpu" && shouldFailWebGpu) {
        throw new AcceleratorUnavailableError("GPU unavailable");
      }
    },
    fetchModel: async (modelUrl, onProgress, signal) => {
      fetches.push(modelUrl.toString());
      if (signal !== undefined) {
        fetchSignals.push(signal);
      }
      await waitForFetch(signal);
      onProgress?.({ loadedBytes: 3, totalBytes: 7 });
      onProgress?.({ loadedBytes: 7, totalBytes: 7 });
      return { bytes: 7 };
    },
    modelFromChunks: async (chunks) => ({
      bytes: chunks.reduce((total, chunk) => total + chunk.byteLength, 0),
    }),
    createEngine: async (_model, accelerator, contextLength) => {
      if (accelerator === "webgpu" && shouldFailWebGpu) {
        throw new AcceleratorUnavailableError("GPU unavailable");
      }
      if (shouldFailEngineCreation) {
        throw new DOMException("engine creation failed", "OperationError");
      }
      created.push({ accelerator, contextLength });
      const engine: LlmEngine = {
        generate: async (): Promise<LlmGeneration> => {
          await engineGenerateGate;
          let stopped = false;
          const stream = (async function* () {
            await generationGate;
            if (shouldFailGeneration) {
              throw new DOMException("decode failed", "OperationError");
            }
            for (const delta of deltas) {
              if (stopped) {
                return;
              }
              yield delta;
            }
          })();
          return {
            stream,
            cancel: () => {
              stopped = true;
              cancelled += 1;
              releaseGate?.();
            },
            dispose: async () => {
              disposedGenerations += 1;
            },
          };
        },
        delete: async () => {
          deletedEngines += 1;
        },
      };
      return engine;
    },
  };

  return {
    runtime,
    initialized,
    fetches,
    fetchSignals,
    created,
    cancelled: () => cancelled,
    disposedGenerations: () => disposedGenerations,
    deletedEngines: () => deletedEngines,
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
    holdFetch: () => {
      fetchGate = new Promise((resolve) => {
        releaseFetchGate = resolve;
      });
    },
    releaseFetch: () => {
      releaseFetchGate?.();
      fetchGate = undefined;
      releaseFetchGate = undefined;
    },
    failWebGpu: () => {
      shouldFailWebGpu = true;
    },
    failEngineCreation: () => {
      shouldFailEngineCreation = true;
    },
    failGeneration: () => {
      shouldFailGeneration = true;
    },
    setDeltas: (values) => {
      deltas = values;
    },
    holdGeneration: () => {
      generationGate = new Promise((resolve) => {
        releaseGate = resolve;
      });
    },
    releaseGeneration: () => {
      releaseGate?.();
      generationGate = undefined;
      releaseGate = undefined;
    },
    holdEngineGenerate: () => {
      engineGenerateGate = new Promise((resolve) => {
        releaseEngineGenerateGate = resolve;
      });
    },
    releaseEngineGenerate: () => {
      releaseEngineGenerateGate?.();
      engineGenerateGate = undefined;
      releaseEngineGenerateGate = undefined;
    },
  };
};

const load = async (
  control = createLlmRuntime(),
  loadOptions: LlmLoadOptions = options,
  metadata: Record<string, unknown> = litertLmMetadata(),
) => ({
  control,
  llm: await loadLlm(
    metadataUrl,
    loadOptions,
    control.runtime,
    async () => parseMetadata(metadata),
    new RuntimeInitializer(),
  ),
});

describe("XybridLlm lifecycle", () => {
  test("rejects reflective construction without the module-private token", () => {
    expect(() => Reflect.construct(XybridLlm, [{}])).toThrow(RuntimeConfigurationError);
  });

  test("rejects malformed public load options with typed errors", async () => {
    for (const malformed of [
      null,
      undefined,
      {},
      { wasmPath: "/litert-lm", accelerator: "invalid" },
      { wasmPath: "/litert-lm", accelerator: "auto", onDownloadProgress: "notify" },
    ]) {
      await expect(
        Reflect.apply(XybridLlm.load, XybridLlm, [metadataUrl, malformed]),
      ).rejects.toBeInstanceOf(RuntimeConfigurationError);
    }
  });

  test("routes template mismatches in both directions to typed errors", async () => {
    await expect(load(createLlmRuntime(), options, tfliteMetadata())).rejects.toThrow(
      UnsupportedTemplateError,
    );
    const tensorRuntime = createRuntime();
    await expect(
      loadWithDependencies(metadataUrl, options, tensorRuntime.runtime, async () =>
        parseMetadata(litertLmMetadata()),
      ),
    ).rejects.toThrow(UnsupportedTemplateError);
  });

  test("wraps model download failures in the typed error hierarchy", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = Object.assign(async () => new Response("missing", { status: 404 }), {
      preconnect: originalFetch.preconnect,
    });
    try {
      await expect(
        liteRtLmRuntime.fetchModel(new URL("https://models.example/missing.litertlm"), undefined),
      ).rejects.toBeInstanceOf(RuntimeInitializationError);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("rejects malformed LiteRtLm metadata before downloading the model", async () => {
    const missingFile = litertLmMetadata({
      execution_template: { type: "LiteRtLm" },
    });
    const control = createLlmRuntime();
    await expect(load(control, options, missingFile)).rejects.toBeInstanceOf(InvalidMetadataError);
    for (const contextLength of [0, -1, 1.5, "2048", 2 ** 53]) {
      const malformedContext = litertLmMetadata({
        execution_template: {
          type: "LiteRtLm",
          model_file: "model.litertlm",
          context_length: contextLength,
        },
      });
      await expect(load(control, options, malformedContext)).rejects.toBeInstanceOf(
        InvalidMetadataError,
      );
    }
    expect(control.fetches).toHaveLength(0);
  });

  test("surfaces a caller abort from fetchModel and allows a fresh load", async () => {
    const controller = new AbortController();
    const control = createLlmRuntime();
    control.holdFetch();
    const pending = load(control, {
      ...options,
      accelerator: "wasm",
      signal: controller.signal,
    });
    await Bun.sleep(0);
    expect(control.fetchSignals).toHaveLength(1);
    controller.abort();
    let caught: unknown;
    try {
      await pending;
    } catch (error: unknown) {
      caught = error;
    }
    expect(caught).toMatchObject({ name: "AbortError" });
    expect(caught).not.toBeInstanceOf(RuntimeInitializationError);
    expect(control.fetchSignals[0]?.aborted).toBe(true);

    const fresh = createLlmRuntime();
    await expect(load(fresh, { ...options, accelerator: "wasm" })).resolves.toBeDefined();
  });

  test("does not poison a shared initializer when LLM metadata validation fails", async () => {
    const initializer = new RuntimeInitializer();
    const invalidRuntime = createLlmRuntime();
    await expect(
      loadLlm(
        metadataUrl,
        { wasmPath: "/litert-lm-a", accelerator: "wasm" },
        invalidRuntime.runtime,
        async () => parseMetadata(litertLmMetadata({ execution_template: { type: "LiteRtLm" } })),
        initializer,
      ),
    ).rejects.toBeInstanceOf(InvalidMetadataError);
    expect(invalidRuntime.initialized).toHaveLength(0);

    const validRuntime = createLlmRuntime();
    await expect(
      loadLlm(
        metadataUrl,
        { wasmPath: "/litert-lm-b", accelerator: "wasm" },
        validRuntime.runtime,
        async () => parseMetadata(litertLmMetadata()),
        initializer,
      ),
    ).resolves.toBeDefined();
    expect(validRuntime.initialized).toEqual(["/litert-lm-b"]);
  });

  test("downloads the model once and reuses it across the auto fallback", async () => {
    const control = createLlmRuntime();
    control.failWebGpu();
    const progress: DownloadProgress[] = [];
    const { llm } = await load(control, {
      ...options,
      onDownloadProgress: (value) => progress.push(value),
    });
    expect(llm.accelerator).toBe("wasm");
    expect(control.fetches).toEqual(["https://models.example/smollm2/model.litertlm"]);
    expect(control.created).toEqual([{ accelerator: "wasm", contextLength: 2048 }]);
    expect(progress).toEqual([
      { loadedBytes: 3, totalBytes: 7 },
      { loadedBytes: 7, totalBytes: 7 },
    ]);

    const explicit = createLlmRuntime();
    explicit.failWebGpu();
    await expect(
      load(explicit, { wasmPath: "/litert-lm", accelerator: "webgpu" }),
    ).rejects.toBeInstanceOf(RuntimeInitializationError);
    expect(explicit.fetches).toHaveLength(0);

    const failed = createLlmRuntime();
    failed.failEngineCreation();
    let caught: unknown;
    try {
      await load(failed);
    } catch (error: unknown) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(RuntimeInitializationError);
    if (!(caught instanceof RuntimeInitializationError)) {
      return;
    }
    expect(caught.causeValue).toBeInstanceOf(AggregateError);
    const aggregate = caught.causeValue as AggregateError;
    expect(aggregate.errors).toHaveLength(2);
    expect(aggregate.errors[0]).toBeInstanceOf(RuntimeInitializationError);
    expect(aggregate.errors[1]).toBeInstanceOf(RuntimeInitializationError);
  });

  test("overlaps initialization and model fetch after metadata resolves", async () => {
    const control = createLlmRuntime();
    control.holdInitialization();
    control.holdFetch();
    let resolveMetadata: ((metadata: ReturnType<typeof parseMetadata>) => void) | undefined;
    const metadata = new Promise<ReturnType<typeof parseMetadata>>((resolve) => {
      resolveMetadata = resolve;
    });
    const loading = loadLlm(
      metadataUrl,
      options,
      control.runtime,
      async () => metadata,
      new RuntimeInitializer(),
    );

    await Bun.sleep(0);
    expect(control.initialized).toHaveLength(0);
    expect(control.fetches).toHaveLength(0);

    resolveMetadata?.(parseMetadata(litertLmMetadata()));
    await Bun.sleep(0);
    expect(control.fetches).toEqual(["https://models.example/smollm2/model.litertlm"]);
    expect(control.created).toHaveLength(0);

    control.releaseFetch();
    control.releaseInitialization();
    await loading;
    expect(control.created).toEqual([{ accelerator: "webgpu", contextLength: 2048 }]);
  });

  test("generates full responses and streams incremental deltas", async () => {
    const { llm } = await load();
    await expect(llm.generate("Tell me a story.")).resolves.toBe("Hello, world.");
    const deltas: string[] = [];
    for await (const delta of llm.generateStream("Tell me a story.")) {
      deltas.push(delta);
    }
    expect(deltas).toEqual(["Hello", ", world."]);
  });

  test("validates prompts and generation options with typed errors", async () => {
    const { llm } = await load();
    await expect(Reflect.apply(llm.generate, llm, [42])).rejects.toBeInstanceOf(
      InputValidationError,
    );
    await expect(llm.generate("")).rejects.toBeInstanceOf(InputValidationError);
    await expect(llm.generate("hi", { maxOutputTokens: 0 })).rejects.toBeInstanceOf(
      InputValidationError,
    );
    await expect(llm.generate("hi", { maxOutputTokens: 1.5 })).rejects.toBeInstanceOf(
      InputValidationError,
    );
  });

  test("wraps runtime generation failures in the typed hierarchy", async () => {
    const { llm, control } = await load();
    control.failGeneration();
    await expect(llm.generate("hi")).rejects.toBeInstanceOf(InferenceError);
    control.setDeltas(["recovered"]);
  });

  test("rejects overlapping generations while one is in flight", async () => {
    const { llm, control } = await load();
    control.holdGeneration();
    const first = llm.generate("hi");
    await Bun.sleep(0);
    expect(() => llm.generateStream("again")).toThrow(ConcurrentRunError);
    await expect(llm.generate("again")).rejects.toBeInstanceOf(ConcurrentRunError);
    control.releaseGeneration();
    await expect(first).resolves.toBe("Hello, world.");
    await expect(llm.generate("after")).resolves.toBe("Hello, world.");
  });

  test("stops decoding when iteration is abandoned early", async () => {
    const { llm, control } = await load();
    for await (const delta of llm.generateStream("hi")) {
      expect(delta).toBe("Hello");
      break;
    }
    expect(control.cancelled()).toBeGreaterThan(0);
    await expect(llm.generate("after")).resolves.toBe("Hello, world.");
  });

  test("disposes a paused stream and closes its iterator", async () => {
    const { llm } = await load();
    const stream = llm.generateStream("hi");
    await expect(stream.next()).resolves.toEqual({ value: "Hello", done: false });

    await expect(
      Promise.race([
        llm.dispose(),
        new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error("dispose timed out")), 500);
        }),
      ]),
    ).resolves.toBeUndefined();
    await expect(stream.next()).resolves.toMatchObject({ done: true });
    await expect(llm.generate("after")).rejects.toBeInstanceOf(DisposedError);
  });

  test("dispose cancels in-flight work, deletes once, and rejects new work", async () => {
    const { llm, control } = await load();
    control.holdGeneration();
    const running = llm.generate("hi");
    await Bun.sleep(0);
    const disposing = llm.dispose();
    await expect(llm.dispose()).resolves.toBeUndefined();
    await disposing;
    expect(control.cancelled()).toBeGreaterThan(0);
    expect(control.deletedEngines()).toBe(1);
    await expect(running).resolves.toBe("");
    expect(() => llm.generateStream("after")).toThrow(DisposedError);
    await expect(llm.generate("after")).rejects.toBeInstanceOf(DisposedError);
  });

  test("disposes a generation exactly once when disposal races generation startup", async () => {
    const { llm, control } = await load();
    control.holdEngineGenerate();
    const running = llm.generate("hi");
    await Bun.sleep(0);
    const disposing = llm.dispose();
    control.releaseEngineGenerate();
    await expect(running).rejects.toBeInstanceOf(DisposedError);
    await expect(disposing).resolves.toBeUndefined();
    expect(control.disposedGenerations()).toBe(1);
  });

  test("un-iterated generators observe disposal on first pull", async () => {
    const { llm } = await load();
    const pending = llm.generateStream("hi");
    await llm.dispose();
    await expect(pending.next()).rejects.toBeInstanceOf(DisposedError);
  });
});
