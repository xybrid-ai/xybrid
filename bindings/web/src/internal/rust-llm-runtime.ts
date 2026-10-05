import { InferenceError, RuntimeInitializationError } from "../errors.ts";
import type { GenerateOptions } from "../types.ts";
import { abortReason, throwIfAborted } from "./loading.ts";
import { loadModelBytes } from "./model-download.ts";
import type { LoadMetrics, Memory, Request, Response, RunMetrics } from "./protocol.ts";
import {
  AcceleratorUnavailableError,
  type LlmEngine,
  type LlmGeneration,
  type LlmRuntime,
} from "./runtime.ts";

type GgufBytes = { bytes: Uint8Array<ArrayBuffer>; downloadMs: number };
type Pending = { resolve(response: Response): void; reject(error: unknown): void };

/** One Rust model/context and one serialized native call stack per worker. */
class WorkerEngine implements LlmEngine {
  loaded: LoadMetrics | undefined;
  lastRun: RunMetrics | undefined;
  releasedMemory: Memory | undefined;
  #worker: Worker;
  #id = 0;
  #pending = new Map<number, Pending>();
  #tokens: ((text: string) => void) | undefined;
  #generation: Promise<Response> | undefined;
  #failure: Error | undefined;

  constructor() {
    // This module is bundled into dist/index.js; worker.js is a separate entry.
    this.#worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.#worker.onmessage = ({ data }: MessageEvent<Response>) => {
      if (data.type === "token") {
        this.#tokens?.(data.text);
        return;
      }
      if (data.type === "progress") return;
      const pending = this.#pending.get(data.id);
      if (pending === undefined) return;
      this.#pending.delete(data.id);
      if (data.type === "error") pending.reject(new Error(data.message));
      else pending.resolve(data);
    };
    const fail = (message: string): void => {
      this.#failure = new Error(message);
      for (const pending of this.#pending.values()) pending.reject(this.#failure);
      this.#pending.clear();
    };
    this.#worker.onerror = (event) => fail(event.message);
    this.#worker.onmessageerror = () => fail("The runtime worker sent an unreadable message.");
  }

  #request(request: Request & { id: number }, transfer: Transferable[] = []): Promise<Response> {
    if (this.#failure !== undefined) return Promise.reject(this.#failure);
    return new Promise((resolve, reject) => {
      this.#pending.set(request.id, { resolve, reject });
      try {
        this.#worker.postMessage(request, transfer);
      } catch (error) {
        this.#pending.delete(request.id);
        reject(error);
      }
    });
  }

  async load(
    request: Omit<Extract<Request, { type: "load" }>, "id">,
    signal?: AbortSignal,
  ): Promise<void> {
    throwIfAborted(signal);
    const aborted = (): void => {
      for (const pending of this.#pending.values())
        pending.reject(abortReason(signal as AbortSignal));
      this.#pending.clear();
      this.#worker.terminate();
    };
    signal?.addEventListener("abort", aborted, { once: true });
    try {
      const response = await this.#request({ ...request, id: ++this.#id }, [request.bytes]);
      if (response.type !== "loaded") throw new Error("Unexpected runtime load response.");
      this.loaded = response.metrics;
    } catch (error) {
      this.#worker.terminate();
      throw error;
    } finally {
      signal?.removeEventListener("abort", aborted);
    }
  }

  async generate(prompt: string, options: GenerateOptions): Promise<LlmGeneration> {
    const queue: string[] = [];
    let wake: (() => void) | undefined;
    let done = false;
    let failure: unknown;
    this.#tokens = (text) => {
      queue.push(text);
      wake?.();
    };
    const operation = this.#request({
      type: "generate",
      id: ++this.#id,
      prompt,
      maxTokens: options.maxOutputTokens ?? 64,
    });
    this.#generation = operation;
    void operation.then(
      (response) => {
        if (response.type === "complete") this.lastRun = response.metrics;
        done = true;
        wake?.();
      },
      (error: unknown) => {
        failure = new InferenceError(error);
        done = true;
        wake?.();
      },
    );

    return {
      stream: (async function* () {
        while (!done || queue.length > 0) {
          if (queue.length === 0)
            await new Promise<void>((resolve) => {
              wake = resolve;
            });
          const text = queue.shift();
          if (text !== undefined) yield text;
        }
        if (failure !== undefined) throw failure;
      })(),
      cancel: () => this.#worker.postMessage({ type: "cancel" } satisfies Request),
      dispose: async () => {
        await operation.catch(() => undefined);
        if (this.#generation === operation) {
          this.#generation = undefined;
          this.#tokens = undefined;
        }
      },
    };
  }

  async delete(): Promise<void> {
    try {
      this.#worker.postMessage({ type: "cancel" } satisfies Request);
      await this.#generation?.catch(() => undefined);
      const response = await this.#request({ type: "dispose", id: ++this.#id });
      if (response.type !== "disposed") throw new Error("Unexpected runtime disposal response.");
      this.releasedMemory = response.memory;
    } finally {
      this.#worker.terminate();
    }
  }
}

let wasmPath: URL | undefined;

export const rustLlmRuntime: LlmRuntime<GgufBytes> = {
  initialize: async (config) => {
    wasmPath = new URL(config.wasmPath);
  },
  probeAccelerator: async () => {
    const gpu = (
      navigator as unknown as {
        gpu?: { requestAdapter(): Promise<{ features: { has(feature: string): boolean } } | null> };
      }
    ).gpu;
    if (!(await gpu?.requestAdapter())?.features.has("shader-f16")) {
      throw new AcceleratorUnavailableError(
        "WebGPU with shader-f16 is unavailable in this browser.",
      );
    }
  },
  fetchModel: async (url, progress, signal) => {
    const started = performance.now();
    const bytes = await loadModelBytes(url, signal, progress);
    return { bytes, downloadMs: performance.now() - started };
  },
  modelFromChunks: async (chunks, downloadMs = 0) => {
    const bytes = new Uint8Array(chunks.reduce((total, chunk) => total + chunk.byteLength, 0));
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return { bytes, downloadMs };
  },
  createEngine: async (model, accelerator, contextLength, signal) => {
    if (wasmPath === undefined)
      throw new RuntimeInitializationError("Initialize the runtime first.");
    const engine = new WorkerEngine();
    // Keep the source available for auto's GPU-to-CPU fallback without a second download.
    const bytes = model.bytes.slice().buffer;
    await engine.load(
      {
        type: "load",
        accelerator,
        bytes,
        contextLength: contextLength ?? 512,
        wasmPath: wasmPath.href,
        downloadMs: model.downloadMs,
      },
      signal,
    );
    return engine;
  },
};
