import type {
  Accelerator,
  LoadMetrics,
  Memory,
  Request,
  Response,
  RunMetrics,
} from "./protocol.ts";

type Pending = {
  resolve(message: Response): void;
  reject(error: Error): void;
};

/** Experimental browser handle over the existing Rust llama.cpp wrappers. */
export class RustGgufModel {
  readonly loaded: LoadMetrics;
  lastRun: RunMetrics | undefined;
  #worker: Worker;
  #id = 0;
  #pending = new Map<number, Pending>();
  #tokens: ((text: string) => void) | undefined;
  #generation: Promise<Response> | undefined;
  #disposal: Promise<Memory> | undefined;

  private constructor(worker: Worker, loaded: LoadMetrics) {
    this.#worker = worker;
    this.loaded = loaded;
    worker.onmessage = ({ data }: MessageEvent<Response>) => {
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
    worker.onerror = (event) => {
      for (const pending of this.#pending.values()) pending.reject(new Error(event.message));
      this.#pending.clear();
    };
  }

  static async load(
    options: {
      accelerator?: Accelerator | "auto";
      onDownloadProgress?: (loadedBytes: number, totalBytes: number) => void;
    } = {},
  ): Promise<RustGgufModel> {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    try {
      const loaded = await new Promise<LoadMetrics>((resolve, reject) => {
        worker.onmessage = ({ data }: MessageEvent<Response>) => {
          if (data.type === "progress")
            options.onDownloadProgress?.(data.loadedBytes, data.totalBytes);
          if (data.type === "loaded") resolve(data.metrics);
          if (data.type === "error") reject(new Error(data.message));
        };
        worker.onerror = (event) => reject(new Error(event.message));
        worker.postMessage({
          type: "load",
          id: 0,
          accelerator: options.accelerator ?? "wasm",
        } satisfies Request);
      });
      return new RustGgufModel(worker, loaded);
    } catch (error) {
      worker.terminate();
      throw error;
    }
  }

  #request(request: Request & { id: number }): Promise<Response> {
    return new Promise((resolve, reject) => {
      this.#pending.set(request.id, { resolve, reject });
      this.#worker.postMessage(request);
    });
  }

  async *generateStream(
    prompt: string,
    options: { maxTokens?: number } = {},
  ): AsyncGenerator<string> {
    if (this.#disposal !== undefined) throw new Error("Model is disposed.");
    if (this.#generation !== undefined) throw new Error("Generation is already in flight.");
    if (typeof prompt !== "string" || prompt.length === 0)
      throw new Error("A nonempty prompt is required.");
    const maxTokens = options.maxTokens ?? 64;
    if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 256)
      throw new Error("maxTokens must be 1..=256.");
    const queue: string[] = [];
    let wake: (() => void) | undefined;
    let done = false;
    let failure: unknown;
    this.#tokens = (text) => {
      queue.push(text);
      wake?.();
    };
    const generation = this.#request({ type: "generate", id: ++this.#id, prompt, maxTokens });
    this.#generation = generation;
    void generation.then(
      (message) => {
        if (message.type === "complete") this.lastRun = message.metrics;
        done = true;
        wake?.();
      },
      (error: unknown) => {
        failure = error;
        done = true;
        wake?.();
      },
    );
    try {
      while (!done || queue.length > 0) {
        if (queue.length === 0)
          await new Promise<void>((resolve) => {
            wake = resolve;
          });
        const text = queue.shift();
        if (text !== undefined) yield text;
      }
      if (failure !== undefined) throw failure;
    } finally {
      this.#worker.postMessage({ type: "cancel" } satisfies Request);
      await generation.catch(() => {});
      this.#tokens = undefined;
      this.#generation = undefined;
    }
  }

  async cancel(): Promise<void> {
    this.#worker.postMessage({ type: "cancel" } satisfies Request);
    await this.#generation;
  }

  dispose(): Promise<Memory> {
    this.#disposal ??= (async () => {
      await this.cancel().catch(() => {});
      try {
        const response = await this.#request({ type: "dispose", id: ++this.#id });
        if (response.type !== "disposed") throw new Error("Unexpected disposal response.");
        return response.memory;
      } finally {
        this.#worker.terminate();
      }
    })();
    return this.#disposal;
  }
}
