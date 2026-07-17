import { Backend, type Conversation, Engine, getOrLoadGlobalLiteRtLm } from "@litert-lm/core";

import { RuntimeInitializationError } from "../errors.ts";
import type { DownloadProgress, SelectedAccelerator } from "../types.ts";
import { loadModelChunks } from "./model-download.ts";
import {
  AcceleratorUnavailableError,
  type LlmEngine,
  type LlmGeneration,
  type LlmRuntime,
  type RuntimeInitConfig,
} from "./runtime.ts";

type StreamedMessage = {
  readonly content?: string | readonly { readonly type?: string; readonly text?: string }[];
};

const messageText = (message: StreamedMessage): string => {
  const content = message.content;
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  let text = "";
  for (const part of content as StreamedMessage["content"] & readonly unknown[]) {
    if (
      typeof part === "object" &&
      part !== null &&
      "type" in part &&
      part.type === "text" &&
      "text" in part &&
      typeof part.text === "string"
    ) {
      text += part.text;
    }
  }
  return text;
};

const requireWebGpuAdapter = async (): Promise<void> => {
  const gpu =
    typeof navigator === "undefined"
      ? undefined
      : (navigator as { gpu?: { requestAdapter?: () => Promise<unknown> } }).gpu;
  if (gpu === undefined || typeof gpu.requestAdapter !== "function") {
    throw new AcceleratorUnavailableError("WebGPU is unavailable.");
  }
  let adapter: unknown;
  try {
    adapter = await gpu.requestAdapter();
  } catch (error: unknown) {
    throw new AcceleratorUnavailableError(`WebGPU adapter request failed: ${String(error)}`);
  }
  if (adapter === null || adapter === undefined) {
    throw new AcceleratorUnavailableError("WebGPU reported no compatible adapter.");
  }
};

const probeAccelerator = async (accelerator: SelectedAccelerator): Promise<void> => {
  if (accelerator === "webgpu") {
    await requireWebGpuAdapter();
  }
};

const openGeneration = async (
  engine: Engine,
  prompt: string,
  options: { readonly maxOutputTokens?: number },
): Promise<LlmGeneration> => {
  const conversation: Conversation = await engine.createConversation(
    options.maxOutputTokens === undefined
      ? undefined
      : { sessionConfig: { maxOutputTokens: options.maxOutputTokens } },
  );
  let reader: ReadableStreamDefaultReader<StreamedMessage> | undefined;
  let cleaned = false;
  let cleanupPromise: Promise<void> | undefined;
  const cleanup = async (): Promise<void> => {
    if (cleaned) {
      if (cleanupPromise !== undefined) {
        await cleanupPromise;
      }
      return;
    }
    cleaned = true;
    cleanupPromise = (async () => {
      if (reader !== undefined) {
        await reader.cancel().catch(() => undefined);
      }
      await conversation.delete().catch(() => undefined);
    })();
    await cleanupPromise;
  };
  try {
    reader = conversation.sendMessageStreaming(prompt).getReader();
  } catch (error: unknown) {
    await cleanup();
    throw error;
  }
  const activeReader = reader;
  if (activeReader === undefined) {
    await cleanup();
    throw new RuntimeInitializationError("LiteRT-LM did not return a stream reader.");
  }
  const stream = (async function* () {
    try {
      while (true) {
        const { done, value } = await activeReader.read();
        if (done) {
          return;
        }
        const delta = messageText(value);
        if (delta.length > 0) {
          yield delta;
        }
      }
    } finally {
      // Stop any in-flight decoding before deleting; delete waits on the
      // engine mutex, so an abandoned iteration must cancel first.
      await cleanup();
    }
  })();
  return {
    stream,
    cancel: () => {
      void cleanup();
    },
    dispose: cleanup,
  };
};

export const liteRtLmRuntime: LlmRuntime<Blob> = {
  initialize: async (config: RuntimeInitConfig) => {
    await getOrLoadGlobalLiteRtLm(config.wasmPath.toString());
  },
  probeAccelerator,
  fetchModel: async (
    modelUrl: URL,
    onProgress: ((progress: DownloadProgress) => void) | undefined,
    signal?: AbortSignal,
  ) => loadModelChunks(modelUrl, onProgress, signal).then((chunks) => new Blob(chunks)),
  modelFromChunks: async (chunks) => new Blob([...chunks] as BlobPart[]),
  createEngine: async (
    model: Blob,
    accelerator: SelectedAccelerator,
    contextLength: number | undefined,
  ): Promise<LlmEngine> => {
    // The default GPU_ARTISAN backend streams the model and cannot stream
    // compressed tokenizer sections; the plain GPU and CPU backends load
    // through the wasm filesystem and support every published section type.
    const engine = await Engine.create({
      model,
      backend: accelerator === "webgpu" ? Backend.GPU : Backend.CPU,
      ...(contextLength === undefined
        ? {}
        : { mainExecutorSettings: { maxNumTokens: contextLength } }),
    });
    return {
      generate: (prompt, options) => openGeneration(engine, prompt, options),
      delete: () => engine.delete(),
    };
  },
};
