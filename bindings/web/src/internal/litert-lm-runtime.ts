import { Backend, type Conversation, Engine, getOrLoadGlobalLiteRtLm } from "@litert-lm/core";
import ky from "ky";

import { RuntimeInitializationError } from "../errors.ts";
import type { DownloadProgress, SelectedAccelerator } from "../types.ts";
import { readResponseBytes } from "./response.ts";
import {
  AcceleratorUnavailableError,
  type LlmEngine,
  type LlmGeneration,
  type LlmRuntime,
  type RuntimeInitConfig,
} from "./runtime.ts";

const MAX_MODEL_BYTES = 512 * 1024 * 1024;

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
  const gpu = (navigator as { gpu?: { requestAdapter?: () => Promise<unknown> } }).gpu;
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
  let reader: ReadableStreamDefaultReader<StreamedMessage>;
  try {
    reader = conversation.sendMessageStreaming(prompt).getReader();
  } catch (error: unknown) {
    await conversation.delete().catch(() => undefined);
    throw error;
  }
  const stream = (async function* () {
    try {
      while (true) {
        const { done, value } = await reader.read();
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
      await reader.cancel().catch(() => undefined);
      await conversation.delete().catch(() => undefined);
    }
  })();
  return {
    stream,
    cancel: () => {
      void reader.cancel().catch(() => undefined);
    },
  };
};

export const liteRtLmRuntime: LlmRuntime = {
  initialize: async (config: RuntimeInitConfig) => {
    await getOrLoadGlobalLiteRtLm(config.wasmPath.toString());
  },
  fetchModel: async (
    modelUrl: URL,
    onProgress: ((progress: DownloadProgress) => void) | undefined,
  ) => {
    const response = await ky.get(modelUrl);
    try {
      const bytes = await readResponseBytes(
        response,
        MAX_MODEL_BYTES,
        "Model exceeds the 512 MiB browser limit.",
        onProgress === undefined
          ? undefined
          : (loadedBytes, totalBytes) => onProgress({ loadedBytes, totalBytes }),
      );
      return new Blob([bytes as Uint8Array<ArrayBuffer>]);
    } catch (error: unknown) {
      throw new RuntimeInitializationError(error);
    }
  },
  createEngine: async (
    model: unknown,
    accelerator: SelectedAccelerator,
    contextLength: number | undefined,
  ): Promise<LlmEngine> => {
    if (!(model instanceof Blob)) {
      throw new RuntimeInitializationError("Unexpected LiteRT-LM model payload.");
    }
    if (accelerator === "webgpu") {
      await requireWebGpuAdapter();
    }
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
