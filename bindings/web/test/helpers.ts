import { RuntimeInitializer } from "../src/internal/initialization.ts";
import type { MetadataLoader } from "../src/internal/loading.ts";
import type { LlmRuntime } from "../src/internal/runtime.ts";
import { loadLlm } from "../src/llm.ts";
import type { LlmLoadOptions } from "../src/types.ts";

export const ggufMetadata = (overrides: Record<string, unknown> = {}): Record<string, unknown> => ({
  model_id: "browser-smollm2",
  version: "1",
  execution_template: { type: "Gguf", model_file: "model.gguf", context_length: 512 },
  files: ["model.gguf"],
  preprocessing: [],
  postprocessing: [],
  ...overrides,
});

export const createRuntime = () => {
  const initialized: string[] = [];
  const runtime: LlmRuntime<number> = {
    initialize: async (config) => {
      initialized.push(config.wasmPath.toString());
    },
    probeAccelerator: async () => undefined,
    fetchModel: async () => 1,
    modelFromChunks: async (chunks) => chunks.reduce((n, chunk) => n + chunk.byteLength, 0),
    createEngine: async () => ({
      generate: async () => ({
        stream: (async function* () {})(),
        cancel: () => undefined,
        dispose: async () => undefined,
      }),
      delete: async () => undefined,
    }),
  };
  return { runtime, initialized };
};

export const loadWithDependencies = (
  url: URL,
  options: LlmLoadOptions,
  runtime: LlmRuntime<number>,
  loader: MetadataLoader,
) => loadLlm(url, options, runtime, loader, new RuntimeInitializer());
