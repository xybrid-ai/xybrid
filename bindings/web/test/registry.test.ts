import { describe, expect, test } from "bun:test";

import { IntegrityError, RegistryError, RuntimeConfigurationError } from "../src/errors.ts";
import { RuntimeInitializer } from "../src/internal/initialization.ts";
import { normalizeBaseLoadOptions, normalizeRegistryLoadOptions } from "../src/internal/loading.ts";
import { parseRegistryResponse, resolveRegistryModel } from "../src/internal/registry.ts";
import type { LlmEngine, LlmGeneration, LlmRuntime } from "../src/internal/runtime.ts";
import { downloadVerifiedModel } from "../src/internal/verified-download.ts";
import { loadLlmFromRegistry } from "../src/llm.ts";
import type { DownloadProgress } from "../src/types.ts";
import { tfliteMetadata } from "./helpers.ts";

const REGISTRY_URL = "https://registry.test";
const MODEL_URL = "https://models.test/model.tflite";
const SHA256 = "a".repeat(64);

const validEnvelope = (resolved: Record<string, unknown> = {}): Record<string, unknown> => ({
  mask: "demo-model",
  platform: "web",
  resolved: {
    hf_repo: "example/demo-model",
    file: "model.tflite",
    download_url: MODEL_URL,
    format: "tflite",
    quantization: "none",
    size_bytes: 4,
    sha256: SHA256,
    passthrough: true,
    artifacts: [],
    model_metadata: tfliteMetadata(),
    ...resolved,
  },
});

const llmMetadata = (): Record<string, unknown> => ({
  model_id: "demo-llm",
  version: "1",
  execution_template: {
    type: "LiteRtLm",
    model_file: "model.litertlm",
    context_length: 2048,
  },
  preprocessing: [],
  postprocessing: [],
  files: ["model.litertlm"],
});

const installFetch = (
  handler: (url: URL, init: RequestInit | undefined) => Promise<Response>,
): (() => void) => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = Object.assign(
    (input: RequestInfo | URL, init?: RequestInit) => {
      const url = input instanceof Request ? input.url : input.toString();
      return handler(new URL(url), init);
    },
    { preconnect: originalFetch.preconnect },
  );
  return () => {
    globalThis.fetch = originalFetch;
  };
};

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    headers: { "content-type": "application/json" },
    status,
  });

type RegistryLlmControl = {
  readonly runtime: LlmRuntime<number>;
  readonly initialized: readonly string[];
  readonly created: readonly { accelerator: string; contextLength: number | undefined }[];
};

const createRegistryLlmRuntime = (): RegistryLlmControl => {
  const initialized: string[] = [];
  const created: { accelerator: string; contextLength: number | undefined }[] = [];
  const runtime: LlmRuntime<number> = {
    initialize: async (config) => {
      initialized.push(config.wasmPath.toString());
    },
    fetchModel: async () => 0,
    modelFromChunks: async (chunks) => chunks.reduce((total, chunk) => total + chunk.byteLength, 0),
    createEngine: async (_model, accelerator, contextLength): Promise<LlmEngine> => {
      created.push({ accelerator, contextLength });
      const generation: LlmGeneration = {
        stream: (async function* () {})(),
        cancel: () => undefined,
        dispose: async () => undefined,
      };
      return {
        generate: async () => generation,
        delete: async () => undefined,
      };
    },
  };
  return { runtime, initialized, created };
};

describe("registry resolution and verified downloads", () => {
  test("rejects the live GGUF leak and independently rejects its empty sha256", async () => {
    const fixture = await Bun.file("test/fixtures/registry-resolve-gguf-leak.json").json();
    expect(() => parseRegistryResponse(fixture, "litertlm")).toThrow(RegistryError);
    expect(() => parseRegistryResponse(fixture, "litertlm")).toThrow(/format/);

    const forcedFormat = structuredClone(fixture) as Record<string, unknown> & {
      resolved: Record<string, unknown>;
    };
    forcedFormat.resolved["format"] = "litertlm";
    expect(() => parseRegistryResponse(forcedFormat, "litertlm")).toThrow(RegistryError);
    expect(() => parseRegistryResponse(forcedFormat, "litertlm")).toThrow(/sha256/);
  });

  test("rejects missing, short, and non-hex sha256 values", () => {
    const missing = validEnvelope();
    delete (missing["resolved"] as Record<string, unknown>)["sha256"];
    expect(() => parseRegistryResponse(missing, "tflite")).toThrow(RegistryError);

    for (const sha256 of ["abc", "g".repeat(64), "A".repeat(64)]) {
      expect(() => parseRegistryResponse(validEnvelope({ sha256 }), "tflite")).toThrow(
        RegistryError,
      );
    }
  });

  test("rejects an oversized model before downloading", () => {
    expect(() =>
      parseRegistryResponse(validEnvelope({ size_bytes: 512 * 1024 * 1024 + 1 }), "tflite"),
    ).toThrow(RegistryError);
  });

  test("rejects passthrough false", () => {
    expect(() => parseRegistryResponse(validEnvelope({ passthrough: false }), "tflite")).toThrow(
      RegistryError,
    );
  });

  test("rejects non-empty artifacts", () => {
    expect(() =>
      parseRegistryResponse(validEnvelope({ artifacts: [{ file: "tokenizer.json" }] }), "tflite"),
    ).toThrow(RegistryError);
  });

  test("rejects missing model_metadata", () => {
    const body = validEnvelope();
    delete (body["resolved"] as Record<string, unknown>)["model_metadata"];
    expect(() => parseRegistryResponse(body, "tflite")).toThrow(RegistryError);
  });

  test("rejects a file that differs from execution_template.model_file", () => {
    expect(() =>
      parseRegistryResponse(validEnvelope({ file: "different.tflite" }), "tflite"),
    ).toThrow(RegistryError);
  });

  test("rejects an envelope without resolved", () => {
    expect(() => parseRegistryResponse({ mask: "demo-model", platform: "web" }, "tflite")).toThrow(
      RegistryError,
    );
  });

  test("resolves, verifies, and creates the mocked LLM engine", async () => {
    const bytes = new Uint8Array([11, 22, 33, 44, 55, 66]);
    const { sha256 } = await import("hash-wasm");
    const expectedSha256 = await sha256(bytes);
    const body = {
      ...validEnvelope({
        file: "model.litertlm",
        format: "litertlm",
        model_metadata: llmMetadata(),
        size_bytes: bytes.byteLength,
        sha256: expectedSha256,
      }),
    };
    const progress: DownloadProgress[] = [];
    let downloadRequests = 0;
    const restoreFetch = installFetch(async (url) => {
      if (url.pathname.endsWith("/resolve")) {
        return jsonResponse(body);
      }
      downloadRequests += 1;
      return new Response(bytes);
    });
    try {
      const normalizedOptions = normalizeRegistryLoadOptions(
        {
          accelerator: "wasm",
          onDownloadProgress: (value: DownloadProgress) => progress.push(value),
          registryUrl: REGISTRY_URL,
          wasmPath: "/llm-runtime",
        },
        "https://app.test/",
        "/xybrid/llm-runtime",
      );
      const resolution = await resolveRegistryModel("demo-llm", "litertlm", normalizedOptions);
      const control = createRegistryLlmRuntime();
      const session = await loadLlmFromRegistry(
        resolution,
        normalizedOptions,
        control.runtime,
        new RuntimeInitializer(),
      );

      expect(session.accelerator).toBe("wasm");
      expect(control.initialized).toEqual(["https://app.test/llm-runtime"]);
      expect(control.created).toEqual([{ accelerator: "wasm", contextLength: 2048 }]);
      expect(downloadRequests).toBe(1);
      expect(progress.length).toBeGreaterThan(0);
      expect(progress.every((value) => value.totalBytes === bytes.byteLength)).toBe(true);
      expect(progress.at(-1)).toEqual({
        loadedBytes: bytes.byteLength,
        totalBytes: bytes.byteLength,
      });
      await session.dispose();
    } finally {
      restoreFetch();
    }
  });

  test("reports hash and byte-count mismatches as IntegrityError", async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const { sha256 } = await import("hash-wasm");
    const actualSha256 = await sha256(bytes);

    let restoreFetch = installFetch(async () => new Response(bytes));
    try {
      await expect(
        downloadVerifiedModel(new URL(MODEL_URL), {
          sha256: "b".repeat(64),
          sizeBytes: bytes.byteLength,
        }),
      ).rejects.toBeInstanceOf(IntegrityError);
    } finally {
      restoreFetch();
    }

    restoreFetch = installFetch(async () => new Response(bytes));
    try {
      await expect(
        downloadVerifiedModel(new URL(MODEL_URL), {
          sha256: actualSha256,
          sizeBytes: bytes.byteLength + 1,
        }),
      ).rejects.toBeInstanceOf(IntegrityError);
    } finally {
      restoreFetch();
    }

    restoreFetch = installFetch(async () => new Response(bytes));
    try {
      await expect(
        downloadVerifiedModel(new URL(MODEL_URL), {
          sha256: actualSha256,
          sizeBytes: bytes.byteLength - 1,
        }),
      ).rejects.toBeInstanceOf(IntegrityError);
    } finally {
      restoreFetch();
    }
  });

  test("turns registry 404 into an id-specific RegistryError", async () => {
    const restoreFetch = installFetch(async () => new Response("missing", { status: 404 }));
    try {
      await expect(
        resolveRegistryModel("missing-model", "tflite", { registryUrl: REGISTRY_URL }),
      ).rejects.toThrow(/missing-model/);
    } finally {
      restoreFetch();
    }
  });

  test("includes version in the resolve query string", async () => {
    let requestedUrl: URL | undefined;
    const restoreFetch = installFetch(async (url) => {
      requestedUrl = url;
      return jsonResponse(validEnvelope());
    });
    try {
      await resolveRegistryModel("demo-model", "tflite", {
        registryUrl: REGISTRY_URL,
        version: "2026.07.13",
      });
      expect(requestedUrl?.searchParams.get("platform")).toBe("web");
      expect(requestedUrl?.searchParams.get("format")).toBe("tflite");
      expect(requestedUrl?.searchParams.get("version")).toBe("2026.07.13");
    } finally {
      restoreFetch();
    }
  });

  test("falls back after a primary network error but not after an HTTP error", async () => {
    const hosts: string[] = [];
    const restoreFetch = installFetch(async (url) => {
      hosts.push(url.host);
      if (url.host === "registry.xybrid.dev") {
        throw new TypeError("network unavailable");
      }
      return jsonResponse(validEnvelope());
    });
    try {
      await resolveRegistryModel("demo-model", "tflite");
      expect(hosts).toEqual(["registry.xybrid.dev", "r2.xybrid.dev"]);
    } finally {
      restoreFetch();
    }

    const explicitHosts: string[] = [];
    const restoreExplicitFetch = installFetch(async (url) => {
      explicitHosts.push(url.host);
      throw new TypeError("network unavailable");
    });
    try {
      await expect(
        resolveRegistryModel("demo-model", "tflite", { registryUrl: REGISTRY_URL }),
      ).rejects.toBeInstanceOf(RegistryError);
      expect(explicitHosts).toEqual(["registry.test"]);
    } finally {
      restoreExplicitFetch();
    }
  });

  test("aborts and cancels an in-flight verified download", async () => {
    const controller = new AbortController();
    let cancelled = false;
    let releasePull: (() => void) | undefined;
    const stream = new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(new Uint8Array([1]));
      },
      pull() {
        return new Promise<void>((resolve) => {
          releasePull = resolve;
        });
      },
      cancel() {
        cancelled = true;
        releasePull?.();
      },
    });
    const restoreFetch = installFetch(async () => new Response(stream));
    try {
      const pending = downloadVerifiedModel(new URL(MODEL_URL), {
        sha256: "c".repeat(64),
        signal: controller.signal,
        sizeBytes: 2,
      });
      await Bun.sleep(0);
      controller.abort();
      await expect(pending).rejects.toBeDefined();
      expect(cancelled).toBe(true);
    } finally {
      controller.abort();
      restoreFetch();
    }
  });

  test("defaults load wasm paths while retaining same-origin validation", () => {
    const normalized = normalizeBaseLoadOptions(
      { accelerator: "wasm" },
      "https://app.test/models/",
      "/xybrid/litert",
    );
    expect(normalized.accelerator).toBe("wasm");
    expect(normalized.wasmPath.href).toBe("https://app.test/xybrid/litert");
    expect(() =>
      normalizeBaseLoadOptions(
        { accelerator: "wasm", wasmPath: "https://cdn.test/litert" },
        "https://app.test/models/",
        "/xybrid/litert",
      ),
    ).toThrow(RuntimeConfigurationError);
  });
});
