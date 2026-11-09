import { describe, expect, test } from "bun:test";

import { HuggingFaceError, IntegrityError } from "../src/errors.ts";
import {
  resolveHuggingFaceModel,
  validateHfRepo,
  validateRevision,
} from "../src/internal/huggingface.ts";
import { RuntimeInitializer } from "../src/internal/initialization.ts";
import { normalizeHuggingFaceLoadOptions } from "../src/internal/loading.ts";
import type { LlmEngine, LlmGeneration, LlmRuntime } from "../src/internal/runtime.ts";
import { downloadVerifiedModel } from "../src/internal/verified-download.ts";
import { loadLlmFromResolution } from "../src/llm.ts";

const REPO = "QuantFactory/SmolLM2-135M-Instruct-GGUF";
const MODEL_FILE = "SmolLM2-135M-Instruct.Q4_0.gguf";
const HF_BASE = "https://huggingface.co";
const FIXTURE_PATH = "test/fixtures/hf-tree-smollm2-gguf.json";

type TreeEntry = {
  type: "file" | "directory";
  path: string;
  size?: number;
  lfs?: { oid: string; size: number; pointerSize: number };
};

const fixtureTree = async (): Promise<TreeEntry[]> =>
  structuredClone((await Bun.file(FIXTURE_PATH).json()) as TreeEntry[]);

const fileEntry = (path: string, size = 4): TreeEntry => ({
  type: "file",
  path,
  size,
});

const metadata = (modelFile: string, type = "Gguf") => ({
  model_id: "metadata-model",
  version: "1",
  execution_template: { type, model_file: modelFile },
  preprocessing: [],
  postprocessing: [],
  files: [modelFile],
});

const installFetch = (
  handler: (url: URL, init: RequestInit | undefined, request: Request) => Promise<Response>,
): (() => void) => {
  const originalFetch = globalThis.fetch;
  const originalRequest = globalThis.Request;
  class TrackedRequest extends originalRequest {
    private readonly trackedCredentials: RequestCredentials;

    constructor(input: RequestInfo | URL, init?: RequestInit) {
      super(input, init);
      this.trackedCredentials =
        init?.credentials ?? (input instanceof originalRequest ? input.credentials : "same-origin");
    }

    override get credentials(): RequestCredentials {
      return this.trackedCredentials;
    }
  }
  globalThis.Request = TrackedRequest;
  globalThis.fetch = Object.assign(
    (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init);
      const requestInit =
        input instanceof Request ? { ...init, signal: init?.signal ?? request.signal } : init;
      return handler(new URL(request.url), requestInit, request);
    },
    { preconnect: originalFetch.preconnect },
  );
  return () => {
    globalThis.fetch = originalFetch;
    globalThis.Request = originalRequest;
  };
};

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    headers: { "content-type": "application/json" },
    status,
  });

type LlmControl = {
  readonly runtime: LlmRuntime<number>;
  readonly initialized: readonly string[];
  readonly created: readonly { accelerator: string; contextLength: number | undefined }[];
};

const createLlmRuntime = (): LlmControl => {
  const initialized: string[] = [];
  const created: { accelerator: string; contextLength: number | undefined }[] = [];
  const runtime: LlmRuntime<number> = {
    initialize: async (config) => {
      initialized.push(config.wasmPath.toString());
    },
    probeAccelerator: async () => undefined,
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

describe("HuggingFace resolution", () => {
  test("validates repository names", () => {
    for (const repo of ["org", "org/", "/name", "org/na me", "org/../x", "a/b/c"]) {
      expect(() => validateHfRepo(repo)).toThrow(HuggingFaceError);
    }
    expect(validateHfRepo(REPO).name).toBe("SmolLM2-135M-Instruct-GGUF");
  });

  test("validates revisions and defaults to main", () => {
    for (const revision of ["a/b", "..", "%2e", ""]) {
      expect(() => validateRevision(revision)).toThrow(HuggingFaceError);
    }
    expect(validateRevision()).toBe("main");
  });

  test("selects the real fixture, synthesizes metadata, and verifies the model", async () => {
    const bytes = new Uint8Array([11, 22, 33, 44, 55, 66]);
    const { sha256 } = await import("hash-wasm");
    const expectedSha256 = await sha256(bytes);
    const tree = await fixtureTree();
    const modelEntry = tree.find((entry) => entry.path === MODEL_FILE);
    if (modelEntry === undefined) {
      throw new Error("fixture must contain the model file");
    }
    expect(modelEntry.size).toBe(91_726_912);
    expect(modelEntry.lfs?.size).toBe(91_726_912);
    expect(modelEntry.lfs?.oid).toBe(
      "f68203bfb98b1b1e8c64fac75fab10c4e36acac081609573b4df0fcc19c90dd9",
    );
    modelEntry.size = bytes.byteLength;
    modelEntry.lfs = {
      oid: expectedSha256,
      size: bytes.byteLength,
      pointerSize: 134,
    };
    const requests: URL[] = [];
    const requestCredentials: string[] = [];
    const restoreFetch = installFetch(async (url, _init, request) => {
      requests.push(url);
      requestCredentials.push(request.credentials);
      if (url.pathname.includes("/api/models/")) {
        return jsonResponse(tree);
      }
      return new Response(bytes);
    });
    try {
      const options = normalizeHuggingFaceLoadOptions(
        { accelerator: "wasm", wasmPath: "/llm-runtime" },
        "https://app.test/",
        "/xybrid/llm-runtime",
      );
      const resolution = await resolveHuggingFaceModel(REPO, "gguf", options);
      expect(resolution.sizeBytes).toBe(bytes.byteLength);
      expect(resolution.sha256).toBe(expectedSha256);
      expect(resolution.metadata.modelId).toBe("smollm2-135m-instruct-gguf");
      expect(resolution.metadata.version).toBe("main");
      expect(resolution.metadata.template).toEqual({
        type: "Gguf",
        modelFile: MODEL_FILE,
        contextLength: undefined,
        chatTemplate: undefined,
        generationParams: undefined,
      });

      const control = createLlmRuntime();
      const session = await loadLlmFromResolution(
        resolution,
        options,
        control.runtime,
        new RuntimeInitializer(),
      );
      expect(control.initialized).toEqual(["https://app.test/llm-runtime"]);
      expect(control.created).toEqual([{ accelerator: "wasm", contextLength: undefined }]);
      expect(requests.map((request) => request.href)).toEqual([
        `${HF_BASE}/api/models/${REPO}/tree/main`,
        `${HF_BASE}/${REPO}/resolve/main/${MODEL_FILE}`,
      ]);
      expect(requestCredentials).toEqual(["omit", "omit"]);
      await session.dispose();
    } finally {
      restoreFetch();
    }
  });

  test("rejects ambiguous files and honors options.file", async () => {
    const tree = [fileEntry("first.gguf"), fileEntry("second.gguf")];
    const restoreFetch = installFetch(async () => jsonResponse(tree));
    try {
      await expect(resolveHuggingFaceModel("org/model", "gguf")).rejects.toThrow(
        /first\.gguf.*second\.gguf.*options\.file/,
      );
      const selected = await resolveHuggingFaceModel("org/model", "gguf", {
        file: "second.gguf",
      });
      expect(selected.modelUrl.pathname).toBe("/org/model/resolve/main/second.gguf");
      await expect(
        resolveHuggingFaceModel("org/model", "gguf", { file: "missing.gguf" }),
      ).rejects.toThrow(HuggingFaceError);
      await expect(
        resolveHuggingFaceModel("org/model", "gguf", { file: "second.litertlm" }),
      ).rejects.toThrow(HuggingFaceError);
    } finally {
      restoreFetch();
    }
  });

  test("rejects repositories containing only legacy browser formats", async () => {
    const tree = [fileEntry("model.tflite"), fileEntry("model.litertlm")];
    const restoreFetch = installFetch(async () => jsonResponse(tree));
    try {
      await expect(resolveHuggingFaceModel(REPO, "gguf")).rejects.toThrow(/no \.gguf file in repo/);
    } finally {
      restoreFetch();
    }
  });

  test("ignores directories and nested paths", async () => {
    const tree: TreeEntry[] = [
      { type: "directory", path: "nested.gguf" },
      { type: "file", path: "sub/nested.gguf", size: 5 },
      fileEntry("top.gguf", 5),
    ];
    const restoreFetch = installFetch(async () => jsonResponse(tree));
    try {
      const resolution = await resolveHuggingFaceModel("org/model", "gguf");
      expect(resolution.modelUrl.pathname).toBe("/org/model/resolve/main/top.gguf");
    } finally {
      restoreFetch();
    }
  });

  test("loads metadata from the repository and validates its model file", async () => {
    const tree = [fileEntry("actual.gguf"), fileEntry("model_metadata.json")];
    const requests: URL[] = [];
    const requestCredentials: string[] = [];
    const restoreFetch = installFetch(async (url, _init, request) => {
      requests.push(url);
      requestCredentials.push(request.credentials);
      if (url.pathname.endsWith("model_metadata.json")) {
        return jsonResponse(metadata("actual.gguf"));
      }
      return jsonResponse(tree);
    });
    try {
      const resolution = await resolveHuggingFaceModel("org/model", "gguf");
      expect(resolution.metadata.modelId).toBe("metadata-model");
      expect(resolution.modelUrl.pathname).toBe("/org/model/resolve/main/actual.gguf");
      expect(requests.map((request) => request.pathname)).toEqual([
        "/api/models/org/model/tree/main",
        "/org/model/resolve/main/model_metadata.json",
      ]);
      expect(requestCredentials).toEqual(["omit", "omit"]);
    } finally {
      restoreFetch();
    }
  });

  test("rejects missing and conflicting metadata model files", async () => {
    const restoreMissing = installFetch(async (url) =>
      url.pathname.endsWith("model_metadata.json")
        ? jsonResponse(metadata("missing.gguf"))
        : jsonResponse([fileEntry("model_metadata.json"), fileEntry("actual.gguf")]),
    );
    try {
      await expect(resolveHuggingFaceModel("org/model", "gguf")).rejects.toThrow(
        /missing\.gguf.*not found/,
      );
    } finally {
      restoreMissing();
    }

    const restoreConflict = installFetch(async (url) =>
      url.pathname.endsWith("model_metadata.json")
        ? jsonResponse(metadata("actual.gguf"))
        : jsonResponse([
            fileEntry("model_metadata.json"),
            fileEntry("actual.gguf"),
            fileEntry("other.gguf"),
          ]),
    );
    try {
      await expect(
        resolveHuggingFaceModel("org/model", "gguf", { file: "other.gguf" }),
      ).rejects.toThrow(/conflicts.*actual\.gguf/);
    } finally {
      restoreConflict();
    }
  });

  test("verifies non-LFS files by size only", async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const resolutionTree = [fileEntry("model.gguf", bytes.byteLength)];
    const restoreTree = installFetch(async () => jsonResponse(resolutionTree));
    let resolution: Awaited<ReturnType<typeof resolveHuggingFaceModel>> | undefined;
    try {
      resolution = await resolveHuggingFaceModel("org/model", "gguf");
    } finally {
      restoreTree();
    }
    if (resolution === undefined) {
      throw new Error("HuggingFace resolution should have completed");
    }
    expect(resolution.sha256).toBeUndefined();

    let restoreDownload = installFetch(async () => new Response(bytes));
    try {
      const chunks = await downloadVerifiedModel(resolution.modelUrl, resolution);
      expect(chunks.reduce((total, chunk) => total + chunk.byteLength, 0)).toBe(bytes.byteLength);
    } finally {
      restoreDownload();
    }

    restoreDownload = installFetch(async () => new Response(new Uint8Array([1, 2])));
    try {
      await expect(downloadVerifiedModel(resolution.modelUrl, resolution)).rejects.toBeInstanceOf(
        IntegrityError,
      );
    } finally {
      restoreDownload();
    }
  });

  test("rejects oversized files before a model download", async () => {
    let modelDownloads = 0;
    const restoreFetch = installFetch(async (url) => {
      if (!url.pathname.includes("/api/models/")) {
        modelDownloads += 1;
      }
      return jsonResponse([fileEntry("model.gguf", 512 * 1024 * 1024 + 1)]);
    });
    try {
      await expect(resolveHuggingFaceModel("org/model", "gguf")).rejects.toThrow(HuggingFaceError);
      expect(modelDownloads).toBe(0);
    } finally {
      restoreFetch();
    }
  });

  test("turns a tree 404 into a repo and revision-specific error", async () => {
    const restoreFetch = installFetch(async () => new Response("missing", { status: 404 }));
    try {
      await expect(
        resolveHuggingFaceModel("org/model", "gguf", { revision: "v1" }),
      ).rejects.toThrow(/org\/model.*v1/);
    } finally {
      restoreFetch();
    }
  });

  test("aborts an in-flight tree request", async () => {
    const controller = new AbortController();
    let started = false;
    const restoreFetch = installFetch(async (_url, init) => {
      started = true;
      const signal = init?.signal ?? controller.signal;
      return new Promise<Response>((_resolve, reject) => {
        signal?.addEventListener(
          "abort",
          () =>
            reject(signal.reason ?? new DOMException("The operation was aborted.", "AbortError")),
          { once: true },
        );
      });
    });
    try {
      const pending = resolveHuggingFaceModel("org/model", "gguf", {
        signal: controller.signal,
      });
      await Bun.sleep(0);
      expect(started).toBe(true);
      controller.abort();
      await expect(pending).rejects.toBeDefined();
    } finally {
      controller.abort();
      restoreFetch();
    }
  });

  test("passes the caller signal to an in-flight metadata request", async () => {
    const controller = new AbortController();
    const tree = [fileEntry("model.gguf"), fileEntry("model_metadata.json")];
    let metadataSignal: AbortSignal | undefined;
    let resolveMetadataStarted: (() => void) | undefined;
    const metadataStarted = new Promise<void>((resolve) => {
      resolveMetadataStarted = resolve;
    });
    const restoreFetch = installFetch(async (url, init) => {
      if (url.pathname.includes("/api/models/")) {
        return jsonResponse(tree);
      }
      metadataSignal = init?.signal ?? undefined;
      resolveMetadataStarted?.();
      return new Promise<Response>((_resolve, reject) => {
        metadataSignal?.addEventListener(
          "abort",
          () =>
            reject(
              metadataSignal?.reason ??
                new DOMException("The operation was aborted.", "AbortError"),
            ),
          { once: true },
        );
      });
    });
    try {
      const pending = resolveHuggingFaceModel("org/model", "gguf", {
        signal: controller.signal,
      });
      await metadataStarted;
      expect(metadataSignal).toBeDefined();
      controller.abort();
      await expect(pending).rejects.toMatchObject({ name: "AbortError" });
      expect(metadataSignal?.aborted).toBe(true);
    } finally {
      controller.abort();
      restoreFetch();
    }
  });
});
