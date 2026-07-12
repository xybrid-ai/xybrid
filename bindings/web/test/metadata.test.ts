import { describe, expect, test } from "bun:test";
import {
  InvalidMetadataError,
  UnsupportedFeatureError,
  UnsupportedTemplateError,
  XybridError,
} from "../src/errors.ts";
import { readResponseBytes, readResponseChunks } from "../src/internal/response.ts";
import { resolveMetadataUrl, resolveWasmPath } from "../src/internal/url.ts";
import {
  parseMetadata,
  resolveModelUrl,
  validateBrowserMetadata,
  validateLlmBrowserMetadata,
} from "../src/metadata.ts";
import { createRuntime, loadWithDependencies, tfliteMetadata } from "./helpers.ts";

const metadataUrl = new URL("https://models.example/add/model_metadata.json");

const metadataWithContextLength = (
  template: "TfLite" | "LiteRtLm",
  contextLength?: unknown,
): Record<string, unknown> => ({
  ...tfliteMetadata(),
  execution_template: {
    type: template,
    model_file: "model.tflite",
    ...(contextLength === undefined ? {} : { context_length: contextLength }),
  },
});

describe("metadata boundary", () => {
  test("parses existing metadata with future fields and routes non-TfLite to a typed error", async () => {
    const fixture = await Bun.file(
      "../../integration-tests/fixtures/models/mnist/model_metadata.json",
    ).json();
    const parsed = parseMetadata({ ...fixture, future_browser_hint: { retained: true } });
    expect(parsed.modelId).toBe("mnist-digit-recognition");

    const runtime = createRuntime();
    await expect(
      loadWithDependencies(
        metadataUrl,
        { wasmPath: "/wasm", accelerator: "wasm" },
        runtime.runtime,
        async () => parsed,
      ),
    ).rejects.toBeInstanceOf(UnsupportedTemplateError);
  });

  test("leaves TfLite context_length unvalidated", () => {
    for (const contextLength of [2_097_152, 0, 4096.5]) {
      const parsed = parseMetadata(metadataWithContextLength("TfLite", contextLength));

      expect(validateBrowserMetadata(parsed)).toBe("model.tflite");
    }
  });

  test("validates context_length only for LiteRtLm", () => {
    for (const contextLength of [2_097_152, 0, 4096.5]) {
      const parsed = parseMetadata(metadataWithContextLength("LiteRtLm", contextLength));

      expect(() => validateLlmBrowserMetadata(parsed)).toThrow(InvalidMetadataError);
    }

    const valid = parseMetadata(metadataWithContextLength("LiteRtLm", 2048));
    expect(validateLlmBrowserMetadata(valid)).toEqual({
      modelFile: "model.tflite",
      contextLength: 2048,
    });

    const withoutContextLength = parseMetadata(metadataWithContextLength("LiteRtLm"));
    expect(validateLlmBrowserMetadata(withoutContextLength).contextLength).toBeUndefined();
  });

  test("stops reading chunked responses at the configured byte limit", async () => {
    let cancelled = false;
    const response = new Response(
      new ReadableStream<Uint8Array>({
        pull(controller) {
          controller.enqueue(new Uint8Array(4));
        },
        cancel() {
          cancelled = true;
        },
      }),
    );
    await expect(readResponseBytes(response, 7, "too large")).rejects.toThrow("too large");
    expect(cancelled).toBe(true);
  });

  test("returns streamed response chunks with progress without concatenating them", async () => {
    const progress: [number, number | undefined][] = [];
    const response = new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new Uint8Array([1, 2]));
          controller.enqueue(new Uint8Array([3]));
          controller.close();
        },
      }),
      { headers: { "content-length": "3" } },
    );

    const chunks = await readResponseChunks(response, 3, "too large", (loaded, total) => {
      progress.push([loaded, total]);
    });

    expect(chunks.map((chunk) => Array.from(chunk))).toEqual([[1, 2], [3]]);
    expect(progress).toEqual([
      [2, 3],
      [3, 3],
    ]);
  });

  test("rejects unsupported processing and attached model surfaces before runtime initialization", async () => {
    for (const metadata of [
      tfliteMetadata({ preprocessing: [{ type: "Normalize" }] }),
      tfliteMetadata({ postprocessing: [{ type: "Softmax" }] }),
      tfliteMetadata({ voices: { format: "embedded" } }),
      tfliteMetadata({ vision_encoder: { file: "vision.gguf" } }),
    ]) {
      const runtime = createRuntime();
      await expect(
        loadWithDependencies(
          metadataUrl,
          { wasmPath: "/wasm", accelerator: "wasm" },
          runtime.runtime,
          async () => parseMetadata(metadata),
        ),
      ).rejects.toBeInstanceOf(UnsupportedFeatureError);
      expect(runtime.initialized).toHaveLength(0);
    }
  });

  test("only resolves listed same-directory model paths", () => {
    for (const modelFile of [
      "../model.tflite",
      "/model.tflite",
      "//host/model.tflite",
      "https://host/model.tflite",
      "%2e%2e%2fmodel.tflite",
      "model.tflite?variant=1",
      "model.tflite#fragment",
      " model.tflite ",
      "model\t.tflite",
      "model\n.tflite",
      "model\r.tflite",
    ]) {
      expect(() => resolveModelUrl(metadataUrl, modelFile, [modelFile])).toThrow(XybridError);
    }
    expect(() => resolveModelUrl(metadataUrl, "model.tflite", ["other.tflite"])).toThrow(
      XybridError,
    );
    expect(() =>
      resolveModelUrl(metadataUrl, "nested/model.tflite", ["nested/model.tflite"]),
    ).toThrow(XybridError);
    expect(resolveModelUrl(metadataUrl, "model.tflite", ["model.tflite"]).href).toBe(
      "https://models.example/add/model.tflite",
    );
  });

  test("accepts browser-relative metadata URLs at the public boundary", () => {
    expect(resolveMetadataUrl("/model_metadata.json", "https://example.test/example/").href).toBe(
      "https://example.test/model_metadata.json",
    );
  });

  test("keeps executable wasm assets on the page origin", () => {
    const page = "https://app.example.test/models/";
    expect(resolveWasmPath("/litert", page).href).toBe("https://app.example.test/litert");
    expect(() => resolveWasmPath("https://cdn.example.test/litert", page)).toThrow(XybridError);
    expect(() => resolveWasmPath("data:text/javascript,alert(1)", page)).toThrow(XybridError);
  });

  test("wraps metadata transport failures in the public typed error hierarchy", async () => {
    const runtime = createRuntime();
    await expect(
      loadWithDependencies(
        metadataUrl,
        { wasmPath: "/wasm", accelerator: "wasm" },
        runtime.runtime,
        async () => {
          throw new DOMException("network unavailable", "NetworkError");
        },
      ),
    ).rejects.toBeInstanceOf(InvalidMetadataError);
    expect(runtime.initialized).toHaveLength(0);
  });
});
