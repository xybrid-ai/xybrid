import { MODEL } from "./model.ts";
import type { Accelerator, LoadMetrics, Memory, Request, Response } from "./protocol.ts";

type Runtime = {
  HEAPU8: Uint8Array;
  FS: { writeFile(path: string, bytes: Uint8Array): void; unlink(path: string): void };
  UTF8ToString(pointer: number): string;
  stringToNewUTF8(text: string): number;
  ccall<T extends "number" | null>(
    name: string,
    returnType: T,
    argumentTypes: readonly string[],
    arguments_: readonly number[],
    options: { async: true },
  ): Promise<T extends "number" ? number : void>;
  _xybrid_web_create(): number;
  _xybrid_web_error(handle: number): number;
  _xybrid_web_version(): number;
  _xybrid_web_prompt_tokens(handle: number): number;
  _xybrid_web_generated_tokens(handle: number): number;
  _xybrid_web_allocated_bytes(): number;
  _free(pointer: number): void;
  xybridCancelled: boolean;
  xybridToken(tokenId: number, text: string): void;
};
type Adapter = {
  features: { has(feature: string): boolean };
  info?: { vendor?: string; architecture?: string; device?: string; description?: string };
};
type GpuNavigator = { gpu?: { requestAdapter(): Promise<Adapter | null> } };
const scope = globalThis as unknown as {
  onmessage: ((event: MessageEvent<Request>) => void) | null;
  postMessage(message: Response): void;
};
const send = (message: Response): void => scope.postMessage(message);
let runtime: Runtime | undefined;
let handle = 0;
let busy = false;

const memory = (): Memory => ({
  heapBytes: runtime?.HEAPU8.byteLength ?? 0,
  allocatedBytes: runtime?._xybrid_web_allocated_bytes() ?? 0,
});

const requireRuntime = (): Runtime => {
  if (runtime === undefined || handle === 0) throw new Error("Load a model first.");
  return runtime;
};

const destroy = async (): Promise<void> => {
  const module = runtime;
  const ownedHandle = handle;
  handle = 0;
  if (module === undefined || ownedHandle === 0) return;
  await module.ccall("xybrid_web_destroy", null, ["number"], [ownedHandle], { async: true });
};

const webGpuAdapter = async (): Promise<Adapter | null> => {
  const gpu = (navigator as unknown as GpuNavigator).gpu;
  const adapter = await gpu?.requestAdapter();
  return adapter?.features.has("shader-f16") ? adapter : null;
};

const download = async (): Promise<Uint8Array> => {
  const response = await fetch("/model.gguf", { credentials: "omit" });
  if (!response.ok || response.body === null) throw new Error(`Model download: ${response.status}`);
  const bytes = new Uint8Array(MODEL.bytes);
  const reader = response.body.getReader();
  let offset = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value.byteLength > bytes.byteLength - offset) throw new Error("Oversized GGUF download.");
      bytes.set(value, offset);
      offset += value.byteLength;
      send({ type: "progress", loadedBytes: offset, totalBytes: MODEL.bytes });
    }
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
  if (offset !== MODEL.bytes) throw new Error("Truncated GGUF download.");
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  const digest = Array.from(hash, (byte) => byte.toString(16).padStart(2, "0")).join("");
  if (digest !== MODEL.sha256) throw new Error("GGUF checksum mismatch.");
  return bytes;
};

const load = async (preference: Accelerator | "auto"): Promise<LoadMetrics> => {
  if (handle !== 0)
    throw new Error("The spike worker owns one model; dispose it before reloading.");
  const adapter = preference === "wasm" ? null : await webGpuAdapter();
  if (preference === "webgpu" && adapter === null) {
    throw new Error("WebGPU with shader-f16 is unavailable in this browser.");
  }
  const accelerator = adapter === null ? "wasm" : "webgpu";
  const downloadStart = performance.now();
  const bytes = await download();
  const downloadMs = performance.now() - downloadStart;
  const initializationStart = performance.now();
  const moduleUrl = new URL(`/runtime/${accelerator}/xybrid_spike.js`, location.origin).href;
  const { default: createModule } = (await import(/* @vite-ignore */ moduleUrl)) as {
    default: (options: Record<string, unknown>) => Promise<Runtime>;
  };
  runtime = await createModule({
    locateFile: (filename: string) => new URL(filename, moduleUrl).href,
    print: (text: string) => console.log(`[llama.cpp] ${text}`),
    printErr: (text: string) => console.log(`[llama.cpp] ${text}`),
  });
  runtime.xybridCancelled = false;
  runtime.xybridToken = () => {};
  const initializationMs = performance.now() - initializationStart;
  handle = runtime._xybrid_web_create();
  if (handle === 0) throw new Error("Rust engine allocation failed.");
  const loadStart = performance.now();
  runtime.FS.writeFile("/model.gguf", bytes);
  const path = runtime.stringToNewUTF8("/model.gguf");
  try {
    const status = await runtime.ccall(
      "xybrid_web_load",
      "number",
      ["number", "number", "number"],
      [handle, path, accelerator === "webgpu" ? 99 : 0],
      { async: true },
    );
    if (status !== 0) throw new Error(runtime.UTF8ToString(runtime._xybrid_web_error(handle)));
  } catch (error) {
    await destroy();
    throw error;
  } finally {
    runtime._free(path);
    // llama.cpp has consumed the file; release the separate MEMFS copy.
    runtime.FS.unlink("/model.gguf");
  }
  return {
    accelerator,
    rustVersion: runtime.UTF8ToString(runtime._xybrid_web_version()),
    adapter:
      adapter === null
        ? null
        : [
            adapter.info?.vendor,
            adapter.info?.architecture,
            adapter.info?.device,
            adapter.info?.description,
          ]
            .filter(Boolean)
            .join(" / "),
    modelBytes: MODEL.bytes,
    downloadMs,
    initializationMs,
    loadMs: performance.now() - loadStart,
    memory: memory(),
  };
};

const generate = async (request: Extract<Request, { type: "generate" }>): Promise<void> => {
  const module = requireRuntime();
  module.xybridCancelled = false;
  const started = performance.now();
  let firstTokenAt: number | null = null;
  let lastTokenAt: number | null = null;
  const peakMemory = memory();
  module.xybridToken = (tokenId, text) => {
    const now = performance.now();
    firstTokenAt ??= now;
    lastTokenAt = now;
    const current = memory();
    peakMemory.heapBytes = Math.max(peakMemory.heapBytes, current.heapBytes);
    peakMemory.allocatedBytes = Math.max(peakMemory.allocatedBytes, current.allocatedBytes);
    send({ type: "token", id: request.id, tokenId, text });
  };
  const prompt = module.stringToNewUTF8(request.prompt);
  let status: number;
  try {
    status = await module.ccall(
      "xybrid_web_generate",
      "number",
      ["number", "number", "number"],
      [handle, prompt, request.maxTokens],
      { async: true },
    );
    if (status < 0) throw new Error(module.UTF8ToString(module._xybrid_web_error(handle)));
  } finally {
    module._free(prompt);
    module.xybridToken = () => {};
  }
  const generatedTokens = module._xybrid_web_generated_tokens(handle);
  const decodeMs = firstTokenAt === null || lastTokenAt === null ? 0 : lastTokenAt - firstTokenAt;
  send({
    type: "complete",
    id: request.id,
    metrics: {
      cancelled: status === 1,
      promptTokens: module._xybrid_web_prompt_tokens(handle),
      generatedTokens,
      firstTokenMs: firstTokenAt === null ? null : firstTokenAt - started,
      totalMs: performance.now() - started,
      decodeTokensPerSecond:
        generatedTokens > 1 && decodeMs > 0 ? ((generatedTokens - 1) * 1000) / decodeMs : null,
      peakMemory,
      memory: memory(),
    },
  });
};

scope.onmessage = (event) => {
  const request = event.data;
  if (request.type === "cancel") {
    if (runtime !== undefined) runtime.xybridCancelled = true;
    return;
  }
  if (busy) {
    send({ type: "error", id: request.id, message: "Another runtime operation is in flight." });
    return;
  }
  busy = true;
  void (async () => {
    try {
      switch (request.type) {
        case "load":
          send({ type: "loaded", id: request.id, metrics: await load(request.accelerator) });
          break;
        case "generate":
          await generate(request);
          break;
        case "dispose":
          await destroy();
          send({ type: "disposed", id: request.id, memory: memory() });
          break;
      }
    } catch (error) {
      console.error(error);
      send({
        type: "error",
        id: request.id,
        message: error instanceof Error ? error.message : String(error),
      });
    } finally {
      busy = false;
    }
  })();
};
