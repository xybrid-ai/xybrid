import { access, copyFile, open, unlink, writeFile } from "node:fs/promises";
import { arch, cpus, platform } from "node:os";
import { expect, test } from "@playwright/test";
import { SDK_VERSION } from "../../src/version.ts";
import { MODEL } from "../model.ts";

test("streams real GGUF tokens through Rust, cancels, reuses and releases the model", async ({
  page,
  browser,
}, testInfo) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const moduleUrl = "/client.ts";
    const { RustGgufModel } = (await import(moduleUrl)) as typeof import("../client.ts");
    const model = await RustGgufModel.load({ accelerator: "wasm" });
    const runs = [];
    for (let index = 0; index < 3; index += 1) {
      let output = "";
      let deltas = 0;
      for await (const text of model.generateStream(
        "Tell me a short story about a fox who learns to share.",
        { maxTokens: 32 },
      )) {
        output += text;
        deltas += 1;
      }
      runs.push({ output, deltas, metrics: model.lastRun });
    }
    let cancelledDeltas = 0;
    let cancelStarted = 0;
    for await (const _text of model.generateStream("Tell me a long story about a fox.", {
      maxTokens: 256,
    })) {
      cancelledDeltas += 1;
      if (cancelledDeltas === 5) {
        cancelStarted = performance.now();
        break;
      }
    }
    const cancelMs = performance.now() - cancelStarted;
    const cancelled = model.lastRun;
    let resumed = "";
    for await (const text of model.generateStream("Say hello.", { maxTokens: 8 })) resumed += text;
    const reused = model.lastRun;
    const paused = model.generateStream("Tell me another story.", { maxTokens: 256 });
    const first = await paused.next();
    const disposeStarted = performance.now();
    const disposed = await model.dispose();
    const disposeMs = performance.now() - disposeStarted;
    await paused.return(undefined);
    return {
      loaded: model.loaded,
      runs,
      cancelledDeltas,
      cancelled,
      cancelMs,
      resumed,
      reused,
      first,
      disposed,
      disposeMs,
      userAgent: navigator.userAgent,
    };
  });
  expect(result.loaded.rustVersion).toBe(SDK_VERSION);
  expect(result.loaded.accelerator).toBe("wasm");
  expect(result.loaded.modelBytes).toBe(91_726_912);
  expect(result.loaded.memory.allocatedBytes).toBeGreaterThan(50_000_000);
  for (const run of result.runs) {
    expect(run.output.trim().length).toBeGreaterThan(10);
    expect(run.deltas).toBeGreaterThan(1);
    expect(run.metrics?.cancelled).toBe(false);
    expect(run.metrics?.generatedTokens).toBe(run.deltas);
    expect(run.metrics?.firstTokenMs).toBeGreaterThan(0);
    expect(run.metrics?.decodeTokensPerSecond).toBeGreaterThan(0);
  }
  expect(result.cancelledDeltas).toBe(5);
  expect(result.cancelled?.cancelled).toBe(true);
  expect(result.cancelled?.generatedTokens).toBeLessThan(256);
  expect(result.cancelMs).toBeLessThan(5000);
  expect(result.resumed.length).toBeGreaterThan(0);
  expect(result.reused?.cancelled).toBe(false);
  expect(result.first.done).toBe(false);
  expect(result.disposeMs).toBeLessThan(5000);
  expect(result.disposed.allocatedBytes).toBeLessThan(result.loaded.memory.allocatedBytes / 2);
  const processors = cpus();
  const report = {
    model: MODEL,
    host: {
      platform: platform(),
      architecture: arch(),
      cpu: processors[0]?.model,
      logicalCpus: processors.length,
    },
    browserVersion: browser.version(),
    ...result,
  };
  await testInfo.attach("rust-gguf-measurements", {
    body: JSON.stringify(report, null, 2),
    contentType: "application/json",
  });
  const reportPath = process.env["XYBRID_SPIKE_REPORT"];
  if (reportPath !== undefined) await writeFile(reportPath, JSON.stringify(report, null, 2));
});

test("rejects an overlapping generation and remains usable", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const moduleUrl = "/client.ts";
    const { RustGgufModel } = (await import(moduleUrl)) as typeof import("../client.ts");
    const model = await RustGgufModel.load();
    const active = model.generateStream("Tell me a story about a fox.", { maxTokens: 256 });
    await active.next();
    const overlap = model.generateStream("Hello.");
    let failure = "";
    try {
      await overlap.next();
    } catch (error) {
      failure = String(error);
    }
    await active.return(undefined);
    let output = "";
    for await (const text of model.generateStream("Say hello.", { maxTokens: 8 })) output += text;
    await model.dispose();
    return { failure, output };
  });
  expect(result.failure).toContain("already in flight");
  expect(result.output.length).toBeGreaterThan(0);
});

test("rejects a corrupt GGUF before native loading", async ({ page }) => {
  // Serve a real corrupt file over HTTP without forwarding a 92 MB response
  // body through Chromium's debugging protocol.
  const corruptPath = new URL("../public/corrupt-test.gguf", import.meta.url);
  await copyFile(new URL("../public/model.gguf", import.meta.url), corruptPath);
  try {
    const file = await open(corruptPath, "r+");
    try {
      const offset = (await file.stat()).size - 1;
      const byte = new Uint8Array(1);
      await file.read(byte, 0, 1, offset);
      byte[0] = (byte[0] ?? 0) ^ 1;
      await file.write(byte, 0, 1, offset);
    } finally {
      await file.close();
    }
    await page.route("**/model.gguf", async (route) => {
      await route.continue({ url: new URL("/corrupt-test.gguf", route.request().url()).href });
    });
    await page.goto("/");
    await page.click("#load");
    await expect(page.locator("#status")).toHaveAttribute("data-state", "error", {
      timeout: 60_000,
    });
    await expect(page.locator("#status")).toContainText("checksum mismatch");
  } finally {
    await unlink(corruptPath);
  }
});

test("page stays responsive while generating and the Stop button cancels", async ({ page }) => {
  await page.goto("/");
  await page.click("#load");
  await expect(page.locator("#run")).toBeEnabled({ timeout: 60_000 });
  await page.fill("#max-tokens", "256");
  await page.click("#run");
  await expect(page.locator("#output")).not.toBeEmpty({ timeout: 60_000 });
  await page.click("#stop");
  await expect(page.locator("#status")).toHaveAttribute("data-state", "cancelled", {
    timeout: 10_000,
  });
  await page.click("#dispose");
  await expect(page.locator("#status")).toHaveAttribute("data-state", "disposed");
});

test("reports an unavailable WebGPU adapter before downloading the model", async ({ page }) => {
  await page.goto("/");
  const available = await page.evaluate(async () => {
    const gpu = (
      navigator as unknown as {
        gpu?: { requestAdapter(): Promise<{ features: { has(feature: string): boolean } } | null> };
      }
    ).gpu;
    return (await gpu?.requestAdapter())?.features.has("shader-f16") ?? false;
  });
  test.skip(available, "This browser has a shader-f16-capable WebGPU adapter.");
  let downloaded = false;
  page.on("request", (request) => {
    if (request.url().endsWith("/model.gguf")) downloaded = true;
  });
  await page.selectOption("#accelerator", "webgpu");
  await page.click("#load");
  await expect(page.locator("#status")).toHaveAttribute("data-state", "error");
  await expect(page.locator("#status")).toContainText("WebGPU with shader-f16");
  expect(downloaded).toBe(false);
});

test("WebGPU artifact exposes Rust handles and asynchronous calls", async ({ page }, testInfo) => {
  try {
    await access(new URL("../public/runtime/webgpu/xybrid_spike.wasm", import.meta.url));
  } catch {
    test.skip(true, "Prepare the optional WebGPU artifact first.");
    return;
  }
  await page.goto("/");
  const result = await page.evaluate(async () => {
    // Exercise the built GPU module without initializing a GPU backend. An
    // unloaded Engine safely returns a Rust error, even on a software adapter.
    const source = `
      const url = new URL("/runtime/webgpu/xybrid_spike.js", self.location.origin).href;
      const { default: createModule } = await import(url);
      const module = await createModule({ locateFile: file => new URL(file, url).href });
      const handle = module._xybrid_web_create();
      const prompt = module.stringToNewUTF8("Hello.");
      const generation = module.ccall("xybrid_web_generate", "number",
        ["number", "number", "number"], [handle, prompt, 8], { async: true });
      const status = await generation;
      module._free(prompt);
      const error = module.UTF8ToString(module._xybrid_web_error(handle));
      const destruction = module.ccall("xybrid_web_destroy", null,
        ["number"], [handle], { async: true });
      await destruction;
      self.postMessage({
        rustVersion: module.UTF8ToString(module._xybrid_web_version()),
        generationIsPromise: generation instanceof Promise,
        destroyIsPromise: destruction instanceof Promise,
        status,
        error,
      });
    `;
    const url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
    const worker = new Worker(url, { type: "module" });
    try {
      return await new Promise<{
        rustVersion: string;
        generationIsPromise: boolean;
        destroyIsPromise: boolean;
        status: number;
        error: string;
      }>((resolve, reject) => {
        worker.onmessage = (event) => resolve(event.data);
        worker.onerror = (event) => reject(new Error(event.message));
      });
    } finally {
      worker.terminate();
      URL.revokeObjectURL(url);
    }
  });
  expect(result.rustVersion).toBe(SDK_VERSION);
  expect(result.generationIsPromise).toBe(true);
  expect(result.destroyIsPromise).toBe(true);
  expect(result.status).toBe(-1);
  expect(result.error).toContain("model is not loaded");
  await testInfo.attach("webgpu-rust-artifact", {
    body: JSON.stringify(result, null, 2),
    contentType: "application/json",
  });
});

test("WebGPU generates through the Rust wrapper", async ({ page }) => {
  test.skip(
    process.env["XYBRID_SPIKE_WEBGPU"] !== "1",
    "Opt in after preparing WebGPU assets on a shader-f16-capable adapter.",
  );
  await page.goto("/");
  await page.selectOption("#accelerator", "webgpu");
  await page.click("#load");
  await expect(page.locator("#run")).toBeEnabled({ timeout: 120_000 });
  await page.fill("#max-tokens", "16");
  await page.click("#run");
  await expect(page.locator("#status")).toHaveAttribute("data-state", "done", { timeout: 120_000 });
  await expect(page.locator("#output")).not.toBeEmpty();
  const report = JSON.parse((await page.locator("#report").textContent()) ?? "null") as {
    load: { accelerator: string; rustVersion: string };
  };
  expect(report.load.accelerator).toBe("webgpu");
  expect(report.load.rustVersion).toBe(SDK_VERSION);
  await page.click("#dispose");
});
