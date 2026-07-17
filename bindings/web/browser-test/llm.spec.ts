import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

// The example app's vite root is example/, so the SDK source outside that
// root is only reachable through vite's /@fs/ endpoint.
const sdkModuleUrl = `/@fs${fileURLToPath(new URL("../src/index.ts", import.meta.url))}`;

for (const accelerator of ["wasm", "auto"] as const) {
  test(`generates text from the pinned language model through ${accelerator}`, async ({ page }) => {
    test.setTimeout(300_000);
    await page.goto("/");
    await page.selectOption("#accelerator", accelerator);
    await page.fill("#prompt", "Reply with one short sentence about foxes.");
    await page.click("#run-button");

    const status = page.locator("#status");
    await expect(status.locator("strong")).toHaveText("DONE", { timeout: 240_000 });
    await expect(status).toContainText(/(?:wasm|webgpu) engine/);
    await expect(page.locator("#output-view .output-stream")).not.toBeEmpty();
    await expect(page.locator("#run-button")).toBeEnabled();
  });

  test(`disposes a paused stream through ${accelerator}`, async ({ page }) => {
    test.setTimeout(300_000);
    await page.goto("/");
    await page.selectOption("#accelerator", accelerator);

    await page.evaluate(
      async ({
        moduleUrl,
        selectedAccelerator,
      }: {
        moduleUrl: string;
        selectedAccelerator: "wasm" | "auto";
      }) => {
        const { XybridLlm } = await import(moduleUrl);
        const llm = await XybridLlm.load("/llm/model_metadata.json", {
          wasmPath: "/llm-runtime",
          accelerator: selectedAccelerator,
        });
        const stream = llm.generateStream("Reply with one short sentence about foxes.");
        const first = await stream.next();
        if (first.done) {
          throw new Error("expected the stream to produce a delta");
        }
        await Promise.race([
          llm.dispose(),
          new Promise<never>((_, reject) => {
            setTimeout(() => reject(new Error("dispose timed out")), 500);
          }),
        ]);
      },
      { moduleUrl: sdkModuleUrl, selectedAccelerator: accelerator },
    );
  });
}
