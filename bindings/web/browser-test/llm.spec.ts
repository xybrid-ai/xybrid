import { expect, test } from "@playwright/test";

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
}
