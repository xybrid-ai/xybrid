import { expect, test } from "@playwright/test";

for (const accelerator of ["wasm", "auto"] as const) {
  test(`runs the pinned LiteRT model through ${accelerator}`, async ({ page }) => {
    await page.goto("/tensor.html");
    await page.selectOption("#accelerator", accelerator);
    await page.click("#run-button");

    const status = page.locator("#status");
    await expect(status.locator("strong")).toHaveText("PASS");
    await expect(status).toContainText("100 float32 values matched a+b");
    await expect(status).toContainText(/(?:wasm|webgpu) compile path/);
    await expect(page.locator("#run-button")).toBeEnabled();
  });
}
