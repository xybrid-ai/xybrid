import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "browser-test",
  // Playwright clears this directory; preserve the checked npm archive beside it.
  outputDir: "test-results/browser",
  fullyParallel: false,
  workers: 1,
  reporter: "line",
  timeout: 180_000,
  use: {
    ...devices["Desktop Chrome"],
    baseURL: "http://127.0.0.1:4173",
    launchOptions: {
      ...(process.env["CHROME_PATH"] ? { executablePath: process.env["CHROME_PATH"] } : {}),
    },
  },
  webServer: {
    command:
      "pnpm exec vite preview --config example/vite.config.ts --host 127.0.0.1 --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
