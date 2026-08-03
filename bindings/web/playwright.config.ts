import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "browser-test",
  fullyParallel: false,
  workers: 1,
  reporter: "line",
  use: {
    ...devices["Desktop Chrome"],
    baseURL: "http://127.0.0.1:4173",
  },
  webServer: {
    command: "pnpm dev:example --host 127.0.0.1 --port 4173",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: false,
    // The predev asset step downloads the 136 MB LiteRT-LM model on first run.
    timeout: 600_000,
  },
});
