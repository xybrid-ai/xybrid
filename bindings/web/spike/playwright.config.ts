import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "browser-test",
  fullyParallel: false,
  workers: 1,
  reporter: "line",
  timeout: 180_000,
  use: {
    viewport: { width: 1280, height: 720 },
    baseURL: "http://127.0.0.1:4174",
    launchOptions: {
      ...(process.env["CHROME_PATH"] ? { executablePath: process.env["CHROME_PATH"] } : {}),
    },
  },
  webServer: {
    command: "pnpm spike:dev --host 127.0.0.1 --port 4174",
    url: "http://127.0.0.1:4174",
    reuseExistingServer: !process.env["CI"],
  },
});
