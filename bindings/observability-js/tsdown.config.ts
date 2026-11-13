import { defineConfig } from "tsdown";

// The core and /otlp entries stay unbundled against OpenTelemetry: applications share one
// copy of @opentelemetry/api (it holds the global registry) and pick the SDK packages.
export default defineConfig({
  clean: true,
  dts: true,
  entry: { index: "src/index.ts", otlp: "src/otlp/index.ts" },
  format: ["esm"],
  platform: "neutral",
  sourcemap: true,
});
