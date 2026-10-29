import { defineConfig } from "tsdown";

export default defineConfig({
  clean: true,
  dts: true,
  entry: { index: "src/index.ts", worker: "src/worker.ts" },
  noExternal: ["ky", "zod", "hash-wasm"],
  format: ["esm"],
  platform: "browser",
  minify: true,
  sourcemap: true,
});
