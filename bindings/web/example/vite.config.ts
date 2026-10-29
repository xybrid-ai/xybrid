import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const packedPackage = process.env["XYBRID_WEB_PACKAGE"];
const sdkEntry =
  packedPackage === undefined
    ? fileURLToPath(new URL("../dist/index.js", import.meta.url))
    : resolve(fileURLToPath(new URL("../", import.meta.url)), packedPackage, "dist/index.js");

export default defineConfig({
  root: import.meta.dirname,
  resolve: {
    alias: { "@xybrid/web": sdkEntry },
  },
  worker: { format: "es" },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: fileURLToPath(new URL("index.html", import.meta.url)),
      },
    },
  },
});
