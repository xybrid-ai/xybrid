import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  root: import.meta.dirname,
  resolve: {
    alias: { "@xybrid/web": fileURLToPath(new URL("../dist/index.js", import.meta.url)) },
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
