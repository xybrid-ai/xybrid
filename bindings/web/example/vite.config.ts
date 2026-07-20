import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  root: import.meta.dirname,
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: fileURLToPath(new URL("index.html", import.meta.url)),
        tensor: fileURLToPath(new URL("tensor.html", import.meta.url)),
      },
    },
  },
});
