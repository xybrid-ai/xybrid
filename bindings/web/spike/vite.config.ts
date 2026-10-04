import { defineConfig } from "vite";

export default defineConfig({
  root: import.meta.dirname,
  server: { port: 4174 },
  build: { outDir: "dist", emptyOutDir: true },
  worker: { format: "es" },
});
