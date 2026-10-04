import { spawnSync } from "node:child_process";
import { cp } from "node:fs/promises";
import { buildRuntime } from "./build-runtime.ts";

await buildRuntime();
const bundle = spawnSync("pnpm", ["exec", "tsdown"], { stdio: "inherit" });
if (bundle.error !== undefined) throw bundle.error;
if (bundle.status !== 0) throw new Error(`Web SDK bundle failed (${bundle.status}).`);
await cp(
  new URL("../runtime/artifacts/", import.meta.url),
  new URL("../dist/runtime/", import.meta.url),
  {
    recursive: true,
  },
);
console.log("Built @xybrid/web with its worker and CPU/WebGPU Rust runtimes.");
