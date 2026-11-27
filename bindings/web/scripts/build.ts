import { spawnSync } from "node:child_process";
import { copyFile, cp, readFile, writeFile } from "node:fs/promises";
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
await copyFile(
  new URL("../../../LICENSE", import.meta.url),
  new URL("../LICENSE", import.meta.url),
);
// The npm archive has no crates/ directory. Carry the complete native license
// texts with the package instead of leaving NOTICE's source reference dangling.
const packageNotice = await Promise.all(
  ["../../../NOTICE", "../../../crates/llama-cpp-sys/NOTICE"].map((path) =>
    readFile(new URL(path, import.meta.url), "utf8"),
  ),
);
await writeFile(new URL("../NOTICE", import.meta.url), packageNotice.join("\n\n"));
const licenses = [
  ["llama.cpp", "../../../vendor/llama-cpp/LICENSE"],
  ["JSON for Modern C++", "../../../vendor/llama-cpp/licenses/LICENSE-jsonhpp"],
  ["ky", "../node_modules/ky/license"],
  ["zod", "../node_modules/zod/LICENSE"],
  ["hash-wasm", "../node_modules/hash-wasm/LICENSE"],
] as const;
const notices = await Promise.all(
  licenses.map(
    async ([name, path]) => `${name}\n${await readFile(new URL(path, import.meta.url), "utf8")}`,
  ),
);
await writeFile(new URL("../dist/THIRD_PARTY_LICENSES.txt", import.meta.url), notices.join("\n\n"));
console.log("Built @xybrid/web with its worker and CPU/WebGPU Rust runtimes.");
