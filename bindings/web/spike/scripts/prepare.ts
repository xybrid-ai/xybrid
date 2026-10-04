import { spawnSync } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import ky from "ky";
import { MODEL } from "../model.ts";

const repository = fileURLToPath(new URL("../../../../", import.meta.url));
const publicDirectory = new URL("../public/", import.meta.url);
const backends = process.argv.includes("--webgpu") ? ["wasm", "webgpu"] : ["wasm"];
await mkdir(publicDirectory, { recursive: true });
for (const backend of backends) {
  const args = ["build", "-c", "opt", "--jobs=6", "//bindings/web/spike:runtime"];
  if (backend === "webgpu") args.push("--//bindings/web/spike:webgpu=true");
  const result = spawnSync("bazel", args, { cwd: repository, stdio: "inherit" });
  if (result.error !== undefined) throw result.error;
  if (result.status !== 0) throw new Error(`Bazel ${backend} build failed (${result.status}).`);
  const target = new URL(`runtime/${backend}/`, publicDirectory);
  await mkdir(target, { recursive: true });
  for (const filename of ["xybrid_spike.js", "xybrid_spike.wasm"]) {
    const source = Bun.file(`${repository}bazel-bin/bindings/web/spike/runtime/${filename}`);
    if (!(await source.exists())) throw new Error(`Missing Bazel output ${filename}.`);
    await Bun.write(new URL(filename, target), source);
  }
}

const verified = (bytes: Uint8Array): boolean =>
  bytes.byteLength === MODEL.bytes &&
  new Bun.CryptoHasher("sha256").update(bytes).digest("hex") === MODEL.sha256;
const modelFile = Bun.file(new URL("model.gguf", publicDirectory));
if (!(await modelFile.exists()) || !verified(new Uint8Array(await modelFile.arrayBuffer()))) {
  console.log(`Downloading ${MODEL.name} (${MODEL.bytes} bytes)...`);
  const bytes = new Uint8Array(
    await ky
      .get(MODEL.url, {
        timeout: 600_000,
        retry: { limit: 5, delay: (attempt) => 2 ** attempt * 1000, maxRetryAfter: 60_000 },
      })
      .arrayBuffer(),
  );
  if (!verified(bytes)) throw new Error("GGUF size or SHA-256 did not match the pinned artifact.");
  await Bun.write(modelFile, bytes);
}
console.log(`Prepared ${backends.join(" + ")} runtime and verified ${MODEL.name}.`);
