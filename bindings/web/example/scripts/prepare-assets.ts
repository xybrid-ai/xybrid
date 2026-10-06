import { spawnSync } from "node:child_process";
import { cp, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import ky from "ky";
import { MODEL } from "../model.ts";

const packageDirectory = new URL("../../", import.meta.url);
// Release/CI browser tests serve the extracted npm tarball, including its worker.
const packedPackage = process.env["XYBRID_WEB_PACKAGE"];
const sdkDirectory =
  packedPackage === undefined
    ? new URL("dist/", packageDirectory)
    : resolve(fileURLToPath(packageDirectory), packedPackage, "dist");
if (
  packedPackage === undefined &&
  !(await Bun.file(new URL("dist/index.js", packageDirectory)).exists())
) {
  const build = spawnSync("pnpm", ["build"], { cwd: packageDirectory, stdio: "inherit" });
  if (build.error !== undefined) throw build.error;
  if (build.status !== 0) throw new Error(`SDK build failed (${build.status}).`);
}
const publicDirectory = new URL("../public/", import.meta.url);
await mkdir(publicDirectory, { recursive: true });
await cp(sdkDirectory, new URL("sdk/", publicDirectory), { recursive: true });
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
console.log(`Prepared packaged SDK assets and verified ${MODEL.name}.`);
