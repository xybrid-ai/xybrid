import { spawnSync } from "node:child_process";
import { copyFile } from "node:fs/promises";

const bundle = spawnSync("pnpm", ["exec", "tsdown"], { stdio: "inherit" });
if (bundle.error !== undefined) throw bundle.error;
if (bundle.status !== 0) throw new Error(`Observability bundle failed (${bundle.status}).`);
for (const file of ["LICENSE", "NOTICE"]) {
  await copyFile(
    new URL(`../../../${file}`, import.meta.url),
    new URL(`../${file}`, import.meta.url),
  );
}
console.log("Built @xybrid/observability (core and /otlp).");
