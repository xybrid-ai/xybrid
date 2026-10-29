import { spawnSync } from "node:child_process";
import { copyFile, mkdir } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repository = fileURLToPath(new URL("../../../", import.meta.url));

/** Compile both backends with the pinned Bazel Rust and Emscripten toolchains. */
export const buildRuntime = async (): Promise<void> => {
  for (const backend of ["wasm", "webgpu"]) {
    const options = ["-c", "opt", "--jobs=6"];
    if (backend === "webgpu") options.push("--//bindings/web/runtime:webgpu=true");
    const target = "//bindings/web/runtime:runtime";
    const build = spawnSync("bazel", ["build", ...options, target], {
      cwd: repository,
      stdio: "inherit",
    });
    if (build.error !== undefined) throw build.error;
    if (build.status !== 0) throw new Error(`Bazel ${backend} build failed (${build.status}).`);
    const query = spawnSync("bazel", ["cquery", ...options, target, "--output=files"], {
      cwd: repository,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "inherit"],
    });
    if (query.error !== undefined) throw query.error;
    if (query.status !== 0) throw new Error(`Bazel ${backend} output resolution failed.`);
    const outputs = query.stdout.trim().split("\n");
    const directory = new URL(`../runtime/artifacts/${backend}/`, import.meta.url);
    await mkdir(directory, { recursive: true });
    for (const name of ["xybrid_runtime.js", "xybrid_runtime.wasm"]) {
      const output = outputs.find((path) => basename(path) === name);
      if (output === undefined) throw new Error(`Missing Bazel output ${name}.`);
      await copyFile(resolve(repository, output), new URL(name, directory));
    }
  }
};

if (import.meta.main) await buildRuntime();
