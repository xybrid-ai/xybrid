// Packaging regressions use tiny valid WASM modules, not runtime builds.
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { chmod, cp, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { before, test } from "node:test";
import { fileURLToPath } from "node:url";

const web = fileURLToPath(new URL("../", import.meta.url));
const observability = fileURLToPath(new URL("../../observability-js/", import.meta.url));
const manifest = JSON.parse(await readFile(join(web, "package.json"), "utf8"));
const dependencyManifest = JSON.parse(await readFile(join(observability, "package.json"), "utf8"));
let dependencyArchive;
before(() => {
  execFileSync(process.execPath, [join(observability, "scripts/check-pack.mjs")], {
    cwd: observability,
    stdio: "pipe",
  });
  dependencyArchive = join(
    observability,
    `test-results/npm/xybrid-observability-${dependencyManifest.version}.tgz`,
  );
});

async function fixture(t, version = manifest.version) {
  const directory = await mkdtemp(join(tmpdir(), "xybrid-release-regression-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const root = join(directory, "web");
  const sibling = join(directory, "observability-js");
  await mkdir(join(root, "scripts"), { recursive: true });
  await mkdir(sibling);
  await writeFile(join(sibling, "package.json"), JSON.stringify(dependencyManifest));
  for (const script of ["check-pack.mjs", "prepare-observability.mjs"]) {
    await cp(join(web, "scripts", script), join(root, "scripts", script));
  }
  await symlink(join(web, "node_modules"), join(root, "node_modules"), "dir");
  await writeFile(
    join(root, "package.json"),
    JSON.stringify({ ...manifest, version, dependencies: { "@xybrid/observability": version } }),
  );
  for (const name of ["README.md", "LICENSE", "NOTICE"]) await writeFile(join(root, name), name);
  await mkdir(join(root, "dist"));
  await writeFile(
    join(root, "dist/index.js"),
    "export class XybridLlm { static fromUrl() {} } export class XybridError extends Error {}",
  );
  await writeFile(join(root, "dist/worker.js"), "export {};");
  await writeFile(join(root, "dist/THIRD_PARTY_LICENSES.txt"), "fixture");
  await writeFile(
    join(root, "dist/index.d.ts"),
    `import type { Observer, ObservationRef } from "@xybrid/observability";
export interface GgufLoadOptions { accelerator: "wasm"; wasmPath: string; observer: Observer; }
export interface RunMetrics { observation?: ObservationRef; }
export declare class XybridLlm {
  static fromUrl(url: string, options: GgufLoadOptions): Promise<XybridLlm>;
  generateStream(input: string, options: { maxOutputTokens: number; observation: { conversationId: string; trace: string } }): AsyncIterable<string>;
  lastRun?: RunMetrics;
  cancel(): Promise<void>;
  dispose(): Promise<void>;
}`,
  );
  for (const backend of ["wasm", "webgpu"]) {
    const path = join(root, "dist/runtime", backend);
    await mkdir(path, { recursive: true });
    await writeFile(join(path, "xybrid_runtime.js"), "export {};");
    await writeFile(join(path, "xybrid_runtime.wasm"), Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]));
  }
  return { directory, root, sibling, archive: pack(root, directory) };
}

function pack(root, directory) {
  const [report] = JSON.parse(
    execFileSync("npm", ["pack", "--ignore-scripts", "--json", "--pack-destination", directory], {
      cwd: root,
      encoding: "utf8",
    }),
  );
  return join(directory, report.filename);
}

function verify(root, args, env = {}) {
  return spawnSync(process.execPath, [join(root, "scripts/check-pack.mjs"), ...args], {
    cwd: root,
    encoding: "utf8",
    env: { ...process.env, ...env },
  });
}

function passed(result) {
  assert.equal(result.status, 0, result.stdout + result.stderr);
}

test("downloaded archives verify with no sibling install or build", async (t) => {
  const { root, archive } = await fixture(t);
  passed(verify(root, [archive, manifest.version, dependencyArchive]));
});

test("downloaded dependency must match the packed web version, not the checkout", async (t) => {
  const { root, archive } = await fixture(t, "0.0.1-release-test");
  const result = verify(root, [archive, "0.0.1-release-test", dependencyArchive]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Observability archive version differs/);
});

test("the packed web dependency cannot drift behind its package version", async (t) => {
  const { root, directory } = await fixture(t);
  const path = join(root, "package.json");
  const source = JSON.parse(await readFile(path, "utf8"));
  source.dependencies["@xybrid/observability"] = "0.0.1";
  await writeFile(path, JSON.stringify(source));
  const result = verify(root, [pack(root, directory), manifest.version, dependencyArchive]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /must depend on the same released/);
});

test("direct source packing prepares a clean sibling without lifecycle hooks", async (t) => {
  const { root, directory, sibling } = await fixture(t);
  const template = join(directory, "template");
  await mkdir(template);
  execFileSync("tar", ["-xzf", dependencyArchive, "-C", template]);
  await mkdir(join(sibling, "src"));
  await writeFile(join(sibling, "src/version.ts"), "// source fixture");
  const bin = join(directory, "bin");
  await mkdir(bin);
  const calls = join(directory, "calls.jsonl");
  const shim = join(bin, "pnpm");
  await writeFile(
    shim,
    `#!/usr/bin/env node
const { appendFileSync, cpSync, mkdirSync, writeFileSync } = require("node:fs");
const { join } = require("node:path");
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(calls)}, JSON.stringify(args) + "\\n");
if (args[0] === "install") {
  mkdirSync("node_modules"); writeFileSync("node_modules/.modules.yaml", "fixture");
} else if (args.join(" ") === "run build") {
  for (const name of ["dist", "LICENSE", "NOTICE"]) cpSync(join(${JSON.stringify(template)}, "package", name), name, { recursive: true });
} else { throw new Error("Unexpected pnpm command " + args); }
`,
  );
  await chmod(shim, 0o755);
  passed(
    verify(root, [], {
      PATH: `${bin}:${process.env.PATH}`,
      XYBRID_OBSERVABILITY_ARCHIVE: undefined,
    }),
  );
  assert.deepEqual((await readFile(calls, "utf8")).trim().split("\n").map(JSON.parse), [
    ["install", "--frozen-lockfile"],
    ["run", "build"],
  ]);
});
