// Build first, then pack and verify the actual npm archive in an isolated consumer.
// An archive argument verifies a downloaded release asset without rebuilding it.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import {
  cp,
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  rm,
  stat,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const expected = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
// The release demo can verify another published version without changing the checkout.
const expectedVersion = process.argv[3] ?? expected.version;
const building = process.argv[2] === undefined;
// Direct invocations (including release verification) do not run pnpm pre* hooks.
// Only source packing prepares the sibling; archive verification consumes released bytes.
if (building) {
  execFileSync(process.execPath, [join(root, "scripts/prepare-observability.mjs")], {
    cwd: root,
    stdio: "inherit",
  });
}
const output = join(root, "test-results/npm");
await mkdir(output, { recursive: true });
let archive = process.argv[2];
if (archive === undefined) {
  const [report] = JSON.parse(
    execFileSync("npm", ["pack", "--ignore-scripts", "--json", "--pack-destination", output], {
      cwd: root,
      encoding: "utf8",
    }),
  );
  archive = join(output, report.filename);
}
archive = resolve(archive);
const paths = execFileSync("tar", ["-tzf", archive], { encoding: "utf8" })
  .trim()
  .split("\n")
  .filter((path) => !path.endsWith("/"))
  .map((path) => {
    assert(path.startsWith("package/"), `Unexpected archive entry: ${path}`);
    return path.slice("package/".length);
  });
const required = [
  "package.json",
  "README.md",
  "LICENSE",
  "NOTICE",
  "dist/index.js",
  "dist/index.d.ts",
  "dist/worker.js",
  "dist/THIRD_PARTY_LICENSES.txt",
  ...["wasm", "webgpu"].flatMap((backend) =>
    ["js", "wasm"].map((extension) => `dist/runtime/${backend}/xybrid_runtime.${extension}`),
  ),
];
const allowed =
  /^(package\.json|README\.md|LICENSE|NOTICE|dist\/[^/]+\.(js|js\.map|d\.ts)|dist\/THIRD_PARTY_LICENSES\.txt|dist\/runtime\/(wasm|webgpu)\/xybrid_runtime\.(js|wasm))$/;
for (const path of required) assert(paths.includes(path), `Missing from the npm tarball: ${path}`);
for (const path of paths) assert(allowed.test(path), `Must not ship in the npm tarball: ${path}`);

const temporary = await mkdtemp(join(tmpdir(), "xybrid-web-package-"));
try {
  execFileSync("tar", ["-xzf", archive, "-C", temporary]);
  const packed = join(temporary, "package");
  const manifest = JSON.parse(await readFile(join(packed, "package.json"), "utf8"));
  assert.equal(manifest.name, "@xybrid/web");
  assert.equal(
    manifest.version,
    expectedVersion,
    "Release archive version differs from the expected version",
  );
  assert.notEqual(manifest.private, true, "The web package must be public");
  assert.equal(manifest.publishConfig?.access, "public");
  assert.equal(manifest.license, "Apache-2.0");
  assert.equal(manifest.exports["."].import, "./dist/index.js");
  assert.equal(manifest.exports["."].types, "./dist/index.d.ts");
  assert.equal(manifest.exports["./worker.js"], "./dist/worker.js");
  assert.equal(manifest.exports["./runtime/*"], "./dist/runtime/*");
  // The development link (pnpm-workspace.yaml override) must never reach the registry manifest.
  const observabilityVersion = manifest.dependencies?.["@xybrid/observability"];
  // Older released web archives predate observability; demo:release can still verify them.
  if (observabilityVersion !== undefined || expectedVersion === expected.version) {
    assert.equal(
      observabilityVersion,
      manifest.version,
      "@xybrid/web must depend on the same released @xybrid/observability version",
    );
  }
  for (const [name, range] of Object.entries(manifest.dependencies ?? {})) {
    assert.doesNotMatch(range, /^(link|file|workspace|portal):/, `Unpublishable range for ${name}`);
  }
  for (const path of required)
    assert((await stat(join(packed, path))).size > 0, `Empty asset: ${path}`);
  for (const backend of ["wasm", "webgpu"]) {
    await WebAssembly.compile(
      await readFile(join(packed, `dist/runtime/${backend}/xybrid_runtime.wasm`)),
    );
  }

  // No dependencies are installed beside this copy: the ESM must be self-contained.
  const sdk = await import(pathToFileURL(join(packed, "dist/index.js")).href);
  assert.equal(typeof sdk.XybridLlm.fromUrl, "function");
  assert.equal(typeof sdk.XybridError, "function");
  // The declarations reference @xybrid/observability (types only). Install its packed archive
  // and its @opentelemetry/api peer where Node and TypeScript resolve from the extracted copy.
  const shared = join(temporary, "node_modules");
  await mkdir(join(shared, "@xybrid"), { recursive: true });
  await mkdir(join(shared, "@opentelemetry"), { recursive: true });
  if (observabilityVersion !== undefined) {
    // Release jobs pass the checksummed sibling asset. A downloaded web archive can
    // also be checked against its exact published dependency, without a sibling build.
    let observabilityArchive = process.argv[4] ?? process.env.XYBRID_OBSERVABILITY_ARCHIVE;
    if (observabilityArchive === undefined) {
      const [report] = JSON.parse(
        execFileSync(
          "npm",
          [
            "pack",
            ...(building ? [] : [`@xybrid/observability@${observabilityVersion}`]),
            "--ignore-scripts",
            "--json",
            "--pack-destination",
            output,
          ],
          { cwd: building ? join(root, "../observability-js") : root, encoding: "utf8" },
        ),
      );
      observabilityArchive = join(output, report.filename);
    }
    const observabilityRoot = join(temporary, "observability");
    await mkdir(observabilityRoot);
    execFileSync("tar", ["-xzf", resolve(observabilityArchive), "-C", observabilityRoot]);
    const dependency = JSON.parse(
      await readFile(join(observabilityRoot, "package/package.json"), "utf8"),
    );
    assert.equal(dependency.name, "@xybrid/observability");
    assert.equal(dependency.version, observabilityVersion, "Observability archive version differs");
    assert((await stat(join(observabilityRoot, "package/dist/index.d.ts"))).size > 0);
    await cp(join(observabilityRoot, "package"), join(shared, "@xybrid/observability"), {
      recursive: true,
    });
  }
  for (const name of ["@opentelemetry/api", "@opentelemetry/api-logs"]) {
    await symlink(await realpath(join(root, "node_modules", name)), join(shared, name), "dir");
  }
  const consumer = join(temporary, "consumer");
  await mkdir(join(consumer, "node_modules/@xybrid"), { recursive: true });
  await symlink(packed, join(consumer, "node_modules/@xybrid/web"), "dir");
  await writeFile(
    join(consumer, "package.json"),
    JSON.stringify({ type: "module", private: true }),
  );
  await writeFile(
    join(consumer, "index.ts"),
    observabilityVersion === undefined
      ? `import { XybridLlm } from "@xybrid/web"; void XybridLlm.fromUrl;`
      : `import { XybridLlm, type GgufLoadOptions, type RunMetrics } from "@xybrid/web";
import { Observer, type ObservationRef } from "@xybrid/observability";
const options: GgufLoadOptions = {
  accelerator: "wasm",
  wasmPath: "/xybrid/runtime",
  observer: new Observer({ content: { capture: "no_content" } }),
};
const model = await XybridLlm.fromUrl("/model.gguf", options);
for await (const token of model.generateStream("Hello", {
  maxOutputTokens: 8,
  observation: { conversationId: "c1", trace: "default" },
}))
  void token;
const metrics: RunMetrics | undefined = model.lastRun;
const observation: ObservationRef | undefined = metrics?.observation;
void observation;
await model.cancel();
await model.dispose();
`,
  );
  await writeFile(
    join(consumer, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: {
        strict: true,
        noEmit: true,
        target: "ES2022",
        module: "NodeNext",
        moduleResolution: "NodeNext",
        lib: ["ES2022", "DOM", "DOM.Iterable"],
        types: [],
      },
      include: ["index.ts"],
    }),
  );
  execFileSync(process.execPath, [join(root, "node_modules/typescript/bin/tsc"), "-p", consumer], {
    stdio: "inherit",
  });

  // The browser example subsequently serves this verified copy of the archive.
  const staged = join(output, "package");
  await rm(staged, { recursive: true, force: true });
  await cp(packed, staged, { recursive: true });
  console.log(
    `${manifest.name}@${manifest.version}: ${paths.length} files, ${(await stat(archive)).size} bytes packed`,
  );
  console.log(`Verified ESM imports, TypeScript exports, worker and both WASM assets: ${archive}`);
} finally {
  await rm(temporary, { recursive: true, force: true });
}
