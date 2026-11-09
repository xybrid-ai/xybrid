// Build first, then pack and verify the actual npm archive in an isolated consumer.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { cp, mkdir, mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const expected = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
// Calling this script directly must be enough after a frozen development install.
// Downloaded archives are verified as-is; never rebuild approved release artifacts.
if (process.argv[2] === undefined) {
  execFileSync("pnpm", ["run", "build"], { cwd: root, stdio: "inherit" });
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
  "dist/otlp.js",
  "dist/otlp.d.ts",
];
const allowed = /^(package\.json|README\.md|LICENSE|NOTICE|dist\/[^/]+\.(js|js\.map|d\.ts))$/;
for (const path of required) assert(paths.includes(path), `Missing from the npm tarball: ${path}`);
for (const path of paths) assert(allowed.test(path), `Must not ship in the npm tarball: ${path}`);

const temporary = await mkdtemp(join(tmpdir(), "xybrid-observability-package-"));
try {
  execFileSync("tar", ["-xzf", archive, "-C", temporary]);
  const packed = join(temporary, "package");
  const manifest = JSON.parse(await readFile(join(packed, "package.json"), "utf8"));
  assert.equal(manifest.name, "@xybrid/observability");
  assert.equal(manifest.version, process.argv[3] ?? expected.version);
  assert.equal(manifest.license, "Apache-2.0");
  assert.equal(manifest.publishConfig?.access, "public");
  assert.deepEqual(Object.keys(manifest.dependencies), ["@opentelemetry/api-logs"]);
  for (const range of Object.values(manifest.dependencies)) {
    assert.match(range, /^\d+\.\d+\.\d+$/, "Dependencies are pinned exactly");
  }

  // The consumer gets the archive plus only the packages each entry point may import.
  const consumer = join(temporary, "consumer");
  const modules = join(consumer, "node_modules");
  await mkdir(join(modules, "@xybrid"), { recursive: true });
  await mkdir(join(modules, "@opentelemetry"), { recursive: true });
  // A copy, not a symlink: Node resolves imports from the real path of the importing file.
  await cp(packed, join(modules, "@xybrid/observability"), { recursive: true });
  const link = async (name) =>
    symlink(await realpath(join(root, "node_modules", name)), join(modules, name), "dir");
  await link("@opentelemetry/api");
  await link("@opentelemetry/api-logs");
  await writeFile(
    join(consumer, "package.json"),
    JSON.stringify({ type: "module", private: true }),
  );
  await writeFile(
    join(consumer, "core.mjs"),
    `import { Observer, ObservationContext, VERSION } from "@xybrid/observability";
if (VERSION !== ${JSON.stringify(process.argv[3] ?? expected.version)}) throw new Error("Telemetry version differs from package version");
const op = new Observer().startInference({ operation: "chat", provider: "p", model: "m" });
op.recordChunk();
op.end();
if (ObservationContext.fromTraceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01") === undefined) process.exit(1);
`,
  );
  // Core must run with nothing but @opentelemetry/api and api-logs installed.
  execFileSync(process.execPath, [join(consumer, "core.mjs")], { cwd: consumer, stdio: "inherit" });

  for (const name of [
    "@opentelemetry/core",
    "@opentelemetry/otlp-transformer",
    "@opentelemetry/resources",
    "@opentelemetry/sdk-logs",
    "@opentelemetry/sdk-metrics",
    "@opentelemetry/sdk-trace",
  ]) {
    await link(name);
  }
  await writeFile(
    join(consumer, "index.ts"),
    `import { Observer, type ObservationRef } from "@xybrid/observability";
import { createOtlpPipeline, xybridTracesPreset } from "@xybrid/observability/otlp";
const pipeline = createOtlpPipeline(xybridTracesPreset("key", { endpoint: "http://127.0.0.1:9/v1/traces" }));
const observer: Observer = pipeline.observer({ capture: "no_content" });
const op = observer.startInference({ operation: "chat", provider: "p", model: "m", location: "local" });
op.recordUsage({ inputTokens: 1 });
op.end();
const ref: ObservationRef | undefined = op.observationRef();
void ref;
await pipeline.shutdown(100);
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
        lib: ["ES2022", "DOM"],
        types: [],
      },
      include: ["index.ts"],
    }),
  );
  execFileSync(process.execPath, [join(root, "node_modules/typescript/bin/tsc"), "-p", consumer], {
    stdio: "inherit",
  });
  console.log(
    `${manifest.name}@${manifest.version}: ${paths.length} files verified in ${dirname(archive)}`,
  );
} finally {
  await rm(temporary, { recursive: true, force: true });
}
