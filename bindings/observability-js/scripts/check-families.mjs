// Real archive consumers and full source tests against each aligned OTel family.
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const manifest = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
execFileSync("pnpm", ["build"], { cwd: root, stdio: "inherit" });
const temp = await mkdtemp(join(tmpdir(), "xybrid-otel-families-"));
const run = (command, args, cwd) => execFileSync(command, args, { cwd, stdio: "inherit" });
try {
  const [packed] = JSON.parse(
    execFileSync("npm", ["pack", "--ignore-scripts", "--json", "--pack-destination", temp], {
      cwd: root,
      encoding: "utf8",
    }),
  );
  const archive = join(temp, packed.filename);
  const consumer = async (name, dependencies) => {
    const dir = join(temp, name, "bindings/observability-js");
    await mkdir(dir, { recursive: true });
    await writeFile(
      join(dir, "package.json"),
      JSON.stringify({
        private: true,
        type: "module",
        dependencies: {
          "@xybrid/observability": `file:${archive}`,
          "@opentelemetry/api": "1.10.0",
          ...dependencies,
        },
      }),
    );
    run("npm", ["install", "--ignore-scripts", "--no-audit", "--no-fund"], dir);
    return dir;
  };
  const core = await consumer("core", {});
  run(
    process.execPath,
    [
      "--input-type=module",
      "-e",
      `import assert from "node:assert/strict";
import { Observer } from "@xybrid/observability";
const op = new Observer().startWorkflow("test"); op.end();
for (const name of ["core", "resources", "sdk-trace", "sdk-logs", "sdk-metrics", "otlp-transformer"]) {
  assert.throws(() => import.meta.resolve("@opentelemetry/" + name), { code: "ERR_MODULE_NOT_FOUND" });
}`,
    ],
    core,
  );
  const smoke = await readFile(join(root, "scripts/packed-smoke.mjs"), "utf8");
  for (const [sdk, transformer] of [
    ["3.0.0", "0.300.0"],
    ["3.1.0", "0.301.0"],
  ]) {
    const dependencies = { ...manifest.devDependencies };
    for (const key of Object.keys(dependencies)) {
      if (key.startsWith("@opentelemetry/") && key !== "@opentelemetry/api")
        dependencies[key] = key.endsWith("otlp-transformer") ? transformer : sdk;
    }
    const dir = await consumer(sdk, dependencies);
    await writeFile(join(dir, "packed-smoke.mjs"), smoke);
    await cp(join(root, "test/support/otlp-decode.ts"), join(dir, "otlp-decode.ts"));
    run("bun", ["build", "./otlp-decode.ts", "--target=node", "--outfile=otlp-decode.mjs"], dir);
    run(process.execPath, ["packed-smoke.mjs"], dir);
    for (const path of ["src", "test", "examples", "tsconfig.json"])
      await cp(join(root, path), join(dir, path), { recursive: true });
    await cp(
      join(root, "../../crates/xybrid-observability/contract"),
      join(temp, sdk, "crates/xybrid-observability/contract"),
      { recursive: true },
    );
    run("bun", ["test", "test/"], dir);
    run(process.execPath, [join(dir, "node_modules/typescript/bin/tsc"), "--noEmit"], dir);
    console.log(`Packed consumer and full suite passed: SDK ${sdk}, transformer ${transformer}`);
  }
  // SDK 2.x (D33). A complete API 1.9 + SDK 2.x tree, valid on its own, so any rejection
  // below comes from our package and not from an unrelated conflict or a missing module.
  const unsupported = join(temp, "unsupported");
  await mkdir(unsupported);
  const legacy = {
    private: true,
    type: "module",
    dependencies: {
      "@opentelemetry/api": "1.9.0",
      "@opentelemetry/core": "2.12.0",
      "@opentelemetry/resources": "2.12.0",
      "@opentelemetry/sdk-trace": "2.12.0",
      "@opentelemetry/sdk-logs": "0.223.0",
      "@opentelemetry/sdk-metrics": "2.12.0",
      "@opentelemetry/otlp-transformer": "0.223.0",
    },
  };
  const npmInstall = ["install", "--ignore-scripts", "--no-audit", "--no-fund"];
  await writeFile(join(unsupported, "package.json"), JSON.stringify(legacy));
  run("npm", npmInstall, unsupported);
  run(
    process.execPath,
    [
      "--input-type=module",
      "-e",
      `for (const name of ["core", "resources", "sdk-trace", "sdk-logs", "sdk-metrics", "otlp-transformer"])
  await import("@opentelemetry/" + name);`,
    ],
    unsupported,
  );
  // Leg 1, npm's peer check: refusing the package must be a peer conflict naming it.
  legacy.dependencies["@xybrid/observability"] = `file:${archive}`;
  await writeFile(join(unsupported, "package.json"), JSON.stringify(legacy));
  const strict = spawnSync("npm", npmInstall, { cwd: unsupported, encoding: "utf8" });
  if (strict.error) throw strict.error;
  let peerCheck = "npm accepted the conflicting optional peers";
  if (strict.status !== 0) {
    assert.match(
      strict.stderr,
      /ERESOLVE/,
      "only a package peer conflict is an acceptable refusal",
    );
    assert.match(
      strict.stderr,
      /@xybrid\/observability/,
      "the peer conflict must name the package",
    );
    peerCheck = "npm refuses the package's peers (ERESOLVE)";
    // Leg 2 needs the tree npm could also accept: force it in, as other installers do.
    run("npm", [...npmInstall, "--legacy-peer-deps"], unsupported);
  }
  // Leg 2, always: with SDK 2.x loaded, every OTLP entry point throws the package's own
  // rejection. A missing module, link error or other exception fails this check.
  run(
    process.execPath,
    [
      "--input-type=module",
      "-e",
      `import assert from "node:assert/strict";
import { SDK_INFO } from "@opentelemetry/core";
assert.match(SDK_INFO["telemetry.sdk.version"], /^2\\./);
const otlp = await import("@xybrid/observability/otlp");
const options = {
  environment: null,
  traces: { endpoint: "http://127.0.0.1:1/v1/traces" },
  metrics: { endpoint: "http://127.0.0.1:1/v1/metrics" },
  logs: { endpoint: "http://127.0.0.1:1/v1/logs" },
};
const message = "@xybrid/observability/otlp requires OpenTelemetry SDK 3.0 or 3.1.";
const entries = {
  createOtlpPipeline: () => otlp.createOtlpPipeline(options),
  createOtlpExporters: () => otlp.createOtlpExporters(options),
  OtlpTraceExporter: () => new otlp.OtlpTraceExporter(options),
  OtlpLogExporter: () => new otlp.OtlpLogExporter(options),
  OtlpMetricExporter: () => new otlp.OtlpMetricExporter(options),
};
for (const [name, construct] of Object.entries(entries)) {
  assert.throws(construct, (error) => error instanceof TypeError && error.message === message, name);
}`,
    ],
    unsupported,
  );
  console.log(`SDK 2.x baseline installs alone; ${peerCheck}; every OTLP entry point rejects it.`);
} finally {
  await rm(temp, { recursive: true, force: true });
}
