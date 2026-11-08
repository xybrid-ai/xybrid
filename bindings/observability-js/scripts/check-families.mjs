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
  // This API/SDK combination is valid on its own: rejection must come from our package.
  const unsupported = join(temp, "unsupported");
  await mkdir(unsupported);
  const legacy = {
    private: true,
    type: "module",
    dependencies: {
      "@opentelemetry/api": "1.9.0",
      "@opentelemetry/core": "2.6.0",
      "@opentelemetry/sdk-trace": "2.12.0",
    },
  };
  await writeFile(join(unsupported, "package.json"), JSON.stringify(legacy));
  run("npm", ["install", "--ignore-scripts", "--no-audit", "--no-fund"], unsupported);
  run(
    process.execPath,
    ["--input-type=module", "-e", 'await import("@opentelemetry/sdk-trace");'],
    unsupported,
  );
  legacy.dependencies["@xybrid/observability"] = `file:${archive}`;
  await writeFile(join(unsupported, "package.json"), JSON.stringify(legacy));
  const installed = spawnSync("npm", ["install", "--ignore-scripts", "--no-audit", "--no-fund"], {
    cwd: unsupported,
    encoding: "utf8",
  });
  if (installed.error) throw installed.error;
  if (installed.status !== 0) {
    assert.match(
      installed.stderr,
      /ERESOLVE/,
      "only a package peer conflict is an acceptable rejection",
    );
  } else {
    // npm can accept incompatible optional peers; the OTLP boundary must reject SDK 2.x too.
    const loaded = spawnSync(
      process.execPath,
      [
        "--input-type=module",
        "-e",
        `
      const { createOtlpPipeline } = await import("@xybrid/observability/otlp");
      await createOtlpPipeline({ environment: null }).shutdown();
    `,
      ],
      { cwd: unsupported, encoding: "utf8" },
    );
    if (loaded.error) throw loaded.error;
    assert.notEqual(loaded.status, 0, "OTLP must reject a realistic SDK 2.x consumer");
    assert.match(
      loaded.stderr,
      /OpenTelemetry|opentelemetry/,
      "rejection must identify the incompatible SDK",
    );
  }
  console.log("Realistic SDK 2.x baseline installs; the package's OTLP boundary rejects it.");
} finally {
  await rm(temp, { recursive: true, force: true });
}
