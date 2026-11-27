import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { arch, platform, release } from "node:os";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";
import type { PageResult, StagingConfig } from "./staging-page.ts";

// The browser lane (`web-mac`) of the step 6 staging scenario,
// tools/scripts/observability_staging/README.md; staging-page.ts is the scenario itself.
// run_staging.py writes config.json, which holds the staging key, and names it in
// XYBRID_STAGING_CONFIG. The key reaches the page only as an evaluate argument: never in a URL,
// a console line or a file vite serves. A trace would record that argument and the bearer header,
// so this file is never traced.
test.use({ trace: "off" });

/** Reads config.json and deletes it before anything else can fail: it holds the staging key. */
const takeConfig = async (path: string): Promise<StagingConfig> => {
  const text = await readFile(path, "utf8");
  await rm(path);
  try {
    return JSON.parse(text) as StagingConfig;
  } catch {
    // V8 quotes its input in JSON errors, and this input holds the key.
    throw new Error(`${path} is not valid JSON.`);
  }
};

/**
 * staging-page.ts as one ES module, with the application side it needs bundled in:
 * @xybrid/observability and its OpenTelemetry SDK. The SDK under test stays out; the page
 * imports it from /sdk/index.js, the build `vite preview` serves.
 */
const scenarioModule = async (): Promise<string> => {
  const { build } = await import("vite");
  const output = await build({
    configFile: false,
    root: fileURLToPath(new URL(".", import.meta.url)),
    logLevel: "warn",
    publicDir: false,
    build: {
      write: false,
      emptyOutDir: false,
      minify: false,
      lib: {
        entry: fileURLToPath(new URL("staging-page.ts", import.meta.url)),
        formats: ["es"],
        fileName: "staging-page",
      },
      rollupOptions: { output: { inlineDynamicImports: true } },
    },
  });
  const chunk = (Array.isArray(output) ? output : [output])
    .flatMap((bundle) => ("output" in bundle ? bundle.output : []))
    .find((item) => item.type === "chunk" && item.isEntry);
  if (chunk?.type !== "chunk") throw new Error("Vite produced no scenario module.");
  return chunk.code;
};

/** Writes expectations.json atomically. A staging key never belongs there; any is replaced. */
const writeExpectations = async (
  path: string,
  value: unknown,
  keys: readonly string[],
): Promise<void> => {
  let text = `${JSON.stringify(value, null, 2)}\n`;
  for (const key of keys) if (key.length > 0) text = text.replaceAll(key, "<removed>");
  await mkdir(dirname(path), { recursive: true });
  await writeFile(`${path}.tmp`, text);
  await rename(`${path}.tmp`, path);
};

test("staging scenario exports both sessions from the page to Xybrid staging", async ({
  page,
  browser,
}) => {
  const configPath = process.env["XYBRID_STAGING_CONFIG"];
  test.skip(configPath === undefined, "run_staging.py sets XYBRID_STAGING_CONFIG (lane web-mac).");
  if (configPath === undefined) return;
  test.setTimeout(300_000);
  const config = await takeConfig(configPath);
  const expectationsPath = process.env["XYBRID_STAGING_EXPECTATIONS"];
  if (expectationsPath === undefined) throw new Error("XYBRID_STAGING_EXPECTATIONS is not set.");

  let result: PageResult = { sdk_version: null, sessions: {}, scenario_error: null };
  try {
    const source = await scenarioModule();
    await page.goto("/");
    result = await page.evaluate(
      async ({ source, config }) => {
        const moduleUrl = "/sdk/index.js";
        const sdk = (await import(moduleUrl)) as typeof import("../src/index.ts");
        const url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
        try {
          const { runStaging } = (await import(url)) as typeof import("./staging-page.ts");
          return await runStaging(sdk, config);
        } finally {
          URL.revokeObjectURL(url);
        }
      },
      { source, config },
    );
  } catch (error) {
    result = { ...result, scenario_error: error instanceof Error ? error.message : String(error) };
  }
  await writeExpectations(
    expectationsPath,
    {
      schema: "xybrid-staging-scenario/1",
      binding: config.binding,
      lane: config.lane,
      run: config.run,
      sdk_version: result.sdk_version,
      model_id: config.model_id,
      device: {
        system: platform(),
        release: release(),
        machine: arch(),
        browser: `${browser.browserType().name()} ${browser.version()}`,
      },
      sessions: result.sessions,
      scenario_error: result.scenario_error,
    },
    Object.values(config.sessions).map((values) => values.api_key),
  );
  const summary = Object.entries(result.sessions).map(([session, values]) => [
    session,
    {
      references: values.references,
      stopped: values.stopped,
      completed: values.completed,
      quality: values.quality,
    },
  ]);
  console.log(JSON.stringify(Object.fromEntries(summary), null, 1));
  expect(result.scenario_error, "the scenario error, also in expectations.json").toBeNull();
});
