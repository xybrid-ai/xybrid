// Read-only staging console check for the step 6 launches (README, "Console").
//
//   node tools/scripts/observability_staging/console.mjs <evidence lanes root> [key file]
//
// Opens Chrome at https://staging.xybrid.dev and waits (up to 15 minutes) for
// you to sign in and open the launch project: the check starts once that
// workspace shows one of the launches' traces. The profile is temporary and
// deleted at the end, unless XYBRID_CONSOLE_PROFILE names a directory (keep it
// outside the evidence, mode 0700): that profile is kept, so its sign-in and
// last-opened project carry over to the next run.
// Then, for each launch trace in <root>/traces.json (run_staging.py summary),
// it compares the trace page with the staging API's answer, read with the
// launch key: span rows, request and span counts, linked quality records,
// a span link, and the overview KPIs against GET /v1/telemetry/stats. Writes
// report.json and screenshots to <root>/../console-<time>/; the key and the
// session never reach them.
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../../..");
const require = createRequire(path.join(repo, "bindings/web/package.json"));
// bindings/web depends on @playwright/test, which re-exports the browsers.
const { chromium, expect } = require("@playwright/test");

const [root, keyPath = path.join(os.homedir(), ".xybrid-step6/key")] = process.argv.slice(2);
assert(root, "usage: node console.mjs <evidence lanes root> [key file]");
const key = fs.readFileSync(keyPath, "utf8").trim();
const origin = "https://staging.xybrid.dev";
const api = "https://staging.api.xybrid.dev";
const traces = JSON.parse(fs.readFileSync(path.join(root, "traces.json"), "utf8"));
assert(traces.length > 0, "traces.json lists no launch traces; run run_staging.py summary first");
const out = path.join(path.dirname(path.resolve(root)), `console-${new Date().toISOString().replace(/[:.]/g, "-")}`);
fs.mkdirSync(out, { recursive: true });
const keptProfile = process.env.XYBRID_CONSOLE_PROFILE;
if (keptProfile) fs.mkdirSync(keptProfile, { recursive: true, mode: 0o700 });
const profile = keptProfile || fs.mkdtempSync(path.join(os.tmpdir(), "xybrid-step6-console-"));
const report = { passed: false, origin, started_at: new Date().toISOString(), checks: [], browserErrors: [] };
const scrub = (text) => String(text).replaceAll(key, "[REDACTED]");
const save = (name, value) => fs.writeFileSync(path.join(out, name), scrub(JSON.stringify(value, null, 2)));
const pass = (name, details = {}) => {
  report.checks.push({ name, passed: true, ...details });
  console.log(`PASS ${name}`);
};

// `notFound` is returned for a 404; any other non-200 answer fails the check.
async function apiRead(route, { notFound } = {}) {
  const response = await fetch(api + route, {
    headers: { Authorization: `Bearer ${key}` },
    redirect: "error",
    signal: AbortSignal.timeout(20_000),
  });
  if (notFound !== undefined && response.status === 404) return notFound;
  assert.equal(response.status, 200, `API ${route}`);
  return response.json();
}

const context = await chromium.launchPersistentContext(profile, {
  channel: "chrome",
  headless: false,
  viewport: { width: 1500, height: 1000 },
  locale: "en-US",
  timezoneId: "UTC",
});
const page = context.pages()[0] ?? (await context.newPage());
page.on("pageerror", (error) => report.browserErrors.push({ name: error.name, message: scrub(error.message).slice(0, 500) }));
const field = (scope, label) => scope.locator("dt").filter({ hasText: new RegExp(`^${label}$`) }).locator("..").locator("dd");
const qualityList = () => page.getByRole("list", { name: "Feedback and evaluation results" });

try {
  // XYBRID_CONSOLE_WORKSPACE (`/o/<org>/<project>`) opens that workspace first;
  // it still has to pass the check below.
  const start = process.env.XYBRID_CONSOLE_WORKSPACE ?? "";
  assert(start === "" || /^\/o\/[^/]+\/[^/]+$/.test(start), "XYBRID_CONSOLE_WORKSPACE is /o/<org>/<project>");
  await page.goto(`${origin}${start}/traces`, { waitUntil: "domcontentloaded", timeout: 60_000 });
  console.log("Sign in to the staging console in the Chrome window and open the launch project's Traces page.");
  const probe = traces[0];
  // Another project may hold spans under the same trace id (the isolation run's
  // probes do), so the launch project is the one showing the launch key's spans.
  const probeSpans = (await apiRead(`/v1/telemetry/traces/${probe.trace_id}/spans`)).spans ?? [];
  assert(probeSpans.length > 0, `the launch key reads no spans for ${probe.trace_id}`);
  let workspace = null;
  const deadline = Date.now() + 15 * 60_000;
  const rejected = new Set();
  while (workspace === null && Date.now() < deadline) {
    await page.waitForTimeout(5_000);
    // Sign-in may happen in any tab of the window: try every workspace open in it.
    const candidates = new Set(
      context.pages().flatMap((tab) => {
        try {
          const match = /^(\/o\/[^/]+\/[^/]+)(\/|$)/.exec(new URL(tab.url()).pathname);
          return match ? [match[1]] : [];
        } catch {
          return [];
        }
      }),
    );
    for (const candidate of candidates) {
      if (workspace !== null || rejected.has(candidate)) continue;
      const check = await context.newPage();
      const response = await check.goto(`${origin}${candidate}/traces/${probe.trace_id}`, {
        waitUntil: "domcontentloaded",
        timeout: 60_000,
      });
      // The heading shows the requested id even when the workspace cannot load
      // the trace: require the loaded view (its summary with a Requests field)
      // with the launch's span rows.
      const loaded =
        response?.status() === 200 &&
        (await check.getByRole("heading", { name: probe.trace_id, exact: true }).isVisible({ timeout: 15_000 }).catch(() => false)) &&
        (await check.locator("dt").filter({ hasText: /^Requests$/ }).first().isVisible({ timeout: 15_000 }).catch(() => false));
      const found =
        loaded &&
        (await expect(check.locator("[data-span-row]"))
          .toHaveCount(probeSpans.length, { timeout: 15_000 })
          .then(() => true, () => false));
      await check.close();
      if (found) workspace = candidate;
      // A loaded trace with other spans is another project's copy: never this one.
      else if (loaded) {
        rejected.add(candidate);
        console.log(`${candidate} holds other spans under the probe trace id; open the launch project.`);
      }
    }
  }
  assert(workspace, "no workspace showed the launch traces within 15 minutes");
  report.workspace = workspace;
  assert(!(await context.cookies()).some((c) => c.name === "xybrid_demo_mode" && c.value === "1"), "demo mode is on");
  pass("Signed in to the launch project, demo mode off", { workspace });

  for (const trace of traces) {
    const answer = await apiRead(`/v1/telemetry/traces/${trace.trace_id}/spans`);
    const evaluations = await apiRead(`/v1/telemetry/traces/${trace.trace_id}/evaluations`, {
      notFound: { items: [] },
    });
    const spans = answer.spans ?? [];
    const requests = spans.filter((span) => span.counted_request === true).length;
    const route = `${workspace}/traces/${trace.trace_id}`;
    const response = await page.goto(origin + route, { waitUntil: "domcontentloaded", timeout: 60_000 });
    assert.equal(response?.status(), 200, `${trace.lane}/${trace.session}: trace page`);
    await expect(page.getByRole("heading", { name: trace.trace_id, exact: true })).toBeVisible({ timeout: 30_000 });
    await expect(page.locator("[data-span-row]")).toHaveCount(spans.length, { timeout: 30_000 });
    const summary = page.locator("dl").filter({ has: page.locator("dt").filter({ hasText: /^Requests$/ }) });
    await expect(field(summary, "Requests")).toHaveText(String(requests));
    await expect(field(summary, "Spans")).toHaveText(String(spans.length));
    const records = evaluations.items ?? [];
    if (records.length > 0) {
      await expect(qualityList().locator(":scope > li")).toHaveCount(records.length, { timeout: 30_000 });
      for (const record of records) {
        const row = qualityList().locator(":scope > li").filter({ has: page.getByText(record.name, { exact: true }) });
        await expect(row).toHaveCount(1);
        await expect(row.getByRole("link", { name: /^Span:/ })).toHaveAttribute("href", `${route}?span=${record.span_id}`);
      }
      assert(!(await page.locator("body").innerText()).includes("xybrid-staging-secret-5e8b"), "the secret is shown");
    }
    if (trace.run_span_id) {
      await page.goto(`${origin}${route}?span=${trace.run_span_id}`, { waitUntil: "domcontentloaded", timeout: 60_000 });
      await expect(page.locator("#trace-span-details")).toContainText(trace.run_span_id, { timeout: 30_000 });
    }
    await page.screenshot({ path: path.join(out, `${trace.lane}-${trace.session}.png`), fullPage: true });
    pass(`${trace.lane}/${trace.session}: spans, requests, quality records and span link`, {
      trace_id: trace.trace_id,
      spans: spans.length,
      requests,
      evaluations: records.length,
    });
  }

  const stats = await apiRead("/v1/telemetry/stats");
  save("dashboard-api.json", stats);
  await page.goto(`${origin}${workspace}/overview`, { waitUntil: "domcontentloaded", timeout: 60_000 });
  const kpi = (label) =>
    page
      .locator('[data-slot="card"]')
      .filter({ has: page.locator('[data-slot="card-description"]').filter({ hasText: new RegExp(`^${label}$`) }) });
  await expect(kpi("Executions").locator('[data-slot="card-title"]')).toHaveText(String(stats.total_executions_24h), {
    timeout: 30_000,
  });
  await expect(kpi("On-device share")).toContainText(
    `${stats.executions_on_device} on-device, ${stats.executions_on_cloud} cloud`,
  );
  await page.screenshot({ path: path.join(out, "overview.png"), fullPage: true });
  pass("Overview KPIs equal the staging API", {
    executions_24h: stats.total_executions_24h,
    on_device: stats.executions_on_device,
    cloud: stats.executions_on_cloud,
  });
  assert.equal(report.browserErrors.length, 0, "browser JavaScript errors were observed");
  pass("No browser JavaScript exceptions");
  report.passed = true;
} catch (error) {
  report.error = scrub(error.stack || error);
  await page.screenshot({ path: path.join(out, "failure.png"), fullPage: true }).catch(() => {});
  process.exitCode = 1;
} finally {
  report.finished_at = new Date().toISOString();
  save("report.json", report);
  await context.close();
  if (!keptProfile) fs.rmSync(profile, { recursive: true, force: true });
  console.log(`${report.passed ? "PASS" : "FAIL"}: ${report.checks.length} checks; evidence ${out}`);
}
