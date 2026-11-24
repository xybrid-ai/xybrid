// @xybrid/web depends on @xybrid/observability, linked from ../observability-js during
// development (see pnpm-workspace.yaml). Type declarations and the test harness resolve the
// sibling's dist/, so install and build it when missing or older than its sources.
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const sibling = fileURLToPath(new URL("../../observability-js/", import.meta.url));
const run = (...args) => execFileSync("pnpm", args, { cwd: sibling, stdio: "inherit" });

const newest = (path) => {
  const stat = statSync(path);
  if (!stat.isDirectory()) return stat.mtimeMs;
  return Math.max(0, ...readdirSync(path).map((entry) => newest(join(path, entry))));
};

if (!existsSync(join(sibling, "node_modules/.modules.yaml"))) {
  run("install", "--frozen-lockfile");
}
const built = join(sibling, "dist/index.d.ts");
const sources = Math.max(newest(join(sibling, "src")), newest(join(sibling, "package.json")));
if (!existsSync(built) || statSync(built).mtimeMs < sources) {
  run("run", "build");
}
