// Run the existing browser example against a verified GitHub release, without Bazel/Rust.
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import ky from "ky";

const root = fileURLToPath(new URL("../", import.meta.url));
const args = process.argv.slice(2).filter((arg) => arg !== "--");
const requestedVersion = args[0]?.startsWith("--") ? undefined : args.shift();
const manifest = await Bun.file(new URL("../package.json", import.meta.url)).json();
const version = (requestedVersion ?? (manifest.version as string)).replace(/^v/, "");
if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
  throw new Error("Usage: pnpm demo:release [version] [Vite options]");
}

const directory = new URL(`../test-results/releases/${version}/`, import.meta.url);
await mkdir(directory, { recursive: true });
const name = `xybrid-web-${version}.tgz`;
const baseUrl = `https://github.com/xybrid-ai/xybrid/releases/download/v${version}/`;
console.log(`Preparing @xybrid/web ${version} from its GitHub release…`);
const checksum = (await ky.get(`${baseUrl}${name}.sha256`).text()).trim().split(/\s+/)[0];
if (checksum === undefined || !/^[a-f0-9]{64}$/.test(checksum)) {
  throw new Error("The release does not provide a valid SHA-256 checksum.");
}
const archive = Bun.file(new URL(name, directory));
const verified = (bytes: Uint8Array): boolean =>
  new Bun.CryptoHasher("sha256").update(bytes).digest("hex") === checksum;
if (!(await archive.exists()) || !verified(new Uint8Array(await archive.arrayBuffer()))) {
  const bytes = new Uint8Array(await ky.get(`${baseUrl}${name}`).arrayBuffer());
  if (!verified(bytes)) throw new Error("Release archive SHA-256 verification failed.");
  await Bun.write(archive, bytes);
}

const env = { ...process.env, XYBRID_WEB_PACKAGE: "test-results/npm/package" };
const run = async (command: string[]): Promise<void> => {
  const child = Bun.spawn(command, {
    cwd: root,
    env,
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
  });
  const code = await child.exited;
  if (code !== 0) throw new Error(`${command[0]} exited with status ${code}.`);
};
await run(["node", "scripts/check-pack.mjs", fileURLToPath(new URL(name, directory)), version]);
await run(["bun", "run", "example/scripts/prepare-assets.ts"]);
console.log(`Testing released @xybrid/web ${version}. Open the URL below, then click Load model.`);
await run(["pnpm", "exec", "vite", "--config", "example/vite.config.ts", ...args]);
