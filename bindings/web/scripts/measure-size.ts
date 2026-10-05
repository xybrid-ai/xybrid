import { appendFile, mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { gzipSync } from "node:zlib";

const directory = new URL("../dist/", import.meta.url);
const files: { file: string; bytes: number; gzipBytes: number }[] = [];
for (const name of await readdir(directory)) {
  if (name.endsWith(".js")) {
    const bytes = await readFile(new URL(name, directory));
    files.push({
      file: name,
      bytes: bytes.byteLength,
      gzipBytes: gzipSync(bytes, { level: 9 }).byteLength,
    });
  }
}
for (const backend of ["wasm", "webgpu"]) {
  for (const extension of ["js", "wasm"]) {
    const file = `runtime/${backend}/xybrid_runtime.${extension}`;
    const bytes = await readFile(new URL(file, directory));
    files.push({
      file,
      bytes: bytes.byteLength,
      gzipBytes: gzipSync(bytes, { level: 9 }).byteLength,
    });
  }
}
await mkdir("test-results", { recursive: true });
await writeFile("test-results/build-sizes.json", JSON.stringify(files, null, 2));
const table = [
  "| Asset | Raw bytes | Gzip bytes |",
  "| --- | ---: | ---: |",
  ...files.map((file) => `| ${file.file} | ${file.bytes} | ${file.gzipBytes} |`),
].join("\n");
console.log(table);
const summary = process.env["GITHUB_STEP_SUMMARY"];
if (summary !== undefined) await appendFile(summary, `\nWeb SDK build sizes\n\n${table}\n`);
