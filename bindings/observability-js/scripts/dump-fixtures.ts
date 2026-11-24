// Writes the normalized output of every contract fixture to .context/fixture-output/js/ at the
// repository root, for cross-language comparison with the Rust crate's dump.
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { loadFixtures, runFixture } from "../test/support/fixtures.ts";

const output = fileURLToPath(new URL("../../../.context/fixture-output/js/", import.meta.url));
await mkdir(output, { recursive: true });
for (const fixture of loadFixtures()) {
  const normalized = await runFixture(fixture);
  await writeFile(`${output}${fixture.name}.json`, `${JSON.stringify(normalized, null, 2)}\n`);
  console.log(
    `${fixture.name}: ${normalized.spans.length} spans, ${normalized.metrics.length} metrics, ${normalized.logs.length} logs`,
  );
}
console.log(`Wrote ${output}`);
