# MLX Embedding Baseline

This document records MLX SafeTensors embedding throughput for the staged
`nomic-ai/nomic-embed-text-v1.5` fixture. It is intentionally separate from
`llm-backends-baseline.md`, which compares generative decode throughput against
llama.cpp.

## Methodology

- Fixture: `nomic-embed-text-v1.5`, resolved from
  `integration-tests/fixtures/models/nomic-embed-text-v1.5` or
  `XYBRID_MLX_NOMIC_DIR`.
- Runtime: `MlxEmbeddingAdapter` with mean pooling, L2 normalization, and
  `max_seq_len = 128`.
- Inputs: five fixed `search_document:` / `search_query:` strings.
- Rounds: 5 measurement rounds; round 0 is discarded.
- Output: `target/benchmark-results/mlx_embedding.md`.
- Warm-up: `tools/scripts/bench-mlx-embedding.sh` runs a non-reporting warm-up
  pass first unless `--measure-only` is passed.
- Host and fixture skips: unsupported hosts and missing staged fixtures are
  non-fatal by default. Pass `--strict` when the Nomic bundle is staged and a
  skipped or failed benchmark should fail the command.

## Reference Runs

Unfilled rows are expected until an Apple Silicon runtime host has the staged
Nomic fixture available. Do not infer embedding runtime speed from the
compile-only CI gate.

### Apple M4 Max (128 GB unified memory, macOS 26.4.1)

| fixture | vectors/s | ms/input | avg input tokens | embedding dim | peak-mem-mb | date | commit |
|---------|-----------|----------|------------------|---------------|-------------|------|--------|
| nomic-embed-text-v1.5 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | - | - |

## How To Update

1. Stage the Nomic SafeTensors bundle outside the repo or under the fixture
   directory:

   ```bash
   export XYBRID_MLX_NOMIC_DIR=/path/to/nomic-embed-text-v1.5
   ```

2. Run the harness:

   ```bash
   tools/scripts/bench-mlx-embedding.sh
   ```

   Use `tools/scripts/bench-mlx-embedding.sh --strict` when the staged Nomic
   fixture is expected to be present. Strict mode turns unsupported-host skips,
   missing-fixture skips, and failed benchmark runs into a non-zero exit.

3. Copy the generated row from `target/benchmark-results/mlx_embedding.md` into
   the table above, then record the date and short commit SHA.
