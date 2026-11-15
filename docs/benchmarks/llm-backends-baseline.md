# LLM Backend Baseline — MLX vs llama.cpp

This document records the reference decode-throughput numbers used by the
regression gate in `crates/xybrid-core/benches/llm_backend_compare.rs`.
New results should be produced by running
`tools/scripts/bench-llm-backends.sh` on the relevant hardware and
appending a row to the table below before updating the pass floor.
MLX embedding throughput is tracked separately in
[`mlx-embedding-baseline.md`](mlx-embedding-baseline.md).

## Why a baseline?

The bench records MLX and llama.cpp throughput side by side. It only
asserts a decode-parity floor for rows marked `enforced`, which must use
the same model, host, prompt, and commit. Cross-model fallback rows are
kept as informational reference points until exact pairs are pinned. The
current Qwen 3 4B BF16 baseline shows parity, not a shipping speedup.
Committing the numbers before tightening the gate means:

1. Future regressions are attributable to a specific commit rather than
   a drifting environment.
2. New reviewers can see at a glance whether any speedup claim is based
   on best-case tuning or sustained steady-state behaviour.
3. The methodology (thermal policy, cold-cache handling, per-round
   repetitions) is visible alongside the numbers.

## Methodology

See the module doc of
[`benches/llm_backend_compare.rs`](../../crates/xybrid-core/benches/llm_backend_compare.rs)
for the full spec. Summary:

- Greedy decode, 256-token prompt, 128-token output budget. Raw-completion
  rows pass the fixed prompt directly; chat-only instruct pairs wrap the
  same prompt body in each backend's chat template.
- 5 measurement rounds; the first round is discarded; the remaining 4
  rounds' median lands in the table.
- Cold-cache mitigation: `bench-llm-backends.sh` runs a throwaway
  warm-up pass before measurement so Metal kernels are already compiled
  when the measurement timer starts. The warm-up pass sets
  `XYBRID_BENCH_WARMUP_ONLY=1`, so it writes no report and performs no
  parity gate; only the measurement pass updates
  `target/benchmark-results/llm_backend_compare.md`.
- Host and fixture skips: unsupported hosts and missing staged fixtures
  are treated as non-fatal skips by default so Linux CI and laptops
  without local model bundles can still run the wrapper. Pass `--strict`
  when all required fixtures are staged and a skipped or failed benchmark
  should fail the command.
- Thermal mitigation: start from idle, mains power, lid open, nothing
  else on the GPU.

## Reference runs

Unfilled rows are expected to be populated once the hardware is
available. Each row records a single (model, backend, fixture, host)
tuple so you can diff two baselines cleanly. Rows marked
`informational` do not feed the parity gate.

### Apple M2 Pro (16 GB unified memory, macOS 15)

| model | backend | fixture | gate | prompt-tokens/s | decode-tokens/s | ttft-ms | peak-mem-mb | date | commit |
|-------|---------|---------|-----------|-----------------|-----------------|---------|-------------|------|--------|
| qwen3-4b | mlx | qwen3-4b-mlx | enforced | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| qwen3-4b | llama.cpp | qwen3-4b-bf16-gguf | enforced | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| gemma4-2b | mlx | gemma4-2b | informational | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| gemma4-2b | llama.cpp | gemma4-2b-bf16-gguf | informational | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| lfm2.5-1.2b-instruct | mlx | lfm2.5-1.2b-instruct-mlx | informational | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| lfm2.5-1.2b-instruct | llama.cpp | lfm2.5-1.2b-instruct-bf16-gguf | informational | _pending_ | _pending_ | _pending_ | _pending_ | — | — |

> The M2 Pro baseline is intended to become the canonical gate once
> both MLX and llama.cpp rows contain measured data. Until then, the
> Qwen 3 4B pair is gate-ready but still must not be treated as a
> measured shipping speedup claim.

### Apple M4 Max (128 GB unified memory, macOS 26.4.1)

| model | backend | fixture | gate | prompt-tokens/s | decode-tokens/s | ttft-ms | peak-mem-mb | date | commit |
|-------|---------|---------|-----------|-----------------|-----------------|---------|-------------|------|--------|
| qwen3-4b | mlx | qwen3-4b-mlx | informational: Q4_K_M compare | 1256.3 | 50.2 | 181.5 | 15893.6 | 2026-05-16 | e78714c |
| qwen3-4b | mlx | qwen3-4b-mlx | enforced | 1266.7 | 50.3 | 180.0 | 15828.2 | 2026-05-16 | e78714c |
| qwen3-4b | llama.cpp | qwen3-4b-bf16-gguf | enforced | 6423.8 | 48.5 | 35.5 | 24020.3 | 2026-05-16 | e78714c |
| qwen3-4b | mlx | qwen3-4b-mlx | enforced | 1172.2 | 48.5 | 194.5 | 15828.7 | 2026-05-17 | e78714c+dirty |
| qwen3-4b | llama.cpp | qwen3-4b-bf16-gguf | enforced | 6711.7 | 44.5 | 34.0 | 20228.8 | 2026-05-17 | e78714c+dirty |
| qwen3-4b | mlx | qwen3-4b-mlx | enforced | 1203.2 | 50.1 | 189.5 | 15831.4 | 2026-05-17 | e78714c+dirty |
| qwen3-4b | llama.cpp | qwen3-4b-bf16-gguf | enforced | 6333.3 | 46.0 | 36.0 | 24027.9 | 2026-05-17 | e78714c+dirty |
| qwen3-4b | llama.cpp | qwen3-4b-gguf | informational: Q4_K_M | 16912.1 | 104.8 | 13.5 | 15104.8 | 2026-05-16 | e78714c |
| gemma4-2b | mlx | gemma4-2b | informational | 866.1 | 14.1 | 274.0 | 13568.5 | 2026-05-17 | e78714c+dirty |
| gemma4-2b | llama.cpp | gemma4-2b-bf16-gguf | informational | 10249.2 | 72.7 | 24.5 | 22497.9 | 2026-05-17 | e78714c+dirty |
| lfm2.5-1.2b-instruct | mlx | lfm2.5-1.2b-instruct-mlx | informational | 3033.5 | 38.0 | 81.5 | 18054.9 | 2026-05-17 | e78714c+dirty |
| lfm2.5-1.2b-instruct | llama.cpp | lfm2.5-1.2b-instruct-bf16-gguf | informational | 4940.0 | 175.7 | 50.0 | 20313.4 | 2026-05-17 | e78714c+dirty |
| lfm2.5-1.2b-instruct | mlx | lfm2.5-1.2b-instruct-mlx | informational | 1216.8 | 32.0 | 203.0 | 18078.4 | 2026-05-17 | e78714c+dirty |
| lfm2.5-1.2b-instruct | llama.cpp | lfm2.5-1.2b-instruct-bf16-gguf | informational | 1757.4 | 123.0 | 141.0 | 20338.6 | 2026-05-17 | e78714c+dirty |

> This run compared BF16 MLX weights against Q4_K_M GGUF. It is useful
> as a quantized llama.cpp reference, but it is not an enforced
> same-precision gate. The enforced Qwen row uses `qwen3-4b-bf16-gguf`.

> The same-precision BF16 runs pass the current 0.95x parity floor but
> do not support a 1.30x speedup claim. The latest dirty-tree rerun
> measured Qwen 3 4B MLX decode throughput at 50.1 tokens/s versus
> llama.cpp BF16 at 46.0 tokens/s, or 1.09x llama.cpp; the earlier clean
> HEAD run measured 1.04x. Treat this as parity evidence, not a shipping
> speedup claim.

> The Gemma 4 exact BF16 pair uses each backend's chat template and is
> recorded as informational because current MLX decode throughput
> measured 14.1 decode tokens/s versus llama.cpp at 72.7 decode
> tokens/s (0.19x). This pins the fair comparison but is not parity-gate
> evidence yet.

> The LFM2.5 exact BF16 pair uses each backend's chat template so both
> sides produce the full 128-token decode budget. It is recorded as
> informational because the current MLX short-conv runtime measured
> 32.0 decode tokens/s versus llama.cpp at 123.0 decode tokens/s
> (0.26x). This is useful performance evidence, but it is not a parity
> gate and should not be used for an MLX speed claim.

### Apple M4 Max, 4-bit pairs (128 GB unified memory, macOS 27.2.1)

The rows above compare the engines on identical BF16 weights. These compare
what each backend's users download: MLX 4-bit SafeTensors against a Q4_K_M
GGUF of the same model. They are two quantization schemes (Q4_K_M keeps more
bits per weight and is the larger file), so read the rows as "what picking a
backend gets you", not as kernel parity. None is gated.

Measured 2026-11-15 at `001378d3`, MLX v0.32.4, mlx-c v0.8.0, llama.cpp
`8642541`. Decode tokens/s is the median of rounds 2 to 5; each column is one
pass of `tools/scripts/bench-llm-backends.sh`. The Gemma pair was added after
the first pass.

| model | backend | fixture | decode-tokens/s, pass 1 / 2 / 3 / 4 | ttft-ms | peak-mem-mb |
|-------|---------|---------|-------------------------------------|---------|-------------|
| qwen3-0.6b-4bit | mlx | qwen3-0.6b-4bit-mlx | 627.2 / 654.7 / 651.4 / 622.8 | 31 | 781 |
| qwen3-0.6b-4bit | llama.cpp | qwen3-0.6b-q4km-gguf | 428.5 / 245.4 / 159.8 / 233.6 | 3 to 5.5 | 1095 |
| gemma-4-e2b-4bit | mlx | gemma-4-e2b-it-4bit-mlx | n/a / 104.8 / 103.8 / 109.4 | 135 | 7005 |
| gemma-4-e2b-4bit | llama.cpp | gemma-4-e2b-it-q4km-gguf | n/a / 56.7 / 73.3 / 53.6 | 13 to 25 | 3596 |
| lfm2.5-1.2b-instruct-4bit | mlx | lfm2.5-1.2b-instruct-4bit-mlx | 508.0 / 502.0 / 506.0 / 503.0 | 51 | 1485 |
| lfm2.5-1.2b-instruct-4bit | llama.cpp | lfm2.5-1.2b-instruct-q4km-gguf | 447.7 / 491.3 / 257.0 / 490.3 | 47 to 53 | 1349 |

> **The machine was not idle, and it shows on one side only.** MLX decoded at
> the same rate in every pass, within 5%. llama.cpp's median moved by up to
> 2.7x between passes: macOS was running photo analysis (`mediaanalysisd`, about
> 85% of a core plus GPU work) beside two browsers during passes 2 to 4.
> Within those passes llama.cpp's first round, which the harness discards as
> warm-up, was its fastest (Qwen 282 tokens/s wall, then 177 to 241; Gemma
> 121, then 48 to 67). Repeat this on an idle machine before quoting a ratio.

> **What the passes support.** Against llama.cpp's best pass, MLX decodes
> Qwen 3 0.6B about 1.5x faster and LFM2.5 1.2B at parity (1.02x). For
> Gemma 4 E2B MLX's median is above every llama.cpp median, but llama.cpp's
> discarded first rounds reached MLX's rate, so no ratio is claimed.

> **Where llama.cpp is ahead whatever the load.** Time to first token on a
> repeated prompt: llama.cpp reuses the prompt it evaluated last time, MLX
> evaluates it again (5.5 ms against 31 ms for Qwen, 13 to 25 ms against
> 135 ms for Gemma). And memory for Gemma 4 E2B: 7.0 GB peak on MLX against
> 3.6 GB on llama.cpp.

## How to update

1. Materialize the MLX staged fixture and download the GGUF reference
   fixture for the model row you're updating. The benchmark accepts either
   a fixture directory under `integration-tests/fixtures/models/<id>` or
   the staged env var declared in `models.json` (`XYBRID_MLX_QWEN_4B_DIR`,
   `XYBRID_MLX_GEMMA_DIR`, or `XYBRID_MLX_LFM25_DIR` for benchmark rows;
   `XYBRID_MLX_QWEN_DIR` and `XYBRID_MLX_LFM_DIR` are the smaller smoke-test
   fixtures):

   ```bash
   # Qwen exact enforced pair:
   # Create integration-tests/fixtures/models/qwen3-4b-mlx/ from
   # Qwen/Qwen3-4B with config.json, tokenizer.json, tokenizer_config.json,
   # model.safetensors.index.json, and referenced model-*.safetensors shards.
   ./integration-tests/download.sh qwen3-4b-mlx || true
   export XYBRID_MLX_QWEN_4B_DIR="$PWD/integration-tests/fixtures/models/qwen3-4b-mlx"

   ./integration-tests/download.sh qwen3-4b-bf16-gguf

   # Staged exact Gemma benchmark pair:
   # Create integration-tests/fixtures/models/gemma4-2b/ from
   # mlx-community/gemma-4-e2b-it-bf16 with config.json,
   # tokenizer.json, tokenizer_config.json, chat_template.jinja,
   # model.safetensors.index.json, and referenced model-*.safetensors shards.
   ./integration-tests/download.sh gemma4-2b || true
   export XYBRID_MLX_GEMMA_DIR="$PWD/integration-tests/fixtures/models/gemma4-2b"

   ./integration-tests/download.sh gemma4-2b-bf16-gguf

   # Staged exact LFM2.5 benchmark pair:
   # export XYBRID_MLX_LFM25_DIR=/path/to/lfm2.5-1.2b-instruct-mlx
   ./integration-tests/download.sh lfm2.5-1.2b-instruct-bf16-gguf
   ```

   The 4-bit pairs are all downloadable, pinned by revision and checksum:

   ```bash
   for fixture in \
       qwen3-0.6b-4bit-mlx qwen3-0.6b-q4km-gguf \
       lfm2.5-1.2b-instruct-4bit-mlx lfm2.5-1.2b-instruct-q4km-gguf \
       gemma-4-e2b-it-4bit-mlx gemma-4-e2b-it-q4km-gguf; do
     ./integration-tests/download.sh "$fixture"
   done
   ```

   The exact same-precision pairs need full-precision (BF16) SafeTensors on
   the MLX side; an MLX 4-bit bundle has no same-precision GGUF. GGUF
   fixtures provide the llama.cpp comparison input.

2. Run the harness:

   ```bash
   tools/scripts/bench-llm-backends.sh
   ```

   Use `tools/scripts/bench-llm-backends.sh --strict` for an enforced
   gate update when every row you expect to measure has its staged fixture
   available. In strict mode, unsupported-host skips, missing-fixture
   skips, and failed benchmark cells all exit non-zero.

3. Copy the generated markdown table from
   `target/benchmark-results/llm_backend_compare.md` into the
   appropriate section above, append the date (ISO 8601) and the short
   commit SHA you measured against, and commit the update.

4. If a measured exact same-model ratio supports a tighter or looser
   floor, update the bench gate policy and floor in the **same commit**
   as the recorded baseline. Never tighten the floor without measured
   rows supporting it.

## Deferred work

- Gemma 4 and the LFM family now have synthetic-covered incremental MLX
  runtime paths. The public LFM2 BF16 bundle, public LFM2.5 1.2B Instruct
  BF16 bundle, and public sharded Gemma 4 BF16 bundle have env-gated real
  generation smokes when staged locally. Gemma 4 and LFM2.5 both have
  exact BF16 GGUF benchmark counterparts and measured rows on Apple M4
  Max; both remain informational because current MLX decode throughput
  is below the parity floor.
- Qwen 3 4B now has exact MLX and BF16 GGUF fixtures and measured rows
  on Apple M4 Max. The same-precision result is in the 1.04x-1.09x
  llama.cpp range across the recorded runs, which passes the 0.95x
  parity floor but does not justify a speedup claim. The earlier Q4_K_M
  llama.cpp row is retained as an informational quantized reference, not
  the enforced parity gate.
- Gemma 4 comparison rows stay informational until measured MLX decode
  throughput improves enough to justify parity gating against the pinned
  exact BF16 GGUF counterpart.
- The BF16 rows for Gemma 4 and LFM2.5 above are from May 2026, on MLX
  v0.31.1 and an earlier adapter, and have not been re-measured. The 4-bit
  rows are the current evidence for those two families.
- LFM3 is not an MLX row: it is a different architecture from LFM2 and the
  adapter refuses it. llama.cpp runs it from GGUF.
- The M2 Pro baseline row is unfilled until a runtime-enabled measurement
  machine with the MLX xcframework installed is available (tracked in
  US-001 / US-002 notes).
