# LLM Backend Baseline — MLX vs llama.cpp

This document records the reference decode-throughput numbers used by the
regression gate in `crates/xybrid-core/benches/llm_backend_compare.rs`.
New results should be produced by running
`tools/scripts/bench-llm-backends.sh` on the relevant hardware and
appending a row to the table below before updating the pass threshold.

## Why a baseline?

The bench asserts `MLX decode_tps >= 1.30 * llama.cpp decode_tps` for
every (model, host) row where both backends ran. Committing the numbers
that produced that ratio means:

1. Future regressions are attributable to a specific commit rather than
   a drifting environment.
2. New reviewers can see at a glance whether the 1.30× claim is based on
   best-case tuning or sustained steady-state behaviour.
3. The methodology (thermal policy, cold-cache handling, per-round
   repetitions) is visible alongside the numbers.

## Methodology

See the module doc of
[`benches/llm_backend_compare.rs`](../../crates/xybrid-core/benches/llm_backend_compare.rs)
for the full spec. Summary:

- Greedy decode, 256-token prompt, 128-token output budget.
- 5 measurement rounds; the first round is discarded; the remaining 4
  rounds' median lands in the table.
- Cold-cache mitigation: `bench-llm-backends.sh` runs a throwaway
  warm-up pass before measurement so Metal kernels are already compiled
  when the measurement timer starts.
- Thermal mitigation: start from idle, mains power, lid open, nothing
  else on the GPU.

## Reference runs

Unfilled rows are expected to be populated once the hardware is
available. Each row records a single (model, backend, host) triple so
you can diff two baselines cleanly.

### Apple M2 Pro (16 GB unified memory, macOS 15)

| model | backend | prompt-tokens/s | decode-tokens/s | ttft-ms | peak-mem-mb | date | commit |
|-------|---------|-----------------|-----------------|---------|-------------|------|--------|
| qwen3.5-3b | mlx | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| qwen3.5-3b | llama.cpp | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| gemma4-2b | mlx | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| gemma4-2b | llama.cpp | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| lfm3.5-1.5b | mlx | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| lfm3.5-1.5b | llama.cpp | _pending_ | _pending_ | _pending_ | _pending_ | — | — |

> The M2 Pro baseline is the canonical gate — the bench assertion
> `LLAMA_CPP_DECODE_SPEEDUP_THRESHOLD = 1.30` is calibrated against
> this row. M4/M3 rows below are informational only; they are not used
> to fail the bench.

### Apple M4 (16 GB unified memory, macOS 15)

| model | backend | prompt-tokens/s | decode-tokens/s | ttft-ms | peak-mem-mb | date | commit |
|-------|---------|-----------------|-----------------|---------|-------------|------|--------|
| qwen3.5-3b | mlx | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| qwen3.5-3b | llama.cpp | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| gemma4-2b | mlx | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| gemma4-2b | llama.cpp | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| lfm3.5-1.5b | mlx | _pending_ | _pending_ | _pending_ | _pending_ | — | — |
| lfm3.5-1.5b | llama.cpp | _pending_ | _pending_ | _pending_ | _pending_ | — | — |

## How to update

1. Fetch both bundles for the model row you're updating:

   ```bash
   cd integration-tests && ./download.sh qwen3.5-3b qwen3.5-2b
   ```

   The MLX fixture is the mlx-community bundle; the GGUF fixture
   doubles as the llama.cpp comparison input. Dir names match
   `integration-tests/fixtures/models/<id>/`.

2. Run the harness:

   ```bash
   tools/scripts/bench-llm-backends.sh
   ```

3. Copy the generated markdown table from
   `target/benchmark-results/llm_backend_compare.md` into the
   appropriate section above, append the date (ISO 8601) and the short
   commit SHA you measured against, and commit the update.

4. If the new ratio is stably >1.30×, you may raise
   `LLAMA_CPP_DECODE_SPEEDUP_THRESHOLD` in the bench source in the
   **same commit** — never tighten the threshold without a recorded
   baseline supporting it.

## Deferred work

- The runtime forward pass for Gemma 4 and LFM 3.5 is still staged
  behind `NotImplemented` in `runtime_adapter/mlx/arch/{gemma4,lfm35}.rs`
  pending US-012 / US-013 follow-ups. Those rows will surface as
  `skipped` in the bench output until the runtime lands.
- The M2 Pro baseline row is unfilled until the `mlx.xcframework` CI
  artefact has landed a signed release and a measurement machine is
  available (tracked in US-001 / US-002 notes).
