# Observability performance and memory benchmarks

What telemetry costs an application that streams tokens from a local model,
measured against a disabled baseline, plus the bounded-resource claims of the
export pipeline under bursts, a blocked exporter and saturation: the
performance acceptance of the observability stack. It records distributions
and gates them on the agreed bounds in `thresholds.json`: for e2e, mostly the
median delta of each mode against `off` over many rounds, plus a limit on the
worst shutdown time; for the micro benches, the measured value. A failed bound
makes `e2e` and `micro` exit 1; `"status": "proposed"` in `thresholds.json`
turns the gate off (see [Thresholds](#thresholds)).

| File | Role |
|---|---|
| `run_perf.py` | Entry point: `fetch-model`, `e2e`, `micro`, `report`. |
| `micro.py` | Runs and collects the Rust Criterion benches and the Node bench. |
| `perf_common.py` | Environment capture, distributions, bootstrap comparisons, thresholds, Markdown. |
| `thresholds.json` | The **agreed** thresholds with their rationale (`status: agreed`, 2026-11-09). |
| `fake_driver.py` | Stand-in for the real driver: same arguments and JSON, no model (dry runs, tests). |
| [`crates/xybrid-sdk/examples/observability_perf.rs`](../../../crates/xybrid-sdk/examples/observability_perf.rs) | The e2e driver: one process, one telemetry mode, a real GGUF through `XybridModel::run_streaming`. |
| [`crates/xybrid-observability/benches/`](../../../crates/xybrid-observability/benches/) | Criterion benches `stream_overhead`, `content_pipeline`, `saturation`. |
| [`bindings/observability-js/scripts/bench.mjs`](../../../bindings/observability-js/scripts/bench.mjs) | The same op/chunk/content workloads for `@xybrid/observability` in Node. |

Python 3.9+, standard library only. Unit tests:
`tools/scripts/tests/test_observability_perf.py` (runs a whole e2e against
`fake_driver.py`; part of CI's `tools/scripts/tests` discovery).

## Running it on a Mac

Prepare the machine: on AC power, Low Power Mode off, other heavy apps
closed, the machine idle for a minute. OrbStack/Docker is not needed. Nothing
contacts a hosted service: every endpoint is local, the driver's environment
loses `OTEL_*`, the Xybrid API keys (`XYBRID_API_KEY`,
`XYBRID_STAGING_API_KEY`, `XYBRID_PLATFORM_API_KEY`) and
`XYBRID_RESOURCE_TELEMETRY`, and the model is a local directory.

```bash
# 1. The model, once: the pinned Qwen3.5-0.8B Q4_K_M GGUF (533 MB) from
#    Hugging Face, SHA-256 verified, plus model_metadata.json from models.json.
python3 tools/scripts/observability_perf/run_perf.py fetch-model

# 2. End to end: builds the driver once (release, --features llm-llamacpp),
#    then 10 rounds x 4 modes, one process each (~10-15 minutes).
python3 tools/scripts/observability_perf/run_perf.py e2e
#    -> target/observability-perf/e2e-<stamp>/report.{json,md}

# 3. Micro benchmarks: Rust Criterion benches (release `bench` profile) and
#    the Node bench (~15-20 minutes). The e2e report supplies the per-token
#    and time-to-first-token references for the relative thresholds.
#    `pnpm exec tsdown` builds dist/ without bun (`pnpm build` runs through
#    bun). Without a built package `micro` fails and lists J1/J2 as not run;
#    pass --skip-js to leave the JS bench out on purpose.
(cd bindings/observability-js && pnpm install --frozen-lockfile && pnpm exec tsdown)
python3 tools/scripts/observability_perf/run_perf.py micro \
    --e2e-report target/observability-perf/e2e-<stamp>/report.json
#    -> target/observability-perf/micro-<stamp>/micro-report.{json,md}

# Re-render both after editing thresholds.json (nothing re-runs):
python3 tools/scripts/observability_perf/run_perf.py report \
    --e2e target/observability-perf/e2e-<stamp> --micro target/observability-perf/micro-<stamp>
```

Every report states the chip, core counts, memory, macOS version and build,
power source, Low Power Mode, thermal state, `rustc -vV`, Node version, and
the commit with a dirty flag. `raw.json` keeps every process's measurements;
`runs/` keeps each driver's JSON, stdout and stderr (with the
`/usr/bin/time -l` block).

Useful `e2e` options: `--rounds N` (processes per mode), `--runs-per-process`,
`--warmup`, `--max-tokens` (128), `--prompt-bytes` (2048), `--modes`,
`--cooldown-s` (2), `--shuffle-seed` (shuffle each round instead of the
balanced order), `--process-timeout-s` (600; kills the driver's whole
process group, the `/usr/bin/time` wrapper included), `--driver PATH` (skip
the build), `--model-dir` with `--allow-unpinned-model` (another model; its
SHA-256 is recorded and the report says UNPINNED). `micro` takes `--quick`,
`--benches`, `--criterion-args`, `--skip-rust`, `--skip-js`, `--collect-only`.

The model is the smallest text-generation GGUF the repo tracks with a
Qwen3.5-0.8B recipe (`integration-tests/fixtures/models/models.json`
`qwen3.5-0.8b`; the CI lane "Qwen3.5-0.8B GGUF (regenerate or verify)"
converts the same model). `integration-tests/download.sh qwen3.5-0.8b`
fetches the same file from the unpinned `main` revision; `fetch-model`
pins revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` and SHA-256
`bd258782…dc517`. `e2e` hashes and records the file that `model_metadata.json`
names (`execution_template.model_file`), exits when it names none or a missing
one, and without `--allow-unpinned-model` refuses anything but the pinned
bytes.

## End to end (`e2e`)

One process per (round, mode). Each process loads the model, runs
`--warmup` discarded streaming runs, then `--runs-per-process` measured
runs (greedy decoding, `max_tokens` 128, a ~2 KiB prompt plus a system
prompt), then shuts telemetry down and writes its JSON. The process runs
under `/usr/bin/time -l` for `maximum resident set size` and
`peak memory footprint` (the latter includes Metal/IOKit memory that RSS
misses). Modes:

| mode | telemetry |
|---|---|
| `off` | No session: the SDK observer is disabled. |
| `metadata` | Packaged OTLP pipeline, all signals, default limits, metadata-only content, to a local `capture_server.py`. |
| `content` | The same with the default content capture (`span_only`), to its own `capture_server.py`. |
| `blocked` | `content` against a local endpoint that accepts connections and never answers. |

Every round runs each mode once, in the order of a Williams design: over
every four rounds (four modes) each mode takes each position once, so slow
drift (thermal, page cache, background work) spreads over every mode, and
within a round each mode directly follows each other mode once, so no mode
keeps the same predecessor. A plain rotation would put `off`, the baseline,
right after `blocked` almost every time and fold that process's carryover
(GPU clocks, heat, its slow exit) into every comparison. The step from one
round to the next is not balanced, so `workload.predecessors` in the report
records how often each mode followed each other one. `--shuffle-seed`
shuffles each round instead.

Per measured run, from the application's token callback:
`first_callback_ms` (run start to first callback), the gaps between
consecutive callbacks, the callback-derived decode rate, and wall time; from
the runtime: `ttft_ms`, `tokens_per_second`, `tokens_generated`. The SDK
reads the chunk's time just before invoking the application's callback and,
once the callback accepts the token, records the chunk at that time
(`InferenceOperation::record_chunk_at`; a token the callback refuses is not
counted). The per-chunk cost therefore lands between consecutive callbacks,
in every callback gap; the last chunk's falls after the final callback, in
wall time only. Per process: peak RSS and footprint, model load time,
shutdown time (timed separately; never counted as inference) and the
pipeline counters before and after shutdown.

The report gives per-mode distributions (p5/p50/p95 and more over runs or
processes), the pooled gap distribution, and for each enabled mode the median
delta against `off` with a bootstrap 95% interval (2000 resamples, fixed
seed). Sanity checks confirm each mode measured what it claims: the same
callback count in every run of every mode, the sinks received traces, content
requests are larger than metadata requests, the blocked endpoint got
connections and nothing was acknowledged, `off` has no counters. `e2e` exits
1 when a process failed or a sanity check did not hold (the report is still
written; the JSON line on stdout names the failed checks).

## Micro benchmarks (`micro`)

`cargo bench -p xybrid-observability` (the `saturation` bench reads
`SignalStats::buffered_bytes` from `ObservabilityPipeline::stats`, the running
total the 32 MiB byte budget admits against). Each bench writes a
JSON side report next to Criterion's output (`XYBRID_OBS_BENCH_DIR`, default
`target/observability-bench`).

- **`stream_overhead`**: one streaming inference as the SDK records it
  (start, 512 B system instructions, 1 KiB input, `N` chunks with a measured
  first token, usage, finish reason, a `4 x N` byte answer, end) for
  `N` = 1, 32, 256, and one `record_chunk` on a running operation. Setups:
  `disabled`; `memory_metadata` and `memory_content` (SDK tracer with a simple
  processor into an in-memory exporter, meter and logger providers);
  `otlp_blackhole` (the packaged pipeline to a local endpoint answering 200 at
  once); `otlp_blocked` (to one that never answers). The report derives the
  fixed per-operation overhead (`N`=1 minus disabled) and the per-chunk
  overhead (slope over `N` minus the disabled slope). The side report times
  every `record_chunk` call (2000 operations x 256 chunks per setup) for
  p50/p99/p99.9/max: the latency a token callback sees.
- **`content_pipeline`**: `record_input_messages` alone (capture gate,
  preflight, clone, callback redactor, declarative rules, truncation to the
  16 KiB attribute budget, retained-operation accounting) for one message of
  256 B to 256 KiB, 32 x 4 KiB, the largest 4 KiB-message and 64 B-message
  histories within the 480 KiB traversal budget, and one message over it.
  Policies: `metadata_only`, `default`, `rules` (three `replace_literal`, one
  `omit_attribute` on another key), `hook_and_rules` (a substring callback
  redactor first), `unsampled` (the default policy under a sampler that drops
  every span). Content for a span that is not recording is not processed, so `unsampled`
  measures the capture gate and should cost about what `metadata_only` does;
  before that gate it cost the same as `default` (Rust 256 KiB: 140 µs vs
  141 µs; JS 3.27 vs 3.38 ms, 2026-11-09 Mac run).
  Every (policy, workload) outcome (kept, truncated, failed closed, skipped)
  is asserted before timing, so a bench cannot silently measure another
  path. Under `hook_and_rules` the near-budget histories fail
  closed: `[REDACTED]` is longer than the words it replaces and pushes them
  over the budget (CONTRACT.md step 6).
- **`saturation`**: 8 threads x 1250 short streaming operations (16 chunks,
  1 KiB in/out) against a blocked endpoint; the same with 15 KiB in/out (the
  byte budget binds before the 2048-item queue); 64 threads x 250; and 8 x
  1250 against the answering endpoint. Per scenario (5 repetitions, fresh
  pipeline each): burst wall time, per-operation latency percentiles,
  accepted/dropped/pending, the sampled peak of library-owned bytes
  (`buffered_bytes`) against the 32 MiB budget and that none are held after
  shutdown, pending against the queue, process RSS before / sampled
  peak / after shutdown, shutdown time. Criterion's `burst` group times the
  bursts.
- **JS** (`pnpm bench`): `disabled`, `memory_metadata` and `memory_content`
  over OpenTelemetry JS SDK in-memory providers, the same stream operation
  and `recordChunk` measurements, and the same content workloads and policies
  (asserted the same way).

`cargo test -p xybrid-observability --features test-util --benches` runs
every benchmark once with smaller side reports (a smoke check); `micro
--profile dev --quick` does a fast but unoptimized pass.

## Thresholds

Agreed on 2026-11-09 (`thresholds.json` has `status: agreed`) after two
real-model runs on an Apple M5 Max (10 and 20 rounds per mode) and the micro
benches, in which every bound held. Each e2e row compares the **regression**
of a mode against `off` (slower, later or larger is positive) with the bound.
With both an absolute and a relative allowance, the larger applies.

The gate is `within`: the median delta against `off` for relative e2e rows,
the row's statistic against its limit for absolute ones (E8, E9: the maximum
shutdown time over processes), the measured value for micro rows. A failed gate makes `e2e` and `micro` exit 1 and lists the rows
under `thresholds_failed`. `within at CI worst case` (the end of the bootstrap
95% interval in the regressing direction) is reported for information only:
on a desktop the e2e noise floor is a few percent (the `blocked` mode, which
does no telemetry work on the token path, spanned -1.9% to +3.7% tokens/s over
20 rounds), far above the per-token cost the micro benches measure (about
0.003% of a token), so the interval end mostly measures the machine. Run on an
idle machine, and add rounds when the intervals are wide. A file with
`"status": "proposed"` turns the gate off again; `report` re-renders without
re-running.

The anchor is the measured per-token time. On an M-series Mac a 0.8B Q4 model
decodes at roughly 100-250 tokens/s (4-10 ms per token), while `record_chunk`
costs well under a microsecond, so per-token telemetry is expected at
~0.01% of a token; the bounds sit one to two orders of magnitude above that,
tight enough to catch a regression that puts real work (allocation storms,
locks shared with an exporter, I/O, a flush) on the token path, loose enough
not to flag run-to-run noise.

| id | what | bound | rationale |
|---|---|---|---|
| E1, E2 | tokens/s (runtime; callback-derived), enabled modes vs `off` | <= 1% slower | per-token cost ~0.01% expected; 1% catches ~50-100 us per token |
| E3 | first callback | <= max(2 ms, 2%) later | operation start plus one prompt through the content pipeline (tens of us for 2 KiB) |
| E4 | median callback gap | <= max(50 us, 1%) longer | the per-chunk cost lands here directly |
| E5 | p99 callback gap | <= max(0.5 ms, 5%) longer | tails catch blocking; `blocked` must look like any enabled mode |
| E6, E7 | peak RSS; peak footprint | <= 16 MiB larger | runtime threads, HTTP client, a few spans; content retained <= 416 KiB per operation (the default budget over OTLP/protobuf, clamped to fit the 480 KiB request bound; 208 KiB over OTLP/JSON); the 32 MiB buffer cannot fill with a handful of operations |
| E8 | shutdown, `blocked` | <= timeout + 2.5 s | timeout plus the pipeline's settle grace (2 x 1 s) and 250 ms slack |
| E9 | shutdown, answering sink | <= 1 s | a final flush of a few spans and one metrics collection |
| M1, J1 | per-chunk overhead / e2e per-token time | <= 0.1% | ~5-10 us at 100-200 tok/s, 10x the expected cost |
| M2 | fixed per-operation overhead / e2e time to first token | <= 0.5% | start + 1.5 KiB content + end against a prefill of tens of ms |
| M3 | `record_chunk` p99.9 | <= 20 us | no call should wait, even while exporting or blocked |
| M4, J2 | near-budget histories (~480 KiB), `rules`, p99 | <= 10 ms (Rust), <= 20 ms (JS) | once per request; below 1% of the prefill such a prompt needs |
| M5 | 1-17 KiB messages, `rules`, p99 | <= 250 us | typical prompts and answers, truncation included |
| M6 | saturation bounds | must hold | invariant: bytes <= 32 MiB, pending <= 2048, every span counted |
| M7 | p99 op latency, blocked vs answering endpoint | <= 1.5x | the caller never waits on export |

M6 is a correctness invariant, not a tunable: a violation is a bug.
Absolute micro bounds (M3-M5) are for the M5 Max class of machine; derive
others from a run there.
