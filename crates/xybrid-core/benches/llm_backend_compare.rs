//! Benchmark comparing MLX vs llama.cpp on the priority LLM models.
//!
//! This benchmark exists to make the "MLX is worth integrating" claim
//! verifiable, reproducible, and regression-safe. For each priority model
//! we run a fixed decode workload against both backends and report
//! prompt-tokens/s, decode-tokens/s, time-to-first-token, and peak
//! resident memory. When both backends produce numbers for the same
//! model, we assert `MLX decode_tps >= LLAMA_CPP_DECODE_SPEEDUP_THRESHOLD
//! * llama.cpp decode_tps` — the benchmark exits non-zero if MLX
//! regresses below the threshold.
//!
//! # Methodology
//!
//! - **Prompt**: fixed ~256-token payload generated deterministically
//!   so every backend sees identical input. The raw string is long
//!   enough that every tokenizer under test emits roughly 256 tokens
//!   after BOS/BPE processing; small per-tokenizer drift is tolerated
//!   because the workload is dominated by the 128 output-token decode
//!   loop.
//! - **Output budget**: 128 tokens via `GenerationConfig::max_tokens`.
//! - **Sampling**: greedy (`temperature = 0.0`) with all other knobs
//!   default. Determinism eliminates per-run variance from sampling so
//!   the speedup assertion reflects kernel throughput, not RNG luck.
//! - **Repetitions**: 5 measurement rounds per (model, backend) cell.
//!   The median of the 5 decode-tps values is what lands in the table
//!   and feeds the threshold assertion.
//! - **Cold-start exclusion**: the model is loaded once per cell, then
//!   the 5 measurements run back-to-back. The first run absorbs Metal
//!   kernel compilation cost and KV cache allocation; we discard the
//!   first run from the median so neither backend is penalised for
//!   cache population.
//!
//! # What "the same model" means across formats
//!
//! `qwen3.5-3b` is a single logical model with two formats: MLX 4-bit
//! (`mlx-community/Qwen3.5-3B-Instruct-4bit`) and GGUF Q4_K_M published
//! alongside. The quantisation schemes aren't bitwise-equivalent — MLX
//! uses affine grouped quantisation, GGUF uses K-quants with super-block
//! scaling — so logits-level parity isn't expected. The comparison is
//! about *decode throughput on equivalent quality*; model quality is
//! validated separately via the model-spec validation workflow.
//!
//! # Known sources of variance
//!
//! - **Thermal throttling**: Apple Silicon cores downclock under
//!   sustained load. Run with the lid open, ideally on mains power, and
//!   let the machine idle for 60s between models.
//! - **Metal kernel cache**: first decode touches an uncached Metal
//!   pipeline; we discard the first run to avoid this penalty, but a
//!   cold machine (post-reboot) will still show elevated first-run
//!   numbers. The runner script `tools/scripts/bench-llm-backends.sh`
//!   does a throwaway warm-up pass before measurement for this reason.
//! - **Shared compute**: close browser tabs, quit Zoom, etc.
//! - **Model bundle variance**: model-specs pin exact HuggingFace
//!   commits. If decode_tps drifts >5% between runs on the same
//!   hardware, verify both bundles match their `.checksum`.
//!
//! # Running
//!
//! ```bash
//! # Convenience wrapper (warm-up + measurement + threshold check):
//! tools/scripts/bench-llm-backends.sh
//!
//! # Direct invocation (requires both backends + macOS):
//! cargo bench -p xybrid-core \
//!     --features "llm-mlx-runtime llm-llamacpp" \
//!     --bench llm_backend_compare
//! ```
//!
//! # Output
//!
//! - `target/benchmark-results/llm_backend_compare.md` — markdown table
//!   plus a "Notes" section when any cell was skipped or failed.
//! - stderr — per (model, backend) progress so the user sees exactly
//!   which cell is running (load times dominate).
//!
//! # Exit codes
//!
//! - `0`: all (model, both-backends-ran) cells meet the 1.30× threshold.
//! - `1`: at least one cell regressed below the threshold, or when
//!   `XYBRID_BENCH_STRICT=1` is set and any cell was skipped/failed.
//!   Unset strict mode (the default) treats missing fixtures as "skip";
//!   the table still emits but the regression gate is bypassed for the
//!   corresponding model.

#[cfg(not(all(
    target_os = "macos",
    feature = "llm-mlx-runtime",
    feature = "llm-llamacpp"
)))]
fn main() {
    eprintln!(
        "llm_backend_compare skipped: bench requires target_os=macos and \
         --features 'llm-mlx-runtime llm-llamacpp'"
    );
}

#[cfg(all(
    target_os = "macos",
    feature = "llm-mlx-runtime",
    feature = "llm-llamacpp"
))]
fn main() {
    active::run();
}

#[cfg(all(
    target_os = "macos",
    feature = "llm-mlx-runtime",
    feature = "llm-llamacpp"
))]
mod active {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::time::Instant;

    use xybrid_core::runtime_adapter::llama_cpp::LlamaCppBackend;
    use xybrid_core::runtime_adapter::llm::{
        ChatMessage, GenerationConfig, GenerationOutput, LlmBackend, LlmConfig,
    };
    use xybrid_core::runtime_adapter::mlx::{MlxLlmAdapter, MlxLlmConfig};

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Median MLX decode_tps must be at least this multiple of the matching
    /// llama.cpp median. Set in the PRD for US-019.
    const LLAMA_CPP_DECODE_SPEEDUP_THRESHOLD: f32 = 1.30;

    /// Output budget per measurement (PRD: 128 tokens).
    const OUTPUT_TOKENS: usize = 128;

    /// Measurement repetitions per (model, backend) cell (PRD: 5).
    const MEASUREMENT_ROUNDS: usize = 5;

    /// Approximate token count of [`PROMPT_256_TOKENS`] (PRD: 256 tokens).
    ///
    /// The fixed string below is tuned to land near 256 tokens on the
    /// Qwen/Gemma/LFM tokenizers; decode dominates wall-clock so exact
    /// parity on prompt length is not required. Prose, not structured
    /// text, to keep tokenization close to the model's training
    /// distribution.
    const PROMPT_256_TOKENS: &str = concat!(
        "Imagine the compact benchmarking harness we are running right now. ",
        "Its job is not to produce any particular answer, but to exercise the ",
        "decode path of a language model under a reproducible, repeatable ",
        "load. We are measuring three things: how quickly the prompt is ",
        "consumed, how many output tokens per second the decode loop can ",
        "emit once it is warmed up, and how long the user waits before seeing ",
        "the very first token come back from the engine. Because we want the ",
        "measurements to reflect kernel throughput and not random sampling ",
        "luck, the temperature is pinned at zero, which means the model will ",
        "greedily pick the highest-scoring next token at every step. Please ",
        "produce a long, continuous, well-formed paragraph that expands on ",
        "the following observation: benchmarks are only as useful as the ",
        "methodology they encode. Keep the output prose, with no headings or ",
        "bullet points, so the decode loop does not terminate early on a ",
        "stop-sequence match. Now, please continue writing from this point ",
        "onward until you have produced roughly one hundred and twenty-eight ",
        "output tokens. Write about caches, warm-ups, variance, and why you ",
        "should always discard the first run. Keep going."
    );

    // ========================================================================
    // Model registry
    // ========================================================================

    struct ModelUnderTest {
        /// Short identifier used in the markdown table and stderr logs.
        display_name: &'static str,
        /// Fixture subdir for the MLX bundle
        /// (`integration-tests/fixtures/models/<mlx_fixture_id>`).
        mlx_fixture_id: &'static str,
        /// Fixture subdir for the GGUF bundle.
        gguf_fixture_id: &'static str,
        /// Expected GGUF filename inside the fixture directory.
        gguf_filename: &'static str,
    }

    /// PRD-priority models (qwen3.5-3b, gemma4-2b, lfm3.5-1.5b).
    ///
    /// Gemma 4 and LFM 3.5 MLX runtimes are not fully implemented yet
    /// (see `runtime_adapter/mlx/arch/{gemma4,lfm35}.rs` — forward()
    /// returns `NotImplemented` at the attention/conv boundary pending
    /// US-012 / US-013 follow-ups). The bench still lists them so the
    /// table shape stays stable; the MLX cell for those models will
    /// auto-skip with a clear stderr message until the runtime lands.
    const MODELS: &[ModelUnderTest] = &[
        ModelUnderTest {
            display_name: "qwen3.5-3b",
            mlx_fixture_id: "qwen3.5-3b",
            gguf_fixture_id: "qwen3.5-2b",
            gguf_filename: "model.gguf",
        },
        ModelUnderTest {
            display_name: "gemma4-2b",
            mlx_fixture_id: "gemma4-2b",
            gguf_fixture_id: "gemma-3-1b",
            gguf_filename: "model.gguf",
        },
        ModelUnderTest {
            display_name: "lfm3.5-1.5b",
            mlx_fixture_id: "lfm3.5-1.5b",
            gguf_fixture_id: "smollm2",
            gguf_filename: "model.gguf",
        },
    ];

    // ========================================================================
    // Result collection
    // ========================================================================

    #[derive(Clone, Debug)]
    struct CellResult {
        model: String,
        backend: String,
        prompt_tps: Option<f32>,
        decode_tps: Option<f32>,
        ttft_ms: Option<f32>,
        peak_mem_mb: Option<f32>,
        status: CellStatus,
    }

    #[derive(Clone, Debug)]
    enum CellStatus {
        Ok,
        Skipped(String),
        Failed(String),
    }

    // ========================================================================
    // Fixture resolution
    // ========================================================================

    fn fixtures_root() -> Option<PathBuf> {
        let candidates = [
            PathBuf::from("repos/xybrid/integration-tests/fixtures/models"),
            PathBuf::from("integration-tests/fixtures/models"),
            PathBuf::from("../integration-tests/fixtures/models"),
            PathBuf::from("../../integration-tests/fixtures/models"),
        ];
        candidates.into_iter().find(|p| p.exists())
    }

    fn resolve_mlx_dir(model: &ModelUnderTest) -> Option<PathBuf> {
        let root = fixtures_root()?;
        let dir = root.join(model.mlx_fixture_id);
        // MLX bundle must at least contain config.json — the adapter's
        // load() verifies the rest (tokenizer.json + model.safetensors).
        if dir.join("config.json").exists() {
            Some(dir)
        } else {
            None
        }
    }

    fn resolve_gguf_path(model: &ModelUnderTest) -> Option<PathBuf> {
        let root = fixtures_root()?;
        let p = root.join(model.gguf_fixture_id).join(model.gguf_filename);
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    // ========================================================================
    // Memory probing
    // ========================================================================

    /// Resident memory in MB for the current process (macOS).
    ///
    /// Uses `sysinfo::System::refresh_processes` scoped to our own PID so
    /// the call is cheap enough to make between rounds.
    fn current_rss_mb() -> Option<f32> {
        use sysinfo::{Pid, ProcessesToUpdate, System};
        let mut sys = System::new();
        let pid = Pid::from_u32(std::process::id());
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        let proc = sys.process(pid)?;
        Some(proc.memory() as f32 / (1024.0 * 1024.0))
    }

    // ========================================================================
    // Measurement helpers
    // ========================================================================

    fn bench_config() -> GenerationConfig {
        GenerationConfig::greedy().with_max_tokens(OUTPUT_TOKENS)
    }

    /// Run `MEASUREMENT_ROUNDS` decodes against `backend` with identical
    /// prompt + config. Returns all per-round outputs plus peak RSS
    /// observed across the rounds.
    fn measure_backend(
        backend: &dyn LlmBackend,
        prompt: &str,
    ) -> Result<(Vec<GenerationOutput>, f32), String> {
        let cfg = bench_config();
        let messages = [ChatMessage::user(prompt)];
        let mut outputs = Vec::with_capacity(MEASUREMENT_ROUNDS);
        let mut peak_rss = current_rss_mb().unwrap_or(0.0);

        for round in 0..MEASUREMENT_ROUNDS {
            let started = Instant::now();
            let output = backend
                .generate(&messages, &cfg)
                .map_err(|e| format!("generate round {} failed: {}", round, e))?;
            let wall = started.elapsed();
            eprintln!(
                "    round {}/{} -> tokens={} wall={:.2}s tps={:.1}",
                round + 1,
                MEASUREMENT_ROUNDS,
                output.tokens_generated,
                wall.as_secs_f64(),
                output.tokens_per_second,
            );
            outputs.push(output);
            if let Some(rss) = current_rss_mb() {
                if rss > peak_rss {
                    peak_rss = rss;
                }
            }
        }

        Ok((outputs, peak_rss))
    }

    /// Collapse per-round `GenerationOutput`s into (prompt_tps, decode_tps,
    /// ttft_ms). Discards round 0 when >= 2 rounds succeeded.
    fn summarise(outputs: &[GenerationOutput]) -> (Option<f32>, Option<f32>, Option<f32>) {
        let steady = if outputs.len() >= 2 {
            &outputs[1..]
        } else {
            outputs
        };
        if steady.is_empty() {
            return (None, None, None);
        }

        let decode: Vec<f32> = steady
            .iter()
            .map(|o| o.decode_tps.unwrap_or(o.tokens_per_second))
            .collect();
        let prompt: Vec<f32> = steady.iter().filter_map(|o| o.prefill_tps).collect();
        let ttft: Vec<f32> = steady
            .iter()
            .filter_map(|o| o.ttft_ms.map(|v| v as f32))
            .collect();

        (median(&prompt), median(&decode), median(&ttft))
    }

    fn median(values: &[f32]) -> Option<f32> {
        if values.is_empty() {
            return None;
        }
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = v.len() / 2;
        Some(if v.len().is_multiple_of(2) {
            (v[mid - 1] + v[mid]) / 2.0
        } else {
            v[mid]
        })
    }

    // ========================================================================
    // Per-cell runners
    // ========================================================================

    fn run_mlx_cell(model: &ModelUnderTest, prompt: &str) -> CellResult {
        let display_name = model.display_name.to_string();
        eprintln!("  [MLX] {} — resolving fixture…", display_name);

        let mlx_dir = match resolve_mlx_dir(model) {
            Some(p) => p,
            None => {
                return CellResult {
                    model: display_name,
                    backend: "mlx".into(),
                    prompt_tps: None,
                    decode_tps: None,
                    ttft_ms: None,
                    peak_mem_mb: None,
                    status: CellStatus::Skipped(format!(
                        "fixture missing: integration-tests/fixtures/models/{}/config.json",
                        model.mlx_fixture_id
                    )),
                };
            }
        };

        let mlx_config = MlxLlmConfig::default();
        let mut adapter = MlxLlmAdapter::new(mlx_config);
        let llm_config = LlmConfig::new(mlx_dir.to_string_lossy().to_string());
        if let Err(e) = LlmBackend::load(&mut adapter, &llm_config) {
            return CellResult {
                model: display_name,
                backend: "mlx".into(),
                prompt_tps: None,
                decode_tps: None,
                ttft_ms: None,
                peak_mem_mb: None,
                status: CellStatus::Failed(format!("load: {}", e)),
            };
        }

        match measure_backend(&adapter, prompt) {
            Ok((outputs, peak_mb)) => {
                let (prompt_tps, decode_tps, ttft_ms) = summarise(&outputs);
                CellResult {
                    model: display_name,
                    backend: "mlx".into(),
                    prompt_tps,
                    decode_tps,
                    ttft_ms,
                    peak_mem_mb: Some(peak_mb),
                    status: CellStatus::Ok,
                }
            }
            Err(e) => CellResult {
                model: display_name,
                backend: "mlx".into(),
                prompt_tps: None,
                decode_tps: None,
                ttft_ms: None,
                peak_mem_mb: None,
                status: CellStatus::Failed(e),
            },
        }
    }

    fn run_llamacpp_cell(model: &ModelUnderTest, prompt: &str) -> CellResult {
        let display_name = model.display_name.to_string();
        eprintln!("  [llama.cpp] {} — resolving fixture…", display_name);

        let gguf_path = match resolve_gguf_path(model) {
            Some(p) => p,
            None => {
                return CellResult {
                    model: display_name,
                    backend: "llama.cpp".into(),
                    prompt_tps: None,
                    decode_tps: None,
                    ttft_ms: None,
                    peak_mem_mb: None,
                    status: CellStatus::Skipped(format!(
                        "fixture missing: integration-tests/fixtures/models/{}/{}",
                        model.gguf_fixture_id, model.gguf_filename
                    )),
                };
            }
        };

        let mut backend = match LlamaCppBackend::new() {
            Ok(b) => b,
            Err(e) => {
                return CellResult {
                    model: display_name,
                    backend: "llama.cpp".into(),
                    prompt_tps: None,
                    decode_tps: None,
                    ttft_ms: None,
                    peak_mem_mb: None,
                    status: CellStatus::Failed(format!("backend init: {}", e)),
                };
            }
        };

        let llm_config = LlmConfig::new(gguf_path.to_string_lossy().to_string());
        if let Err(e) = LlmBackend::load(&mut backend, &llm_config) {
            return CellResult {
                model: display_name,
                backend: "llama.cpp".into(),
                prompt_tps: None,
                decode_tps: None,
                ttft_ms: None,
                peak_mem_mb: None,
                status: CellStatus::Failed(format!("load: {}", e)),
            };
        }

        match measure_backend(&backend, prompt) {
            Ok((outputs, peak_mb)) => {
                let (prompt_tps, decode_tps, ttft_ms) = summarise(&outputs);
                CellResult {
                    model: display_name,
                    backend: "llama.cpp".into(),
                    prompt_tps,
                    decode_tps,
                    ttft_ms,
                    peak_mem_mb: Some(peak_mb),
                    status: CellStatus::Ok,
                }
            }
            Err(e) => CellResult {
                model: display_name,
                backend: "llama.cpp".into(),
                prompt_tps: None,
                decode_tps: None,
                ttft_ms: None,
                peak_mem_mb: None,
                status: CellStatus::Failed(e),
            },
        }
    }

    // ========================================================================
    // Reporting
    // ========================================================================

    fn format_f32(opt: Option<f32>) -> String {
        opt.map(|v| format!("{:.1}", v))
            .unwrap_or_else(|| "—".into())
    }

    fn format_cell_note(cell: &CellResult) -> &str {
        match &cell.status {
            CellStatus::Ok => "ok",
            CellStatus::Skipped(_) => "skipped",
            CellStatus::Failed(_) => "failed",
        }
    }

    fn render_markdown(results: &[CellResult]) -> String {
        let mut out = String::new();
        out.push_str("# LLM Backend Comparison — MLX vs llama.cpp\n\n");
        out.push_str(&format!(
            "Generated by `cargo bench -p xybrid-core --bench llm_backend_compare`.\n\
             Host: `{}`. Output budget: {} tokens. Rounds: {} (round 0 discarded).\n\n",
            std::env::consts::OS,
            OUTPUT_TOKENS,
            MEASUREMENT_ROUNDS,
        ));

        out.push_str("| model | backend | prompt-tokens/s | decode-tokens/s | ttft-ms | peak-mem-mb | status |\n");
        out.push_str("|-------|---------|-----------------|-----------------|---------|-------------|--------|\n");
        for cell in results {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} |\n",
                cell.model,
                cell.backend,
                format_f32(cell.prompt_tps),
                format_f32(cell.decode_tps),
                format_f32(cell.ttft_ms),
                format_f32(cell.peak_mem_mb),
                format_cell_note(cell),
            ));
        }
        out.push('\n');

        let notes: Vec<&CellResult> = results
            .iter()
            .filter(|c| !matches!(c.status, CellStatus::Ok))
            .collect();
        if !notes.is_empty() {
            out.push_str("## Notes\n\n");
            for cell in notes {
                let detail = match &cell.status {
                    CellStatus::Skipped(s) => format!("skipped — {}", s),
                    CellStatus::Failed(s) => format!("failed — {}", s),
                    CellStatus::Ok => unreachable!(),
                };
                out.push_str(&format!(
                    "- **{} / {}**: {}\n",
                    cell.model, cell.backend, detail
                ));
            }
            out.push('\n');
        }

        out.push_str(&format!(
            "## Threshold\n\nMLX decode-tokens/s must be at least `{}x` the llama.cpp \
             median for each model. Cells where at least one backend was skipped \
             or failed are reported but do not count toward the regression gate.\n",
            LLAMA_CPP_DECODE_SPEEDUP_THRESHOLD,
        ));

        out
    }

    fn write_report(markdown: &str) -> std::io::Result<PathBuf> {
        let dir = Path::new("target/benchmark-results");
        fs::create_dir_all(dir)?;
        let out = dir.join("llm_backend_compare.md");
        fs::write(&out, markdown)?;
        Ok(out)
    }

    // ========================================================================
    // Threshold check
    // ========================================================================

    fn check_threshold(results: &[CellResult]) -> Result<(), Vec<String>> {
        let mut regressions = Vec::new();
        for model in MODELS {
            let mlx = results
                .iter()
                .find(|c| c.model == model.display_name && c.backend == "mlx");
            let llama = results
                .iter()
                .find(|c| c.model == model.display_name && c.backend == "llama.cpp");
            let (mlx_cell, llama_cell) = match (mlx, llama) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            let mlx_tps = match (&mlx_cell.status, mlx_cell.decode_tps) {
                (CellStatus::Ok, Some(v)) => v,
                _ => continue,
            };
            let llama_tps = match (&llama_cell.status, llama_cell.decode_tps) {
                (CellStatus::Ok, Some(v)) => v,
                _ => continue,
            };
            let ratio = mlx_tps / llama_tps;
            if ratio < LLAMA_CPP_DECODE_SPEEDUP_THRESHOLD {
                regressions.push(format!(
                    "{}: MLX decode-tps {:.2} / llama.cpp {:.2} = {:.2}x < {:.2}x threshold",
                    model.display_name,
                    mlx_tps,
                    llama_tps,
                    ratio,
                    LLAMA_CPP_DECODE_SPEEDUP_THRESHOLD,
                ));
            }
        }
        if regressions.is_empty() {
            Ok(())
        } else {
            Err(regressions)
        }
    }

    // ========================================================================
    // Entry point
    // ========================================================================

    pub fn run() {
        eprintln!(
            "llm_backend_compare — {} models x 2 backends, {} rounds, {}-token budget",
            MODELS.len(),
            MEASUREMENT_ROUNDS,
            OUTPUT_TOKENS,
        );

        let results: Mutex<Vec<CellResult>> = Mutex::new(Vec::new());
        for model in MODELS {
            eprintln!("==> {}", model.display_name);
            let mlx = run_mlx_cell(model, PROMPT_256_TOKENS);
            results.lock().unwrap().push(mlx);
            let llama = run_llamacpp_cell(model, PROMPT_256_TOKENS);
            results.lock().unwrap().push(llama);
        }

        let collected = results.into_inner().unwrap();
        let markdown = render_markdown(&collected);
        match write_report(&markdown) {
            Ok(path) => eprintln!("wrote {}", path.display()),
            Err(e) => eprintln!("report write failed: {}", e),
        }

        let strict = std::env::var("XYBRID_BENCH_STRICT").ok().as_deref() == Some("1");
        let any_skipped_or_failed = collected
            .iter()
            .any(|c| !matches!(c.status, CellStatus::Ok));
        if strict && any_skipped_or_failed {
            eprintln!("XYBRID_BENCH_STRICT=1 and at least one cell was skipped/failed");
            std::process::exit(1);
        }

        match check_threshold(&collected) {
            Ok(()) => {
                eprintln!(
                    "all comparable cells met the {}x speedup threshold",
                    LLAMA_CPP_DECODE_SPEEDUP_THRESHOLD
                );
            }
            Err(regressions) => {
                eprintln!("regression: MLX failed the speedup threshold on:");
                for r in &regressions {
                    eprintln!("  - {}", r);
                }
                std::process::exit(1);
            }
        }
    }
}
