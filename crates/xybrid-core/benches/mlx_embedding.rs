//! Benchmark MLX SafeTensors embedding throughput.
//!
//! This harness covers the staged `nomic-ai/nomic-embed-text-v1.5`
//! SafeTensors path so embedding runtime performance is tracked separately from
//! the LLM decode-throughput benchmark. The fixture is intentionally resolved
//! through the same staged fixture metadata used by runtime smoke tests:
//! `integration-tests/fixtures/models/nomic-embed-text-v1.5` or
//! `$XYBRID_MLX_NOMIC_DIR`.
//!
//! # Running
//!
//! ```bash
//! export XYBRID_MLX_NOMIC_DIR=/path/to/nomic-embed-text-v1.5
//! cargo bench -p xybrid-core --no-default-features --features llm-mlx-runtime --bench mlx_embedding
//! ```
//!
//! # Output
//!
//! - `target/benchmark-results/mlx_embedding.md` — measurement table plus notes.
//! - `XYBRID_BENCH_WARMUP_ONLY=1` runs the fixture but suppresses report writing.
//! - `XYBRID_BENCH_STRICT=1` turns a missing fixture or failed run into a non-zero
//!   exit code.

#[cfg(not(all(
    target_os = "macos",
    target_arch = "aarch64",
    feature = "llm-mlx-runtime"
)))]
fn main() {
    eprintln!(
        "mlx_embedding skipped: bench requires Apple Silicon macOS and \
         --features 'llm-mlx-runtime'"
    );
}

#[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    feature = "llm-mlx-runtime"
))]
fn main() {
    active::run();
}

#[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    feature = "llm-mlx-runtime"
))]
mod active {
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    use xybrid_core::ir::EnvelopeKind;
    use xybrid_core::runtime_adapter::mlx::bench_fixtures;
    use xybrid_core::runtime_adapter::mlx::{MlxEmbeddingAdapter, MlxEmbeddingConfig, Pooling};

    const FIXTURE_ID: &str = "nomic-embed-text-v1.5";
    const MEASUREMENT_ROUNDS: usize = 5;
    const INPUTS: &[&str] = &[
        "search_document: Xybrid routes inference between device and cloud.",
        "search_document: MLX SafeTensors embeddings should reuse resident weights.",
        "search_document: The benchmark measures steady-state local embedding latency.",
        "search_query: hybrid edge cloud inference latency",
        "search_query: mlx safetensors embedding backend",
    ];

    #[derive(Debug)]
    struct RoundMetrics {
        elapsed_ms: f64,
        vectors: usize,
        dims: usize,
        avg_tokens: f64,
    }

    #[derive(Debug)]
    enum BenchStatus {
        Ok,
        Skipped(String),
        Failed(String),
    }

    #[derive(Debug)]
    struct BenchResult {
        fixture_id: &'static str,
        rounds: usize,
        inputs_per_round: usize,
        vectors_per_second: Option<f64>,
        ms_per_input: Option<f64>,
        avg_tokens: Option<f64>,
        embedding_dim: Option<usize>,
        peak_mem_mb: Option<f32>,
        status: BenchStatus,
    }

    fn current_rss_mb() -> Option<f32> {
        use sysinfo::{Pid, ProcessesToUpdate, System};
        let mut sys = System::new();
        let pid = Pid::from_u32(std::process::id());
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        let proc = sys.process(pid)?;
        Some(proc.memory() as f32 / (1024.0 * 1024.0))
    }

    fn median_f64(values: &[f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        Some(if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        })
    }

    fn measure_round(adapter: &MlxEmbeddingAdapter) -> Result<RoundMetrics, String> {
        let started = Instant::now();
        let mut dims = None;
        for input in INPUTS {
            let envelope = adapter
                .embed(input)
                .map_err(|e| format!("embed `{input}`: {e}"))?;
            let EnvelopeKind::Embedding(values) = envelope.kind else {
                return Err("adapter returned a non-embedding envelope".into());
            };
            if values.iter().any(|value| !value.is_finite()) {
                return Err("adapter returned a non-finite embedding value".into());
            }
            let len = values.len();
            if let Some(expected) = dims {
                if expected != len {
                    return Err(format!(
                        "embedding dimension changed from {expected} to {len}"
                    ));
                }
            } else {
                dims = Some(len);
            }
        }

        let tokenizer = adapter
            .tokenizer()
            .ok_or_else(|| "adapter loaded without tokenizer".to_string())?;
        let avg_tokens = INPUTS
            .iter()
            .map(|input| {
                tokenizer
                    .encode(*input, true)
                    .map(|encoding| encoding.get_ids().len() as f64)
                    .map_err(|e| e.to_string())
            })
            .sum::<Result<f64, _>>()?
            / INPUTS.len() as f64;

        Ok(RoundMetrics {
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            vectors: INPUTS.len(),
            dims: dims.unwrap_or(0),
            avg_tokens,
        })
    }

    fn summarise(
        rounds: &[RoundMetrics],
    ) -> (Option<f64>, Option<f64>, Option<f64>, Option<usize>) {
        let steady = if rounds.len() >= 2 {
            &rounds[1..]
        } else {
            rounds
        };
        if steady.is_empty() {
            return (None, None, None, None);
        }

        let vectors_per_second: Vec<f64> = steady
            .iter()
            .map(|round| round.vectors as f64 / (round.elapsed_ms / 1000.0))
            .collect();
        let ms_per_input: Vec<f64> = steady
            .iter()
            .map(|round| round.elapsed_ms / round.vectors as f64)
            .collect();
        let avg_tokens: Vec<f64> = steady.iter().map(|round| round.avg_tokens).collect();
        let dim = steady.iter().map(|round| round.dims).find(|dim| *dim > 0);

        (
            median_f64(&vectors_per_second),
            median_f64(&ms_per_input),
            median_f64(&avg_tokens),
            dim,
        )
    }

    fn run_bench() -> BenchResult {
        let Some(model_dir) = bench_fixtures::resolve_mlx_dir(FIXTURE_ID) else {
            return BenchResult {
                fixture_id: FIXTURE_ID,
                rounds: MEASUREMENT_ROUNDS,
                inputs_per_round: INPUTS.len(),
                vectors_per_second: None,
                ms_per_input: None,
                avg_tokens: None,
                embedding_dim: None,
                peak_mem_mb: None,
                status: BenchStatus::Skipped(
                    "fixture missing: set XYBRID_MLX_NOMIC_DIR or stage integration-tests/fixtures/models/nomic-embed-text-v1.5".into(),
                ),
            };
        };

        eprintln!("mlx_embedding — loading {}", model_dir.display());
        let config = MlxEmbeddingConfig {
            max_seq_len: 128,
            pooling: Pooling::Mean,
            normalize: true,
        };
        let adapter = match MlxEmbeddingAdapter::load(&model_dir, &config) {
            Ok(adapter) => adapter,
            Err(e) => {
                return BenchResult {
                    fixture_id: FIXTURE_ID,
                    rounds: MEASUREMENT_ROUNDS,
                    inputs_per_round: INPUTS.len(),
                    vectors_per_second: None,
                    ms_per_input: None,
                    avg_tokens: None,
                    embedding_dim: None,
                    peak_mem_mb: None,
                    status: BenchStatus::Failed(format!("load: {e}")),
                };
            }
        };

        let mut rounds = Vec::with_capacity(MEASUREMENT_ROUNDS);
        let mut peak_mem_mb = current_rss_mb().unwrap_or(0.0);
        for round in 0..MEASUREMENT_ROUNDS {
            match measure_round(&adapter) {
                Ok(metrics) => {
                    eprintln!(
                        "  round {}/{} -> inputs={} dim={} elapsed={:.1}ms vectors/s={:.1}",
                        round + 1,
                        MEASUREMENT_ROUNDS,
                        metrics.vectors,
                        metrics.dims,
                        metrics.elapsed_ms,
                        metrics.vectors as f64 / (metrics.elapsed_ms / 1000.0),
                    );
                    rounds.push(metrics);
                    if let Some(rss) = current_rss_mb() {
                        if rss > peak_mem_mb {
                            peak_mem_mb = rss;
                        }
                    }
                }
                Err(e) => {
                    return BenchResult {
                        fixture_id: FIXTURE_ID,
                        rounds: MEASUREMENT_ROUNDS,
                        inputs_per_round: INPUTS.len(),
                        vectors_per_second: None,
                        ms_per_input: None,
                        avg_tokens: None,
                        embedding_dim: None,
                        peak_mem_mb: Some(peak_mem_mb),
                        status: BenchStatus::Failed(e),
                    };
                }
            }
        }

        let (vectors_per_second, ms_per_input, avg_tokens, embedding_dim) = summarise(&rounds);
        BenchResult {
            fixture_id: FIXTURE_ID,
            rounds: MEASUREMENT_ROUNDS,
            inputs_per_round: INPUTS.len(),
            vectors_per_second,
            ms_per_input,
            avg_tokens,
            embedding_dim,
            peak_mem_mb: Some(peak_mem_mb),
            status: BenchStatus::Ok,
        }
    }

    fn format_f64(value: Option<f64>) -> String {
        value
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "-".into())
    }

    fn format_usize(value: Option<usize>) -> String {
        value
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".into())
    }

    fn format_f32(value: Option<f32>) -> String {
        value
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "-".into())
    }

    fn render_markdown(result: &BenchResult) -> String {
        let mut out = String::new();
        out.push_str("# MLX Embedding Benchmark\n\n");
        out.push_str(&format!(
            "Generated by `cargo bench -p xybrid-core --no-default-features \
             --features llm-mlx-runtime --bench mlx_embedding`. \
             Fixture: `{}`. Rounds: {} (round 0 discarded). Inputs per round: {}.\n\n",
            result.fixture_id, result.rounds, result.inputs_per_round,
        ));

        out.push_str("| fixture | vectors/s | ms/input | avg input tokens | embedding dim | peak-mem-mb | status |\n");
        out.push_str("|---------|-----------|----------|------------------|---------------|-------------|--------|\n");
        let status = match &result.status {
            BenchStatus::Ok => "ok",
            BenchStatus::Skipped(_) => "skipped",
            BenchStatus::Failed(_) => "failed",
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n\n",
            result.fixture_id,
            format_f64(result.vectors_per_second),
            format_f64(result.ms_per_input),
            format_f64(result.avg_tokens),
            format_usize(result.embedding_dim),
            format_f32(result.peak_mem_mb),
            status,
        ));

        match &result.status {
            BenchStatus::Ok => {}
            BenchStatus::Skipped(reason) => {
                out.push_str("## Notes\n\n");
                out.push_str(&format!("- skipped — {reason}\n"));
            }
            BenchStatus::Failed(reason) => {
                out.push_str("## Notes\n\n");
                out.push_str(&format!("- failed — {reason}\n"));
            }
        }

        out
    }

    fn write_report(markdown: &str) -> std::io::Result<PathBuf> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|crates_dir| crates_dir.parent())
            .unwrap_or(&manifest_dir);
        let dir = repo_root.join("target/benchmark-results");
        fs::create_dir_all(&dir)?;
        let out = dir.join("mlx_embedding.md");
        fs::write(&out, markdown)?;
        Ok(out)
    }

    fn is_warmup_only() -> bool {
        std::env::var("XYBRID_BENCH_WARMUP_ONLY").ok().as_deref() == Some("1")
    }

    fn strict_enabled() -> bool {
        std::env::var("XYBRID_BENCH_STRICT").ok().as_deref() == Some("1")
    }

    pub fn run() {
        eprintln!(
            "mlx_embedding — fixture={}, rounds={}, inputs/round={}",
            FIXTURE_ID,
            MEASUREMENT_ROUNDS,
            INPUTS.len(),
        );
        let result = run_bench();

        if is_warmup_only() {
            eprintln!("XYBRID_BENCH_WARMUP_ONLY=1; skipping benchmark report");
            return;
        }

        let markdown = render_markdown(&result);
        match write_report(&markdown) {
            Ok(path) => eprintln!("wrote {}", path.display()),
            Err(e) => eprintln!("report write failed: {e}"),
        }

        if strict_enabled() && !matches!(result.status, BenchStatus::Ok) {
            eprintln!("XYBRID_BENCH_STRICT=1 and mlx_embedding did not complete");
            std::process::exit(1);
        }
    }
}
