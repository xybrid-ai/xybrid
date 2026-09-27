//! Benchmark MLX streaming cancellation unwind latency.
//!
//! This harness measures the narrow cancellation path: the elapsed time from
//! the user streaming callback returning a typed [`CloudFallbackAbort`] to
//! `generate_streaming` returning that abort to the Rust caller. It uses a
//! tiny synthetic Qwen bundle so the gate is not polluted by real-model
//! prefill or first-token latency.
//!
//! # Running
//!
//! ```bash
//! cargo bench -p xybrid-core --no-default-features \
//!   --features llm-mlx-runtime --bench mlx_cancellation_latency
//! ```
//!
//! # Output
//!
//! - `target/benchmark-results/mlx_cancellation_latency.md`
//! - `XYBRID_BENCH_WARMUP_ONLY=1` runs warmups and suppresses report writing.

#[cfg(not(all(
    target_os = "macos",
    target_arch = "aarch64",
    feature = "llm-mlx-runtime"
)))]
fn main() {
    eprintln!(
        "mlx_cancellation_latency skipped: bench requires Apple Silicon macOS and \
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
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use xybrid_core::abort::{AbortReason, CloudFallbackAbort};
    use xybrid_core::runtime_adapter::llm::{GenerationConfig, PartialToken, StreamingCallback};
    use xybrid_core::runtime_adapter::mlx::{
        generate::{self, GenerateParams},
        MlxLlmAdapter, MlxLlmConfig,
    };
    use xybrid_core::runtime_adapter::AdapterError;

    const WARMUP_ROUNDS: usize = 3;
    const MEASUREMENT_ROUNDS: usize = 30;
    const P95_BUDGET_MS: f64 = 50.0;

    struct OwnedTensor {
        name: String,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    fn push_tensor(tensors: &mut Vec<OwnedTensor>, name: &str, shape: &[usize], value: f32) {
        let element_count = shape.iter().product::<usize>();
        let mut bytes = Vec::with_capacity(element_count * std::mem::size_of::<f32>());
        for _ in 0..element_count {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        tensors.push(OwnedTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            bytes,
        });
    }

    fn write_synthetic_qwen_bundle(dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("create bundle dir: {e}"))?;

        let tok_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("qwen_tokenizer.json");
        std::fs::copy(&tok_src, dir.join("tokenizer.json"))
            .map_err(|e| format!("copy tokenizer: {e}"))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tok_src)
            .map_err(|e| format!("load tokenizer: {e}"))?;
        let vocab_size = tokenizer.get_vocab_size(true);

        const HIDDEN: usize = 16;
        const HEADS: usize = 4;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const INTERMEDIATE: usize = 32;

        let cfg = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "intermediate_size": INTERMEDIATE,
            "vocab_size": vocab_size,
            "max_position_embeddings": 128,
            "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1.0e-6,
            "tie_word_embeddings": true,
            "head_dim": HEAD_DIM
        });
        std::fs::write(dir.join("config.json"), cfg.to_string())
            .map_err(|e| format!("write config: {e}"))?;

        let mut tensors = Vec::new();
        push_tensor(
            &mut tensors,
            "model.embed_tokens.weight",
            &[vocab_size, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            "model.layers.0.input_layernorm.weight",
            &[HIDDEN],
            1.0,
        );
        push_tensor(
            &mut tensors,
            "model.layers.0.self_attn.q_proj.weight",
            &[HEADS * HEAD_DIM, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            "model.layers.0.self_attn.k_proj.weight",
            &[KV_HEADS * HEAD_DIM, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            "model.layers.0.self_attn.v_proj.weight",
            &[KV_HEADS * HEAD_DIM, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            "model.layers.0.self_attn.o_proj.weight",
            &[HIDDEN, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            "model.layers.0.self_attn.q_norm.weight",
            &[HEAD_DIM],
            1.0,
        );
        push_tensor(
            &mut tensors,
            "model.layers.0.self_attn.k_norm.weight",
            &[HEAD_DIM],
            1.0,
        );
        push_tensor(
            &mut tensors,
            "model.layers.0.post_attention_layernorm.weight",
            &[HIDDEN],
            1.0,
        );
        push_tensor(
            &mut tensors,
            "model.layers.0.mlp.gate_proj.weight",
            &[INTERMEDIATE, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            "model.layers.0.mlp.up_proj.weight",
            &[INTERMEDIATE, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            "model.layers.0.mlp.down_proj.weight",
            &[HIDDEN, INTERMEDIATE],
            0.0,
        );
        push_tensor(&mut tensors, "model.norm.weight", &[HIDDEN], 1.0);

        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        tensor.shape.clone(),
                        &tensor.bytes,
                    )
                    .map_err(|e| e.to_string()),
                )
            })
            .map(|(name, view)| view.map(|view| (name, view)))
            .collect::<Result<_, _>>()?;
        safetensors::serialize_to_file(views, &None, &dir.join("model.safetensors"))
            .map_err(|e| format!("write safetensors: {e}"))
    }

    fn config() -> GenerationConfig {
        GenerationConfig {
            max_tokens: 4,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            stop_sequences: Vec::new(),
            seed: None,
            ..Default::default()
        }
    }

    fn measure_abort_latency(
        adapter: &MlxLlmAdapter,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Duration, String> {
        let callback_started = Arc::new(Mutex::new(None::<Instant>));
        let callback_started_for_cb = callback_started.clone();
        let cb: StreamingCallback<'_> = Box::new(move |_token: PartialToken| {
            *callback_started_for_cb.lock().unwrap() = Some(Instant::now());
            Err(Box::new(CloudFallbackAbort::new(AbortReason::StressMemory)))
        });

        let result =
            generate::generate_tokens(adapter, prompt, GenerateParams::new(config), Some(cb));
        let started = callback_started
            .lock()
            .unwrap()
            .ok_or_else(|| "streaming callback did not fire".to_string())?;
        let elapsed = started.elapsed();

        let err = AdapterError::from(result.expect_err("callback abort must return an error"));
        if err.cloud_fallback_abort_reason() != Some(AbortReason::StressMemory) {
            return Err(format!("unexpected callback error: {err}"));
        }

        Ok(elapsed)
    }

    fn percentile(sorted_ms: &[f64], percentile: f64) -> f64 {
        let rank = ((percentile / 100.0) * sorted_ms.len() as f64).ceil() as usize;
        let index = rank.saturating_sub(1).min(sorted_ms.len() - 1);
        sorted_ms[index]
    }

    fn median(sorted_ms: &[f64]) -> f64 {
        let mid = sorted_ms.len() / 2;
        if sorted_ms.len().is_multiple_of(2) {
            (sorted_ms[mid - 1] + sorted_ms[mid]) / 2.0
        } else {
            sorted_ms[mid]
        }
    }

    fn write_report(samples_ms: &[f64], median_ms: f64, p95_ms: f64, max_ms: f64) {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|crates_dir| crates_dir.parent())
            .unwrap_or(&manifest_dir);
        let out_dir = repo_root.join("target/benchmark-results");
        fs::create_dir_all(&out_dir).expect("create benchmark-results");
        let report = out_dir.join("mlx_cancellation_latency.md");
        let status = if p95_ms <= P95_BUDGET_MS {
            "PASS"
        } else {
            "FAIL"
        };
        let body = format!(
            "# MLX Cancellation Latency\n\n\
             Synthetic Qwen callback-unwind latency. Timing starts inside the \
             streaming callback immediately before returning `CloudFallbackAbort` \
             and stops after `generate_streaming` returns the typed abort.\n\n\
             | samples | median_ms | p95_ms | max_ms | budget_ms | status |\n\
             |---:|---:|---:|---:|---:|---|\n\
             | {} | {:.3} | {:.3} | {:.3} | {:.1} | {} |\n\n",
            samples_ms.len(),
            median_ms,
            p95_ms,
            max_ms,
            P95_BUDGET_MS,
            status
        );
        fs::write(&report, body).expect("write latency report");
        eprintln!("report: {}", report.display());
    }

    pub fn run() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_synthetic_qwen_bundle(tmp.path()).expect("write synthetic qwen bundle");
        let adapter =
            MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load synthetic qwen");
        let config = config();
        let prompt = "Hello";

        eprintln!("mlx_cancellation_latency — warmups: {WARMUP_ROUNDS}");
        for _ in 0..WARMUP_ROUNDS {
            let _ = measure_abort_latency(&adapter, prompt, &config).expect("warmup abort");
        }

        if std::env::var("XYBRID_BENCH_WARMUP_ONLY").ok().as_deref() == Some("1") {
            eprintln!("XYBRID_BENCH_WARMUP_ONLY=1; skipping benchmark report and p95 gate");
            return;
        }

        eprintln!("mlx_cancellation_latency — samples: {MEASUREMENT_ROUNDS}");
        let mut samples_ms = Vec::with_capacity(MEASUREMENT_ROUNDS);
        for _ in 0..MEASUREMENT_ROUNDS {
            samples_ms.push(
                measure_abort_latency(&adapter, prompt, &config)
                    .expect("measure abort")
                    .as_secs_f64()
                    * 1000.0,
            );
        }
        samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_ms = median(&samples_ms);
        let p95_ms = percentile(&samples_ms, 95.0);
        let max_ms = *samples_ms.last().unwrap();
        println!(
            "mlx_cancellation_latency: samples={} median_ms={:.3} p95_ms={:.3} max_ms={:.3} budget_ms={:.1}",
            samples_ms.len(),
            median_ms,
            p95_ms,
            max_ms,
            P95_BUDGET_MS
        );
        write_report(&samples_ms, median_ms, p95_ms, max_ms);

        if p95_ms > P95_BUDGET_MS {
            eprintln!(
                "MLX cancellation latency p95 exceeded budget: {:.3}ms > {:.1}ms",
                p95_ms, P95_BUDGET_MS
            );
            std::process::exit(1);
        }
    }
}
