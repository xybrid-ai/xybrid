//! MLX generation orchestrator (US-014).
//!
//! Wires together the pieces the [`MlxLlmAdapter`] doesn't own directly:
//! chat-template rendering, tokeniser encoding, the arch-specific forward
//! pass, the sampler, optional streaming-callback fan-out, and telemetry
//! emission. The adapter's [`LlmBackend::generate`] /
//! [`LlmBackend::generate_raw`] / [`LlmBackend::generate_streaming`]
//! methods are thin shims over [`generate_tokens`] (for the token-by-token
//! loop) and [`render_prompt`] (for chat-template application).
//!
//! # Split of responsibility
//!
//! - **`generate.rs` (this file)** — cross-platform orchestration: prompt
//!   build, token append/emit, sampler dispatch, stop-condition checks,
//!   telemetry.  Unit-testable without MLX.
//! - **Arch builders (`arch/qwen35::runtime` etc.)** — transformer forward
//!   pass.  Gated on `llm-mlx-runtime` + Apple because they touch
//!   `xybrid_mlx::MlxArray`.
//!
//! When `llm-mlx-runtime` is off (or on a non-Apple target), the forward-
//! pass entry returns [`MlxLlmError::NotImplemented`] — the orchestration
//! remains fully covered by unit tests (sampler, chat-template, stop-
//! sequence logic) and the error points the caller at the missing feature
//! gate.

use crate::runtime_adapter::llm::{
    ChatMessage, GenerationConfig, GenerationOutput, StreamingCallback,
};

use super::chat_template::RenderOptions;
use super::model::{MlxLlmAdapter, MlxLlmError, MlxLlmResult};
use super::sampler::Sampler;

/// Default seed used by the non-streaming `generate` path when the caller
/// hasn't requested a seeded sampler. Matches OpenAI's public seed
/// convention — callers that need reproducibility should call
/// [`generate_tokens`] directly with their own `Sampler::seeded(seed)`.
const DEFAULT_SAMPLER_SEED: u64 = 0;

/// Render a chat-message list into the raw prompt string the tokenizer
/// sees. Loads the bundle's chat template from `tokenizer_config.json`
/// (cached on the adapter's [`LoadedState`](super::model)), applies it with
/// `add_generation_prompt = true` so the LLM continues mid-turn.
pub fn render_prompt(adapter: &MlxLlmAdapter, messages: &[ChatMessage]) -> MlxLlmResult<String> {
    let template = adapter
        .chat_template()
        .ok_or(MlxLlmError::NotLoaded)?
        .clone();
    template
        .render(messages, &RenderOptions::default())
        .map_err(|e| MlxLlmError::ConfigInvalid(format!("chat template render failed: {e}")))
}

/// High-level generation parameters shared across generate / stream / chat.
/// Exposed as a builder so the adapter shims don't have to duplicate the
/// sampler-construction logic.
#[derive(Debug)]
pub struct GenerateParams<'cfg> {
    pub config: &'cfg GenerationConfig,
    /// Pre-constructed sampler. Callers that want a deterministic seed
    /// construct `Sampler::seeded(N)` themselves; the adapter's default
    /// `LlmBackend::generate` path uses [`DEFAULT_SAMPLER_SEED`] for
    /// reproducibility across the streaming-vs-batched parity test.
    pub sampler: Sampler,
}

impl<'cfg> GenerateParams<'cfg> {
    pub fn new(config: &'cfg GenerationConfig) -> Self {
        Self {
            config,
            sampler: Sampler::seeded(DEFAULT_SAMPLER_SEED),
        }
    }

    pub fn with_sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = sampler;
        self
    }
}

/// Orchestrate the token-by-token generation loop for a pre-rendered
/// prompt. The arch-specific forward pass is dispatched internally based on
/// [`MlxLlmAdapter::architecture`].
///
/// When `callback` is `Some`, it is invoked for each generated token with a
/// [`PartialToken`]. Returning `Err(_)` from the callback aborts generation
/// with `finish_reason = "error"`.
pub fn generate_tokens<'cb>(
    adapter: &MlxLlmAdapter,
    prompt: &str,
    params: GenerateParams<'_>,
    callback: Option<StreamingCallback<'cb>>,
) -> MlxLlmResult<GenerationOutput> {
    let tokenizer = adapter.tokenizer().ok_or(MlxLlmError::NotLoaded)?;

    // Encode the prompt. The BOS / EOS are handled by the chat template
    // (they're baked into the rendered string for models that need them),
    // so we pass `add_special_tokens = false` through the tokenizer.
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| MlxLlmError::TokenizerLoad(format!("prompt encode failed: {e}")))?;
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&u| i64::from(u)).collect();

    let arch = adapter.architecture().ok_or(MlxLlmError::NotLoaded)?;

    // Budget check — we require at least 1 token of headroom for generation.
    let ctx_len = adapter.config().max_seq_len;
    if prompt_ids.len() >= ctx_len {
        return Err(MlxLlmError::ConfigInvalid(format!(
            "prompt length ({}) exceeds model context window ({})",
            prompt_ids.len(),
            ctx_len
        )));
    }
    let max_decode = params
        .config
        .max_tokens
        .min(ctx_len - prompt_ids.len())
        .max(1);

    // Resolve EOS token(s) from the tokenizer.
    let eos_tokens = resolve_eos_tokens(tokenizer);

    // Non-runtime build: surface an actionable error instead of pretending
    // to decode. The rest of the orchestration is still exercised by the
    // unit tests that construct a mock adapter.
    #[cfg(not(all(
        feature = "llm-mlx-runtime",
        any(target_os = "macos", target_os = "ios")
    )))]
    {
        let _ = (params, callback, eos_tokens, max_decode, arch);
        Err(MlxLlmError::NotImplemented {
            feature: "MLX forward pass (build with `--features llm-mlx-runtime` on an Apple host, \
                 and fetch mlx.xcframework via tools/scripts/fetch-mlx-xcframework.sh)",
            story: "US-014",
        })
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        any(target_os = "macos", target_os = "ios")
    ))]
    {
        runtime::generate_runtime(
            adapter,
            arch,
            &prompt_ids,
            max_decode,
            &eos_tokens,
            params,
            callback,
        )
    }
}

/// Extract EOS token IDs from the loaded tokenizer. Returns every id the
/// tokenizer marks as EOS — some models (Qwen 3) register multiple (the
/// primary `<|im_end|>` plus `<|endoftext|>` as a secondary stop).
fn resolve_eos_tokens(tokenizer: &tokenizers::Tokenizer) -> Vec<i64> {
    // tokenizers 0.19 exposes the EOS token as part of the added-tokens map,
    // not a dedicated accessor. Look for the HF-canonical names.
    const EOS_NAMES: &[&str] = &["<|endoftext|>", "<|im_end|>", "</s>", "<|end_of_text|>"];
    let mut out: Vec<i64> = Vec::new();
    for name in EOS_NAMES {
        if let Some(id) = tokenizer.token_to_id(name) {
            out.push(i64::from(id));
        }
    }
    out
}

/// The MLX-linked decode loop. Only compiled when the xcframework is
/// linkable (`llm-mlx-runtime` on an Apple target) — everything below
/// touches `xybrid_mlx::MlxArray` either directly or via the arch builder.
#[cfg(all(
    feature = "llm-mlx-runtime",
    any(target_os = "macos", target_os = "ios")
))]
mod runtime {
    use super::super::model::ModelArchitecture;
    use super::*;
    use crate::runtime_adapter::types::PartialToken;
    use std::time::Instant;
    use tracing::{debug, info};

    pub(super) fn generate_runtime<'cb>(
        adapter: &MlxLlmAdapter,
        arch: ModelArchitecture,
        prompt_ids: &[i64],
        max_decode: usize,
        eos_tokens: &[i64],
        mut params: GenerateParams<'_>,
        mut callback: Option<StreamingCallback<'cb>>,
    ) -> MlxLlmResult<GenerationOutput> {
        let model_dir = adapter.model_dir().ok_or(MlxLlmError::NotLoaded)?;
        let weights_path = model_dir.join("model.safetensors");

        let start = Instant::now();
        let mut first_token_time: Option<Instant> = None;
        let mut inter_chunk_ms: Vec<u32> = Vec::new();
        let mut last_tick = Instant::now();

        // Currently only Qwen 3.5 has a completed forward pass. Gemma 4 /
        // LFM 3.5 builders return NotImplemented at their attention/FFN
        // boundary; those land in follow-up stories.
        match arch {
            ModelArchitecture::Qwen35 => {}
            ModelArchitecture::Gemma4 => {
                return Err(MlxLlmError::NotImplemented {
                    feature: "Gemma 4 generation (attention boundary still open in arch/gemma4)",
                    story: "US-012 follow-up",
                });
            }
            ModelArchitecture::Lfm35 => {
                return Err(MlxLlmError::NotImplemented {
                    feature:
                        "LFM 3.5 generation (conv/attention boundary still open in arch/lfm35)",
                    story: "US-013 follow-up",
                });
            }
        }

        // Build weights + config from the bundle. Each generate() call
        // rebuilds the weight handles — this is cheap on MLX because the
        // underlying MTLBuffer / safetensors data hits the FS cache and the
        // MlxArray constructors are reference-bumps. A future optimisation
        // keeps the built weights on the adapter; we defer that to avoid
        // caching concerns on the first pass.
        let qwen_cfg = super::super::arch::qwen35::Qwen3Config::from_model_dir(model_dir)?;
        let weights = super::super::arch::qwen35::runtime::build(&qwen_cfg, &weights_path)?;

        let stream = None; // default CPU stream — GPU dispatch is a separate knob

        // Generated token history: starts with the prompt, appended as we
        // decode so the sampler's repetition_penalty sees the whole history.
        let mut all_tokens: Vec<i64> = prompt_ids.to_vec();
        let mut generated: Vec<i64> = Vec::with_capacity(max_decode);
        let mut cumulative_text = String::new();
        let mut finish_reason = "length";

        info!(
            model_id = adapter.model_id_or_default(),
            prompt_tokens = prompt_ids.len(),
            max_tokens = max_decode,
            "mlx.generate.start"
        );

        // Prefill: run the forward pass on the full prompt, then slice the
        // last token's logits.
        let prefill_ids = xybrid_mlx::MlxArray::from_slice_i32(
            &prompt_ids.iter().map(|&t| t as i32).collect::<Vec<i32>>(),
            &[1, prompt_ids.len() as i32],
        )?;
        let logits = super::super::arch::qwen35::runtime::forward(
            &qwen_cfg,
            &weights,
            &prefill_ids,
            0,
            stream,
        )?;
        let mut next_logits_row = last_token_logits(&logits, qwen_cfg.vocab_size)?;

        for step in 0..max_decode {
            let next_token = params
                .sampler
                .sample(&next_logits_row, params.config, &all_tokens)?;

            all_tokens.push(next_token);
            generated.push(next_token);

            // Decode the new token to text. HuggingFace tokenizers return
            // bytes for partial-UTF-8 sequences; we tolerate that by falling
            // back to the replacement character.
            let token_text = adapter
                .tokenizer()
                .unwrap()
                .decode(&[next_token as u32], true)
                .unwrap_or_else(|_| "\u{FFFD}".to_string());
            cumulative_text.push_str(&token_text);

            // Timing: first-token latency + inter-chunk gaps.
            let now = Instant::now();
            if first_token_time.is_none() {
                first_token_time = Some(now);
            } else {
                let dt = now
                    .duration_since(last_tick)
                    .as_millis()
                    .min(u32::MAX as u128) as u32;
                inter_chunk_ms.push(dt);
            }
            last_tick = now;

            // EOS check.
            if eos_tokens.contains(&next_token) {
                finish_reason = "stop";
                emit_callback(
                    callback.as_mut(),
                    PartialToken::new(token_text.clone(), step, cumulative_text.clone())
                        .with_token_id(next_token)
                        .with_finish_reason("stop"),
                )?;
                break;
            }

            // Stop-sequence check on cumulative text. We don't strip the
            // stop-sequence from the returned text — matches the llama.cpp
            // adapter's contract.
            let mut hit_stop = false;
            for stop in &params.config.stop_sequences {
                if !stop.is_empty() && cumulative_text.contains(stop) {
                    hit_stop = true;
                    break;
                }
            }
            if hit_stop {
                finish_reason = "stop";
                emit_callback(
                    callback.as_mut(),
                    PartialToken::new(token_text.clone(), step, cumulative_text.clone())
                        .with_token_id(next_token)
                        .with_finish_reason("stop"),
                )?;
                break;
            }

            // Non-terminal token → emit without finish_reason.
            emit_callback(
                callback.as_mut(),
                PartialToken::new(token_text, step, cumulative_text.clone())
                    .with_token_id(next_token),
            )?;

            // Decode step: forward the single new token.
            let one_tok = xybrid_mlx::MlxArray::from_slice_i32(&[next_token as i32], &[1, 1])?;
            let position_offset = (prompt_ids.len() + step) as i32;
            let step_logits = super::super::arch::qwen35::runtime::forward(
                &qwen_cfg,
                &weights,
                &one_tok,
                position_offset,
                stream,
            )?;
            next_logits_row = last_token_logits(&step_logits, qwen_cfg.vocab_size)?;
        }

        let elapsed = start.elapsed();
        let generation_time_ms = elapsed.as_millis().min(u64::MAX as u128) as u64;
        let tokens_generated = generated.len();
        let tokens_per_second = if elapsed.as_secs_f32() > 0.0 {
            tokens_generated as f32 / elapsed.as_secs_f32()
        } else {
            0.0
        };
        let ttft_ms = first_token_time
            .map(|t| t.duration_since(start).as_millis().min(u64::MAX as u128) as u64);
        let (mean_itl_ms, p95_itl_ms) = summarise_inter_chunk(&inter_chunk_ms);

        info!(
            model_id = adapter.model_id_or_default(),
            tokens_generated, tokens_per_second, finish_reason, "mlx.generate.done"
        );
        debug!(
            prefill_tokens = prompt_ids.len(),
            decode_tokens = tokens_generated,
            "mlx.generate.summary"
        );

        // If a stop was hit *before* we emitted anything (extremely short
        // generation), cumulative_text may be empty — still return a valid
        // GenerationOutput so callers can inspect telemetry.
        Ok(GenerationOutput {
            text: cumulative_text,
            tokens_generated,
            generation_time_ms,
            tokens_per_second,
            finish_reason: finish_reason.to_string(),
            ttft_ms,
            mean_itl_ms,
            p95_itl_ms,
            emitted_chunks: Some(tokens_generated as u32),
            inter_chunk_ms,
            decode_tps: None,
            prefill_tps: None,
        })
    }

    /// Extract the last-position logits from a `[batch=1, seq_len, vocab]`
    /// tensor as a plain `Vec<f32>`. The sampler works on CPU-side probs, so
    /// this is the handoff point between MLX and the generate loop.
    fn last_token_logits(logits: &xybrid_mlx::MlxArray, vocab: usize) -> MlxLlmResult<Vec<f32>> {
        // Flattened readback. The logits tensor is `[1, T, V]`, and we want
        // the last `V` slice. For T=1 (decode step) this is just to_vec_f32;
        // for T>1 (prefill) we take the tail.
        let data = logits.to_vec_f32()?;
        if data.len() < vocab {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "logits read back {} elements, expected at least {}",
                data.len(),
                vocab
            )));
        }
        Ok(data[data.len() - vocab..].to_vec())
    }

    fn emit_callback<'cb>(
        cb: Option<&mut StreamingCallback<'cb>>,
        token: PartialToken,
    ) -> MlxLlmResult<()> {
        if let Some(cb) = cb {
            // `cb: &mut Box<dyn FnMut...>` — auto-deref invokes the FnMut.
            cb(token).map_err(|e| {
                MlxLlmError::ConfigInvalid(format!("streaming callback error: {e}"))
            })?;
        }
        Ok(())
    }

    fn summarise_inter_chunk(gaps: &[u32]) -> (Option<f32>, Option<u32>) {
        if gaps.is_empty() {
            return (None, None);
        }
        let mean = gaps.iter().map(|&g| g as f64).sum::<f64>() / gaps.len() as f64;
        let mut sorted: Vec<u32> = gaps.to_vec();
        sorted.sort_unstable();
        let p95_idx = ((sorted.len() as f32) * 0.95).ceil() as usize;
        let p95 = sorted.get(p95_idx.min(sorted.len() - 1)).copied();
        (Some(mean as f32), p95)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_adapter::llm::LlmBackend;
    use crate::runtime_adapter::mlx::{MlxLlmAdapter, MlxLlmConfig};

    /// Helper: build a fresh unloaded adapter and assert generate returns
    /// NotLoaded — the wiring-sanity check for the trait impls.
    #[test]
    fn generate_without_loaded_model_errors() {
        let adapter = MlxLlmAdapter::new(MlxLlmConfig::default());
        let err = LlmBackend::generate(
            &adapter,
            &[ChatMessage::user("hi")],
            &GenerationConfig::default(),
        )
        .expect_err("no model loaded");
        // The trait returns AdapterError; we just check a non-empty message.
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn generate_params_default_seed_is_deterministic() {
        // Two runs of Sampler::seeded(DEFAULT_SAMPLER_SEED) should produce
        // the same token sequence on the same logits — the invariant the
        // streaming-vs-batched parity test depends on.
        let logits = vec![0.5, 1.0, 1.5, 2.0];
        let cfg = GenerationConfig {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 8,
            stop_sequences: vec![],
        };
        let mut p1 = GenerateParams::new(&cfg);
        let mut p2 = GenerateParams::new(&cfg);
        for _ in 0..16 {
            let t1 = p1.sampler.sample(&logits, p1.config, &[]).unwrap();
            let t2 = p2.sampler.sample(&logits, p2.config, &[]).unwrap();
            assert_eq!(t1, t2);
        }
    }

    /// Integration-style test of the generate path on a non-runtime build
    /// (i.e. when `llm-mlx-runtime` is OFF). Loads a dummy Qwen 3 bundle,
    /// calls `LlmBackend::generate`, and asserts the returned error carries
    /// the "build with llm-mlx-runtime" hint. This is the cross-platform
    /// half of the AC's "integration test runs a 4-turn chat with Qwen
    /// 3.5" — the Apple-linked half lives in `tests/mlx_llm_chat.rs` and
    /// is cfg-gated to macOS + `llm-mlx-runtime`.
    #[cfg(not(all(
        feature = "llm-mlx-runtime",
        any(target_os = "macos", target_os = "ios")
    )))]
    #[test]
    fn generate_on_non_runtime_build_returns_actionable_error() {
        use crate::runtime_adapter::mlx::model::MlxLlmError;
        use std::{fs, path::Path};
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        write_qwen3_chat_bundle(tmp.path());

        let adapter =
            MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load qwen3 bundle");

        // Render the 4-turn chat through the template and generate_raw it.
        // Non-runtime path errors at the forward-pass boundary with a
        // pointed message that names the feature gate and the fetch script.
        let prompt = super::render_prompt(
            &adapter,
            &[
                ChatMessage::system("You are a helpful assistant."),
                ChatMessage::user("Hello!"),
                ChatMessage::assistant("Hi there!"),
                ChatMessage::user("What's 2+2?"),
            ],
        )
        .expect("prompt render");
        assert!(
            prompt.contains("<|im_start|>user\nHello!<|im_end|>"),
            "render output missing expected user turn: {prompt}"
        );
        assert!(
            prompt.ends_with("<|im_start|>assistant\n"),
            "render must end with trailing assistant scaffold"
        );

        let cfg = GenerationConfig::default();
        let err = super::generate_tokens(&adapter, &prompt, GenerateParams::new(&cfg), None)
            .expect_err("non-runtime build must error");
        match err {
            MlxLlmError::NotImplemented { feature, story } => {
                assert!(
                    feature.contains("llm-mlx-runtime"),
                    "feature hint: {feature}"
                );
                assert!(
                    feature.contains("fetch-mlx-xcframework"),
                    "fetch hint: {feature}"
                );
                assert_eq!(story, "US-014");
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }

        // The cleaning: all helpers live in a module that the test can call.
        fn write_qwen3_chat_bundle(dir: &Path) {
            // Minimal config.json — same shape as the dispatch fixtures in
            // model.rs, plus enough Qwen-3 fields to pass validation.
            let cfg = serde_json::json!({
                "model_type": "qwen3",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "vocab_size": 100,
                "max_position_embeddings": 1024,
                "rope_theta": 1_000_000.0,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": true,
                "head_dim": 4
            });
            std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();

            // Use the Qwen tokenizer fixture for the encode path. The
            // generate loop tokenises the rendered prompt and encoding
            // must succeed BEFORE the non-runtime error fires.
            let tok_src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("fixtures")
                .join("qwen_tokenizer.json");
            fs::copy(&tok_src, dir.join("tokenizer.json")).unwrap();

            // Chat template — Qwen-3 style ChatML. The renderer test in
            // `chat_template.rs` validates the template itself; here we
            // just need SOMETHING loadable so the adapter finds it.
            let tokenizer_cfg = serde_json::json!({
                "chat_template": "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
            });
            std::fs::write(dir.join("tokenizer_config.json"), tokenizer_cfg.to_string()).unwrap();

            // Full weight manifest (same placeholder pattern as model.rs tests).
            use super::super::arch::qwen35::{expected_weight_keys, Qwen3Config};
            use safetensors::{tensor::TensorView, Dtype};
            let q_cfg = Qwen3Config {
                model_type: "qwen3".into(),
                hidden_size: 16,
                num_hidden_layers: 2,
                num_attention_heads: 4,
                num_key_value_heads: Some(2),
                intermediate_size: 32,
                vocab_size: 100,
                max_position_embeddings: 1024,
                rope_theta: 1_000_000.0,
                rms_norm_eps: 1e-6,
                tie_word_embeddings: true,
                head_dim: Some(4),
                quantization: None,
            };
            let keys = expected_weight_keys(&q_cfg);
            let data = vec![0u8; 4];
            let tensors: Vec<(String, TensorView<'_>)> = keys
                .into_iter()
                .map(|k| (k, TensorView::new(Dtype::F32, vec![1], &data).unwrap()))
                .collect();
            safetensors::serialize_to_file(tensors, &None, &dir.join("model.safetensors")).unwrap();
        }
    }
}
