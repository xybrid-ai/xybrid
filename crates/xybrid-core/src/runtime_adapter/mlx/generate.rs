//! MLX generation orchestrator.
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
//!   pass. Gated on `llm-mlx-runtime` + Apple Silicon macOS because they touch
//!   `xybrid_mlx::MlxArray`.
//!
//! When `llm-mlx-runtime` is off (or on a non-runtime target), the forward-pass
//! entry returns [`MlxLlmError::NotImplemented`] as a build-gate error: the
//! orchestration remains fully covered by unit tests (sampler, chat-template,
//! stop-sequence logic), and the error points the caller at the missing runtime
//! feature.

use crate::runtime_adapter::llm::{
    ChatMessage, GenerationConfig, GenerationOutput, StreamingCallback,
};
use crate::runtime_adapter::llm_telemetry::{compute_streaming_fields, StreamingTelemetryFields};

use std::time::Instant;

use super::chat_template::RenderOptions;
use super::model::{MlxLlmAdapter, MlxLlmError, MlxLlmResult};
use super::sampler::Sampler;

/// Default seed used by the MLX generate path when the caller has not set
/// [`GenerationConfig::seed`]. This preserves the historical deterministic
/// default that streaming-vs-batched parity tests rely on.
const DEFAULT_SAMPLER_SEED: u64 = 0;

/// Render a chat-message list into the raw prompt string the tokenizer
/// sees. Loads the bundle's chat template from `tokenizer_config.json`
/// (cached on the adapter's [`LoadedState`](super::model)), applies it with
/// `add_generation_prompt = true` so the LLM continues mid-turn. User-facing
/// generation disables model thinking channels because the CLI/SDK generation
/// APIs return answers rather than reasoning traces.
pub fn render_prompt(adapter: &MlxLlmAdapter, messages: &[ChatMessage]) -> MlxLlmResult<String> {
    let template = adapter
        .chat_template()
        .ok_or(MlxLlmError::NotLoaded)?
        .clone();
    let options = RenderOptions {
        enable_thinking: false,
        ..RenderOptions::default()
    };
    template
        .render(messages, &options)
        .map_err(|e| MlxLlmError::ConfigInvalid(format!("chat template render failed: {e}")))
}

/// High-level generation parameters shared across generate / stream / chat.
/// Exposed as a builder so the adapter shims don't have to duplicate the
/// sampler-construction logic.
#[derive(Debug)]
pub struct GenerateParams<'cfg> {
    pub config: &'cfg GenerationConfig,
    /// Pre-constructed sampler. `GenerateParams::new` honors
    /// [`GenerationConfig::seed`] and otherwise preserves the historical
    /// deterministic default; tests may still inject a custom sampler.
    pub sampler: Sampler,
}

impl<'cfg> GenerateParams<'cfg> {
    pub fn new(config: &'cfg GenerationConfig) -> Self {
        Self {
            config,
            sampler: Sampler::seeded(config.seed.unwrap_or(DEFAULT_SAMPLER_SEED)),
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

    generate_tokens_from_ids(adapter, &prompt_ids, params, callback)
}

/// Generation entry that takes pre-encoded prompt token ids, bypassing the
/// tokenizer encode step. Used by the golden-token parity oracle so prompt
/// tokenization differences are isolated from decode-numerics comparison.
/// Not part of the supported public API.
#[doc(hidden)]
pub fn generate_tokens_from_ids<'cb>(
    adapter: &MlxLlmAdapter,
    prompt_ids: &[i64],
    params: GenerateParams<'_>,
    callback: Option<StreamingCallback<'cb>>,
) -> MlxLlmResult<GenerationOutput> {
    let arch = adapter.architecture().ok_or(MlxLlmError::NotLoaded)?;
    let tokenizer = adapter.tokenizer().ok_or(MlxLlmError::NotLoaded)?;

    // A zero-token prompt would reach MLX prefill as a `[1, 0]` tensor with
    // undefined downstream behavior; reject it as a validation error.
    if prompt_ids.is_empty() {
        return Err(MlxLlmError::ConfigInvalid(
            "prompt encoded to zero tokens; generation requires at least one prompt token".into(),
        ));
    }

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

    // EOS set: config-driven, resolved once at load time (with the
    // name-list fallback applied there for bundles that declare nothing).
    let eos_tokens: Vec<i64> = adapter
        .eos_token_ids()
        .ok_or(MlxLlmError::NotLoaded)?
        .to_vec();

    // Non-runtime build: surface an actionable error instead of pretending
    // to decode. The rest of the orchestration is still exercised by the
    // unit tests that construct a mock adapter.
    #[cfg(not(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    )))]
    {
        let _ = (params, callback, eos_tokens, max_decode, arch, tokenizer);
        Err(MlxLlmError::NotImplemented {
            feature:
                "MLX forward pass (build with `--features llm-mlx-runtime` on Apple Silicon macOS, \
                 then materialize mlx.xcframework via tools/scripts/fetch-mlx-xcframework.sh \
                 or tools/scripts/build-local-mlx-xcframework.sh)",
            story: "llm-mlx-runtime",
        })
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    {
        runtime::generate_runtime(
            adapter,
            tokenizer,
            arch,
            prompt_ids,
            max_decode,
            &eos_tokens,
            params,
            callback,
        )
    }
}

/// Extract EOS token IDs from the loaded tokenizer by scanning HF-canonical
/// token names. Fallback only: the load path prefers config-declared EOS ids
/// (`config.json` `eos_token_id` / `tokenizer_config.json` `eos_token`) and
/// uses this scan when a bundle declares neither.
pub(crate) fn resolve_eos_tokens_by_name(tokenizer: &tokenizers::Tokenizer) -> Vec<i64> {
    // tokenizers 0.19 exposes the EOS token as part of the added-tokens map,
    // not a dedicated accessor. Look for the HF-canonical names.
    const EOS_NAMES: &[&str] = &[
        "<|endoftext|>",
        "<|im_end|>",
        "</s>",
        "<|end_of_text|>",
        "<eos>",
        "<turn|>",
        "<end_of_turn>",
        "<|end_of_turn|>",
    ];
    let mut out: Vec<i64> = Vec::new();
    for name in EOS_NAMES {
        if let Some(id) = tokenizer.token_to_id(name) {
            out.push(i64::from(id));
        }
    }
    out
}

fn token_id_to_u32(label: &str, token_id: i64) -> MlxLlmResult<u32> {
    u32::try_from(token_id).map_err(|_| {
        MlxLlmError::ConfigInvalid(format!(
            "{label} token id {token_id} is outside the tokenizer u32 id range"
        ))
    })
}

fn decode_generated_text(
    tokenizer: &tokenizers::Tokenizer,
    generated_ids: &[i64],
    previous_text: &str,
) -> MlxLlmResult<(String, String)> {
    let ids: Vec<u32> = generated_ids
        .iter()
        .map(|&id| token_id_to_u32("generated", id))
        .collect::<MlxLlmResult<_>>()?;
    let stable_len = stable_decode_prefix_len(tokenizer, &ids);
    let decoded = tokenizer
        .decode(&ids[..stable_len], true)
        .map_err(|e| MlxLlmError::TokenizerLoad(format!("generated decode failed: {e}")))?;
    // Byte-level decoders can surface an incomplete trailing scalar as U+FFFD.
    // Hold that unstable suffix until a later token completes it.
    let decoded = decoded.trim_end_matches('\u{FFFD}').to_string();
    let delta = decoded
        .strip_prefix(previous_text)
        .map(str::to_string)
        .unwrap_or_else(|| decoded.clone());
    Ok((delta, decoded))
}

fn stable_decode_prefix_len(tokenizer: &tokenizers::Tokenizer, ids: &[u32]) -> usize {
    let mut trailing_bytes = Vec::new();
    let mut trailing_count = 0usize;
    for &id in ids.iter().rev() {
        let Some(token) = tokenizer.id_to_token(id) else {
            break;
        };
        let Some(byte) = byte_fallback_token_byte(&token) else {
            break;
        };
        trailing_bytes.push(byte);
        trailing_count += 1;
    }

    if trailing_count == 0 {
        return ids.len();
    }

    trailing_bytes.reverse();
    if String::from_utf8(trailing_bytes).is_ok() {
        ids.len()
    } else {
        ids.len() - trailing_count
    }
}

fn byte_fallback_token_byte(token: &str) -> Option<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        u8::from_str_radix(&token[3..5], 16).ok()
    } else {
        None
    }
}

fn mlx_streaming_fields(
    start: Instant,
    chunk_timestamps: &[Instant],
    prompt_token_count: usize,
    tokens_generated: usize,
) -> StreamingTelemetryFields {
    compute_streaming_fields(
        start,
        chunk_timestamps,
        prompt_token_count,
        tokens_generated,
    )
}

#[cfg(any(
    test,
    all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    )
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeStreamPreference {
    Gpu,
    CpuFallback,
}

#[cfg(any(
    test,
    all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    )
))]
fn runtime_stream_preference(gpu_available: bool) -> RuntimeStreamPreference {
    if gpu_available {
        RuntimeStreamPreference::Gpu
    } else {
        RuntimeStreamPreference::CpuFallback
    }
}

#[cfg(any(
    test,
    all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    )
))]
fn token_count_to_i32(label: &str, len: usize) -> MlxLlmResult<i32> {
    i32::try_from(len).map_err(|_| {
        MlxLlmError::ConfigInvalid(format!(
            "{label} length ({len}) exceeds MLX i32 shape limit"
        ))
    })
}

#[cfg(any(
    test,
    all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    )
))]
fn token_batch_shape(label: &str, len: usize) -> MlxLlmResult<[i32; 2]> {
    Ok([1, token_count_to_i32(label, len)?])
}

#[cfg(any(
    test,
    all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    )
))]
fn position_offset_for_history_len(len: usize) -> MlxLlmResult<i32> {
    token_count_to_i32("kv-cache position offset", len.saturating_sub(1))
}

#[cfg(any(
    test,
    all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    )
))]
fn decode_forward_input_without_kv_cache(token_history: &[i64]) -> MlxLlmResult<(&[i64], i32)> {
    Ok((token_history, 0))
}

#[cfg(any(
    test,
    all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    )
))]
fn decode_forward_input_with_kv_cache(token_history: &[i64]) -> MlxLlmResult<(&[i64], i32)> {
    let last = token_history.len().saturating_sub(1);
    Ok((
        &token_history[last..],
        position_offset_for_history_len(token_history.len())?,
    ))
}

/// The MLX-linked decode loop. Only compiled when the xcframework is linkable
/// (`llm-mlx-runtime` on Apple Silicon macOS) — everything below touches
/// `xybrid_mlx::MlxArray` either directly or via the arch builder.
#[cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod runtime {
    use super::super::model::ModelArchitecture;
    use super::*;
    use crate::runtime_adapter::types::PartialToken;
    use tracing::{debug, info, warn};

    fn clear_mlx_cache(stage: &'static str) {
        match xybrid_mlx::clear_cache() {
            Ok(()) => debug!(stage, "mlx.clear_cache.done"),
            Err(error) => warn!(stage, %error, "mlx.clear_cache.failed"),
        }
    }

    pub(super) fn generate_runtime<'cb>(
        adapter: &MlxLlmAdapter,
        tokenizer: &tokenizers::Tokenizer,
        arch: ModelArchitecture,
        prompt_ids: &[i64],
        max_decode: usize,
        eos_tokens: &[i64],
        mut params: GenerateParams<'_>,
        mut callback: Option<StreamingCallback<'cb>>,
    ) -> MlxLlmResult<GenerationOutput> {
        let start = Instant::now();
        let mut chunk_timestamps: Vec<Instant> = Vec::new();

        let mut forward = match arch {
            ModelArchitecture::Qwen35 => {
                let (cfg, weights) = adapter.qwen3_runtime_state()?;
                ForwardRuntime::Qwen35 { cfg, weights }
            }
            ModelArchitecture::Gemma4 => {
                let (cfg, weights) = adapter.gemma4_runtime_state()?;
                ForwardRuntime::Gemma4 { cfg, weights }
            }
            ModelArchitecture::Lfm35 => {
                let (cfg, weights) = adapter.lfm35_runtime_state()?;
                ForwardRuntime::Lfm35 { cfg, weights }
            }
        };

        let runtime_stream = default_runtime_stream()?;
        let stream = Some(&runtime_stream);
        forward.reset_kv_cache();
        clear_mlx_cache("generation_start");
        xybrid_mlx::reset_perf_counters();

        let result = (|| -> MlxLlmResult<GenerationOutput> {
            // Generated token history: starts with the prompt, appended as we
            // decode so the sampler's repetition_penalty sees the whole history.
            let mut all_tokens: Vec<i64> = prompt_ids.to_vec();
            let mut generated: Vec<i64> = Vec::with_capacity(max_decode);
            let mut cumulative_text = String::new();
            let mut finish_reason = "length";
            // Streaming callbacks and stop-sequence scans need stable text on
            // every token. This no longer disables the async greedy path: the
            // incremental detokenizer below runs on the host *after* the next
            // token's graph has been dispatched, so text work overlaps GPU
            // compute (mlx-lm's `stream_generate` ordering).
            let needs_incremental_text =
                callback.is_some() || !params.config.stop_sequences.is_empty();
            // Incremental detokenizer: O(1) amortized per token. `step` returns
            // `None` while the pending bytes don't yet form complete UTF-8
            // (byte-fallback `<0xNN>` runs, multi-byte scalars split across
            // ids); the held-back text is emitted once a later token completes
            // it. Replaces the previous O(tokens²) full-prefix re-decode.
            let mut detok = needs_incremental_text.then(|| tokenizer.decode_stream(true));
            let longest_stop = params
                .config
                .stop_sequences
                .iter()
                .map(String::len)
                .max()
                .unwrap_or(0);

            info!(
                model_id = adapter.model_id_or_default(),
                prompt_tokens = prompt_ids.len(),
                max_tokens = max_decode,
                "mlx.generate.start"
            );

            // Prefill: run the forward pass on the full prompt, then slice the
            // last token's logits.
            let prefill_shape = token_batch_shape("prefill prompt", prompt_ids.len())?;
            let prefill_ids = xybrid_mlx::MlxArray::from_slice_i64(prompt_ids, &prefill_shape)?;
            let mut next_logits = forward.forward(&prefill_ids, 0, stream)?;
            let greedy_decode = params.config.temperature <= 0.0;
            let use_async_greedy =
                greedy_decode && forward.uses_kv_cache() && forward.supports_async_greedy();
            let mut next_token_ids = if use_async_greedy {
                let ids = greedy_argmax_token_ids(&next_logits, stream)?;
                xybrid_mlx::async_eval(&[&ids])?;
                Some(ids)
            } else {
                None
            };

            for step in 0..max_decode {
                let current_token_ids = next_token_ids.take();
                let next_token = if let Some(token_ids) = current_token_ids.as_ref() {
                    if step + 1 < max_decode {
                        // Queue one token ahead before the host readback. If the
                        // current token is EOS or completes a stop sequence, this
                        // over-prefetches one cache entry that is discarded with
                        // the per-run cache.
                        let position_offset =
                            position_offset_for_history_len(prompt_ids.len() + step + 1)?;
                        let decode_tok =
                            xybrid_mlx::ops::cast(token_ids, xybrid_mlx::MlxDtype::I64, stream)?;
                        let step_logits = forward.forward(&decode_tok, position_offset, stream)?;
                        let ids = greedy_argmax_token_ids(&step_logits, stream)?;
                        xybrid_mlx::async_eval(&[&ids])?;
                        next_token_ids = Some(ids);
                    }
                    read_greedy_token_id(token_ids)?
                } else if greedy_decode {
                    greedy_argmax_token(&next_logits, stream)?
                } else {
                    let next_logits_row =
                        last_token_logits(&next_logits, forward.vocab_size(), stream)?;
                    params
                        .sampler
                        .sample(&next_logits_row, params.config, &all_tokens)?
                };

                all_tokens.push(next_token);
                generated.push(next_token);

                // Streaming callbacks and text stop-sequences need stable text on
                // every token. Plain batch generation can defer tokenizer decode
                // until the end. Either way this host-side work overlaps the GPU
                // computing the already-dispatched next token.
                let token_text = if let Some(detok) = detok.as_mut() {
                    let id = token_id_to_u32("generated", next_token)?;
                    let segment = detok
                        .step(id)
                        .map_err(|e| {
                            MlxLlmError::TokenizerLoad(format!("streaming decode failed: {e}"))
                        })?
                        .unwrap_or_default();
                    cumulative_text.push_str(&segment);
                    segment
                } else {
                    String::new()
                };

                // Timing: first-token latency + inter-chunk gaps.
                let now = Instant::now();
                chunk_timestamps.push(now);

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

                // Stop-sequence check. We don't strip the stop-sequence from
                // the returned text — matches the llama.cpp adapter's contract.
                // Only the tail can contain a *new* match: earlier text was
                // scanned on previous tokens, so the window is the new segment
                // plus enough preceding bytes to catch a match spanning the
                // boundary. Avoids re-scanning the whole text every token.
                let mut hit_stop = false;
                if longest_stop > 0 && !token_text.is_empty() {
                    let mut tail_start = cumulative_text
                        .len()
                        .saturating_sub(token_text.len() + longest_stop);
                    while !cumulative_text.is_char_boundary(tail_start) {
                        tail_start -= 1;
                    }
                    let tail = &cumulative_text[tail_start..];
                    for stop in &params.config.stop_sequences {
                        if !stop.is_empty() && tail.contains(stop) {
                            hit_stop = true;
                            break;
                        }
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

                // Bound MLX's memory-pool growth on long generations. Same
                // cadence as mlx-lm's decode loop (`mx.clear_cache()` every
                // 256 tokens); a no-op cost at short budgets.
                if step > 0 && step % 256 == 0 {
                    clear_mlx_cache("decode_steady_state");
                }

                // Decode step: runtime architectures reset their resident
                // attention/conv caches before prefill, then only forward the
                // newest token at the absolute position.
                if !use_async_greedy {
                    let (decode_ids, position_offset) = if forward.uses_kv_cache() {
                        decode_forward_input_with_kv_cache(&all_tokens)?
                    } else {
                        decode_forward_input_without_kv_cache(&all_tokens)?
                    };
                    let decode_shape = token_batch_shape("decode input", decode_ids.len())?;
                    let decode_tok =
                        xybrid_mlx::MlxArray::from_slice_i64(decode_ids, &decode_shape)?;
                    next_logits = forward.forward(&decode_tok, position_offset, stream)?;
                }
            }

            let tokens_generated = generated.len();
            if !needs_incremental_text {
                cumulative_text = decode_generated_text(tokenizer, &generated, "")?.1;
            }
            let fields =
                mlx_streaming_fields(start, &chunk_timestamps, prompt_ids.len(), tokens_generated);

            info!(
                model_id = adapter.model_id_or_default(),
                tokens_generated,
                tokens_per_second = fields.tokens_per_second,
                finish_reason,
                "mlx.generate.done"
            );
            let perf = xybrid_mlx::perf_counters();
            let generated_denominator = tokens_generated.max(1) as f64;
            debug!(
                eval_calls = perf.eval_calls,
                async_eval_calls = perf.async_eval_calls,
                to_vec_f32_calls = perf.to_vec_f32_calls,
                to_vec_u32_calls = perf.to_vec_u32_calls,
                evals_per_token = perf.eval_calls as f64 / generated_denominator,
                f32_readbacks_per_token = perf.to_vec_f32_calls as f64 / generated_denominator,
                u32_readbacks_per_token = perf.to_vec_u32_calls as f64 / generated_denominator,
                async_greedy = use_async_greedy,
                "mlx.generate.perf"
            );
            if std::env::var_os("XYBRID_MLX_PRINT_PERF").is_some() {
                eprintln!(
                    "mlx perf: eval_calls={} async_eval_calls={} to_vec_f32_calls={} \
                 to_vec_u32_calls={} evals_per_token={:.4} f32_readbacks_per_token={:.4} \
                 u32_readbacks_per_token={:.4} async_greedy={}",
                    perf.eval_calls,
                    perf.async_eval_calls,
                    perf.to_vec_f32_calls,
                    perf.to_vec_u32_calls,
                    perf.eval_calls as f64 / generated_denominator,
                    perf.to_vec_f32_calls as f64 / generated_denominator,
                    perf.to_vec_u32_calls as f64 / generated_denominator,
                    use_async_greedy
                );
            }
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
                generation_time_ms: fields.generation_time_ms,
                tokens_per_second: fields.tokens_per_second,
                finish_reason: finish_reason.to_string(),
                ttft_ms: fields.ttft_ms,
                mean_itl_ms: fields.mean_itl_ms,
                p95_itl_ms: fields.p95_itl_ms,
                emitted_chunks: fields.emitted_chunks,
                inter_chunk_ms: fields.inter_chunk_ms,
                decode_tps: fields.decode_tps,
                prefill_tps: fields.prefill_tps,
                // Text-only MLX path: no image preprocessing leg to time.
                image_preprocess_ms: None,
                // MLX does not split a reasoning channel yet; <think> capture
                // happens in the shared executor marker-table pass.
                reasoning_content: None,
            })
        })();

        forward.reset_kv_cache();
        clear_mlx_cache("generation_end");
        result
    }

    enum ForwardRuntime<'a> {
        Qwen35 {
            cfg: &'a super::super::arch::qwen35::Qwen3Config,
            weights: std::sync::MutexGuard<'a, super::super::arch::qwen35::runtime::Qwen35Weights>,
        },
        Gemma4 {
            cfg: &'a super::super::arch::gemma4::Gemma4Config,
            weights: std::sync::MutexGuard<'a, super::super::arch::gemma4::runtime::Gemma4Weights>,
        },
        Lfm35 {
            cfg: &'a super::super::arch::lfm35::Lfm35Config,
            weights: std::sync::MutexGuard<'a, super::super::arch::lfm35::runtime::Lfm35Weights>,
        },
    }

    impl ForwardRuntime<'_> {
        fn vocab_size(&self) -> usize {
            match self {
                Self::Qwen35 { cfg, .. } => cfg.vocab_size,
                Self::Gemma4 { cfg, .. } => cfg.vocab_size,
                Self::Lfm35 { cfg, .. } => cfg.vocab_size,
            }
        }

        fn reset_kv_cache(&mut self) {
            match self {
                Self::Qwen35 { weights, .. } => weights.reset_kv_cache(),
                Self::Gemma4 { weights, .. } => weights.reset_kv_cache(),
                Self::Lfm35 { weights, .. } => weights.reset_kv_cache(),
            }
        }

        fn uses_kv_cache(&self) -> bool {
            matches!(
                self,
                Self::Qwen35 { .. } | Self::Gemma4 { .. } | Self::Lfm35 { .. }
            )
        }

        fn supports_async_greedy(&self) -> bool {
            matches!(
                self,
                Self::Qwen35 { .. } | Self::Gemma4 { .. } | Self::Lfm35 { .. }
            )
        }

        fn forward(
            &mut self,
            input_ids: &xybrid_mlx::MlxArray,
            position_offset: i32,
            stream: Option<&xybrid_mlx::MlxStream>,
        ) -> MlxLlmResult<xybrid_mlx::MlxArray> {
            match self {
                Self::Qwen35 { cfg, weights } => super::super::arch::qwen35::runtime::forward(
                    cfg,
                    weights,
                    input_ids,
                    position_offset,
                    stream,
                ),
                Self::Gemma4 { cfg, weights } => super::super::arch::gemma4::runtime::forward(
                    cfg,
                    weights,
                    input_ids,
                    position_offset,
                    stream,
                ),
                Self::Lfm35 { cfg, weights } => super::super::arch::lfm35::runtime::forward(
                    cfg,
                    weights,
                    input_ids,
                    position_offset,
                    stream,
                ),
            }
        }
    }

    /// Extract the last-position logits from a `[batch=1, seq_len, vocab]`
    /// tensor as a plain `Vec<f32>`. The sampler works on CPU-side probs, so
    /// this is the handoff point between MLX and the generate loop.
    fn last_token_logits(
        logits: &xybrid_mlx::MlxArray,
        vocab: usize,
        stream: Option<&xybrid_mlx::MlxStream>,
    ) -> MlxLlmResult<Vec<f32>> {
        let logits = if logits.dtype()? == xybrid_mlx::MlxDtype::F32 {
            logits.clone()
        } else {
            xybrid_mlx::ops::cast(logits, xybrid_mlx::MlxDtype::F32, stream)?
        };
        let shape = logits.shape();
        if shape.len() != 3 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "logits must be rank-3 [1, T, V], got shape {shape:?}"
            )));
        }
        if shape[0] != 1 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "MLX generation currently supports batch size 1, got logits shape {shape:?}"
            )));
        }
        let seq_len = shape[1];
        let vocab_i32 = token_count_to_i32("vocab", vocab)?;
        if shape[2] != vocab_i32 {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "logits vocab dimension is {}, expected {vocab_i32}",
                shape[2]
            )));
        }
        let last_pos = seq_len.checked_sub(1).ok_or_else(|| {
            MlxLlmError::ConfigInvalid("logits sequence dimension must be positive".into())
        })?;
        let last = xybrid_mlx::ops::slice(
            &logits,
            &[0, last_pos, 0],
            &[1, seq_len, vocab_i32],
            &[1, 1, 1],
            stream,
        )?;
        let data = last.to_vec_f32_on_stream(stream)?;
        if data.len() != vocab {
            return Err(MlxLlmError::ConfigInvalid(format!(
                "last-token logits read back {} elements, expected {}",
                data.len(),
                vocab
            )));
        }
        Ok(data)
    }

    /// Greedy decode keeps token selection on the MLX stream and reads back
    /// only the selected token id. This matches the existing sampler's
    /// temperature=0 semantics and avoids copying a full vocab row to the CPU
    /// on every token.
    fn greedy_argmax_token(
        logits: &xybrid_mlx::MlxArray,
        stream: Option<&xybrid_mlx::MlxStream>,
    ) -> MlxLlmResult<i64> {
        let ids = greedy_argmax_token_ids(logits, stream)?;
        read_greedy_token_id(&ids)
    }

    fn greedy_argmax_token_ids(
        logits: &xybrid_mlx::MlxArray,
        stream: Option<&xybrid_mlx::MlxStream>,
    ) -> MlxLlmResult<xybrid_mlx::MlxArray> {
        debug_assert_eq!(
            logits.shape().first().copied(),
            Some(1),
            "MLX generation currently supports batch size 1"
        );
        let ids = xybrid_mlx::ops::argmax(logits, -1, stream)?;
        let shape = ids.shape();
        let ids = match shape.as_slice() {
            [1, 1] => ids,
            [1, seq_len] if *seq_len > 1 => {
                let last_pos = seq_len.checked_sub(1).ok_or_else(|| {
                    MlxLlmError::ConfigInvalid(
                        "greedy argmax token ids must have positive sequence length".into(),
                    )
                })?;
                xybrid_mlx::ops::slice(&ids, &[0, last_pos], &[1, *seq_len], &[1, 1], stream)
                    .map_err(MlxLlmError::from)?
            }
            _ => Err(MlxLlmError::ConfigInvalid(format!(
                "greedy argmax returned unexpected shape {shape:?}"
            )))?,
        };
        ids.contiguous_on_stream(stream).map_err(Into::into)
    }

    fn read_greedy_token_id(ids: &xybrid_mlx::MlxArray) -> MlxLlmResult<i64> {
        let ids = ids.to_vec_u32_evaluated()?;
        let token = ids.last().ok_or_else(|| {
            MlxLlmError::ConfigInvalid("greedy argmax returned no token ids".into())
        })?;
        Ok(i64::from(*token))
    }

    fn emit_callback<'cb>(
        cb: Option<&mut StreamingCallback<'cb>>,
        token: PartialToken,
    ) -> MlxLlmResult<()> {
        if let Some(cb) = cb {
            // `cb: &mut Box<dyn FnMut...>` — auto-deref invokes the FnMut.
            cb(token).map_err(MlxLlmError::StreamingCallback)?;
        }
        Ok(())
    }

    fn default_runtime_stream() -> MlxLlmResult<xybrid_mlx::MlxStream> {
        match xybrid_mlx::MlxStream::default_gpu() {
            Ok(stream) => {
                debug!(
                    preference = ?super::runtime_stream_preference(true),
                    "mlx.generate.stream"
                );
                Ok(stream)
            }
            Err(gpu_err) => {
                warn!(
                    error = %gpu_err,
                    preference = ?super::runtime_stream_preference(false),
                    "mlx.generate.gpu_stream_unavailable_falling_back_to_cpu"
                );
                xybrid_mlx::MlxStream::default_cpu().map_err(Into::into)
            }
        }
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
            seed: None,
            stop_sequences: vec![],
            ..Default::default()
        };
        let mut p1 = GenerateParams::new(&cfg);
        let mut p2 = GenerateParams::new(&cfg);
        for _ in 0..16 {
            let t1 = p1.sampler.sample(&logits, p1.config, &[]).unwrap();
            let t2 = p2.sampler.sample(&logits, p2.config, &[]).unwrap();
            assert_eq!(t1, t2);
        }
    }

    #[test]
    fn generate_params_uses_generation_config_seed() {
        let logits = vec![0.5, 1.0, 1.5, 2.0];
        let cfg = GenerationConfig {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 8,
            stop_sequences: vec![],
            seed: Some(0x5eed),
            ..Default::default()
        };
        let mut from_config = GenerateParams::new(&cfg);
        let mut explicit = Sampler::seeded(0x5eed);

        for _ in 0..16 {
            let configured = from_config
                .sampler
                .sample(&logits, from_config.config, &[])
                .unwrap();
            let seeded = explicit.sample(&logits, &cfg, &[]).unwrap();
            assert_eq!(configured, seeded);
        }
    }

    #[test]
    fn mlx_streaming_fields_derives_prefill_and_decode_tps() {
        use std::time::{Duration, Instant};

        let start = Instant::now();
        let chunks = [
            start + Duration::from_millis(50),
            start + Duration::from_millis(75),
            start + Duration::from_millis(100),
        ];

        let fields = super::mlx_streaming_fields(start, &chunks, 20, 3);

        assert_eq!(fields.ttft_ms, Some(50));
        assert_eq!(fields.mean_itl_ms, Some(25.0));
        assert_eq!(fields.p95_itl_ms, Some(25));
        assert_eq!(fields.decode_tps, Some(40.0));
        assert_eq!(fields.prefill_tps, Some(400.0));
        assert_eq!(fields.emitted_chunks, Some(3));
        assert_eq!(fields.inter_chunk_ms, vec![25, 25]);
    }

    #[test]
    fn runtime_stream_policy_prefers_gpu_when_available() {
        assert_eq!(
            super::runtime_stream_preference(true),
            super::RuntimeStreamPreference::Gpu
        );
        assert_eq!(
            super::runtime_stream_preference(false),
            super::RuntimeStreamPreference::CpuFallback
        );
    }

    #[test]
    fn decode_without_kv_cache_forwards_full_context_from_zero_position() {
        let token_history = [10, 20, 30, 40];
        let (forward_tokens, position_offset) =
            super::decode_forward_input_without_kv_cache(&token_history).unwrap();

        assert_eq!(forward_tokens, &token_history);
        assert_eq!(position_offset, 0);
    }

    #[test]
    fn decode_with_kv_cache_forwards_only_new_token_at_absolute_position() {
        let token_history = [10, 20, 30, 40];
        let (forward_tokens, position_offset) =
            super::decode_forward_input_with_kv_cache(&token_history).unwrap();

        assert_eq!(forward_tokens, &[40]);
        assert_eq!(position_offset, 3);
    }

    #[test]
    fn token_batch_shape_rejects_lengths_over_mlx_i32_limit() {
        match super::token_batch_shape("prefill prompt", i32::MAX as usize + 1).unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("prefill prompt"), "got: {msg}");
                assert!(msg.contains("exceeds MLX i32 shape limit"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn kv_cache_position_offset_rejects_lengths_over_mlx_i32_limit() {
        match super::position_offset_for_history_len(i32::MAX as usize + 2).unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("kv-cache position offset"), "got: {msg}");
                assert!(msg.contains("exceeds MLX i32 shape limit"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn token_id_to_u32_rejects_negative_ids() {
        match super::token_id_to_u32("generated", -1).unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("generated"), "got: {msg}");
                assert!(
                    msg.contains("outside the tokenizer u32 id range"),
                    "got: {msg}"
                );
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn token_id_to_u32_rejects_ids_over_u32_limit() {
        match super::token_id_to_u32("generated", i64::from(u32::MAX) + 1).unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("generated"), "got: {msg}");
                assert!(
                    msg.contains("outside the tokenizer u32 id range"),
                    "got: {msg}"
                );
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn generated_text_decode_preserves_byte_fallback_sequences() {
        let tok_src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("qwen_tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(tok_src).expect("load qwen tokenizer");
        let text = "Hello! 😊";
        let encoding = tokenizer.encode(text, false).expect("encode text");
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&u| i64::from(u)).collect();
        let decoded_individually = ids
            .iter()
            .map(|&id| {
                let token_id = super::token_id_to_u32("generated", id).unwrap();
                tokenizer
                    .decode(&[token_id], true)
                    .unwrap_or_else(|_| "\u{FFFD}".to_string())
            })
            .collect::<String>();
        assert!(
            decoded_individually.contains('\u{FFFD}'),
            "fixture must reproduce per-token byte-fallback replacement: {decoded_individually:?}"
        );

        let mut generated = Vec::new();
        let mut cumulative = String::new();
        let mut streamed_deltas = String::new();
        for id in ids {
            generated.push(id);
            let (delta, decoded) =
                super::decode_generated_text(&tokenizer, &generated, &cumulative)
                    .expect("decode generated text");
            streamed_deltas.push_str(&delta);
            cumulative = decoded;
        }

        assert_eq!(cumulative, text);
        assert_eq!(streamed_deltas, text);
    }

    /// The streaming decode path uses `tokenizers::DecodeStream`. It must
    /// reproduce the same byte-fallback guarantee as the batch-side
    /// `decode_generated_text`: multi-id Unicode scalars are held back until
    /// complete instead of surfacing replacement characters.
    #[test]
    fn decode_stream_preserves_byte_fallback_sequences() {
        let tok_src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("qwen_tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(tok_src).expect("load qwen tokenizer");
        let text = "Hello! 😊";
        let encoding = tokenizer.encode(text, false).expect("encode text");

        let mut detok = tokenizer.decode_stream(true);
        let mut streamed = String::new();
        let mut held_back = 0usize;
        for &id in encoding.get_ids() {
            match detok.step(id).expect("decode stream step") {
                Some(segment) => streamed.push_str(&segment),
                None => held_back += 1,
            }
        }

        assert_eq!(streamed, text);
        assert!(
            held_back > 0,
            "fixture must exercise the held-back partial-scalar path"
        );
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
        target_os = "macos",
        target_arch = "aarch64"
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
            prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "Qwen render must disable thinking before generation: {prompt}"
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
                assert!(
                    feature.contains("build-local-mlx-xcframework"),
                    "source-build hint: {feature}"
                );
                assert_eq!(story, "llm-mlx-runtime");
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

            // Chat template — Qwen-3 style ChatML with its no-thinking
            // branch. The generate path must opt out of thinking by passing
            // `enable_thinking = false`.
            let tokenizer_cfg = serde_json::json!({
                "chat_template": "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}{%- endif %}"
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
