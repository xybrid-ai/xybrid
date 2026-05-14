//! LlamaCppBackend - LLM inference using llama.cpp
//!
//! This module provides llama.cpp bindings for LLM inference.
//! It is feature-gated behind `llm-llamacpp`.
//!
//! # Why llama.cpp?
//!
//! llama.cpp has proper Android ARM64 support with runtime SIMD detection,
//! unlike mistral.rs/candle which require compile-time `+fp16` flags that
//! cause SIGILL on devices without ARMv8.2-A FP16 extension.
//!
//! # Architecture
//!
//! ```text
//! LlamaCppBackend (Rust)
//!     │
//!     └── llama_cpp_sys (FFI bindings)
//!             │
//!             └── llama.cpp (C/C++ library)
//!                     │
//!                     └── ggml (tensor library with runtime SIMD detection)
//! ```

mod sys;

// Re-export log control functions for external use
pub use sys::{llama_log_get_verbosity, llama_log_set_verbosity};

use crate::runtime_adapter::llm::{
    ChatMessage, GenerationConfig, GenerationOutput, LlmBackend, LlmConfig, LlmResult,
};
#[cfg(feature = "llm-llamacpp")]
use crate::runtime_adapter::llm_telemetry::StreamingTelemetry;
#[cfg(feature = "llm-llamacpp")]
use crate::runtime_adapter::streaming_postprocess::{
    merge_stop_patterns, strip_thinking_tags, trim_partial_stop_suffix, truncate_at_first_stop,
    StreamingTextFilter, CHAT_STOP_PATTERNS, CHAT_STOP_PATTERNS_BROKEN,
};
use crate::runtime_adapter::AdapterError;
use crate::tracing as xybrid_trace;
use std::sync::Mutex;
#[cfg(feature = "llm-llamacpp")]
use std::sync::Once;

/// Ensures llama_backend_init() is called exactly once, regardless of how many
/// LlamaCppBackend instances are created.
///
/// Note: We intentionally never call llama_backend_free(). The `Once` guard
/// cannot be re-armed, so if we freed the backend when the last instance drops
/// and then created a new instance (e.g., during model swap), the backend
/// would NOT be re-initialized — causing undefined behavior. Since
/// llama_backend_free() only cleans up NUMA info (a no-op on most platforms),
/// skipping it is safe. The OS reclaims all resources at process exit.
#[cfg(feature = "llm-llamacpp")]
static BACKEND_INIT: Once = Once::new();

/// LlamaCppBackend - LLM inference using llama.cpp.
///
/// This backend uses llama.cpp for GGUF model inference with proper
/// Android ARM64 support via runtime SIMD detection.
///
/// # Platform Support
///
/// - **Android**: Full support with runtime NEON/FP16 detection
/// - **iOS**: Supported with Metal acceleration
/// - **macOS**: Supported with Metal acceleration
/// - **Linux/Windows**: Supported with CPU/CUDA
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::runtime_adapter::llama_cpp::LlamaCppBackend;
/// use xybrid_core::runtime_adapter::llm::{LlmBackend, LlmConfig};
///
/// let mut backend = LlamaCppBackend::new()?;
/// backend.load(&LlmConfig::new("model.gguf"))?;
/// ```
#[cfg(feature = "llm-llamacpp")]
pub struct LlamaCppBackend {
    /// Pointer to loaded model (llama_model*)
    model: Option<sys::LlamaModel>,
    /// Pointer to context (llama_context*).
    ///
    /// Wrapped in Mutex because llama_decode() mutates internal state and is
    /// not thread-safe. The LlmBackend trait requires Send + Sync, and
    /// generate() takes &self, so we need a Mutex to serialize context access.
    context: Mutex<Option<sys::LlamaContext>>,
    /// Current configuration
    config: Option<LlmConfig>,
    /// Multi-turn KV cache reuse state. Records the tokenized prompt of
    /// the last `generate*` call so the next call can find the longest
    /// common prefix and skip re-prefilling that portion. Wrapped in
    /// `Mutex` because `generate*` are `&self` and we need cross-call
    /// mutation; lock order is always `context → kv_state` to match the
    /// natural critical section (we already hold the context lock when
    /// we mutate the cache via seq_rm).
    kv_state: Mutex<KvCacheState>,
}

/// Cross-call state for the multi-turn KV cache reuse path. `Default::default()`
/// gives an empty cache (`n_past = 0`, `cached_tokens.is_empty()`), which
/// matches the post-load state of a fresh context.
///
/// `last_prefix_hit` records the prefix length the most recent call was able
/// to reuse — the source of truth for the future `prompt_cached_tokens`
/// telemetry field. Read it post-`generate*` to learn how many tokens were
/// served from cache.
#[cfg(feature = "llm-llamacpp")]
#[derive(Default)]
struct KvCacheState {
    /// The exact token sequence currently sitting in the KV cache. Empty
    /// when the cache is unpopulated (post-load, or post full-clear).
    cached_tokens: Vec<i32>,
    /// Prefix length the most recent prepare call was able to keep from
    /// the prior cached_tokens. `None` when no `generate*` has run since
    /// load; `Some(0)` when the most recent call had no shared prefix
    /// (fully re-prefilled).
    last_prefix_hit: Option<usize>,
}

#[cfg(feature = "llm-llamacpp")]
impl LlamaCppBackend {
    /// Create a new LlamaCppBackend.
    pub fn new() -> LlmResult<Self> {
        // Initialize llama.cpp backend exactly once (idempotent via Once).
        BACKEND_INIT.call_once(|| {
            sys::llama_backend_init();

            // Check for verbosity env var to surface C++ logs during debugging
            if let Ok(level) = std::env::var("XYBRID_LLAMACPP_VERBOSITY") {
                if let Ok(v) = level.parse::<i32>() {
                    sys::llama_log_set_verbosity(v);
                }
            }
        });

        Ok(Self {
            model: None,
            context: Mutex::new(None),
            config: None,
            kv_state: Mutex::new(KvCacheState::default()),
        })
    }
}

#[cfg(feature = "llm-llamacpp")]
impl Drop for LlamaCppBackend {
    fn drop(&mut self) {
        // Drop context first, then model (order matters: context references model).
        // LlamaContext and LlamaModel implement Drop, so take() + drop handles cleanup.
        // get_mut() doesn't lock — safe because Drop has &mut self.
        let _ = self.context.get_mut().unwrap().take(); // drops LlamaContext
        let _ = self.model.take(); // drops LlamaModel

        // Note: We intentionally do NOT call llama_backend_free() here.
        // See BACKEND_INIT comment for rationale.
    }
}

#[cfg(feature = "llm-llamacpp")]
impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create LlamaCppBackend")
    }
}

#[cfg(feature = "llm-llamacpp")]
impl LlamaCppBackend {
    /// Acquire the model + context under the context mutex and hand both
    /// to `f`. Replaces three copies of the same five-line dance across
    /// `generate`, `generate_raw`, and `generate_streaming`. The guard
    /// is held for the duration of `f` — `LlamaContext` is non-`Sync`
    /// and `llama_decode` mutates internal state, so serialization
    /// across threads is required.
    fn with_model_and_context<R, F>(&self, f: F) -> LlmResult<R>
    where
        F: FnOnce(&sys::LlamaModel, &sys::LlamaContext) -> LlmResult<R>,
    {
        let model = self.model.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No model loaded. Call load() first.".to_string())
        })?;
        let ctx_guard = self
            .context
            .lock()
            .map_err(|_| AdapterError::RuntimeError("Context mutex poisoned".to_string()))?;
        let context = ctx_guard.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No context. Call load() first.".to_string())
        })?;
        f(model, context)
    }

    /// Multi-turn KV cache reuse: compute the longest common prefix between
    /// the new prompt's tokens and what's already in the cache from the
    /// previous call, truncate the cache to that prefix, and return the
    /// diverged tail along with the prefix length the caller should pass
    /// as `n_past_in` to [`sys::llama_generate_streaming`].
    ///
    /// On a fresh context (no prior call) or when the prompts share no
    /// prefix, returns `(full_tokens, 0)` and full-clears the cache so the
    /// generate call below starts from scratch — same behaviour as before
    /// this helper existed.
    ///
    /// Safety net: when the prefilled prefix plus new tail plus
    /// `max_tokens` would overflow the context window, fall back to a
    /// full clear so the generation call doesn't trip the
    /// `n_past + n_input >= n_ctx` bail-out in the C wrapper. The simple
    /// LCP path doesn't implement context-window eviction yet — that's
    /// deliberately out of scope, the fallback keeps correctness.
    ///
    /// Updates `kv_state.cached_tokens` to reflect the post-call cache
    /// (which will be `[prefix; new_tail; generated_tokens]` after the
    /// generation runs). Records `last_prefix_hit` for the telemetry
    /// accessor [`Self::last_cached_prefix_len`].
    fn prepare_kv_cache_and_get_tail(
        &self,
        model: &sys::LlamaModel,
        context: &sys::LlamaContext,
        new_tokens: &[i32],
        max_new_tokens: usize,
    ) -> LlmResult<(Vec<i32>, usize)> {
        let mut state = self
            .kv_state
            .lock()
            .map_err(|_| AdapterError::RuntimeError("KV state mutex poisoned".to_string()))?;

        // Models with recurrent state — fully recurrent (Mamba, RWKV)
        // OR hybrid (LFM2 / LFM2MOE, Qwen35 / Qwen35MOE, Granite-hybrid,
        // …) — can't safely have their cache truncated by position.
        // Their recurrent layers accumulate state across positions, so
        // `llama_kv_cache_seq_rm` leaves the residual state inconsistent
        // with the new prefix length and `llama_decode` fails on the
        // diverging tail (wrapper error code -3; surfaced on LFM 2.5
        // second-turn chat).
        //
        // Gating on `has_recurrent_state` (which combines the upstream
        // `is_recurrent` and `is_hybrid` predicates) instead of
        // `is_recurrent` alone is important: LFM2 is classified hybrid,
        // not fully recurrent, so a narrower check missed it the first
        // time round. Skip prefix-reuse entirely on these models and
        // fall back to the pre-INF-99 full-clear path. The cost is
        // per-turn re-prefill of the full conversation — correct, just
        // not the prefix-reuse optimisation.
        if sys::llama_model_has_recurrent_state(model) {
            sys::llama_kv_cache_clear(context);
            state.cached_tokens = new_tokens.to_vec();
            state.last_prefix_hit = Some(0);
            return Ok((new_tokens.to_vec(), 0));
        }

        let n_ctx = sys::llama_n_ctx(context);
        let prefix_len = compute_reusable_prefix_len(&state.cached_tokens, new_tokens);

        // Safety net: if the prefilled prefix + new tail + max_new_tokens
        // would push us past the context window, the simple LCP path
        // can't help — drop the cache and let the standard "input too
        // long" check in the C wrapper produce a clear error. This also
        // covers the eviction case the issue explicitly punts on.
        let would_overflow = prefix_len
            .saturating_add(new_tokens.len() - prefix_len)
            .saturating_add(max_new_tokens)
            >= n_ctx;
        if prefix_len == 0 || would_overflow {
            sys::llama_kv_cache_clear(context);
            state.cached_tokens = new_tokens.to_vec();
            state.last_prefix_hit = Some(0);
            return Ok((new_tokens.to_vec(), 0));
        }

        // Truncate the cache to the prefix. seq_id = 0 because the
        // wrapper's batch.seq_id[..][0] = 0 path uses a single sequence;
        // when we add multi-sequence support the seq_id needs to flow
        // through prepare too.
        sys::llama_kv_cache_seq_rm(context, 0, prefix_len);
        let tail = new_tokens[prefix_len..].to_vec();
        state.cached_tokens = new_tokens.to_vec();
        state.last_prefix_hit = Some(prefix_len);
        Ok((tail, prefix_len))
    }

    fn reset_kv_cache_after_failed_stream(&self, context: &sys::LlamaContext) {
        sys::llama_kv_cache_clear(context);
        self.clear_cached_prefix_state();
    }

    fn clear_cached_prefix_state(&self) {
        if let Ok(mut state) = self.kv_state.lock() {
            *state = KvCacheState::default();
        }
    }
}

/// Longest-common-prefix length between the cached tokens and the new
/// prompt tokens, capped at `new_tokens.len() - 1` so the post-prefill
/// batch always has at least one token to feed the C decoder.
///
/// Pulled out so the LCP arithmetic is unit-testable without needing a
/// real `LlamaContext`. Behaviour:
/// - empty `new_tokens` ⇒ 0 (caller should bail before reaching the helper)
/// - empty `cached` ⇒ 0 (first call)
/// - identical sequences ⇒ `new_tokens.len() - 1` (keep last token for the C decoder)
/// - common prefix shorter than either ⇒ that prefix length
#[cfg(feature = "llm-llamacpp")]
fn compute_reusable_prefix_len(cached: &[i32], new_tokens: &[i32]) -> usize {
    let max_reuse = new_tokens.len().saturating_sub(1);
    cached
        .iter()
        .zip(new_tokens.iter())
        .take(max_reuse)
        .take_while(|(a, b)| a == b)
        .count()
}

#[cfg(feature = "llm-llamacpp")]
impl LlmBackend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama-cpp"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["gguf"]
    }

    fn load(&mut self, config: &LlmConfig) -> LlmResult<()> {
        use std::path::Path;

        let model_path = Path::new(&config.model_path);
        if !model_path.exists() {
            return Err(AdapterError::ModelNotFound(config.model_path.clone()));
        }

        // Find the GGUF file
        let gguf_path = if model_path.is_file() {
            config.model_path.clone()
        } else {
            // Directory provided - look for .gguf files
            let gguf_files: Vec<_> = std::fs::read_dir(model_path)
                .map_err(AdapterError::IOError)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "gguf")
                        .unwrap_or(false)
                })
                .collect();

            if gguf_files.is_empty() {
                return Err(AdapterError::ModelNotFound(format!(
                    "No .gguf files found in {}",
                    config.model_path
                )));
            }

            gguf_files[0].path().to_string_lossy().to_string()
        };

        // Load model
        let model =
            sys::llama_load_model_from_file(&gguf_path, config.gpu_layers).map_err(|e| {
                AdapterError::RuntimeError(format!(
                    "Failed to load model from {}: {}. \
                 This may indicate an unsupported GGUF architecture — \
                 check that the vendored llama.cpp version supports this model's architecture. \
                 Enable verbose logging with XYBRID_LLAMACPP_VERBOSITY=4 for C++ error details.",
                    gguf_path, e
                ))
            })?;

        // Create context with thread and batch configuration
        // n_threads=0 means auto-detect in the C++ layer
        // n_batch=0 means use default (512)
        let context = sys::llama_new_context_with_model(
            &model,
            config.context_length,
            config.n_threads,
            config.n_batch,
            config.flash_attn,
        )
        .map_err(|e| AdapterError::RuntimeError(format!("Failed to create context: {}", e)))?;

        self.model = Some(model);
        *self.context.get_mut().unwrap() = Some(context);
        self.config = Some(config.clone());

        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some() && self.context.lock().unwrap().is_some()
    }

    fn unload(&mut self) -> LlmResult<()> {
        // Drop context first, then model (order matters).
        // LlamaContext and LlamaModel implement Drop, so take() handles cleanup.
        let _ = self.context.get_mut().unwrap().take();
        let _ = self.model.take();
        self.config = None;
        // Reset KV cache reuse state — the next load gets a fresh cache,
        // so any cached prefix from a prior model would be a use-after-free
        // waiting to happen.
        *self.kv_state.get_mut().unwrap() = KvCacheState::default();
        Ok(())
    }

    fn generate(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        self.with_model_and_context(|model, context| {
            // Format messages into prompt using chat template
            let prompt = sys::llama_format_chat(model, messages)?;

            // Tokenize with special token parsing enabled — the chat template contains
            // special tokens like <|im_start|>, <end_of_turn>, etc. that must be
            // recognized as their special token IDs, not as individual characters.
            let tokens = sys::llama_tokenize_special(model, &prompt, true)?;

            // Validate: input tokens must fit within the context window with room to generate
            let n_ctx = sys::llama_n_ctx(context);
            if tokens.len() >= n_ctx {
                return Err(AdapterError::InvalidInput(format!(
                    "Input too long: {} tokens exceeds context window of {} tokens. \
                     Reduce the prompt size or conversation history.",
                    tokens.len(),
                    n_ctx
                )));
            }

            // Multi-turn KV cache reuse: keep the prefix the previous call
            // already prefilled, only re-prefill the diverged tail. On a
            // first turn (or no shared prefix) `tail == tokens` and
            // `n_past == 0` — same observable behaviour as the legacy
            // unconditional clear, just without the duplicate work in
            // multi-turn chats.
            let (tail, n_past) =
                self.prepare_kv_cache_and_get_tail(model, context, &tokens, config.max_tokens)?;

            // Per-chunk timestamps capture the streaming cadence for TTFT +
            // inter-token-latency telemetry. The closure is observation-only
            // (no external emission) — generation still returns the full
            // token vector like `llama_generate_with_stops` did. Keeps the
            // non-streaming contract of this function intact.
            // Capture prompt size up-front so we can attach `tokens_in` to
            // the active span after the loop. The executor opens
            // `llm_inference_streaming` around this call, so this metadata
            // lands on the same span as the rest of the LLM telemetry that
            // `mirror_llm_metrics_to_span` writes post-return.
            let prompt_token_count = tokens.len();
            // Surface prompt size on the active span BEFORE the streaming
            // loop, so cloud-fallback aborts (which short-circuit before
            // tel.finalize runs) still attach tokens_in to LocalAborted.
            // Without this the dashboard's TOKENS column shows `—` for the
            // local leg of every aborted run. Successful runs harmlessly
            // overwrite this with the same value after finalize.
            xybrid_trace::add_metadata("tokens_in", prompt_token_count.to_string());
            let mut tel = StreamingTelemetry::new(prompt_token_count);
            let stream_result = sys::llama_generate_streaming(
                context,
                model,
                &tail,
                config.max_tokens,
                config.temperature,
                config.top_p,
                config.min_p,
                config.top_k,
                config.repetition_penalty,
                &config.stop_sequences,
                |_token_id, _token_text| {
                    tel.record_chunk();
                    Ok(())
                },
                n_past,
            );
            let (output_tokens, stopped_by_callback) = match stream_result {
                Ok(result) => result,
                Err(err) => {
                    self.reset_kv_cache_after_failed_stream(context);
                    return Err(err);
                }
            };

            // Finalize telemetry before the post-processing work below so
            // `generation_time_ms` reflects pure generation wallclock and is
            // not inflated by detokenization / stop-sequence scanning.
            let fields = tel.finalize(output_tokens.len());

            // Log generated token count and last few tokens for debugging
            log::debug!(
                target: "xybrid_core",
                "Generated {} tokens. Last 10: {:?}",
                output_tokens.len(),
                output_tokens.iter().rev().take(10).collect::<Vec<_>>()
            );

            // Decode tokens to text
            let mut text = sys::llama_detokenize(model, &output_tokens)?;

            // Debug: log the raw text and its bytes to understand encoding
            log::debug!(target: "xybrid_core", "LLM raw output ({} chars): {:?}", text.len(), &text[..text.len().min(200)]);
            log::debug!(target: "xybrid_core", "First 100 bytes: {:?}", text.as_bytes().iter().take(100).collect::<Vec<_>>());

            // Stop-pattern truncation + think-tag stripping live in
            // `streaming_postprocess`. The `*_BROKEN` patterns cover
            // tokenizers that split the leading `<` off a chat-template
            // marker — safe only for final-text cleanup, not streaming.
            let final_stop_patterns = {
                let mut extras: Vec<&str> = CHAT_STOP_PATTERNS.to_vec();
                extras.extend_from_slice(CHAT_STOP_PATTERNS_BROKEN);
                merge_stop_patterns(&config.stop_sequences, &extras)
            };
            log::debug!(target: "xybrid_core", "Searching for stop patterns: {:?}", final_stop_patterns);
            let stopped_in_text = truncate_at_first_stop(&mut text, &final_stop_patterns);
            let text = strip_thinking_tags(&text).trim().to_string();
            // `stopped_by_callback` catches the C layer detecting a stop
            // before the Rust post-scan would — e.g. the user-supplied
            // stop sequences that the C layer sees first. Prior code
            // silently dropped this signal and sometimes reported
            // `length` for a clean stop.
            let finish_reason = if stopped_in_text || stopped_by_callback {
                "stop"
            } else {
                "length"
            }
            .to_string();

            // Telemetry derivation (TTFT, mean/p95 ITL, decode_tps, prefill_tps)
            // lives in `llm_telemetry::StreamingTelemetry` and is shared with
            // the mistral backend — llama.cpp's sys bindings don't expose
            // `llama_perf_context`'s `t_p_eval_ms` / `t_eval_ms`, so the
            // numbers are derived from per-chunk timestamps. See
            // `compute_streaming_fields` for formula semantics.
            Ok(GenerationOutput {
                text,
                tokens_generated: output_tokens.len(),
                generation_time_ms: fields.generation_time_ms,
                tokens_per_second: fields.tokens_per_second,
                finish_reason,
                ttft_ms: fields.ttft_ms,
                mean_itl_ms: fields.mean_itl_ms,
                p95_itl_ms: fields.p95_itl_ms,
                emitted_chunks: fields.emitted_chunks,
                inter_chunk_ms: fields.inter_chunk_ms,
                decode_tps: fields.decode_tps,
                prefill_tps: fields.prefill_tps,
            })
        })
    }

    fn generate_raw(&self, prompt: &str, config: &GenerationConfig) -> LlmResult<GenerationOutput> {
        self.with_model_and_context(|model, context| {
            // Tokenize with parse_special=true so boundary tokens like
            // <|SPEECH_GENERATION_START|>, <|TEXT_PROMPT_START|>, <|im_start|>, etc.
            // collapse to single vocab IDs instead of 8-10 subword pieces each.
            // Matches llama-cpp-python's Llama.__call__ default (special=True),
            // which is required for NeuTTS-style codec TTS models.
            let tokens = sys::llama_tokenize_special(model, prompt, true)?;

            let n_ctx = sys::llama_n_ctx(context);
            if tokens.len() >= n_ctx {
                return Err(AdapterError::InvalidInput(format!(
                    "Input too long: {} tokens exceeds context window of {} tokens.",
                    tokens.len(),
                    n_ctx
                )));
            }

            // Multi-turn KV cache reuse: see prepare_kv_cache_and_get_tail
            // for the LCP + truncate-or-clear contract. raw-prompt callers
            // (TTS codec preludes etc.) typically don't share prefixes
            // across calls so the LCP path will mostly clear-and-refill,
            // but the unified helper keeps behaviour consistent.
            let (tail, n_past) =
                self.prepare_kv_cache_and_get_tail(model, context, &tokens, config.max_tokens)?;

            // Use the streaming-capable API with an observation-only
            // callback so raw generation gets the same TTFT / ITL /
            // decode-tps telemetry as `generate()`. Stop handling stays
            // raw — only user-supplied sequences, no chat markers.
            let prompt_token_count = tokens.len();
            // Surface prompt size on the active span BEFORE the streaming
            // loop, so cloud-fallback aborts (which short-circuit before
            // tel.finalize runs) still attach tokens_in to LocalAborted.
            // Without this the dashboard's TOKENS column shows `—` for the
            // local leg of every aborted run. Successful runs harmlessly
            // overwrite this with the same value after finalize.
            xybrid_trace::add_metadata("tokens_in", prompt_token_count.to_string());
            let mut tel = StreamingTelemetry::new(prompt_token_count);
            let (output_tokens, stopped_by_callback) = sys::llama_generate_streaming(
                context,
                model,
                &tail,
                config.max_tokens,
                config.temperature,
                config.top_p,
                config.min_p,
                config.top_k,
                config.repetition_penalty,
                &config.stop_sequences,
                |_token_id, _token_text| {
                    tel.record_chunk();
                    Ok(())
                },
                n_past,
            )?;
            let fields = tel.finalize(output_tokens.len());

            let text = sys::llama_detokenize(model, &output_tokens)?;
            let text = text.trim().to_string();
            let finish_reason = if stopped_by_callback {
                "stop"
            } else {
                "length"
            }
            .to_string();

            Ok(GenerationOutput {
                text,
                tokens_generated: output_tokens.len(),
                generation_time_ms: fields.generation_time_ms,
                tokens_per_second: fields.tokens_per_second,
                finish_reason,
                ttft_ms: fields.ttft_ms,
                mean_itl_ms: fields.mean_itl_ms,
                p95_itl_ms: fields.p95_itl_ms,
                emitted_chunks: fields.emitted_chunks,
                inter_chunk_ms: fields.inter_chunk_ms,
                decode_tps: fields.decode_tps,
                prefill_tps: fields.prefill_tps,
            })
        })
    }

    fn generate_streaming(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        on_token: crate::runtime_adapter::llm::StreamingCallback<'_>,
    ) -> LlmResult<GenerationOutput> {
        use crate::runtime_adapter::llm::PartialToken;
        let mut on_token = on_token;

        self.with_model_and_context(|model, context| {
            // Format messages into prompt using chat template
            let prompt = sys::llama_format_chat(model, messages)?;

            // Tokenize with special token parsing — chat template contains special tokens
            let tokens = sys::llama_tokenize_special(model, &prompt, true)?;

            // Validate: input tokens must fit within the context window with room to generate
            let n_ctx = sys::llama_n_ctx(context);
            if tokens.len() >= n_ctx {
                return Err(AdapterError::InvalidInput(format!(
                    "Input too long: {} tokens exceeds context window of {} tokens. \
                     Reduce the prompt size or conversation history.",
                    tokens.len(),
                    n_ctx
                )));
            }

            // Multi-turn KV cache reuse: keep the prefix the previous call
            // already prefilled, only re-prefill the diverged tail. See
            // prepare_kv_cache_and_get_tail for the full contract.
            let (tail, n_past) =
                self.prepare_kv_cache_and_get_tail(model, context, &tokens, config.max_tokens)?;

            // Shared streaming state: telemetry recorder + text filter.
            // The filter owns cumulative text, think-block state, stop-pattern
            // detection, and safe-prefix buffering — this backend just feeds
            // raw chunks in and emits whatever comes out. See
            // `streaming_postprocess` for the contract.
            //
            // Stop patterns are cloned once so the filter can own them while
            // the C layer and final-text cleanup keep a reference. The
            // `_BROKEN` variants are intentionally excluded from streaming
            // (they false-positive on legitimate text) — they only run in
            // the final cleanup pass below.
            let prompt_token_count = tokens.len();
            // Surface prompt size on the active span BEFORE the streaming
            // loop, so cloud-fallback aborts (which short-circuit before
            // tel.finalize runs) still attach tokens_in to LocalAborted.
            // Without this the dashboard's TOKENS column shows `—` for the
            // local leg of every aborted run. Successful runs harmlessly
            // overwrite this with the same value after finalize.
            xybrid_trace::add_metadata("tokens_in", prompt_token_count.to_string());
            let mut tel = StreamingTelemetry::new(prompt_token_count);
            let stop_patterns = merge_stop_patterns(&config.stop_sequences, CHAT_STOP_PATTERNS);
            let mut filter = StreamingTextFilter::new(stop_patterns.clone());
            let mut token_index = 0usize;

            let (output_tokens, stopped_by_callback) = sys::llama_generate_streaming(
                context,
                model,
                &tail,
                config.max_tokens,
                config.temperature,
                config.top_p,
                config.min_p,
                config.top_k,
                config.repetition_penalty,
                &stop_patterns, // C layer uses these for early stop / llama_vocab_is_eog
                |token_id, token_text| {
                    // Timestamp every C-layer callback, before any filtering —
                    // the stream itself is what's being measured, not the
                    // user-visible emission.
                    tel.record_chunk();

                    if let Some(safe_text) = filter.push(token_text) {
                        let partial = PartialToken::new(
                            safe_text,
                            token_index,
                            filter.cumulative_emitted().to_string(),
                        )
                        .with_token_id(token_id as i64);
                        token_index += 1;
                        on_token(partial)?;
                    }

                    Ok(())
                },
                n_past,
            )?;

            // Finalize telemetry before post-processing so `generation_time_ms`
            // reflects only the generation loop, not detokenization or
            // stop-pattern cleanup. Shared with `generate()` — see
            // `compute_streaming_fields`.
            let fields = tel.finalize(output_tokens.len());

            // Final-output cleanup: detokenize the full token vector (rather
            // than using the filter's cumulative text) as a belt-and-braces
            // guard against chunk-boundary UTF-8 edge cases, then run the
            // same truncate / trim-partial / strip-think passes used by the
            // non-streaming path. The `_BROKEN` fallback patterns are
            // included here because this is final-text only — no streaming
            // false-positive risk.
            let final_patterns = {
                let mut extras: Vec<&str> = CHAT_STOP_PATTERNS.to_vec();
                extras.extend_from_slice(CHAT_STOP_PATTERNS_BROKEN);
                merge_stop_patterns(&config.stop_sequences, &extras)
            };
            let mut text = sys::llama_detokenize(model, &output_tokens)?;
            let stopped_full = truncate_at_first_stop(&mut text, &final_patterns);
            let trimmed_partial = trim_partial_stop_suffix(&mut text, &final_patterns);
            let text = strip_thinking_tags(&text).trim().to_string();
            // `stopped_by_callback` is an independent signal from the C
            // layer that a stop sequence was hit — previously dropped.
            let finish_reason =
                if filter.is_stopped() || stopped_full || trimmed_partial || stopped_by_callback {
                    "stop".to_string()
                } else {
                    "length".to_string()
                };

            // Send final empty token with finish_reason — matches the
            // pre-refactor contract so downstream consumers see a
            // terminal signal. Guarded on `token_index > 0` to avoid
            // emitting a stray terminal chunk when nothing was ever
            // emitted (e.g. immediate stop).
            if token_index > 0 {
                let final_partial = PartialToken::new(String::new(), token_index, text.clone())
                    .with_finish_reason(&finish_reason);
                let _ = on_token(final_partial);
            }

            Ok(GenerationOutput {
                text,
                tokens_generated: output_tokens.len(),
                generation_time_ms: fields.generation_time_ms,
                tokens_per_second: fields.tokens_per_second,
                finish_reason,
                ttft_ms: fields.ttft_ms,
                mean_itl_ms: fields.mean_itl_ms,
                p95_itl_ms: fields.p95_itl_ms,
                emitted_chunks: fields.emitted_chunks,
                inter_chunk_ms: fields.inter_chunk_ms,
                decode_tps: fields.decode_tps,
                prefill_tps: fields.prefill_tps,
            })
        })
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn memory_usage(&self) -> Option<u64> {
        // TODO: Implement via llama_get_state_size or similar
        None
    }

    fn context_length(&self) -> Option<usize> {
        self.config.as_ref().map(|c| c.context_length)
    }

    /// Surface the prefix length the most recent `generate*` reused from
    /// the persisted KV cache. `Some(0)` on a first turn or a totally
    /// divergent prompt, `None` only before any `generate*` has run.
    /// Telemetry hoists this as `prompt_cached_tokens` on inference
    /// events when the value is positive.
    fn last_cached_prefix_len(&self) -> Option<usize> {
        self.kv_state.lock().ok().and_then(|s| s.last_prefix_hit)
    }
}

// =============================================================================
// Stub implementation when llm-llamacpp feature is not enabled
// =============================================================================

#[cfg(not(feature = "llm-llamacpp"))]
pub struct LlamaCppBackend;

#[cfg(not(feature = "llm-llamacpp"))]
impl LlamaCppBackend {
    pub fn new() -> LlmResult<Self> {
        Err(AdapterError::RuntimeError(
            "llm-llamacpp feature not enabled. Build with --features llm-llamacpp".to_string(),
        ))
    }
}

#[cfg(not(feature = "llm-llamacpp"))]
impl LlmBackend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama-cpp"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["gguf"]
    }

    fn load(&mut self, _config: &LlmConfig) -> LlmResult<()> {
        Err(AdapterError::RuntimeError(
            "llm-llamacpp feature not enabled".to_string(),
        ))
    }

    fn is_loaded(&self) -> bool {
        false
    }

    fn unload(&mut self) -> LlmResult<()> {
        Ok(())
    }

    fn generate(
        &self,
        _messages: &[ChatMessage],
        _config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        Err(AdapterError::RuntimeError(
            "llm-llamacpp feature not enabled".to_string(),
        ))
    }

    fn generate_raw(
        &self,
        _prompt: &str,
        _config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        Err(AdapterError::RuntimeError(
            "llm-llamacpp feature not enabled".to_string(),
        ))
    }
}

#[cfg(all(test, feature = "llm-llamacpp"))]
mod tests {
    use super::*;

    #[test]
    fn backend_reports_true_streaming_for_sdk_cancellation_gate() {
        let backend = LlamaCppBackend::new().unwrap();

        assert!(
            backend.supports_streaming(),
            "llama.cpp must stay on the true streaming path so SDK abort checks can interrupt generation"
        );
    }

    #[test]
    fn failed_stream_resets_rust_kv_cache_state() {
        let backend = LlamaCppBackend::new().unwrap();
        {
            let mut state = backend.kv_state.lock().unwrap();
            state.cached_tokens = vec![1, 2, 3];
            state.last_prefix_hit = Some(2);
        }

        backend.clear_cached_prefix_state();

        let state = backend.kv_state.lock().unwrap();
        assert!(
            state.cached_tokens.is_empty(),
            "failed streaming runs must not leave reusable prompt tokens behind"
        );
        assert_eq!(
            state.last_prefix_hit, None,
            "failed streaming runs must clear prefix-hit metadata"
        );
    }

    #[test]
    fn lcp_empty_inputs_return_zero() {
        // First-call shape: nothing cached yet. The KV cache is empty so
        // there's nothing to reuse; caller falls back to a full prefill.
        assert_eq!(compute_reusable_prefix_len(&[], &[1, 2, 3]), 0);
        // Defensive: an empty new prompt would be rejected by the caller
        // before reaching here, but the helper must not panic on it.
        assert_eq!(compute_reusable_prefix_len(&[1, 2, 3], &[]), 0);
        assert_eq!(compute_reusable_prefix_len(&[], &[]), 0);
    }

    #[test]
    fn lcp_identical_prompts_keep_one_token_for_decoder() {
        // Multi-turn replay of the exact same prompt. We could in
        // principle reuse the entire cache, but the C wrapper rejects
        // an empty input slice, so the helper caps reuse at len-1 to
        // guarantee one fresh token feeds the decode loop.
        let same = vec![10, 20, 30, 40];
        assert_eq!(compute_reusable_prefix_len(&same, &same), 3);
    }

    #[test]
    fn lcp_partial_match_returns_shared_length() {
        // Typical multi-turn shape: shared system prompt + earlier turns,
        // diverging at the new user turn. The shared prefix is the part
        // worth keeping in the cache.
        let cached = vec![1, 2, 3, 4, 99, 99];
        let new = vec![1, 2, 3, 4, 50, 60, 70];
        assert_eq!(compute_reusable_prefix_len(&cached, &new), 4);
    }

    #[test]
    fn lcp_no_overlap_returns_zero() {
        // Totally divergent prompts (e.g. user starts a new conversation
        // without clearing context). The caller will full-clear the cache
        // when this returns 0.
        assert_eq!(compute_reusable_prefix_len(&[1, 2, 3], &[9, 8, 7]), 0);
    }

    #[test]
    fn lcp_caps_at_new_tokens_minus_one() {
        // Cached is longer than the new prompt and shares it entirely.
        // We still cap at len-1 so the decoder gets a fresh token to
        // process — otherwise the C wrapper bails on empty input.
        let cached = vec![1, 2, 3, 4, 5, 6];
        let new = vec![1, 2, 3];
        assert_eq!(compute_reusable_prefix_len(&cached, &new), 2);
    }

    #[test]
    fn lcp_single_token_new_prompt_returns_zero() {
        // Edge case: new prompt is one token long (rare but possible —
        // think test fixtures). max_reuse = 0 ⇒ we always re-prefill.
        assert_eq!(compute_reusable_prefix_len(&[1, 2, 3], &[1]), 0);
    }
}
