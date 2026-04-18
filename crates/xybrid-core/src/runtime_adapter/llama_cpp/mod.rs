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
        Ok(())
    }

    fn generate(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
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

        // Clear KV cache to reset context state for new conversation
        // This is essential when reusing the context across multiple queries
        sys::llama_kv_cache_clear(context);

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

        // Per-chunk timestamps capture the streaming cadence for TTFT +
        // inter-token-latency telemetry. The closure is observation-only
        // (no external emission) — generation still returns the full
        // token vector like `llama_generate_with_stops` did. Keeps the
        // non-streaming contract of this function intact.
        let mut tel = StreamingTelemetry::new(tokens.len());
        let (output_tokens, _stopped_by_callback) = sys::llama_generate_streaming(
            context,
            model,
            &tokens,
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
        )?;

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
        let stopped = truncate_at_first_stop(&mut text, &final_stop_patterns);
        let text = strip_thinking_tags(&text).trim().to_string();
        let finish_reason = if stopped { "stop" } else { "length" }.to_string();

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
    }

    fn generate_raw(&self, prompt: &str, config: &GenerationConfig) -> LlmResult<GenerationOutput> {
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

        sys::llama_kv_cache_clear(context);

        // Tokenize directly — no chat template formatting.
        let tokens = sys::llama_tokenize(model, prompt, true)?;

        let n_ctx = sys::llama_n_ctx(context);
        if tokens.len() >= n_ctx {
            return Err(AdapterError::InvalidInput(format!(
                "Input too long: {} tokens exceeds context window of {} tokens.",
                tokens.len(),
                n_ctx
            )));
        }

        let start = std::time::Instant::now();

        let output_tokens = sys::llama_generate_with_stops(
            context,
            model,
            &tokens,
            config.max_tokens,
            config.temperature,
            config.top_p,
            config.min_p,
            config.top_k,
            config.repetition_penalty,
            &config.stop_sequences,
        )?;

        let elapsed = start.elapsed();
        let text = sys::llama_detokenize(model, &output_tokens)?;
        let text = text.trim().to_string();

        let tokens_generated = output_tokens.len();
        let tokens_per_second = if elapsed.as_secs_f32() > 0.0 {
            tokens_generated as f32 / elapsed.as_secs_f32()
        } else {
            0.0
        };

        Ok(GenerationOutput {
            text,
            tokens_generated,
            generation_time_ms: elapsed.as_millis() as u64,
            tokens_per_second,
            finish_reason: "length".to_string(),
            ttft_ms: None,
            mean_itl_ms: None,
            p95_itl_ms: None,
            emitted_chunks: None,
            inter_chunk_ms: Vec::new(),
            decode_tps: None,
            prefill_tps: None,
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

        // Clear KV cache to reset context state for new conversation
        sys::llama_kv_cache_clear(context);

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
        let mut tel = StreamingTelemetry::new(tokens.len());
        let stop_patterns = merge_stop_patterns(&config.stop_sequences, CHAT_STOP_PATTERNS);
        let mut filter = StreamingTextFilter::new(stop_patterns.clone());
        let mut token_index = 0usize;

        let (output_tokens, _stopped_by_callback) = sys::llama_generate_streaming(
            context,
            model,
            &tokens,
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
        let finish_reason = if filter.is_stopped() || stopped_full || trimmed_partial {
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
