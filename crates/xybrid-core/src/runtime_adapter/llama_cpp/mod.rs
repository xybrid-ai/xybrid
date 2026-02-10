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
use crate::runtime_adapter::AdapterError;
use std::sync::Mutex;

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
        // Initialize llama.cpp backend
        sys::llama_backend_init();

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
        // Free context first, then model.
        // get_mut() doesn't lock — safe because Drop has &mut self.
        if let Some(ctx) = self.context.get_mut().unwrap().take() {
            sys::llama_free(ctx);
        }
        if let Some(model) = self.model.take() {
            sys::llama_free_model(model);
        }
        // Note: Don't call llama_backend_free here as other instances may exist
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
        let model = sys::llama_load_model_from_file(&gguf_path, config.gpu_layers)
            .map_err(|e| AdapterError::RuntimeError(format!("Failed to load model: {}", e)))?;

        // Create context with thread and batch configuration
        // n_threads=0 means auto-detect in the C++ layer
        // n_batch=0 means use default (512)
        let context = sys::llama_new_context_with_model(
            &model,
            config.context_length,
            config.n_threads,
            config.n_batch,
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
        if let Some(ctx) = self.context.get_mut().unwrap().take() {
            sys::llama_free(ctx);
        }
        if let Some(model) = self.model.take() {
            sys::llama_free_model(model);
        }
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
        let ctx_guard = self.context.lock().map_err(|_| {
            AdapterError::RuntimeError("Context mutex poisoned".to_string())
        })?;
        let context = ctx_guard.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No context. Call load() first.".to_string())
        })?;

        // Clear KV cache to reset context state for new conversation
        // This is essential when reusing the context across multiple queries
        sys::llama_kv_cache_clear(context);

        // Format messages into prompt using chat template
        let prompt = sys::llama_format_chat(model, messages)?;

        // Tokenize
        let tokens = sys::llama_tokenize(model, &prompt, true)?;

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

        let start = std::time::Instant::now();

        // Generate with stop sequences for early termination
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

        // Apply stop sequence truncation
        // Find the earliest occurrence of any stop sequence and truncate there
        let mut finish_reason = "length".to_string();

        // Build list of patterns to check - include config stop sequences plus common markers
        let mut stop_patterns: Vec<&str> = config.stop_sequences.iter().map(|s| s.as_str()).collect();
        // Always check for these common markers even if not in config
        // Note: <end_of_turn> is Gemma's stop token, <|im_end|> is ChatML (Qwen, Phi)
        // Also include partial markers without "<" for models using ChatML fallback
        let extra_patterns = [
            "<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>", "<end_of_turn>",
            "|im_end|>", "|im_start|>", "|endoftext|>", "end_of_turn>",
        ];
        for p in &extra_patterns {
            if !stop_patterns.contains(p) {
                stop_patterns.push(p);
            }
        }

        log::debug!(target: "xybrid_core", "Searching for stop patterns: {:?}", stop_patterns);

        let mut earliest_pos: Option<usize> = None;
        for stop_seq in &stop_patterns {
            if let Some(pos) = text.find(stop_seq) {
                log::debug!(target: "xybrid_core", "Found '{}' at position {}", stop_seq, pos);
                match earliest_pos {
                    None => earliest_pos = Some(pos),
                    Some(current) if pos < current => earliest_pos = Some(pos),
                    _ => {}
                }
            }
        }
        if let Some(pos) = earliest_pos {
            log::debug!(target: "xybrid_core", "Truncating at position {}", pos);
            text.truncate(pos);
            finish_reason = "stop".to_string();
        } else {
            log::debug!(target: "xybrid_core", "No stop pattern found in text");
        }

        // Trim any trailing whitespace from the response
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
            finish_reason,
        })
    }

    fn generate_raw(&self, prompt: &str, config: &GenerationConfig) -> LlmResult<GenerationOutput> {
        let messages = vec![ChatMessage::user(prompt)];
        self.generate(&messages, config)
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
        let ctx_guard = self.context.lock().map_err(|_| {
            AdapterError::RuntimeError("Context mutex poisoned".to_string())
        })?;
        let context = ctx_guard.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No context. Call load() first.".to_string())
        })?;

        // Clear KV cache to reset context state for new conversation
        sys::llama_kv_cache_clear(context);

        // Format messages into prompt using chat template
        let prompt = sys::llama_format_chat(model, messages)?;

        // Tokenize
        let tokens = sys::llama_tokenize(model, &prompt, true)?;

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

        let start = std::time::Instant::now();

        // Track cumulative text and what we've safely emitted
        let mut cumulative_text = String::new();
        let mut last_emitted_len = 0usize;
        let mut token_index = 0usize;
        let mut hit_stop_pattern = false;

        // Build stop patterns list for early detection during streaming
        // Note: <end_of_turn> is Gemma's stop token, <|im_end|> is ChatML (Qwen, Phi)
        let streaming_stop_patterns: Vec<String> = {
            let mut patterns: Vec<String> = config.stop_sequences.clone();
            for p in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>", "<end_of_turn>"] {
                if !patterns.iter().any(|s| s == p) {
                    patterns.push(p.to_string());
                }
            }
            patterns
        };

        // Partial stop markers that can appear without a leading "<".
        // When a model is prompted with ChatML but doesn't natively use it
        // (e.g., Gemma with ChatML fallback), it may generate malformed stop
        // tokens like "|im_end|>" without the "<". We need to hold back
        // prefixes of these markers too, so they aren't emitted before the
        // full marker is accumulated and caught.
        let partial_holdback_patterns: Vec<&str> = vec![
            "|im_end|>",
            "|im_start|>",
            "|endoftext|>",
            "end_of_turn>",
        ];

        // Helper: find if text ends with a PREFIX of any stop pattern
        // Returns the position where the potential stop sequence starts, or None
        let find_potential_stop_start = |text: &str| -> Option<usize> {
            // Check both full stop patterns and partial holdback patterns
            let full_iter = streaming_stop_patterns.iter().map(|s| s.as_str());
            let partial_iter = partial_holdback_patterns.iter().copied();

            for pattern in full_iter.chain(partial_iter) {
                // Check if text ends with any prefix of this pattern
                for prefix_len in 1..=pattern.len() {
                    let prefix = &pattern[..prefix_len];
                    if text.ends_with(prefix) {
                        return Some(text.len() - prefix_len);
                    }
                }
            }
            None
        };

        // Generate with streaming callback
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
            &streaming_stop_patterns, // Pass full stop patterns to C layer
            |token_id, token_text| {
                // Once we hit a stop pattern, don't emit any more tokens
                if hit_stop_pattern {
                    return Ok(());
                }

                cumulative_text.push_str(token_text);

                // Check if any COMPLETE stop pattern is now in the cumulative text
                for pattern in &streaming_stop_patterns {
                    if cumulative_text.contains(pattern.as_str()) {
                        log::debug!(target: "xybrid_core", "Streaming: detected stop pattern '{}', stopping token emission", pattern);
                        hit_stop_pattern = true;
                        // Truncate the cumulative text at the stop pattern
                        if let Some(pos) = cumulative_text.find(pattern.as_str()) {
                            cumulative_text.truncate(pos);
                        }
                        return Ok(());
                    }
                }

                // Also check for partial/corrupted stop tokens (missing leading "<")
                // This handles cases where the model generates malformed stop tokens
                let partial_stop_markers = ["|im_end|>", "im_end|>", "end_of_turn>", "_of_turn>"];
                for marker in partial_stop_markers {
                    if cumulative_text.contains(marker) {
                        log::debug!(target: "xybrid_core", "Streaming: detected partial stop marker '{}', stopping", marker);
                        hit_stop_pattern = true;
                        if let Some(pos) = cumulative_text.find(marker) {
                            cumulative_text.truncate(pos);
                        }
                        return Ok(());
                    }
                }

                // Find the safe portion to emit (excluding potential stop sequence starts)
                let safe_end = find_potential_stop_start(&cumulative_text)
                    .unwrap_or(cumulative_text.len());

                // Only emit if we have new safe content
                if safe_end > last_emitted_len {
                    let safe_text = &cumulative_text[last_emitted_len..safe_end];
                    let partial = PartialToken::new(
                        safe_text.to_string(),
                        token_index,
                        cumulative_text[..safe_end].to_string(),
                    ).with_token_id(token_id as i64);

                    last_emitted_len = safe_end;
                    token_index += 1;
                    on_token(partial)?;
                }

                Ok(())
            },
        )?;

        let elapsed = start.elapsed();

        // Decode tokens to text (for final output)
        let mut text = sys::llama_detokenize(model, &output_tokens)?;

        // Apply stop sequence truncation (safety net - streaming callback already truncated)
        let mut finish_reason = if hit_stop_pattern { "stop".to_string() } else { "length".to_string() };

        // Use streaming_stop_patterns (already built above) for final text cleanup
        let mut earliest_pos: Option<usize> = None;
        for pattern in &streaming_stop_patterns {
            if let Some(pos) = text.find(pattern.as_str()) {
                match earliest_pos {
                    None => earliest_pos = Some(pos),
                    Some(current) if pos < current => earliest_pos = Some(pos),
                    _ => {}
                }
            }
        }
        if let Some(pos) = earliest_pos {
            text.truncate(pos);
            finish_reason = "stop".to_string();
        }

        // Clean up partial/corrupted stop tokens that may appear at the end
        // This handles cases where the model generates parts of stop tokens
        // that don't exactly match the full pattern (e.g., "|im_end|>" without "<")
        let partial_stop_fragments = [
            // ChatML partial fragments
            "|im_end|>",
            "im_end|>",
            "_end|>",
            "end|>",
            "nd|>",
            "d|>",
            "|>",
            "<|im_end",
            "<|im_en",
            "<|im_e",
            "<|im_",
            "<|im",
            "<|i",
            "<|",
            // Gemma partial fragments
            "end_of_turn>",
            "_of_turn>",
            "of_turn>",
            "f_turn>",
            "_turn>",
            "turn>",
            "urn>",
            "rn>",
            "n>",
            "<end_of_turn",
            "<end_of_tur",
            "<end_of_tu",
            "<end_of_t",
            "<end_of_",
            "<end_of",
            "<end_o",
            "<end_",
            "<end",
            // Common end markers
            "</s",
            "<s>",
            "|endoftext|>",
            "endoftext|>",
        ];

        for fragment in partial_stop_fragments {
            if text.ends_with(fragment) {
                log::debug!(target: "xybrid_core", "Cleaning up partial stop token: '{}'", fragment);
                text.truncate(text.len() - fragment.len());
                finish_reason = "stop".to_string();
                break;
            }
        }

        let text = text.trim().to_string();

        let tokens_generated = output_tokens.len();
        let tokens_per_second = if elapsed.as_secs_f32() > 0.0 {
            tokens_generated as f32 / elapsed.as_secs_f32()
        } else {
            0.0
        };

        // Send final token with finish reason
        if token_index > 0 {
            let final_partial = PartialToken::new(
                String::new(),
                token_index,
                text.clone(),
            ).with_finish_reason(&finish_reason);

            // Ignore error on final notification - generation is complete
            let _ = on_token(final_partial);
        }

        Ok(GenerationOutput {
            text,
            tokens_generated,
            generation_time_ms: elapsed.as_millis() as u64,
            tokens_per_second,
            finish_reason,
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
            "llm-llamacpp feature not enabled. Build with --features llm-llamacpp"
                .to_string(),
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
