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

use crate::runtime_adapter::llm::{
    ChatMessage, GenerationConfig, GenerationOutput, LlmBackend, LlmConfig, LlmResult,
};
use crate::runtime_adapter::AdapterError;

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
    /// Pointer to context (llama_context*)
    context: Option<sys::LlamaContext>,
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
            context: None,
            config: None,
        })
    }
}

#[cfg(feature = "llm-llamacpp")]
impl Drop for LlamaCppBackend {
    fn drop(&mut self) {
        // Free context first, then model
        if let Some(ctx) = self.context.take() {
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

        // Create context
        let context = sys::llama_new_context_with_model(&model, config.context_length)
            .map_err(|e| AdapterError::RuntimeError(format!("Failed to create context: {}", e)))?;

        self.model = Some(model);
        self.context = Some(context);
        self.config = Some(config.clone());

        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some() && self.context.is_some()
    }

    fn unload(&mut self) -> LlmResult<()> {
        if let Some(ctx) = self.context.take() {
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
        let context = self.context.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No context. Call load() first.".to_string())
        })?;

        // Format messages into prompt using chat template
        let prompt = sys::llama_format_chat(model, messages)?;

        // Tokenize
        let tokens = sys::llama_tokenize(model, &prompt, true)?;

        let start = std::time::Instant::now();

        // Generate
        let output_tokens = sys::llama_generate(
            context,
            model,
            &tokens,
            config.max_tokens,
            config.temperature,
            config.top_p,
            config.top_k,
        )?;

        let elapsed = start.elapsed();

        // Decode tokens to text
        let text = sys::llama_detokenize(model, &output_tokens)?;

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
            finish_reason: "stop".to_string(),
        })
    }

    fn generate_raw(&self, prompt: &str, config: &GenerationConfig) -> LlmResult<GenerationOutput> {
        let messages = vec![ChatMessage::user(prompt)];
        self.generate(&messages, config)
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
