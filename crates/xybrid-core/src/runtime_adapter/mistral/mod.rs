//! MistralBackend - LLM inference using mistral.rs
//!
//! This module provides the mistral.rs implementation of the LlmBackend trait.
//! It is feature-gated behind `llm-mistral`.
//!
//! # Note
//!
//! mistral.rs uses candle + gemm which requires `+fp16` on ARM64.
//! This causes SIGILL on Android devices without ARMv8.2-A FP16 extension.
//! For Android, use `llama_cpp` backend instead.

use crate::ir::MessageRole;
use crate::runtime_adapter::llm::{
    ChatMessage, GenerationConfig, GenerationOutput, LlmBackend, LlmConfig, LlmResult,
};
use crate::runtime_adapter::AdapterError;

#[cfg(feature = "llm-mistral")]
use mistralrs::{
    GgufModelBuilder, Model, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole,
};

/// MistralBackend - LLM inference using mistral.rs.
///
/// This backend uses the mistral.rs library for pure-Rust LLM inference.
/// It supports GGUF models and provides efficient inference with features like:
/// - Paged attention for memory efficiency
/// - Metal/CUDA acceleration (via feature flags)
/// - Streaming generation (future)
///
/// # Platform Support
///
/// - **macOS/iOS**: Works with Metal acceleration
/// - **Linux/Windows**: Works with CUDA or CPU
/// - **Android**: NOT SUPPORTED - use `LlamaCppBackend` instead
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::runtime_adapter::mistral::MistralBackend;
/// use xybrid_core::runtime_adapter::llm::{LlmBackend, LlmConfig};
///
/// let mut backend = MistralBackend::new()?;
/// backend.load(&LlmConfig::new("model.gguf"))?;
/// ```
#[cfg(feature = "llm-mistral")]
pub struct MistralBackend {
    /// Loaded model (None if not loaded)
    model: Option<Model>,
    /// Current configuration
    config: Option<LlmConfig>,
}

#[cfg(feature = "llm-mistral")]
impl MistralBackend {
    /// Create a new MistralBackend.
    pub fn new() -> LlmResult<Self> {
        Ok(Self {
            model: None,
            config: None,
        })
    }

    /// Convert our MessageRole to mistral.rs TextMessageRole.
    fn convert_role(role: &MessageRole) -> TextMessageRole {
        match role {
            MessageRole::System => TextMessageRole::System,
            MessageRole::Assistant => TextMessageRole::Assistant,
            MessageRole::User => TextMessageRole::User,
        }
    }
}

#[cfg(feature = "llm-mistral")]
impl Default for MistralBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create MistralBackend")
    }
}

#[cfg(feature = "llm-mistral")]
impl LlmBackend for MistralBackend {
    fn name(&self) -> &str {
        "mistral"
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

        // Determine directory and filename
        let (model_dir, model_file) = if model_path.is_file() {
            let dir = model_path
                .parent()
                .ok_or_else(|| AdapterError::InvalidInput("Invalid model path".to_string()))?;
            let file = model_path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| AdapterError::InvalidInput("Invalid model filename".to_string()))?;
            (dir.to_string_lossy().to_string(), file.to_string())
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

            let file = gguf_files[0].file_name().to_string_lossy().to_string();
            (config.model_path.clone(), file)
        };

        // Build the model using tokio runtime
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AdapterError::RuntimeError(format!("Failed to create runtime: {}", e)))?;

        let model = rt.block_on(async {
            let mut builder = GgufModelBuilder::new(&model_dir, vec![model_file]);

            // Apply chat template if provided
            if let Some(ref template) = config.chat_template {
                builder = builder.with_chat_template(template);
            }

            // Enable logging if requested
            if config.logging {
                builder = builder.with_logging();
            }

            // Enable paged attention if requested
            if config.paged_attention {
                builder = builder
                    .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())
                    .map_err(|e| {
                        AdapterError::RuntimeError(format!("Paged attention setup failed: {}", e))
                    })?;
            }

            builder
                .build()
                .await
                .map_err(|e| AdapterError::RuntimeError(format!("Model loading failed: {}", e)))
        })?;

        self.model = Some(model);
        self.config = Some(config.clone());

        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    fn unload(&mut self) -> LlmResult<()> {
        self.model = None;
        self.config = None;
        Ok(())
    }

    fn generate(
        &self,
        messages: &[ChatMessage],
        _config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        let model = self.model.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No model loaded. Call load() first.".to_string())
        })?;

        // Build the request with messages
        let mut request = RequestBuilder::new();

        for msg in messages {
            request = request.add_message(Self::convert_role(&msg.role), &msg.content);
        }

        let start = std::time::Instant::now();

        // Run inference
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AdapterError::RuntimeError(format!("Failed to create runtime: {}", e)))?;

        let response = rt.block_on(async {
            model
                .send_chat_request(request)
                .await
                .map_err(|e| AdapterError::InferenceFailed(format!("Generation failed: {}", e)))
        })?;

        let elapsed = start.elapsed();

        // Extract text from response
        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.to_string())
            .unwrap_or_default();

        let finish_reason = response
            .choices
            .first()
            .map(|c| c.finish_reason.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let tokens_generated = response.usage.completion_tokens;

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

    // TODO: Implement true streaming for mistral.rs once we verify the streaming API
    // For now, uses the default implementation which falls back to non-streaming.
    // The mistral.rs streaming API (stream_chat_request) has a different response
    // structure that needs investigation.

    fn supports_streaming(&self) -> bool {
        // Return false until true streaming is implemented
        false
    }

    fn memory_usage(&self) -> Option<u64> {
        None
    }

    fn context_length(&self) -> Option<usize> {
        self.config.as_ref().map(|c| c.context_length)
    }
}

// =============================================================================
// Stub implementation when llm-mistral feature is not enabled
// =============================================================================

#[cfg(not(feature = "llm-mistral"))]
pub struct MistralBackend;

#[cfg(not(feature = "llm-mistral"))]
impl MistralBackend {
    pub fn new() -> LlmResult<Self> {
        Err(AdapterError::RuntimeError(
            "llm-mistral feature not enabled. Build with --features llm-mistral".to_string(),
        ))
    }
}

#[cfg(not(feature = "llm-mistral"))]
impl LlmBackend for MistralBackend {
    fn name(&self) -> &str {
        "mistral"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["gguf"]
    }

    fn load(&mut self, _config: &LlmConfig) -> LlmResult<()> {
        Err(AdapterError::RuntimeError(
            "llm-mistral feature not enabled".to_string(),
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
            "llm-mistral feature not enabled".to_string(),
        ))
    }

    fn generate_raw(
        &self,
        _prompt: &str,
        _config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput> {
        Err(AdapterError::RuntimeError(
            "llm-mistral feature not enabled".to_string(),
        ))
    }
}
