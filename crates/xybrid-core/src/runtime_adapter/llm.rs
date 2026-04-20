//! LLM types and adapter - shared abstractions for local LLM inference.
//!
//! This module provides:
//! - `LlmBackend` trait - interface for LLM backends (mistral, llama_cpp)
//! - `LlmConfig` / `GenerationConfig` - configuration types
//! - `LlmRuntimeAdapter` - RuntimeAdapter implementation that uses backends
//!
//! # Architecture
//!
//! ```text
//! LlmRuntimeAdapter (RuntimeAdapter impl)
//!     │
//!     └── LlmBackend (trait - swappable)
//!             │
//!             ├── MistralBackend (mistral.rs - desktop)
//!             └── LlamaCppBackend (llama.cpp - Android + fallback)
//! ```

use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{
    AdapterError, AdapterResult, ModelMetadata, RuntimeAdapter, RuntimeAdapterExt,
};
use std::collections::HashMap;
use std::path::Path;

// Re-export types from the always-available types module
pub use super::types::{
    ChatMessage, GenerationConfig, LlmConfig, PartialToken, StreamingCallback, StreamingError,
};

// =============================================================================
// Result Type
// =============================================================================

/// Result type for LLM backend operations.
pub type LlmResult<T> = Result<T, AdapterError>;

// =============================================================================
// Generation Output
// =============================================================================

/// Generation output from LLM inference.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Tokens per second (wallclock — includes prefill + decode)
    pub tokens_per_second: f32,
    /// Finish reason: "stop", "length", "error"
    pub finish_reason: String,
    /// Time to first emitted chunk (ms). `None` for non-streaming backends.
    pub ttft_ms: Option<u64>,
    /// Mean inter-token latency (ms), computed from inter-chunk gaps.
    pub mean_itl_ms: Option<f32>,
    /// p95 inter-token latency (ms).
    pub p95_itl_ms: Option<u32>,
    /// Number of chunks emitted (≈ tokens when 1 token/chunk).
    pub emitted_chunks: Option<u32>,
    /// Per-gap inter-chunk latencies (ms), for histogram analysis if desired.
    pub inter_chunk_ms: Vec<u32>,
    /// Decode-only tokens/sec (excludes prefill time). Sourced from
    /// mistralrs's terminal `Usage.avg_compl_tok_per_sec`. Steady-state
    /// throughput, not wallclock.
    pub decode_tps: Option<f32>,
    /// Prefill tokens/sec. Sourced from `Usage.avg_prompt_tok_per_sec`.
    pub prefill_tps: Option<f32>,
}

// =============================================================================
// LLM Backend Trait
// =============================================================================

/// Trait for LLM inference backends.
///
/// This abstraction allows the LlmRuntimeAdapter to work with different
/// LLM engines (mistral.rs, llama.cpp, etc.) through a common interface.
pub trait LlmBackend: Send + Sync {
    /// Returns the name of this backend (e.g., "mistral", "llama-cpp").
    fn name(&self) -> &str;

    /// Returns supported model file formats.
    fn supported_formats(&self) -> Vec<&'static str>;

    /// Load a model from the specified configuration.
    fn load(&mut self, config: &LlmConfig) -> LlmResult<()>;

    /// Check if a model is currently loaded.
    fn is_loaded(&self) -> bool;

    /// Unload the current model, freeing resources.
    fn unload(&mut self) -> LlmResult<()>;

    /// Generate text from a list of chat messages.
    fn generate(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> LlmResult<GenerationOutput>;

    /// Generate text from a raw prompt (no chat template).
    fn generate_raw(&self, prompt: &str, config: &GenerationConfig) -> LlmResult<GenerationOutput>;

    /// Generate text with streaming, calling the callback for each token.
    ///
    /// The callback receives a `PartialToken` for each generated token.
    /// If the callback returns an error, generation stops and the error is propagated.
    ///
    /// # Default Implementation
    ///
    /// The default implementation falls back to non-streaming `generate()` and
    /// emits all tokens at once. Override this for true streaming support.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// backend.generate_streaming(&messages, &config, Box::new(|token| {
    ///     print!("{}", token.token);
    ///     std::io::stdout().flush()?;
    ///     Ok(())
    /// }))?;
    /// ```
    fn generate_streaming(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        on_token: StreamingCallback<'_>,
    ) -> LlmResult<GenerationOutput> {
        // Default implementation: fall back to non-streaming and emit all at once
        let output = self.generate(messages, config)?;

        // Emit the entire output as a single "token"
        let partial = PartialToken {
            token: output.text.clone(),
            token_id: None,
            index: 0,
            cumulative_text: output.text.clone(),
            finish_reason: Some(output.finish_reason.clone()),
        };

        let mut callback = on_token;
        callback(partial)
            .map_err(|e| AdapterError::RuntimeError(format!("Streaming callback error: {}", e)))?;

        Ok(output)
    }

    /// Check if this backend supports true streaming.
    ///
    /// Backends that override `generate_streaming` should return `true`.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Get approximate memory usage in bytes.
    fn memory_usage(&self) -> Option<u64> {
        None
    }

    /// Get the maximum context length for the loaded model.
    fn context_length(&self) -> Option<usize> {
        None
    }
}

/// Factory function type for creating LLM backends.
pub type BackendFactory = fn() -> LlmResult<Box<dyn LlmBackend>>;

// =============================================================================
// LLM Runtime Adapter
// =============================================================================

/// LLM Runtime Adapter.
///
/// Provides local LLM inference through the RuntimeAdapter interface.
/// Uses a pluggable backend for actual inference.
pub struct LlmRuntimeAdapter {
    /// The underlying LLM backend
    backend: Box<dyn LlmBackend>,
    /// Model metadata
    metadata: Option<ModelMetadata>,
    /// Current model path
    current_model_path: Option<String>,
    /// Default generation config
    default_generation_config: GenerationConfig,
}

impl LlmRuntimeAdapter {
    /// Create a new LLM Runtime Adapter with the default backend.
    ///
    /// The default backend depends on feature flags:
    /// - `llm-mistral`: Uses MistralBackend (desktop, not Android)
    /// - `llm-llamacpp`: Uses LlamaCppBackend (Android compatible)
    ///
    /// If both are enabled, prefers MistralBackend on non-Android platforms
    /// and LlamaCppBackend on Android.
    #[cfg(feature = "llm-mistral")]
    pub fn new() -> AdapterResult<Self> {
        use crate::runtime_adapter::mistral::MistralBackend;
        let backend = MistralBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter with LlamaCppBackend.
    #[cfg(all(feature = "llm-llamacpp", not(feature = "llm-mistral")))]
    pub fn new() -> AdapterResult<Self> {
        use crate::runtime_adapter::llama_cpp::LlamaCppBackend;
        let backend = LlamaCppBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter with MistralBackend explicitly.
    ///
    /// Use this when you need to force MistralBackend regardless of metadata hints.
    #[cfg(feature = "llm-mistral")]
    pub fn with_mistral() -> AdapterResult<Self> {
        use crate::runtime_adapter::mistral::MistralBackend;
        let backend = MistralBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter with LlamaCppBackend explicitly.
    ///
    /// Use this when you need to force LlamaCppBackend (e.g., for models that
    /// require it like Gemma 3 which isn't supported by mistral.rs GGUF).
    #[cfg(feature = "llm-llamacpp")]
    pub fn with_llamacpp() -> AdapterResult<Self> {
        use crate::runtime_adapter::llama_cpp::LlamaCppBackend;
        let backend = LlamaCppBackend::new()?;
        Ok(Self::with_backend(Box::new(backend)))
    }

    /// Create a new LLM Runtime Adapter based on a backend hint.
    ///
    /// If the hint is "llamacpp" and the feature is available, uses LlamaCppBackend.
    /// Otherwise falls back to the default backend for the platform.
    #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
    pub fn with_backend_hint(hint: Option<&str>) -> AdapterResult<Self> {
        match hint {
            #[cfg(feature = "llm-llamacpp")]
            Some("llamacpp") => Self::with_llamacpp(),
            #[cfg(feature = "llm-mistral")]
            Some("mistral") => Self::with_mistral(),
            _ => Self::new(),
        }
    }

    /// Create a new adapter with a custom backend.
    pub fn with_backend(backend: Box<dyn LlmBackend>) -> Self {
        Self {
            backend,
            metadata: None,
            current_model_path: None,
            default_generation_config: GenerationConfig::default(),
        }
    }

    /// Set the default generation configuration.
    pub fn set_generation_config(&mut self, config: GenerationConfig) {
        self.default_generation_config = config;
    }

    /// Get the current generation configuration.
    pub fn generation_config(&self) -> &GenerationConfig {
        &self.default_generation_config
    }

    /// Get memory usage in bytes (if available).
    pub fn memory_usage(&self) -> Option<u64> {
        self.backend.memory_usage()
    }

    /// Get the context length of the loaded model.
    pub fn context_length(&self) -> Option<usize> {
        self.backend.context_length()
    }

    /// Get a reference to the underlying backend.
    ///
    /// This is useful for accessing backend-specific features like streaming.
    pub fn backend(&self) -> &dyn LlmBackend {
        self.backend.as_ref()
    }

    /// Generate with custom configuration.
    pub fn generate_with_config(
        &self,
        prompt: &str,
        system: Option<&str>,
        config: &GenerationConfig,
    ) -> AdapterResult<GenerationOutput> {
        let mut messages = Vec::new();

        if let Some(sys) = system {
            messages.push(ChatMessage::system(sys));
        }
        messages.push(ChatMessage::user(prompt));

        self.backend.generate(&messages, config)
    }

    /// Extract model ID from path.
    fn extract_model_id(&self, path: &str) -> String {
        Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    }

    /// Parse generation config from envelope metadata.
    fn parse_generation_config(&self, metadata: &HashMap<String, String>) -> GenerationConfig {
        let mut config = self.default_generation_config.clone();

        if let Some(max_tokens) = metadata.get("max_tokens").and_then(|s| s.parse().ok()) {
            config.max_tokens = max_tokens;
        }
        if let Some(temperature) = metadata.get("temperature").and_then(|s| s.parse().ok()) {
            config.temperature = temperature;
        }
        if let Some(top_p) = metadata.get("top_p").and_then(|s| s.parse().ok()) {
            config.top_p = top_p;
        }
        if let Some(min_p) = metadata.get("min_p").and_then(|s| s.parse().ok()) {
            config.min_p = min_p;
        }
        if let Some(top_k) = metadata.get("top_k").and_then(|s| s.parse().ok()) {
            config.top_k = top_k;
        }

        config
    }
}

impl LlmRuntimeAdapter {
    /// Load a model with a full LlmConfig, preserving context_length and other settings.
    pub fn load_model_with_config(&mut self, config: &LlmConfig) -> AdapterResult<()> {
        let path = &config.model_path;
        let model_path = Path::new(path);

        if !model_path.exists() {
            return Err(AdapterError::ModelNotFound(path.to_string()));
        }

        self.backend.load(config)?;

        let model_id = self.extract_model_id(path);

        self.metadata = Some(ModelMetadata {
            model_id: model_id.clone(),
            version: "1.0.0".to_string(),
            runtime_type: self.backend.name().to_string(),
            model_path: path.to_string(),
            input_schema: {
                let mut schema = HashMap::new();
                schema.insert("text".to_string(), vec![1]);
                schema
            },
            output_schema: {
                let mut schema = HashMap::new();
                schema.insert("text".to_string(), vec![1]);
                schema
            },
        });

        self.current_model_path = Some(path.to_string());
        Ok(())
    }
}

impl RuntimeAdapter for LlmRuntimeAdapter {
    fn name(&self) -> &str {
        "llm"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        self.backend.supported_formats()
    }

    fn load_model(&mut self, path: &str) -> AdapterResult<()> {
        self.load_model_with_config(&LlmConfig::new(path))
    }

    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope> {
        if !self.backend.is_loaded() {
            return Err(AdapterError::ModelNotLoaded(
                "No model loaded. Call load_model() first.".to_string(),
            ));
        }

        match &input.kind {
            EnvelopeKind::Text(prompt) => {
                let system = input.metadata.get("system_prompt").map(|s| s.as_str());
                let config = self.parse_generation_config(&input.metadata);

                let mut messages = Vec::new();
                if let Some(sys) = system {
                    messages.push(ChatMessage::system(sys));
                }
                messages.push(ChatMessage::user(prompt));

                let output = self.backend.generate(&messages, &config)?;

                let mut response_metadata = HashMap::new();
                response_metadata.insert(
                    "tokens_generated".to_string(),
                    output.tokens_generated.to_string(),
                );
                response_metadata.insert(
                    "generation_time_ms".to_string(),
                    output.generation_time_ms.to_string(),
                );
                response_metadata.insert(
                    "tokens_per_second".to_string(),
                    format!("{:.2}", output.tokens_per_second),
                );
                response_metadata.insert("finish_reason".to_string(), output.finish_reason.clone());

                // Streaming-path metrics — the canonical keys the platform's
                // span-extraction layer reads. Route to both the envelope
                // metadata (so downstream stages can see them) and
                // `xybrid_core::tracing::add_metadata` (so the span-level
                // wire path picks them up — see plan phase 0.4).
                if let Some(v) = output.ttft_ms {
                    let s = v.to_string();
                    response_metadata.insert("ttft_ms".to_string(), s.clone());
                    crate::tracing::add_metadata("ttft_ms", &s);
                }
                if let Some(v) = output.mean_itl_ms {
                    let s = format!("{:.4}", v);
                    response_metadata.insert("mean_itl_ms".to_string(), s.clone());
                    crate::tracing::add_metadata("mean_itl_ms", &s);
                }
                if let Some(v) = output.p95_itl_ms {
                    let s = v.to_string();
                    response_metadata.insert("p95_itl_ms".to_string(), s.clone());
                    crate::tracing::add_metadata("p95_itl_ms", &s);
                }
                if let Some(v) = output.emitted_chunks {
                    let s = v.to_string();
                    response_metadata.insert("emitted_chunks".to_string(), s.clone());
                    crate::tracing::add_metadata("emitted_chunks", &s);
                }
                if let Some(v) = output.decode_tps {
                    let s = format!("{:.4}", v);
                    response_metadata.insert("decode_tps".to_string(), s.clone());
                    crate::tracing::add_metadata("decode_tps", &s);
                }
                if let Some(v) = output.prefill_tps {
                    let s = format!("{:.4}", v);
                    response_metadata.insert("prefill_tps".to_string(), s.clone());
                    crate::tracing::add_metadata("prefill_tps", &s);
                }
                // Mirror the always-present scalars onto the span metadata
                // too so they reach the platform without relying on the
                // envelope metadata being serialized downstream.
                crate::tracing::add_metadata(
                    "tokens_generated",
                    output.tokens_generated.to_string(),
                );
                // Canonical `tokens_out` key for the analytics backend's
                // span extractor. Equal to `tokens_generated` by
                // construction — the SDK hoist lifts this onto the outer
                // payload so trace dashboards can render the column.
                crate::tracing::add_metadata("tokens_out", output.tokens_generated.to_string());
                crate::tracing::add_metadata(
                    "generation_time_ms",
                    output.generation_time_ms.to_string(),
                );
                crate::tracing::add_metadata(
                    "tokens_per_second",
                    format!("{:.2}", output.tokens_per_second),
                );
                crate::tracing::add_metadata("finish_reason", &output.finish_reason);
                // span_kind tells the console swim-lanes renderer which
                // bar color to use. llama.cpp with Metal compiled in runs
                // the forward pass on the GPU; otherwise it's CPU.
                #[cfg(all(
                    any(feature = "llm-mistral-metal", feature = "llm-llamacpp"),
                    target_os = "macos"
                ))]
                crate::tracing::add_metadata("span_kind", "gpu");
                #[cfg(not(all(
                    any(feature = "llm-mistral-metal", feature = "llm-llamacpp"),
                    target_os = "macos"
                )))]
                crate::tracing::add_metadata("span_kind", "cpu");

                Ok(Envelope {
                    kind: EnvelopeKind::Text(output.text),
                    metadata: response_metadata,
                })
            }
            EnvelopeKind::Audio(_) => Err(AdapterError::InvalidInput(
                "LLM adapter expects Text input, not Audio".to_string(),
            )),
            EnvelopeKind::Embedding(_) => Err(AdapterError::InvalidInput(
                "LLM adapter expects Text input, not Embedding".to_string(),
            )),
        }
    }
}

impl RuntimeAdapterExt for LlmRuntimeAdapter {
    fn is_loaded(&self, _model_id: &str) -> bool {
        self.backend.is_loaded()
    }

    fn get_metadata(&self, _model_id: &str) -> AdapterResult<&ModelMetadata> {
        self.metadata
            .as_ref()
            .ok_or_else(|| AdapterError::ModelNotLoaded("No model loaded".to_string()))
    }

    fn infer(&self, _model_id: &str, input: &Envelope) -> AdapterResult<Envelope> {
        self.execute(input)
    }

    fn unload_model(&mut self, _model_id: &str) -> AdapterResult<()> {
        self.backend.unload()?;
        self.metadata = None;
        self.current_model_path = None;
        Ok(())
    }

    fn list_loaded_models(&self) -> Vec<String> {
        if self.backend.is_loaded() {
            self.metadata
                .as_ref()
                .map(|m| vec![m.model_id.clone()])
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Tests for PartialToken, ChatMessage, GenerationConfig, and LlmConfig
    // are in runtime_adapter/types.rs since those types are now defined there.

    #[test]
    fn test_generation_output_structure() {
        let output = GenerationOutput {
            text: "Hello world".to_string(),
            tokens_generated: 3,
            generation_time_ms: 100,
            tokens_per_second: 30.0,
            finish_reason: "stop".to_string(),
            ttft_ms: None,
            mean_itl_ms: None,
            p95_itl_ms: None,
            emitted_chunks: None,
            inter_chunk_ms: Vec::new(),
            decode_tps: None,
            prefill_tps: None,
        };
        assert_eq!(output.text, "Hello world");
        assert_eq!(output.tokens_generated, 3);
        assert_eq!(output.generation_time_ms, 100);
        assert!((output.tokens_per_second - 30.0).abs() < f32::EPSILON);
        assert_eq!(output.finish_reason, "stop");
    }

    /// Test that the default streaming implementation emits all text as one token
    #[test]
    fn test_default_streaming_implementation() {
        // Create a mock backend that implements LlmBackend with default streaming
        struct MockBackend;

        impl LlmBackend for MockBackend {
            fn name(&self) -> &str {
                "mock"
            }
            fn supported_formats(&self) -> Vec<&'static str> {
                vec!["test"]
            }
            fn load(&mut self, _config: &LlmConfig) -> LlmResult<()> {
                Ok(())
            }
            fn is_loaded(&self) -> bool {
                true
            }
            fn unload(&mut self) -> LlmResult<()> {
                Ok(())
            }
            fn generate(
                &self,
                _messages: &[ChatMessage],
                _config: &GenerationConfig,
            ) -> LlmResult<GenerationOutput> {
                Ok(GenerationOutput {
                    text: "Test response".to_string(),
                    tokens_generated: 2,
                    generation_time_ms: 50,
                    tokens_per_second: 40.0,
                    finish_reason: "stop".to_string(),
                    ttft_ms: None,
                    mean_itl_ms: None,
                    p95_itl_ms: None,
                    emitted_chunks: None,
                    inter_chunk_ms: Vec::new(),
                    decode_tps: None,
                    prefill_tps: None,
                })
            }
            fn generate_raw(
                &self,
                prompt: &str,
                config: &GenerationConfig,
            ) -> LlmResult<GenerationOutput> {
                self.generate(&[ChatMessage::user(prompt)], config)
            }
            // Uses default generate_streaming implementation
        }

        let backend = MockBackend;
        assert!(!backend.supports_streaming()); // Default is false

        let messages = vec![ChatMessage::user("test")];
        let config = GenerationConfig::default();

        let mut received_tokens: Vec<PartialToken> = Vec::new();
        let result = backend.generate_streaming(
            &messages,
            &config,
            Box::new(|token| {
                received_tokens.push(token);
                Ok(())
            }),
        );

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.text, "Test response");

        // Default implementation emits entire text as single "token"
        assert_eq!(received_tokens.len(), 1);
        assert_eq!(received_tokens[0].token, "Test response");
        assert_eq!(received_tokens[0].cumulative_text, "Test response");
        assert_eq!(received_tokens[0].finish_reason, Some("stop".to_string()));
        assert!(received_tokens[0].is_final());
    }

    /// Test that callback errors propagate correctly in default streaming
    #[test]
    fn test_default_streaming_callback_error() {
        struct MockBackend;

        impl LlmBackend for MockBackend {
            fn name(&self) -> &str {
                "mock"
            }
            fn supported_formats(&self) -> Vec<&'static str> {
                vec!["test"]
            }
            fn load(&mut self, _config: &LlmConfig) -> LlmResult<()> {
                Ok(())
            }
            fn is_loaded(&self) -> bool {
                true
            }
            fn unload(&mut self) -> LlmResult<()> {
                Ok(())
            }
            fn generate(
                &self,
                _messages: &[ChatMessage],
                _config: &GenerationConfig,
            ) -> LlmResult<GenerationOutput> {
                Ok(GenerationOutput {
                    text: "Test".to_string(),
                    tokens_generated: 1,
                    generation_time_ms: 10,
                    tokens_per_second: 100.0,
                    finish_reason: "stop".to_string(),
                    ttft_ms: None,
                    mean_itl_ms: None,
                    p95_itl_ms: None,
                    emitted_chunks: None,
                    inter_chunk_ms: Vec::new(),
                    decode_tps: None,
                    prefill_tps: None,
                })
            }
            fn generate_raw(
                &self,
                prompt: &str,
                config: &GenerationConfig,
            ) -> LlmResult<GenerationOutput> {
                self.generate(&[ChatMessage::user(prompt)], config)
            }
        }

        let backend = MockBackend;
        let messages = vec![ChatMessage::user("test")];
        let config = GenerationConfig::default();

        // Callback that returns an error
        let result = backend.generate_streaming(
            &messages,
            &config,
            Box::new(|_token| Err("User cancelled".into())),
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("callback error") || err_msg.contains("User cancelled"));
    }
}
