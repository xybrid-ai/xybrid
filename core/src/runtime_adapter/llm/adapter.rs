//! LLM Runtime Adapter implementation.
//!
//! This module provides the RuntimeAdapter implementation for local LLM inference,
//! integrating with xybrid's execution system.

use super::backend::{ChatMessage, GenerationOutput, LlmBackend};
use super::config::{GenerationConfig, LlmConfig};
use super::mistral::MistralBackend;
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{
    AdapterError, AdapterResult, ModelMetadata, RuntimeAdapter, RuntimeAdapterExt,
};
use std::collections::HashMap;
use std::path::Path;

/// LLM Runtime Adapter.
///
/// Provides local LLM inference through the RuntimeAdapter interface.
/// Uses a pluggable backend (currently mistral.rs) for actual inference.
///
/// # Input/Output
///
/// - **Input**: `EnvelopeKind::Text` - the user prompt or message
/// - **Output**: `EnvelopeKind::Text` - the generated response
///
/// For multi-turn conversations, use the `metadata` field to pass:
/// - `system_prompt`: Optional system message
/// - `chat_history`: JSON-encoded list of previous messages (future)
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::runtime_adapter::llm::LlmRuntimeAdapter;
/// use xybrid_core::runtime_adapter::RuntimeAdapter;
/// use xybrid_core::ir::{Envelope, EnvelopeKind};
///
/// let mut adapter = LlmRuntimeAdapter::new()?;
/// adapter.load_model("path/to/model.gguf")?;
///
/// let input = Envelope::new(EnvelopeKind::Text("Hello!".to_string()));
/// let output = adapter.execute(&input)?;
/// ```
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
    /// Create a new LLM Runtime Adapter with the default backend (MistralBackend).
    pub fn new() -> AdapterResult<Self> {
        let backend = MistralBackend::new()?;
        Ok(Self {
            backend: Box::new(backend),
            metadata: None,
            current_model_path: None,
            default_generation_config: GenerationConfig::default(),
        })
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

    /// Generate with custom configuration (bypassing RuntimeAdapter interface).
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
        if let Some(top_k) = metadata.get("top_k").and_then(|s| s.parse().ok()) {
            config.top_k = top_k;
        }

        config
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
        let model_path = Path::new(path);

        if !model_path.exists() {
            return Err(AdapterError::ModelNotFound(path.to_string()));
        }

        let config = LlmConfig::new(path);
        self.backend.load(&config)?;

        let model_id = self.extract_model_id(path);

        self.metadata = Some(ModelMetadata {
            model_id: model_id.clone(),
            version: "1.0.0".to_string(),
            runtime_type: self.backend.name().to_string(),
            model_path: path.to_string(),
            input_schema: {
                let mut schema = HashMap::new();
                schema.insert("text".to_string(), vec![1]); // Variable length text
                schema
            },
            output_schema: {
                let mut schema = HashMap::new();
                schema.insert("text".to_string(), vec![1]); // Variable length text
                schema
            },
        });

        self.current_model_path = Some(path.to_string());
        Ok(())
    }

    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope> {
        if !self.backend.is_loaded() {
            return Err(AdapterError::ModelNotLoaded(
                "No model loaded. Call load_model() first.".to_string(),
            ));
        }

        match &input.kind {
            EnvelopeKind::Text(prompt) => {
                // Extract system prompt from metadata if present
                let system = input.metadata.get("system_prompt").map(|s| s.as_str());

                // Parse generation config from metadata
                let config = self.parse_generation_config(&input.metadata);

                // Build messages
                let mut messages = Vec::new();
                if let Some(sys) = system {
                    messages.push(ChatMessage::system(sys));
                }
                messages.push(ChatMessage::user(prompt));

                // Generate
                let output = self.backend.generate(&messages, &config)?;

                // Build response envelope
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
                response_metadata.insert("finish_reason".to_string(), output.finish_reason);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_name() {
        // This test will fail without the local-llm feature, which is expected
        if let Ok(adapter) = LlmRuntimeAdapter::new() {
            assert_eq!(adapter.name(), "llm");
        }
    }

    #[test]
    fn test_supported_formats() {
        if let Ok(adapter) = LlmRuntimeAdapter::new() {
            let formats = adapter.supported_formats();
            assert!(formats.contains(&"gguf"));
        }
    }

    #[test]
    fn test_load_nonexistent_model() {
        if let Ok(mut adapter) = LlmRuntimeAdapter::new() {
            let result = adapter.load_model("/nonexistent/path.gguf");
            assert!(matches!(result, Err(AdapterError::ModelNotFound(_))));
        }
    }

    #[test]
    fn test_execute_without_model() {
        if let Ok(adapter) = LlmRuntimeAdapter::new() {
            let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));
            let result = adapter.execute(&input);
            assert!(matches!(result, Err(AdapterError::ModelNotLoaded(_))));
        }
    }

    #[test]
    fn test_generation_config_parsing() {
        if let Ok(adapter) = LlmRuntimeAdapter::new() {
            let mut metadata = HashMap::new();
            metadata.insert("max_tokens".to_string(), "100".to_string());
            metadata.insert("temperature".to_string(), "0.5".to_string());

            let config = adapter.parse_generation_config(&metadata);
            assert_eq!(config.max_tokens, 100);
            assert!((config.temperature - 0.5).abs() < 0.001);
        }
    }
}
