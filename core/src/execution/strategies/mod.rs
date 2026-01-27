//! Execution strategies - Modular execution paths for different model types.
//!
//! This module extracts execution logic into independent, testable strategies:
//!
//! | Strategy | Use Case |
//! |----------|----------|
//! | [`StandardStrategy`] | Single-model ONNX/Candle execution |
//! | [`TtsStrategy`] | Text-to-speech with chunking |
//! | [`BertStrategy`] | BERT-style token-based inference |
//! | [`ModelGraphStrategy`] | Multi-stage DAG execution |
//! | [`LlmStrategy`] | GGUF LLM execution (feature-gated) |
//!
//! ## Design
//!
//! Each strategy implements the `ExecutionStrategy` trait, which provides:
//! - `can_handle()` - Check if this strategy handles the given metadata
//! - `execute()` - Run the execution
//!
//! The executor uses a strategy resolver to select the appropriate strategy
//! based on the model metadata.

mod standard;
mod tts;

pub use standard::StandardStrategy;
pub use tts::TtsStrategy;

use super::template::ModelMetadata;
use super::types::ExecutorResult;
use crate::ir::Envelope;
use crate::runtime_adapter::ModelRuntime;
use std::collections::HashMap;
use std::path::Path;

/// Context provided to strategies during execution.
///
/// This struct bundles the dependencies that strategies need, avoiding
/// tight coupling to the executor's internal state.
pub struct ExecutionContext<'a> {
    /// Base path for resolving model files
    pub base_path: &'a str,
    /// Available runtimes (e.g., "onnx", "candle")
    pub runtimes: &'a mut HashMap<String, Box<dyn ModelRuntime>>,
}

impl<'a> ExecutionContext<'a> {
    /// Resolve a model file path relative to base_path.
    pub fn resolve_path(&self, file: &str) -> std::path::PathBuf {
        Path::new(self.base_path).join(file)
    }

    /// Get a mutable reference to a runtime by name.
    pub fn get_runtime(&mut self, name: &str) -> Option<&mut Box<dyn ModelRuntime>> {
        self.runtimes.get_mut(name)
    }
}

/// Trait for execution strategies.
///
/// Strategies encapsulate the logic for executing different types of models.
/// Each strategy handles a specific execution pattern (TTS, BERT, standard, etc.).
pub trait ExecutionStrategy: Send + Sync {
    /// Check if this strategy can handle the given metadata.
    ///
    /// The executor uses this to select the appropriate strategy.
    fn can_handle(&self, metadata: &ModelMetadata) -> bool;

    /// Execute the model with the given context and input.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Execution context with runtimes and paths
    /// * `metadata` - Model metadata
    /// * `input` - Input envelope
    ///
    /// # Returns
    ///
    /// Output envelope on success
    fn execute(
        &self,
        ctx: &mut ExecutionContext<'_>,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<Envelope>;

    /// Get the name of this strategy for logging/debugging.
    fn name(&self) -> &'static str;
}

/// Strategy resolver that selects the appropriate strategy for a model.
pub struct StrategyResolver {
    strategies: Vec<Box<dyn ExecutionStrategy>>,
}

impl StrategyResolver {
    /// Create a new resolver with the default strategies.
    pub fn new() -> Self {
        let mut strategies: Vec<Box<dyn ExecutionStrategy>> = vec![
            // TTS must be checked before Standard (both handle ONNX)
            Box::new(TtsStrategy::new()),
            // Standard handles generic ONNX/Candle models
            Box::new(StandardStrategy::new()),
        ];

        // Add feature-gated strategies
        #[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
        {
            // LLM strategy would go here when implemented
        }

        Self { strategies }
    }

    /// Find the strategy that can handle the given metadata.
    pub fn resolve(&self, metadata: &ModelMetadata) -> Option<&dyn ExecutionStrategy> {
        self.strategies
            .iter()
            .find(|s| s.can_handle(metadata))
            .map(|s| s.as_ref())
    }
}

impl Default for StrategyResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::template::PreprocessingStep;

    #[test]
    fn test_resolver_selects_tts_for_phonemize() {
        let resolver = StrategyResolver::new();
        let metadata = ModelMetadata::onnx("test-tts", "1.0", "model.onnx")
            .with_preprocessing(PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                backend: Default::default(),
                dict_file: None,
                language: None,
                add_padding: true,
                normalize_text: false,
            });

        let strategy = resolver.resolve(&metadata);
        assert!(strategy.is_some());
        assert_eq!(strategy.unwrap().name(), "tts");
    }

    #[test]
    fn test_resolver_selects_standard_for_onnx() {
        let resolver = StrategyResolver::new();
        let metadata = ModelMetadata::onnx("test-model", "1.0", "model.onnx");

        let strategy = resolver.resolve(&metadata);
        assert!(strategy.is_some());
        assert_eq!(strategy.unwrap().name(), "standard");
    }
}
