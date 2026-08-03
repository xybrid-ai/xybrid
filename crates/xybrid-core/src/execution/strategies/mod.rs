//! Execution strategy trait and the live codec-TTS strategy.
//!
//! Defines the [`ExecutionStrategy`] trait and [`ExecutionContext`], plus the
//! one strategy that is actually wired into the executor: [`CodecTtsStrategy`]
//! (GGUF backbone + ONNX codec decoder). The `llm` submodule provides the LLM
//! inference infrastructure ([`LlmInference`], generation/model config) that
//! [`CodecTtsStrategy`] builds on.
//!
//! `CodecTtsStrategy` implements [`ExecutionStrategy`]; the executor dispatches
//! to it from `TemplateExecutor::execute_impl` (see `executor.rs`).

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp", feature = "llm-mlx"))]
mod llm;

#[cfg(feature = "llm-mlx")]
mod mlx_embedding;

// Codec TTS depends on LLM infrastructure (same feature gate)
mod codec_tts;

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
#[allow(unused_imports)]
pub use llm::{LlmGenerationParams, LlmInference, LlmModelConfig};

#[cfg(feature = "llm-mlx")]
pub use mlx_embedding::MlxEmbeddingStrategy;

#[allow(unused_imports)]
pub use codec_tts::CodecTtsStrategy;

// Always compile the llm module (stubs when features disabled)
#[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp", feature = "llm-mlx")))]
mod llm;
#[cfg(not(any(feature = "llm-mistral", feature = "llm-llamacpp", feature = "llm-mlx")))]
#[allow(unused_imports)]
pub use llm::{LlmGenerationParams, LlmInference, LlmModelConfig};

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
/// Strategies encapsulate the logic for executing a specific model pattern.
/// Implemented today by [`CodecTtsStrategy`] (GGUF backbone + ONNX codec decoder).
pub trait ExecutionStrategy: Send + Sync {
    /// Check if this strategy can handle the given metadata.
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

// The per-model-type Strategy pattern (resolver + Standard/Tts/Llm strategies) was
// removed as dead code: only CodecTtsStrategy was ever wired; the others diverged
// from the live inline execution paths in TemplateExecutor::execute_impl (dispatch
// is by ExecutionTemplate there). If pluggable/swappable execution is ever needed,
// see .context/strategy-reconciliation-scoping.md for the forward-port blueprint
// (evolve ExecutionStrategy for streaming + adapter-cache + multimodal; drive
// LlmRuntimeAdapter, not the cache-less LlmInference).
