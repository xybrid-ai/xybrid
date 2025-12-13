//! Pipeline DSL module - Defines pipeline configuration and execution targets.
//!
//! This module implements the Pipeline DSL as designed in `docs/design/PIPELINE_DSL_DESIGN.md`.
//! It supports multiple execution targets (device, server, integration, auto) and
//! various input/output types (audio, text, image, embedding).
//!
//! ## Example YAML
//!
//! ```yaml
//! name: "Voice Assistant"
//! version: "1.0"
//!
//! input:
//!   type: audio
//!   config:
//!     sample_rate: 16000
//!     channels: 1
//!     format: float32
//!
//! output:
//!   type: text
//!
//! stages:
//!   - id: asr
//!     model: wav2vec2-base-960h
//!     version: "1.0"
//!     target: device
//!
//!   - id: llm
//!     model: gpt-4o-mini
//!     target: integration
//!     provider: openai
//! ```

mod condition;
mod config;
mod input;
mod provider;
mod resolver;
mod runner;
mod stage;
mod target;

pub use condition::{ConditionEvaluator, ConditionResult, StageOutputContext};
pub use config::{PipelineConfig, PipelineMetadata};
pub use input::{
    AudioInputConfig, EmbeddingInputConfig, ImageInputConfig, InputConfig, InputType,
    TextInputConfig,
};
pub use input::{OutputConfig, OutputType};
pub use provider::{
    AnthropicOptions, ElevenLabsOptions, GoogleOptions, IntegrationProvider, OpenAIOptions,
    ProviderConfig, ProviderValidation,
};
pub use resolver::{ResolutionContext, ResolutionError, ResolvedTarget, TargetResolver};
pub use runner::{
    OutputResult, OutputResultType, PipelineResult, PipelineRunner, PipelineRunnerError,
    RunnerConfig, StageResult, StageResultSummary,
};
pub use stage::{FallbackConfig, StageConfig, StageOptions};
pub use target::ExecutionTarget;
