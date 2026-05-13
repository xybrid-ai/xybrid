//! Execution Template module - Metadata-driven model execution strategies.
//!
//! ## Module Organization
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`metadata`] | `ExecutionTemplate`, `ModelMetadata`, `PipelineStage`, `ExecutionMode` |
//! | [`steps`] | `PreprocessingStep`, `PostprocessingStep`, helper types |
//! | [`voice`] | `VoiceConfig`, `VoiceFormat`, `VoiceInfo`, `VoiceLoader` |

mod metadata;
mod steps;
mod voice;

// Re-export metadata types + swim-lane grouping helpers
pub use metadata::{
    backend_label_from_template, normalize_llm_backend_hint, quantization_label_from_metadata,
    span_kind_from_template, stage_kind_from_task, ExecutionMode, ExecutionTemplate,
    GenerationParams, ModelMetadata, PipelineStage, RefinementSchedule,
};

// Re-export step types
pub use steps::{
    InterpolationMethod, MelScaleType, PhonemizerBackend, PostprocessingStep, PreprocessingStep,
    TokenizerType,
};

// Re-export voice types
pub use voice::{VoiceConfig, VoiceFormat, VoiceInfo, VoiceLoader, VoiceSelectionStrategy};
