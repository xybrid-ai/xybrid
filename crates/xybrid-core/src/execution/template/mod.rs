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

// Re-export metadata types
pub use metadata::{
    ExecutionMode, ExecutionTemplate, ModelMetadata, PipelineStage, RefinementSchedule,
};

// Re-export step types
pub use steps::{
    InterpolationMethod, MelScaleType, PhonemizerBackend, PostprocessingStep, PreprocessingStep,
    TokenizerType,
};

// Re-export voice types
pub use voice::{VoiceConfig, VoiceFormat, VoiceInfo, VoiceLoader};
