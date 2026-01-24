//! Execution module - Metadata-driven model execution engine.
//!
//! This module implements the core execution logic that interprets ModelMetadata
//! and runs inference without hard-coding model-specific logic.
//!
//! ## Module Organization
//!
//! The execution module is organized into focused submodules:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`template`] | Model metadata schema (model_metadata.json types) |
//! | [`executor`] | Main TemplateExecutor struct and execute method |
//! | [`types`] | Data types (PreprocessedData, RawOutputs) |
//! | [`preprocessing`] | Preprocessing step implementations |
//! | [`postprocessing`] | Postprocessing step implementations |
//! | [`modes`] | Execution mode implementations (SingleShot, Autoregressive, Whisper, TTS, BERT) |
//!
//! ## Execution Flow
//!
//! ```text
//! Envelope (input)
//!     │
//!     ├── Preprocessing (mel, tokenize, phonemize, etc.)
//!     │         │
//!     │         ▼
//!     ├── PreprocessedData
//!     │         │
//!     ├── Execution (Onnx, SafeTensors, CoreMl, TfLite, ModelGraph)
//!     │         │
//!     │         ▼
//!     ├── RawOutputs
//!     │         │
//!     ├── Postprocessing (decode, argmax, softmax, etc.)
//!     │         │
//!     │         ▼
//!     └── Envelope (output)
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::execution::{TemplateExecutor, template::ModelMetadata};
//! use xybrid_core::ir::{Envelope, EnvelopeKind};
//!
//! let metadata: ModelMetadata = serde_json::from_str(&config_json)?;
//! let mut executor = TemplateExecutor::with_base_path("/path/to/model-dir");
//!
//! let input = Envelope::new(EnvelopeKind::Audio(audio_bytes));
//! let output = executor.execute(&metadata, &input)?;
//! ```

// Template types (model_metadata.json schema)
pub mod template;

// Re-export commonly used template types at execution:: level
pub use template::{
    ExecutionMode, ExecutionTemplate, ModelMetadata, PipelineStage, PostprocessingStep,
    PreprocessingStep, VoiceConfig, VoiceFormat, VoiceInfo, VoiceLoader,
};

// Data types
mod types;
pub use types::{ExecutorResult, PreprocessedData, RawOutputs};

// Main executor
mod executor;
pub use executor::TemplateExecutor;

// Preprocessing steps
pub mod preprocessing;

// Postprocessing steps
pub mod postprocessing;

// Execution modes (SingleShot, Autoregressive, Whisper, TTS, BERT)
pub mod modes;
