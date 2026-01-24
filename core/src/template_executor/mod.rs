//! Template Executor - Metadata-driven model execution engine.
//!
//! This module implements the core execution logic that interprets ModelMetadata
//! and runs inference without hard-coding model-specific logic.
//!
//! ## Module Organization
//!
//! The template executor is organized into focused submodules:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`types`] | Data types (PreprocessedData, RawOutputs) |
//! | [`executor`] | Main TemplateExecutor struct and execute method |
//! | [`preprocessing`] | Preprocessing step implementations |
//! | [`postprocessing`] | Postprocessing step implementations |
//! | [`execution`] | Execution mode implementations (SingleShot, Autoregressive, Whisper) |
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
//! use xybrid_core::template_executor::TemplateExecutor;
//! use xybrid_core::execution_template::ModelMetadata;
//! use xybrid_core::ir::{Envelope, EnvelopeKind};
//!
//! let metadata: ModelMetadata = serde_json::from_str(&config_json)?;
//! let mut executor = TemplateExecutor::with_base_path("/path/to/model-dir");
//!
//! let input = Envelope::new(EnvelopeKind::Audio(audio_bytes));
//! let output = executor.execute(&metadata, &input)?;
//! ```

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

// Execution modes
pub mod execution;
