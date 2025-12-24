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
//!
//! The main implementation (`executor_impl.rs`) contains:
//! - `TemplateExecutor` struct and core execution methods
//! - Preprocessing step functions (mel spectrogram, tokenize, phonemize, etc.)
//! - Postprocessing step functions (decode, argmax, softmax, etc.)
//! - Pipeline execution (SingleShot, Autoregressive, WhisperDecoder modes)
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
//!     ├── Execution (SimpleMode, CandleModel, Pipeline)
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
//! let mut executor = TemplateExecutor::with_base_path("test_models/whisper");
//!
//! let input = Envelope::new(EnvelopeKind::Audio(audio_bytes));
//! let output = executor.execute(&metadata, &input)?;
//! ```

mod types;

// Re-export main types for public API
pub use types::{ExecutorResult, PreprocessedData, RawOutputs};

// Include the main executor implementation
// Note: The impl is included inline because preprocessing/postprocessing functions
// need access to types and are tightly coupled to the executor methods.
// Future refactoring could move these to proper submodules with appropriate pub(crate) visibility.
include!("executor_impl.rs");
