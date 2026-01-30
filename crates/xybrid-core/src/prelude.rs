//! Prelude module for convenient imports.
//!
//! This module re-exports the most commonly used types and traits from xybrid-core,
//! allowing users to quickly get started with a single import.
//!
//! # Example
//!
//! ```rust,ignore
//! use xybrid_core::prelude::*;
//!
//! // Now you have access to common types
//! let input = Envelope::from_text("Hello, world!");
//! let mut executor = TemplateExecutor::with_base_path("models/tts");
//! let output = executor.execute(&metadata, &input)?;
//! ```
//!
//! # What's Included
//!
//! ## Core Types
//! - [`Envelope`], [`EnvelopeKind`] - Data containers for pipeline stages
//! - [`ModelMetadata`] - Model configuration from `model_metadata.json`
//! - [`TemplateExecutor`] - Metadata-driven model inference
//!
//! ## Error Types
//! - [`XybridError`], [`XybridResult`] - Unified public API errors (preferred)
//! - [`InferenceError`], [`PipelineError`] - Specific error categories
//! - [`AdapterError`], [`AdapterResult`] - Legacy adapter errors
//!
//! ## Execution
//! - [`ExecutionTemplate`] - Execution mode configuration
//! - [`PreprocessingStep`], [`PostprocessingStep`] - Pipeline step definitions

// ============================================================================
// Core Data Types
// ============================================================================

pub use crate::ir::{Envelope, EnvelopeKind};

// ============================================================================
// Model Metadata & Execution Templates
// ============================================================================

pub use crate::execution::{
    ExecutionTemplate, ModelMetadata, PostprocessingStep, PreprocessingStep, TemplateExecutor,
};

// ============================================================================
// Error Types
// ============================================================================

// Unified error types (preferred for public API)
pub use crate::error::{InferenceError, PipelineError, XybridError, XybridResult};

// Legacy error types (still used internally by adapters)
pub use crate::runtime_adapter::{AdapterError, AdapterResult};

// ============================================================================
// Context & Metrics
// ============================================================================

pub use crate::context::{DeviceMetrics, StageDescriptor};

// ============================================================================
// Device Detection
// ============================================================================

pub use crate::device::HardwareCapabilities;

// ============================================================================
// Cache Provider
// ============================================================================

pub use crate::cache_provider::{CacheProvider, FilesystemCacheProvider, NoopCacheProvider};
