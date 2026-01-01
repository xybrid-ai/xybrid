//! Xybrid Core - The Rust orchestrator runtime for hybrid cloud-edge AI inference.
//!
//! ## Quick Start
//!
//! Use the [`prelude`] module for common imports:
//!
//! ```rust,ignore
//! use xybrid_core::prelude::*;
//!
//! // Create an executor and run inference
//! let mut executor = TemplateExecutor::with_base_path("models/whisper");
//! let input = Envelope::from_audio(audio_bytes);
//! let output = executor.execute(&metadata, &input)?;
//! ```
//!
//! ## Module Organization
//!
//! The crate is organized into logical groups:
//!
//! ### Core Execution (see `EXECUTION_LAYERS.md`)
//! - [`orchestrator`] - High-level pipeline coordination
//! - [`executor`] - Stage execution with adapter selection
//! - [`template_executor`] - Metadata-driven model inference
//!
//! ### Data Types
//! - [`ir`] - Intermediate representation (Envelope, EnvelopeKind)
//! - [`context`] - Device metrics and stage descriptors
//! - [`execution_template`] - Model metadata schema
//!
//! ### Runtime
//! - [`runtime_adapter`] - ONNX session management
//! - [`pipeline`] - Pipeline configuration and execution
//! - [`streaming`] - Real-time audio streaming
//!
//! ### Bundles
//! - [`bundler`] - .xyb bundle creation and extraction
//!
//! ### High-Level APIs
//! - [`cloud`] - Cloud client (third-party API integrations)
//! - [`tts`] - Text-to-speech client
//! - [`gateway`] - OpenAI-compatible gateway types
//!
//! ## Public vs Internal Modules
//!
//! Modules marked with `#[doc(hidden)]` are internal implementation details
//! and may change without notice. Use the public API modules listed above.

// ============================================================================
// Prelude - Common imports for convenience
// ============================================================================

/// Common imports for xybrid-core users.
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::prelude::*;
/// ```
pub mod prelude;

// ============================================================================
// Core Execution Layer (Orchestrator → Executor → TemplateExecutor)
// See EXECUTION_LAYERS.md for architecture documentation
// ============================================================================

/// High-level pipeline orchestration (policy, routing, streaming)
pub mod orchestrator;

/// Stage execution with adapter selection
pub mod executor;

/// Metadata-driven model inference (preprocessing → ONNX → postprocessing)
pub mod template_executor;

// ============================================================================
// Data Types & Intermediate Representation
// ============================================================================

/// Envelope-based data passing (Audio, Text, Embedding)
pub mod ir;

/// Device metrics and stage descriptors
pub mod context;

/// Model metadata schema (model_metadata.json)
pub mod execution_template;

// ============================================================================
// Runtime & Pipeline Execution
// ============================================================================

/// ONNX runtime adapter and session management
pub mod runtime_adapter;

/// Pipeline configuration, stages, and runners
pub mod pipeline;

/// Real-time audio streaming infrastructure
pub mod streaming;

// ============================================================================
// Audio & Domain APIs
// ============================================================================

/// Audio processing (mel spectrogram, WAV decode)
pub mod audio;

/// .xyb bundle creation and extraction
pub mod bundler;

/// Device detection (CPU, GPU, Metal, NNAPI)
pub mod device;

// ============================================================================
// High-Level Client APIs
// ============================================================================

/// Cloud client (third-party API integrations via gateway or direct)
/// For local/on-device inference, use `target: device` in pipeline YAML.
pub mod cloud;

/// TTS client (text-to-speech)
pub mod tts;

/// OpenAI-compatible gateway types
pub mod gateway;

/// HTTP utilities (retry logic, circuit breakers)
pub mod http;

// ============================================================================
// Internal Modules (implementation details, may change without notice)
// ============================================================================

/// Execution target definitions (local, edge, cloud)
#[doc(hidden)]
pub mod target;

/// Stage name parsing and resolution
#[doc(hidden)]
pub mod stage_resolver;

/// Preprocessing utilities (audio bytes to mel)
#[doc(hidden)]
pub mod preprocessing;

/// Text-to-phoneme conversion for TTS
#[doc(hidden)]
pub mod phonemizer;

/// Device adapter traits
#[doc(hidden)]
pub mod device_adapter;

/// Telemetry collection
#[doc(hidden)]
pub mod telemetry;

/// Distributed tracing
#[doc(hidden)]
pub mod tracing;

/// Event bus for orchestrator events
#[doc(hidden)]
pub mod event_bus;

/// Control plane synchronization
#[doc(hidden)]
pub mod control_sync;

/// Universal architecture system (model configuration)
#[doc(hidden)]
pub mod universal;

/// Testing utilities (mocks, fixtures)
#[doc(hidden)]
pub mod testing;

// ============================================================================
// Crate-Internal Modules
// ============================================================================

/// Cloud LLM provider implementations (internal)
pub(crate) mod cloud_llm;
