//! Xybrid Core - The Rust orchestrator runtime for hybrid cloud-edge AI inference.
//!
//! ## Module Organization
//!
//! The crate is organized into logical groups:
//!
//! ### Core Execution (see [`EXECUTION_LAYERS.md`])
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
//! - [`llm`] - LLM client (local + cloud)
//! - [`tts`] - Text-to-speech client
//! - [`gateway`] - OpenAI-compatible gateway types

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

/// Stream buffering and chunk management
pub mod stream_manager;

/// Execution target definitions (local, edge, cloud)
pub mod target;

/// Stage name parsing and resolution
pub mod stage_resolver;

// ============================================================================
// Audio & Preprocessing
// ============================================================================

/// Audio processing (mel spectrogram, WAV decode)
pub mod audio;

/// Preprocessing utilities (audio bytes to mel)
pub mod preprocessing;

/// Text-to-phoneme conversion for TTS
pub mod phonemizer;

// ============================================================================
// Model Bundles
// ============================================================================

/// .xyb bundle creation and extraction
pub mod bundler;

// ============================================================================
// Policy & Routing
// ============================================================================

/// Policy engine for allow/deny decisions
pub mod policy_engine;

/// Routing engine for local/edge/cloud decisions
pub mod routing_engine;

// ============================================================================
// Device & Hardware
// ============================================================================

/// Device detection (CPU, GPU, Metal, NNAPI)
pub mod device;

/// Device adapter traits
pub mod device_adapter;

// ============================================================================
// Observability & Control
// ============================================================================

/// Telemetry collection
pub mod telemetry;

/// Distributed tracing
pub mod tracing;

/// Event bus for orchestrator events
pub mod event_bus;

/// Control plane synchronization
pub mod control_sync;

// ============================================================================
// High-Level Client APIs
// ============================================================================

/// LLM client (abstracts local vs cloud execution)
pub mod llm;

/// TTS client (text-to-speech)
pub mod tts;

/// OpenAI-compatible gateway types
pub mod gateway;

/// Universal architecture system (model configuration)
pub mod universal;

// ============================================================================
// Internal Modules
// ============================================================================

/// Cloud LLM provider implementations (internal)
pub(crate) mod cloud_llm;
