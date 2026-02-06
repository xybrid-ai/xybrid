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
// Feature Combination Guards
// These compile_error! blocks prevent invalid feature combinations at build time.
// See docs/FEATURE_MATRIX.md for valid combinations.
// ============================================================================

// llm-mistral uses x86 AVX2/FP16 intrinsics that cause SIGILL on Android ARM devices
// without ARMv8.2-A FP16 extensions (which most devices lack).
// Use llm-llamacpp instead - it has runtime SIMD detection.
#[cfg(all(feature = "llm-mistral", target_os = "android"))]
compile_error!(
    "Invalid feature combination: `llm-mistral` is not supported on Android.\n\n\
    Reason: mistral.rs uses x86 AVX2/FP16 intrinsics that cause SIGILL on ARM devices \
    without ARMv8.2-A FP16 extensions (most Android devices lack this).\n\n\
    Solution: Use `llm-llamacpp` instead. It performs runtime SIMD detection and works \
    on all Android devices.\n\n\
    Change: --features llm-mistral -> --features llm-llamacpp"
);

// ort-download and ort-dynamic are mutually exclusive ORT loading strategies.
// ort-download: Downloads prebuilt ONNX Runtime binaries at build time (desktop/iOS).
// ort-dynamic: Loads .so/.dylib at runtime from platform-specific location (Android AAR).
#[cfg(all(feature = "ort-download", feature = "ort-dynamic"))]
compile_error!(
    "Invalid feature combination: `ort-download` and `ort-dynamic` are mutually exclusive.\n\n\
    Reason: These are two different strategies for loading ONNX Runtime:\n\
    - `ort-download`: Downloads prebuilt binaries at build time (for desktop, macOS, iOS)\n\
    - `ort-dynamic`: Loads .so at runtime from AAR/framework (for Android)\n\n\
    Solution: Enable only one based on your target platform:\n\
    - Desktop/macOS/iOS: Use `ort-download` (default)\n\
    - Android: Use `ort-dynamic`\n\n\
    Tip: Use platform presets instead: `platform-macos`, `platform-ios`, `platform-android`"
);

// candle-metal requires Apple's Metal GPU framework (macOS/iOS only).
#[cfg(all(
    feature = "candle-metal",
    not(any(target_os = "macos", target_os = "ios"))
))]
compile_error!(
    "Invalid feature combination: `candle-metal` requires macOS or iOS.\n\n\
    Reason: Metal is Apple's GPU framework and is only available on Apple platforms.\n\n\
    Solution:\n\
    - For macOS/iOS: `candle-metal` is valid\n\
    - For Linux with NVIDIA GPU: Use `candle-cuda` instead\n\
    - For CPU-only: Use `candle` (no GPU acceleration)"
);

// candle-cuda requires NVIDIA CUDA which is not available on Apple platforms.
#[cfg(all(
    feature = "candle-cuda",
    any(target_os = "macos", target_os = "ios")
))]
compile_error!(
    "Invalid feature combination: `candle-cuda` is not supported on macOS or iOS.\n\n\
    Reason: CUDA is NVIDIA's GPU framework and is not available on Apple platforms \
    (Apple uses Metal instead).\n\n\
    Solution:\n\
    - For macOS/iOS with GPU: Use `candle-metal` instead\n\
    - For CPU-only on Apple: Use `candle` (no GPU acceleration)"
);

// ort-coreml enables Apple's CoreML/Neural Engine acceleration (macOS/iOS only).
#[cfg(all(
    feature = "ort-coreml",
    not(any(target_os = "macos", target_os = "ios"))
))]
compile_error!(
    "Invalid feature combination: `ort-coreml` requires macOS or iOS.\n\n\
    Reason: CoreML is Apple's ML framework for Neural Engine acceleration and is only \
    available on Apple platforms.\n\n\
    Solution:\n\
    - For macOS/iOS: `ort-coreml` is valid (enables ANE acceleration)\n\
    - For Android: Use `ort-dynamic` with NNAPI\n\
    - For other platforms: Use `ort-download` (CPU only)"
);

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

/// Unified error types for xybrid-core public API.
///
/// This module provides the canonical error hierarchy:
/// - [`XybridError`](error::XybridError) - Top-level error type
/// - [`InferenceError`](error::InferenceError) - Model inference failures
/// - [`PipelineError`](error::PipelineError) - Pipeline execution failures
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::error::{XybridError, XybridResult};
///
/// fn run_model() -> XybridResult<String> {
///     // ...
/// }
/// ```
pub mod error;
pub use error::{InferenceError, PipelineError, XybridError, XybridResult};

// ============================================================================
// Core Execution Layer (Orchestrator → Executor → TemplateExecutor)
// See EXECUTION_LAYERS.md for architecture documentation
// ============================================================================

/// High-level pipeline orchestration (policy, routing, streaming)
pub mod orchestrator;

/// Stage execution with adapter selection
pub mod executor;

/// Unified execution module (template schema + executor)
///
/// This module contains:
/// - [`execution::template`] - Model metadata schema (model_metadata.json)
/// - [`execution::TemplateExecutor`] - Metadata-driven inference engine
/// - [`execution::modes`] - Execution modes (SingleShot, Autoregressive, Whisper, TTS)
pub mod execution;

// Backwards compatibility: re-export old module paths
// TODO: Deprecate these in a future version
/// Deprecated: use `execution` module instead
#[doc(hidden)]
pub mod template_executor {
    pub use crate::execution::*;
}

/// Cache provider abstraction for model availability checks.
/// Allows Core to check cache without depending on SDK.
pub mod cache_provider;

// ============================================================================
// Data Types & Intermediate Representation
// ============================================================================

/// Envelope-based data passing (Audio, Text, Embedding)
pub mod ir;

/// Conversation context for multi-turn LLM interactions
pub mod conversation;

/// Device metrics and stage descriptors
pub mod context;

// Backwards compatibility: re-export execution::template as execution_template
/// Deprecated: use `execution::template` instead
#[doc(hidden)]
pub mod execution_template {
    pub use crate::execution::template::*;
}

// ============================================================================
// Runtime & Pipeline Execution
// ============================================================================

/// ONNX runtime adapter and session management
pub mod runtime_adapter;

/// Pipeline configuration, stages, and runners
pub mod pipeline;

/// Pipeline DSL configuration (unified schema for CLI and SDK)
pub mod pipeline_config;

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
