//! ONNX Runtime backend module.
//!
//! This module provides ONNX Runtime inference for xybrid models.
//! It includes:
//! - `OnnxRuntimeAdapter`: High-level adapter implementing `RuntimeAdapter` trait
//! - `OnnxBackend`: Low-level backend implementing `InferenceBackend` trait
//! - `ONNXSession`: ONNX Runtime session wrapper
//! - `ONNXMobileRuntimeAdapter`: Mobile-optimized adapter with NNAPI support
//! - `ExecutionProviderKind`: Execution provider selection (CPU, CoreML, etc.)

mod adapter;
mod backend;
mod execution_provider;
mod runtime;
mod session; // New runtime wrapper

#[cfg(any(target_os = "android", test))]
mod mobile;

// Re-exports
pub use adapter::OnnxRuntimeAdapter;
pub use backend::OnnxBackend;
pub use execution_provider::{
    parse_provider_string, select_optimal_provider, ExecutionProviderKind, ModelHints,
};
#[cfg(feature = "ort-coreml")]
pub use execution_provider::{CoreMLComputeUnits, CoreMLConfig};
pub use runtime::OnnxRuntime;
pub use session::ONNXSession;

#[cfg(any(target_os = "android", test))]
pub use mobile::ONNXMobileRuntimeAdapter;
