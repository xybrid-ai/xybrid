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
mod session;
mod runtime; // New runtime wrapper

#[cfg(any(target_os = "android", test))]
mod mobile;

// Re-exports
pub use adapter::OnnxRuntimeAdapter;
pub use backend::OnnxBackend;
pub use execution_provider::ExecutionProviderKind;
#[cfg(feature = "coreml-ep")]
pub use execution_provider::{CoreMLComputeUnits, CoreMLConfig};
pub use session::ONNXSession;
pub use runtime::OnnxRuntime;

#[cfg(any(target_os = "android", test))]
pub use mobile::ONNXMobileRuntimeAdapter;
