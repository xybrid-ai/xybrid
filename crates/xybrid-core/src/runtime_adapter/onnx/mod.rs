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
mod profiling;
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
pub use profiling::ResolvedExecutionProviders;
pub use runtime::OnnxRuntime;
pub use session::{ONNXSession, SessionOptions};

#[cfg(any(target_os = "android", test))]
pub use mobile::ONNXMobileRuntimeAdapter;

/// Test-only probe: is the ONNX Runtime actually loadable in this process?
///
/// Under `load-dynamic` (linux/windows/android — see xybrid-core's per-target
/// ort sections), a missing `libonnxruntime` makes ort PANIC on first touch
/// (`ort::api()`'s internal `.expect`), not return an error — so tests that
/// reach real ort initialization must skip instead. The probe dlopens the
/// same dylib ort would (ORT_DYLIB_PATH override, else the platform default)
/// via libloading, WITHOUT touching ort's global init — no panic to catch,
/// no panic-hook swap that could suppress unrelated parallel-test panics,
/// and no poisoned ort state. This is the "environments without the binary"
/// skip convention, made explicit.
#[cfg(test)]
pub(crate) fn ort_runtime_available() -> bool {
    // macOS/iOS link ort statically (download-binaries) — always available.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        true
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            // Mirror ort's dylib resolution (lib.rs): non-empty ORT_DYLIB_PATH
            // wins, else the platform default name on the loader search path.
            let path = match std::env::var("ORT_DYLIB_PATH") {
                Ok(s) if !s.is_empty() => s,
                _ => {
                    if cfg!(target_os = "windows") {
                        "onnxruntime.dll".to_string()
                    } else {
                        "libonnxruntime.so".to_string()
                    }
                }
            };
            // SAFETY: test-only probe; the library is opened and immediately
            // dropped without calling any symbol from it.
            unsafe { libloading::Library::new(&path) }.is_ok()
        })
    }
}
