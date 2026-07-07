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
/// reach real ort initialization must skip instead. The probe runs once
/// (memoized); on runners without the binary the caught panic only poisons
/// ort's internal state, which is unusable there anyway. This is the
/// "environments without the binary" skip convention, made explicit.
#[cfg(test)]
pub(crate) fn ort_runtime_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        // Silence the probe panic's default backtrace spam in test output.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let ok = std::panic::catch_unwind(|| {
            let _ = ort::api();
        })
        .is_ok();
        std::panic::set_hook(prev_hook);
        ok
    })
}
