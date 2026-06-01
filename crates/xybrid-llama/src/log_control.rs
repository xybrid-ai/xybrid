//! llama.cpp / ggml log verbosity control.
//!
//! Mirrors the pre-refactor `llama_log_{set,get}_verbosity` helpers
//! that lived inside `xybrid-core::runtime_adapter::llama_cpp` before
//! Phase 2 of the crate-split epic.

use crate::ffi;

/// Set the verbosity level for llama.cpp / ggml logging.
///
/// # Levels
/// - `0`: Silent (suppress all library logs) — default
/// - `1`: Errors only
/// - `2`: Errors + Warnings
/// - `3`: Errors + Warnings + Info
/// - `4`: All logs including Debug
pub fn set_verbosity(level: i32) {
    // SAFETY: `llama_log_set_verbosity_c` is callable from any thread at
    // any time. The C side guards its global state with an atomic.
    unsafe { ffi::log_set_verbosity(level) };
}

/// Read the current verbosity level.
pub fn get_verbosity() -> i32 {
    // SAFETY: same as above; pure read of an atomic.
    unsafe { ffi::log_get_verbosity() }
}
