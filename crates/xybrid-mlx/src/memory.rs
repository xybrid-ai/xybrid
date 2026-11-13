//! MLX allocator and cache controls.

use crate::ffi;
use crate::resources::ensure_metallib_resource;
use crate::MlxResult;

/// Clear MLX's process-local allocation cache.
///
/// This releases cached temporary allocations back to the system without
/// invalidating live arrays. Call it at generation/session boundaries, after
/// arrays from the run have been dropped.
pub fn clear_cache() -> MlxResult<()> {
    ensure_metallib_resource()?;
    // SAFETY: MLX exposes this as a process-local allocator cache cleanup.
    // The safe contract above requires callers to use it at session
    // boundaries instead of while depending on temporary lazy graphs.
    unsafe { ffi::clear_cache() }
}

/// Set MLX's process-local allocation cache limit in bytes.
///
/// Returns the previous limit reported by MLX. This is a process-wide policy
/// knob, so callers should set it from runtime/session setup code rather than
/// per tensor operation.
pub fn set_cache_limit(limit_bytes: usize) -> MlxResult<usize> {
    ensure_metallib_resource()?;
    // SAFETY: the function only mutates MLX allocator policy and returns the
    // previous value. The safe wrapper keeps it explicit and fallible.
    unsafe { ffi::set_cache_limit(limit_bytes) }
}

/// Bytes MLX holds for live arrays: model weights, caches, anything not yet
/// dropped.
pub fn active_memory() -> MlxResult<usize> {
    ensure_metallib_resource()?;
    // SAFETY: reads an allocator counter; no preconditions.
    unsafe { ffi::active_memory() }
}

/// Bytes MLX keeps pooled for reuse after their arrays were dropped. They
/// stay allocated to the process until [`clear_cache`] runs.
pub fn cache_memory() -> MlxResult<usize> {
    ensure_metallib_resource()?;
    // SAFETY: reads an allocator counter; no preconditions.
    unsafe { ffi::cache_memory() }
}

/// Version string of the MLX library this process is linked against.
///
/// Reports the *runtime* version (e.g. `"0.31.1"`), which is the
/// authoritative identity for numerics provenance — golden-token captures
/// record it and verification asserts it matches before comparing streams.
pub fn version() -> MlxResult<String> {
    // SAFETY: queries an immutable process-global string; no preconditions.
    unsafe { ffi::version() }
}
