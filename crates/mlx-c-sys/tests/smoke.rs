//! Smoke test — exercises the FFI linkage end-to-end.
//!
//! The PRD (US-004) asks for a test calling `mlx_default_stream()` and
//! asserting a non-null return. That exact symbol does not exist in mlx-c
//! v0.6.0: the closest public C entry points are
//!
//! ```c
//! mlx_stream mlx_default_cpu_stream_new(void);
//! mlx_stream mlx_default_gpu_stream_new(void);
//! int        mlx_get_default_stream(mlx_stream* out, mlx_device dev);
//! ```
//!
//! We call `mlx_default_cpu_stream_new` — it takes no arguments, returns the
//! handle by value, and works on any Apple target (GPU path needs Metal at
//! link time only, not at test time, but CPU exercises less of the runtime
//! and is a cleaner smoke signal). The handle is the opaque
//! `struct { void *ctx; }` used throughout mlx-c, so "non-null" means
//! `stream.ctx != NULL`.
//!
//! Gated on the `bindings` cargo feature so the default workspace test run
//! (no bindings, no xcframework linked) has nothing to execute.

#![cfg(all(feature = "bindings", any(target_os = "macos", target_os = "ios")))]

use mlx_c_sys::bindings;

#[test]
fn default_cpu_stream_is_non_null() {
    // SAFETY: `mlx_default_cpu_stream_new` has no preconditions — it can be
    // called from any thread at any time and returns a fresh handle. We free
    // it immediately after the assertion so the test does not leak.
    unsafe {
        let stream = bindings::mlx_default_cpu_stream_new();
        assert!(
            !stream.ctx.is_null(),
            "mlx_default_cpu_stream_new returned a null handle — the FFI link is \
             present but the MLX runtime failed to initialise"
        );
        bindings::mlx_stream_free(stream);
    }
}
