//! Raw FFI bindings to [mlx-c](https://github.com/ml-explore/mlx-c), the C
//! wrapper over [MLX](https://github.com/ml-explore/mlx).
//!
//! # Activation
//!
//! The real bindings are hidden behind the `bindings` cargo feature. A default
//! build (`cargo build -p mlx-c-sys`) compiles this crate to an empty shell on
//! every target — this keeps the workspace-wide `cargo clippy` / `cargo check`
//! on Linux CI runners green, because MLX links against Metal and Accelerate
//! and cannot be built or linked on non-Apple hosts.
//!
//! To get the actual FFI surface, build with `--features bindings` on an Apple
//! target. The [`tools/scripts/fetch-mlx-xcframework.sh`] script must have
//! populated `vendor/mlx-apple/mlx.xcframework/` first (or `$MLX_XCFRAMEWORK_PATH`
//! must point at a prebuilt one).
//!
//! # Safety and stability
//!
//! This is a `-sys` crate: every item is `unsafe extern "C"` and mirrors the
//! upstream C ABI one-to-one. It is not meant to be consumed directly outside
//! the workspace. The safe wrapper crate `xybrid-mlx` (US-005 onward) owns the
//! RAII, typed-shape, and error-mapping concerns.
//!
//! The generated bindings are intentionally *not* re-exported transitively:
//! downstream crates opt in explicitly by depending on `mlx-c-sys` with the
//! `bindings` feature and using the items behind `bindings::*`.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

// Apple-only compile-time guard. Fires only when a downstream crate has
// enabled the `bindings` feature on a non-Apple target — for instance by
// activating `llm-mlx` on a `linux-x86_64` build. On default workspace
// builds (no `bindings` feature), this guard is inert, so Ubuntu CI stays
// clean.
#[cfg(all(feature = "bindings", not(any(target_os = "macos", target_os = "ios"))))]
compile_error!(
    "mlx-c-sys `bindings` feature requires an Apple target (macOS or iOS). \
     MLX links against Metal and Accelerate, which are Apple-only. Disable \
     the `llm-mlx` feature (or its equivalent) when building for this target."
);

/// Generated bindgen output. Lives in a submodule so it can carry the
/// non-standard lint allowances without polluting the crate root, and so
/// downstream code imports via `mlx_c_sys::bindings::mlx_stream` rather than
/// relying on flattened re-exports.
#[cfg(all(feature = "bindings", any(target_os = "macos", target_os = "ios")))]
pub mod bindings {
    #![allow(
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        dead_code,
        improper_ctypes,
        clippy::all,
        clippy::pedantic,
        clippy::nursery,
        clippy::cargo
    )]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
