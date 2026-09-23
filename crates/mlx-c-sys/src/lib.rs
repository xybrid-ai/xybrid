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
//! To get the actual FFI surface, build with `--features bindings` on
//! Apple Silicon macOS. Populate `vendor/mlx-apple/mlx.xcframework/` first
//! with [`tools/scripts/build-local-mlx-xcframework.sh`] from source pins, or
//! [`tools/scripts/fetch-mlx-xcframework.sh`] when a download pin is available.
//! Alternatively, `$MLX_XCFRAMEWORK_PATH` can point at a prebuilt one.
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

// Apple Silicon macOS-only compile-time guard. Fires only when a downstream
// crate has enabled the `bindings` feature on an unsupported target — for
// instance by activating `llm-mlx-runtime` on Linux or current iOS builds. On
// default workspace builds (no `bindings` feature), this guard is inert, so
// Ubuntu CI stays clean.
#[cfg(all(
    feature = "bindings",
    not(all(target_os = "macos", target_arch = "aarch64"))
))]
compile_error!(
    "mlx-c-sys `bindings` feature is currently supported only on Apple Silicon \
     macOS (`aarch64-apple-darwin`). iOS remains non-linking only until upstream MLX \
     ships a Metal-enabled iOS slice. Disable `llm-mlx-runtime` for this target."
);

/// Generated bindgen output. Lives in a submodule so it can carry the
/// non-standard lint allowances without polluting the crate root, and so
/// downstream code imports via `mlx_c_sys::bindings::mlx_stream` rather than
/// relying on flattened re-exports.
#[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
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
