//! Safe Rust wrappers over [`mlx-c`](https://github.com/ml-explore/mlx-c).
//!
//! This crate is the safe foundation the MLX LLM / embedding adapters in
//! `xybrid-core` build on. It owns the FFI boundary: RAII handles, typed
//! shapes, error mapping. Downstream code never touches `unsafe` or raw
//! `mlx_array` / `mlx_stream` values.
//!
//! # Activation
//!
//! The real implementation lives behind the `bindings` cargo feature. A
//! default build — `cargo build -p xybrid-mlx` — compiles this crate to an
//! empty shell on every target, which keeps `cargo clippy --workspace` on
//! Linux CI runners green. Downstream consumers enable `bindings` only on
//! Apple Silicon macOS targets (the only validated MLX runtime path).
//!
//! ```bash
//! # Real linkage — requires vendor/mlx-apple/mlx.xcframework to be present.
//! cargo check -p xybrid-mlx --features bindings
//! ```
//!
//! # Design departures from `oxiglade/mlx-rs`
//!
//! A few deliberate differences vs. the community `mlx-rs` crate. These are
//! motivated by xybrid's use case (single-process inference inside a larger
//! runtime) rather than general-purpose MLX scripting:
//!
//! 1. **Typed shapes.** Shapes are `Vec<i32>` / `&[i32]`, not opaque shape
//!    objects. This matches the underlying `mlx_array_shape` C API one-to-one
//!    and avoids a redundant wrapper layer. Downstream crates that need
//!    compile-time rank checking can build on top; we do not force it here.
//! 2. **Explicit `Clone` via refcount.** The mlx-c handles have shared
//!    ownership semantics (the underlying C++ `mlx::array` is reference
//!    counted through its `array_desc_`). We expose this as an explicit
//!    [`Clone`] implementation on `MlxArray` and `MlxStream` that calls
//!    `mlx_*_set`, which is the idiomatic retain in mlx-c. We do **not**
//!    derive [`Copy`] — that would silently double-free on [`Drop`].
//! 3. **No global state.** Default streams are cached per thread via
//!    [`thread_local!`] rather than in process-wide statics. This means
//!    spinning up a fresh thread on iOS (where each app is its own process
//!    and memory pressure is real) does not inherit a stale Metal stream
//!    from another thread. Consumers that actually want a global can call
//!    `MlxStream::default_cpu` / `MlxStream::default_gpu` themselves.
//! 4. **Zero `unsafe` in the public surface.** Every unsafe FFI call lives
//!    in `pub(crate)` helpers in `ffi.rs` with `# Safety` doc comments.
//!    Downstream crates (including `xybrid-core`'s MLX adapter) only use
//!    the safe types.
//!
//! # Why only Apple
//!
//! MLX links against Metal, MetalPerformanceShaders, and Accelerate — all
//! Apple-only frameworks. The `bindings` feature therefore currently requires
//! Apple Silicon macOS; enabling it elsewhere fires a [`compile_error!`] below.
//!
//! # Public surface (bindings feature enabled)
//!
//! - `MlxArray` — RAII handle for `mlx_array`. Typed via [`MlxDtype`].
//! - `MlxStream` + `Device` — stream handle and its device placement.
//! - `SharedBuffer` — MTLBuffer-backed zero-copy heap region used by the
//!   Envelope ↔ MlxArray conversion path (US-008).
//! - [`EnvelopePayload`] / [`EnvelopeSource`] /
//!   [`OwnedEnvelopePayload`] — unconditional trait + payload enums that
//!   `xybrid-core::ir::Envelope` implements downstream, keeping this
//!   crate free of the xybrid-core dep.
//! - [`MlxDtype`] — the supported subset of dtypes.
//! - [`MlxError`] / [`MlxResult`] — error surface.

// Apple Silicon macOS-only compile-time guard for the `bindings` feature. Only
// fires when a downstream crate has opted into the feature on an unsupported
// target. The default (no-feature) path is inert.
#[cfg(all(
    feature = "bindings",
    not(all(target_os = "macos", target_arch = "aarch64"))
))]
compile_error!(
    "xybrid-mlx `bindings` feature is currently supported only on Apple Silicon \
     macOS (`aarch64-apple-darwin`). iOS remains non-linking only until upstream MLX \
     ships a Metal-enabled iOS slice. Disable `llm-mlx-runtime` for this target."
);

// Types that do not touch the FFI are unconditionally compiled — they are
// useful stubs for downstream code that wants to construct error values or
// reference dtype names even in a no-bindings build. The mlx-c conversion
// impls are gated behind `bindings`. [`envelope`] is also unconditional:
// the trait and payload types are pure Rust with no MLX touch-points, so
// downstream crates (notably `xybrid-core`) can `impl EnvelopeSource`
// even in a no-bindings build. Attempting to *use* `MlxArray::from_envelope`
// on a no-bindings target is a compile error because `MlxArray` itself
// is cfg-gated below.
mod dtype;
mod envelope;
mod error;

pub use dtype::{bf16_le_bytes_to_f32_vec, f16_le_bytes_to_f32_vec, MlxDtype};
pub use envelope::{EnvelopePayload, EnvelopeSource, OwnedEnvelopePayload};
pub use error::{MlxError, MlxResult};

#[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
mod array;
#[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
mod buffer;
#[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
mod ffi;
#[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
pub mod ops;
#[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
mod resources;
#[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
mod stream;

#[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
pub use array::{async_eval, perf_counters, reset_perf_counters, MlxArray, MlxPerfCounters};
#[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
pub use buffer::SharedBuffer;
#[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
pub use stream::{Device, MlxStream};
