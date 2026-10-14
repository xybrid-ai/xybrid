//! Raw FFI bindings to [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
//!
//! # What makes this crate unusual
//!
//! It links whisper.cpp against the ggml that [`xybrid-llama-sys`] already
//! builds, rather than building its own. `llm-llamacpp` puts
//! `libggml{,-base,-cpu}.a` in every shipped xybrid artifact; a second copy
//! would mean duplicate symbols, a larger binary, and two divergent tensor
//! runtimes in one process. So the `bindings` feature forwards to
//! `xybrid-llama-sys/bindings`, and this crate's build script compiles
//! whisper's single library translation unit against llama's ggml headers.
//!
//! The `links = "whisper"` declaration in `Cargo.toml` enforces the other half:
//! a dependency graph that also pulls crates.io's `whisper-rs-sys` — which
//! bundles its own ggml — fails to resolve instead of silently linking two.
//!
//! # Activation
//!
//! The real bindings are hidden behind the `bindings` cargo feature. A default
//! build compiles this crate to an empty shell on every target, keeping
//! workspace-wide `cargo check` / `cargo clippy` green on runners with no C++
//! toolchain.
//!
//! # Safety and stability
//!
//! This is a `-sys` crate: every item under [`bindings`] is `unsafe extern "C"`
//! and mirrors the upstream C ABI one-to-one. It is not meant to be consumed
//! directly outside the workspace — the safe wrapper crate `xybrid-whisper`
//! owns RAII, typed errors, and parameter validation.
//!
//! [`xybrid-llama-sys`]: https://docs.rs/xybrid-llama-sys

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

#[cfg(feature = "bindings")]
pub mod bindings {
    //! Generated FFI bindings emitted by `bindgen` against `wrapper.h`.
    //!
    //! The allowlist keeps the surface to `whisper_.*` / `WHISPER_.*`;
    //! `ggml_.*` is intentionally excluded because no consumer references a
    //! ggml symbol directly and duplicating those definitions would conflict
    //! with `xybrid-llama-sys`.

    // Mirrors `xybrid-llama-sys::bindings`'s lint allowances. Bindgen output
    // routinely trips dead_code, improper_ctypes, and the broader clippy
    // lints; wide allowances keep the generated module lint-free without
    // post-processing.
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

    // Two sources for the generated bindings:
    //   - cargo (default): fresh bindgen output from build.rs ($OUT_DIR),
    //     which tracks wrapper.h and the pinned whisper.cpp commit.
    //   - `committed-bindings` (the Bazel path, which has no build script and
    //     therefore no bindgen/libclang): the committed snapshot. Regenerate
    //     by building with `--features bindings` and copying
    //     $OUT_DIR/bindings.rs over src/bindings.rs.
    #[cfg(feature = "committed-bindings")]
    include!("bindings.rs");
    #[cfg(not(feature = "committed-bindings"))]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// Hooks into xybrid's Core ML encoder bridge (`native/xybrid-whisper-coreml.h`),
/// on iOS with the `coreml` feature. Hand-written: the bridge is xybrid's.
#[cfg(all(feature = "coreml", target_os = "ios"))]
pub mod coreml {
    use std::os::raw::c_char;

    /// The encoder loaded and matches the model.
    pub const XYBRID_WHISPER_COREML_ACTIVE: i32 = 0;
    /// No encoder init claimed the declaration.
    pub const XYBRID_WHISPER_COREML_NOT_REQUESTED: i32 = 1;
    /// Core ML could not load the encoder.
    pub const XYBRID_WHISPER_COREML_LOAD_FAILED: i32 = 2;
    /// The encoder's input does not match the model.
    pub const XYBRID_WHISPER_COREML_MISMATCH: i32 = 3;
    /// The system is older than iOS 16.
    pub const XYBRID_WHISPER_COREML_UNSUPPORTED_OS: i32 = 4;

    extern "C" {
        /// Declares the encoder path and model dimensions for the next
        /// `whisper_init_state` on this thread. Pair with [`xybrid_whisper_coreml_end_load`].
        pub fn xybrid_whisper_coreml_begin_load(
            encoder_path: *const c_char,
            n_mels: i32,
            n_audio_ctx: i32,
            n_audio_state: i32,
        );

        /// Closes the innermost declaration on this thread and returns its outcome.
        pub fn xybrid_whisper_coreml_end_load() -> i32;

        /// Total failed encoder runs on this thread (monotonic).
        pub fn xybrid_whisper_coreml_failure_count() -> u64;
    }
}
