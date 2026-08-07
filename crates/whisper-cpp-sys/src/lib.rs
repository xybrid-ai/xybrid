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
