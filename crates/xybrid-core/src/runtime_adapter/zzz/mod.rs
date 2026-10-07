//! zzz engine (Kitten TTS 2) adapter glue: feature-gated thin layer over
//! [`xybrid_zzz_sys`], mirroring the `whisper_cpp` module shape.

mod runtime;

pub use runtime::{assert_engine_abi, ZzzDefaults, ZzzKittenRuntime};
