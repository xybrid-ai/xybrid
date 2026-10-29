//! zzz engine (Kitten TTS 2 and Phonon) adapters: feature-gated thin layer over
//! [`xybrid_zzz_sys`], mirroring the `whisper_cpp` module shape.

mod phonon;
mod runtime;
mod stream;

pub use phonon::{PhononDefaults, ZzzPhononRuntime};
pub use runtime::{assert_engine_abi, ZzzDefaults, ZzzKittenRuntime};
