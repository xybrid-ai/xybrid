//! CoreML Runtime backend module.
//!
//! This module provides native Core ML inference for iOS and macOS when the
//! `coreml` feature is enabled. Other builds retain the lightweight stub so
//! cross-platform consumers can compile without linking Apple frameworks.

#[cfg(not(all(feature = "coreml", any(target_os = "macos", target_os = "ios"))))]
mod adapter;
#[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
mod native;

#[cfg(not(all(feature = "coreml", any(target_os = "macos", target_os = "ios"))))]
pub use adapter::CoreMLRuntimeAdapter;
#[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
pub use native::CoreMLRuntimeAdapter;
