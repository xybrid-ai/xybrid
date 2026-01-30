//! Streaming types for platform bindings (Flutter, Kotlin, Swift).
//!
//! This module provides FFI-safe types for streaming ASR sessions.
//! These types are designed to be easily wrapped by platform-specific FFI layers
//! (flutter_rust_bridge, UniFFI) without needing `#[frb]` or other attributes.
//!
//! # Note
//!
//! For the internal SDK streaming API (XybridStream), see the `stream` module.
//! This module provides simple data types for use in FFI bindings.

pub mod session;

// Re-export session types
pub use session::{FfiPartialResult, FfiStreamState, FfiStreamStats, FfiStreamingConfig};
