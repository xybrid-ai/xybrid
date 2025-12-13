//! CoreML Runtime backend module.
//!
//! This module provides CoreML inference for iOS and macOS platforms.
//! Currently a stub implementation - real CoreML integration planned for future versions.

mod adapter;

// Re-exports
pub use adapter::CoreMLRuntimeAdapter;
