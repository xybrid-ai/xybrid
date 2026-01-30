//! Testing utilities for xybrid-core.
//!
//! This module provides mocks, fixtures, and test helpers for unit testing
//! without requiring real model files or external dependencies.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::testing::{fixtures, mocks, model_fixtures};
//!
//! // Create a mock runtime that returns fixed outputs
//! let mock_runtime = mocks::MockRuntime::with_output(vec![0.1, 0.2, 0.3]);
//!
//! // Use test fixtures for common inputs
//! let audio = fixtures::sample_audio_16khz(1.0); // 1 second of silence
//! let envelope = fixtures::text_envelope("Hello world");
//!
//! // Get path to downloaded test models
//! let model_dir = model_fixtures::require_model("kokoro-82m");
//! ```
//!
//! ## Model Fixtures
//!
//! The `model_fixtures` module provides utilities for locating test models:
//!
//! ```rust,ignore
//! use xybrid_core::testing::model_fixtures;
//!
//! // Get model path (panics if not found)
//! let model_dir = model_fixtures::require_model("kokoro-82m");
//!
//! // Check if model is available
//! if model_fixtures::model_available("kokoro-82m") {
//!     // Run integration test
//! }
//!
//! // Skip test if model not available
//! let Some(model_dir) = model_fixtures::model_or_skip("kokoro-82m") else {
//!     return; // Test skipped
//! };
//! ```

pub mod fixtures;
pub mod mocks;
pub mod model_fixtures;

pub use fixtures::*;
pub use mocks::*;
