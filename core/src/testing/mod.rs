//! Testing utilities for xybrid-core.
//!
//! This module provides mocks, fixtures, and test helpers for unit testing
//! without requiring real model files or external dependencies.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::testing::{fixtures, mocks};
//!
//! // Create a mock runtime that returns fixed outputs
//! let mock_runtime = mocks::MockRuntime::with_output(vec![0.1, 0.2, 0.3]);
//!
//! // Use test fixtures for common inputs
//! let audio = fixtures::sample_audio_16khz(1.0); // 1 second of silence
//! let envelope = fixtures::text_envelope("Hello world");
//! ```

pub mod fixtures;
pub mod mocks;

pub use fixtures::*;
pub use mocks::*;
