//! HTTP utilities for production-grade network operations.
//!
//! This module provides retry logic with exponential backoff and circuit breakers
//! for resilient HTTP operations.
//!
//! ## Retry Policy
//!
//! Use [`RetryPolicy`] to configure automatic retries with exponential backoff:
//!
//! ```no_run
//! # fn _example() {
//! use std::time::Duration;
//! use xybrid_core::http::{RetryPolicy, with_retry, RetryableError};
//!
//! #[derive(Debug)]
//! struct MyError;
//! impl RetryableError for MyError {
//!     fn is_retryable(&self) -> bool { true }
//!     fn retry_after(&self) -> Option<Duration> { None }
//!     fn circuit_open() -> Self { MyError }
//! }
//!
//! let policy = RetryPolicy::default();
//! let result = with_retry(&policy, None, || {
//!     Ok::<_, MyError>("success")
//! });
//! # let _ = result;
//! # }
//! ```
//!
//! A server-directed wait (`Retry-After`) is honored only up to
//! [`MAX_RETRY_AFTER`] per call, in total; past that the retry gives up with
//! the server's error instead of sleeping. Loops that sleep on
//! [`RetryableError::retry_after`] themselves track it with a
//! [`RetryAfterBudget`]; [`parse_retry_after`] reads the header.
//!
//! ## Circuit Breaker
//!
//! Use [`CircuitBreaker`] to prevent hammering failing endpoints:
//!
//! ```no_run
//! use xybrid_core::http::{CircuitBreaker, CircuitConfig};
//!
//! # fn make_request() -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
//! let circuit = CircuitBreaker::new(CircuitConfig::default());
//!
//! if circuit.can_execute() {
//!     match make_request() {
//!         Ok(_) => circuit.record_success(),
//!         Err(_) => circuit.record_failure(),
//!     }
//! }
//! ```

mod circuit_breaker;
mod retry;

pub use circuit_breaker::{CircuitBreaker, CircuitConfig, CircuitState};
pub use retry::{
    parse_retry_after, with_retry, RetryAfterBudget, RetryPolicy, RetryResult, RetryableError,
    MAX_RETRY_AFTER,
};
