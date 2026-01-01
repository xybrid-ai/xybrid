//! HTTP utilities for production-grade network operations.
//!
//! This module provides retry logic with exponential backoff and circuit breakers
//! for resilient HTTP operations.
//!
//! ## Retry Policy
//!
//! Use [`RetryPolicy`] to configure automatic retries with exponential backoff:
//!
//! ```rust,ignore
//! use xybrid_core::http::{RetryPolicy, with_retry};
//!
//! let policy = RetryPolicy::default();
//! let result = with_retry(&policy, None, || {
//!     // Your HTTP operation here
//!     Ok::<_, MyError>("success")
//! });
//! ```
//!
//! ## Circuit Breaker
//!
//! Use [`CircuitBreaker`] to prevent hammering failing endpoints:
//!
//! ```rust,ignore
//! use xybrid_core::http::{CircuitBreaker, CircuitConfig};
//!
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
pub use retry::{RetryPolicy, RetryResult, RetryableError, with_retry};
