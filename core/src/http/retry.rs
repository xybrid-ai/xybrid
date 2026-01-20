//! Retry logic with exponential backoff and jitter.
//!
//! Provides automatic retry for transient failures with configurable backoff.

use std::time::Duration;

use super::CircuitBreaker;

/// Default retry configuration values.
const DEFAULT_MAX_ATTEMPTS: u32 = 5;
const DEFAULT_INITIAL_DELAY_MS: u32 = 1000;
const DEFAULT_MAX_DELAY_MS: u32 = 30000;
const DEFAULT_JITTER_FACTOR: f32 = 0.5;

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of attempts (including the initial attempt).
    pub max_attempts: u32,
    /// Initial delay between retries in milliseconds.
    pub initial_delay_ms: u32,
    /// Maximum delay between retries in milliseconds.
    pub max_delay_ms: u32,
    /// Jitter factor (0.0 to 1.0) - randomizes delay by ±(factor * delay).
    pub jitter_factor: f32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            initial_delay_ms: DEFAULT_INITIAL_DELAY_MS,
            max_delay_ms: DEFAULT_MAX_DELAY_MS,
            jitter_factor: DEFAULT_JITTER_FACTOR,
        }
    }
}

impl RetryPolicy {
    /// Create a policy that never retries (single attempt only).
    pub fn no_retry() -> Self {
        Self {
            max_attempts: 1,
            initial_delay_ms: 0,
            max_delay_ms: 0,
            jitter_factor: 0.0,
        }
    }

    /// Create an aggressive retry policy (more attempts, shorter delays).
    /// Useful for critical operations that should retry quickly.
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 7,
            initial_delay_ms: 500,
            max_delay_ms: 10000,
            jitter_factor: 0.3,
        }
    }

    /// Create a conservative retry policy (fewer attempts, longer delays).
    /// Useful for operations where the server might be overloaded.
    pub fn conservative() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 2000,
            max_delay_ms: 60000,
            jitter_factor: 0.5,
        }
    }

    /// Calculate the delay for a given attempt number (0-indexed).
    ///
    /// Uses exponential backoff: delay = initial_delay * 2^attempt
    /// Capped at max_delay, with optional jitter applied.
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        // Calculate base delay with exponential backoff
        let base_delay_ms = self
            .initial_delay_ms
            .saturating_mul(1 << (attempt - 1).min(10));
        let capped_delay_ms = base_delay_ms.min(self.max_delay_ms);

        // Apply jitter if configured
        let final_delay_ms = if self.jitter_factor > 0.0 {
            let jitter_range = (capped_delay_ms as f32 * self.jitter_factor) as u32;
            let jitter = random_u32() % (jitter_range * 2 + 1);
            capped_delay_ms
                .saturating_sub(jitter_range)
                .saturating_add(jitter)
        } else {
            capped_delay_ms
        };

        Duration::from_millis(final_delay_ms as u64)
    }
}

/// Result of a retry operation.
#[derive(Debug)]
pub enum RetryResult<T, E> {
    /// Operation succeeded.
    Success(T),
    /// Operation failed with a non-retryable error.
    Failure(E),
    /// All retry attempts exhausted.
    Exhausted {
        /// The last error encountered.
        last_error: E,
        /// Total number of attempts made.
        attempts: u32,
    },
}

impl<T, E> RetryResult<T, E> {
    /// Convert to a standard Result, treating Exhausted as an error.
    pub fn into_result(self) -> Result<T, E> {
        match self {
            RetryResult::Success(v) => Ok(v),
            RetryResult::Failure(e) => Err(e),
            RetryResult::Exhausted { last_error, .. } => Err(last_error),
        }
    }

    /// Returns true if the operation succeeded.
    pub fn is_success(&self) -> bool {
        matches!(self, RetryResult::Success(_))
    }
}

/// Trait for errors that can indicate whether a retry is appropriate.
pub trait RetryableError {
    /// Returns true if the error is transient and the operation should be retried.
    ///
    /// Retryable errors include:
    /// - 429 (Rate Limited)
    /// - 502 (Bad Gateway)
    /// - 503 (Service Unavailable)
    /// - Connection timeouts
    /// - DNS failures
    /// - Connection reset
    ///
    /// Non-retryable errors include:
    /// - 400 (Bad Request)
    /// - 401 (Unauthorized)
    /// - 403 (Forbidden)
    /// - 404 (Not Found)
    /// - 422 (Unprocessable Entity)
    fn is_retryable(&self) -> bool;

    /// Returns the recommended delay before retrying, if the server specified one.
    ///
    /// This is typically parsed from the `Retry-After` HTTP header.
    fn retry_after(&self) -> Option<Duration>;
}

/// Execute an operation with automatic retry on failure.
///
/// # Arguments
///
/// * `policy` - The retry policy to use
/// * `circuit` - Optional circuit breaker to check before each attempt
/// * `operation` - The operation to execute (will be called multiple times on retry)
///
/// # Returns
///
/// A `RetryResult` indicating success, non-retryable failure, or exhausted retries.
///
/// # Example
///
/// ```rust,ignore
/// use xybrid_core::http::{RetryPolicy, with_retry, RetryableError};
///
/// let policy = RetryPolicy::default();
/// let result = with_retry(&policy, None, || {
///     make_http_request()
/// });
///
/// match result.into_result() {
///     Ok(response) => println!("Success: {:?}", response),
///     Err(e) => eprintln!("Failed: {:?}", e),
/// }
/// ```
pub fn with_retry<T, E, F>(
    policy: &RetryPolicy,
    circuit: Option<&CircuitBreaker>,
    mut operation: F,
) -> RetryResult<T, E>
where
    F: FnMut() -> Result<T, E>,
    E: RetryableError,
{
    let mut last_error: Option<E> = None;

    for attempt in 0..policy.max_attempts {
        // Check circuit breaker if provided
        if let Some(cb) = circuit {
            if !cb.can_execute() {
                // Circuit is open, skip this attempt but don't count it
                // Wait and try again (the circuit might transition to half-open)
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
        }

        // Calculate delay for this attempt (0 for first attempt)
        let delay = if let Some(ref err) = last_error {
            // Use server-specified retry-after if available
            err.retry_after()
                .unwrap_or_else(|| policy.delay_for_attempt(attempt))
        } else {
            policy.delay_for_attempt(attempt)
        };

        if !delay.is_zero() {
            std::thread::sleep(delay);
        }

        // Execute the operation
        match operation() {
            Ok(result) => {
                // Record success with circuit breaker
                if let Some(cb) = circuit {
                    cb.record_success();
                }
                return RetryResult::Success(result);
            }
            Err(err) => {
                // Record failure with circuit breaker
                if let Some(cb) = circuit {
                    cb.record_failure();
                }

                // Check if error is retryable
                if !err.is_retryable() {
                    return RetryResult::Failure(err);
                }

                last_error = Some(err);
            }
        }
    }

    // All attempts exhausted
    RetryResult::Exhausted {
        last_error: last_error.expect("at least one attempt should have been made"),
        attempts: policy.max_attempts,
    }
}

/// Simple pseudo-random number generator for jitter.
/// Uses a basic xorshift algorithm - not cryptographically secure but sufficient for jitter.
fn random_u32() -> u32 {
    use std::cell::Cell;
    use std::time::SystemTime;

    thread_local! {
        static STATE: Cell<u32> = Cell::new(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u32)
                .unwrap_or(12345)
        );
    }

    STATE.with(|state| {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state.set(x);
        x
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test error type that implements RetryableError.
    #[derive(Debug)]
    struct TestError {
        retryable: bool,
        retry_after: Option<Duration>,
    }

    impl RetryableError for TestError {
        fn is_retryable(&self) -> bool {
            self.retryable
        }

        fn retry_after(&self) -> Option<Duration> {
            self.retry_after
        }
    }

    #[test]
    fn test_default_policy() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_attempts, 5);
        assert_eq!(policy.initial_delay_ms, 1000);
        assert_eq!(policy.max_delay_ms, 30000);
        assert_eq!(policy.jitter_factor, 0.5);
    }

    #[test]
    fn test_no_retry_policy() {
        let policy = RetryPolicy::no_retry();
        assert_eq!(policy.max_attempts, 1);
    }

    #[test]
    fn test_delay_calculation() {
        let policy = RetryPolicy {
            max_attempts: 5,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            jitter_factor: 0.0, // No jitter for predictable testing
        };

        // First attempt has no delay
        assert_eq!(policy.delay_for_attempt(0), Duration::ZERO);

        // Exponential backoff: 1s, 2s, 4s, 8s...
        assert_eq!(policy.delay_for_attempt(1), Duration::from_millis(1000));
        assert_eq!(policy.delay_for_attempt(2), Duration::from_millis(2000));
        assert_eq!(policy.delay_for_attempt(3), Duration::from_millis(4000));
        assert_eq!(policy.delay_for_attempt(4), Duration::from_millis(8000));
    }

    #[test]
    fn test_delay_capped_at_max() {
        let policy = RetryPolicy {
            max_attempts: 10,
            initial_delay_ms: 1000,
            max_delay_ms: 5000,
            jitter_factor: 0.0,
        };

        // Should cap at 5000ms
        assert_eq!(policy.delay_for_attempt(5), Duration::from_millis(5000));
        assert_eq!(policy.delay_for_attempt(6), Duration::from_millis(5000));
    }

    #[test]
    fn test_retry_success_first_attempt() {
        let policy = RetryPolicy::no_retry();
        let mut call_count = 0;

        let result: RetryResult<&str, TestError> = with_retry(&policy, None, || {
            call_count += 1;
            Ok("success")
        });

        assert!(result.is_success());
        assert_eq!(call_count, 1);
    }

    #[test]
    fn test_retry_success_after_failures() {
        let policy = RetryPolicy {
            max_attempts: 5,
            initial_delay_ms: 1, // Very short for testing
            max_delay_ms: 10,
            jitter_factor: 0.0,
        };
        let mut call_count = 0;

        let result: RetryResult<&str, TestError> = with_retry(&policy, None, || {
            call_count += 1;
            if call_count < 3 {
                Err(TestError {
                    retryable: true,
                    retry_after: None,
                })
            } else {
                Ok("success")
            }
        });

        assert!(result.is_success());
        assert_eq!(call_count, 3);
    }

    #[test]
    fn test_retry_non_retryable_error() {
        let policy = RetryPolicy::default();
        let mut call_count = 0;

        let result: RetryResult<&str, TestError> = with_retry(&policy, None, || {
            call_count += 1;
            Err(TestError {
                retryable: false, // Non-retryable
                retry_after: None,
            })
        });

        assert!(matches!(result, RetryResult::Failure(_)));
        assert_eq!(call_count, 1); // Should not retry
    }

    #[test]
    fn test_retry_exhausted() {
        let policy = RetryPolicy {
            max_attempts: 3,
            initial_delay_ms: 1,
            max_delay_ms: 10,
            jitter_factor: 0.0,
        };
        let mut call_count = 0;

        let result: RetryResult<&str, TestError> = with_retry(&policy, None, || {
            call_count += 1;
            Err(TestError {
                retryable: true,
                retry_after: None,
            })
        });

        assert!(matches!(result, RetryResult::Exhausted { attempts: 3, .. }));
        assert_eq!(call_count, 3);
    }

    #[test]
    fn test_jitter_produces_variation() {
        let policy = RetryPolicy {
            max_attempts: 5,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            jitter_factor: 0.5,
        };

        // Generate multiple delays and check they're not all identical
        let delays: Vec<_> = (0..10).map(|_| policy.delay_for_attempt(2)).collect();
        let unique_delays: std::collections::HashSet<_> = delays.iter().collect();

        // With 50% jitter on 2000ms, we should see some variation
        // (statistically very unlikely to get all identical values)
        assert!(
            unique_delays.len() > 1,
            "Expected jitter to produce variation"
        );
    }
}
