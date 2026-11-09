//! Retry logic with exponential backoff and jitter.
//!
//! Provides automatic retry for transient failures with configurable backoff.
//! Server-directed waits (`Retry-After`) are honored only within
//! [`MAX_RETRY_AFTER`]; see [`RetryAfterBudget`].

use std::time::Duration;

use super::CircuitBreaker;

/// Default retry configuration values.
const DEFAULT_MAX_ATTEMPTS: u32 = 5;
const DEFAULT_INITIAL_DELAY_MS: u32 = 1000;
const DEFAULT_MAX_DELAY_MS: u32 = 30000;
const DEFAULT_JITTER_FACTOR: f32 = 0.5;

/// Longest a retried call waits on server-directed delays (`Retry-After`), in
/// total.
///
/// The wait blocks the calling thread, and a server can ask for anything: an
/// over-quota gateway once sent `Retry-After` of a day. A minute covers
/// ordinary rate limits (a few seconds each) across a whole retry policy. A
/// `Retry-After` that does not fit in what is left of it ends the loop at once
/// with the server's error: retrying before the server's time would only be
/// refused again, and sleeping past the budget is what this bound prevents.
/// Policy backoff (no `Retry-After`) is not charged against it.
pub const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

/// What remains of [`MAX_RETRY_AFTER`] during one retried call.
///
/// Create one per call (not per attempt) and [`spend`](Self::spend) every
/// server-directed wait through it before sleeping.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use xybrid_core::http::RetryAfterBudget;
///
/// let mut budget = RetryAfterBudget::new();
/// assert!(budget.spend(Duration::from_secs(1)));
/// // A day does not fit: give up instead of sleeping.
/// assert!(!budget.spend(Duration::from_secs(86_400)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryAfterBudget {
    remaining: Duration,
}

impl RetryAfterBudget {
    /// A full budget of [`MAX_RETRY_AFTER`].
    pub const fn new() -> Self {
        Self {
            remaining: MAX_RETRY_AFTER,
        }
    }

    /// A budget of `limit`, for a call with a tighter deadline of its own.
    ///
    /// Never more than [`MAX_RETRY_AFTER`]: a larger `limit` gives the full
    /// budget.
    pub fn with_limit(limit: Duration) -> Self {
        Self {
            remaining: limit.min(MAX_RETRY_AFTER),
        }
    }

    /// Charge `wait` to the budget if it fits, returning whether the caller
    /// may sleep it.
    ///
    /// `false` leaves the budget unchanged and means "give up now": the
    /// server asked for longer than this call will wait.
    pub fn spend(&mut self, wait: Duration) -> bool {
        match self.remaining.checked_sub(wait) {
            Some(left) => {
                self.remaining = left;
                true
            }
            None => false,
        }
    }

    /// Server-directed waiting this call may still do.
    pub fn remaining(&self) -> Duration {
        self.remaining
    }
}

impl Default for RetryAfterBudget {
    fn default() -> Self {
        Self::new()
    }
}

/// Read an HTTP `Retry-After` value as the wait it asks for, as of `now`.
///
/// Both forms RFC 9110 allows are read: delay-seconds (`"120"`) and an
/// HTTP-date (`"Wed, 21 Oct 2015 07:28:00 GMT"`, the time until then; a date
/// already past is no wait). The result is always whole seconds, rounded up,
/// so [`Duration::as_secs`] loses nothing. Two out-of-spec forms are read
/// too, because guessing a default for them could wait far less (or more)
/// than the server asked: a delay too large for `u64` saturates, and a
/// fractional delay (`"1.5"`) rounds up.
///
/// Returns `None` for anything else, so the caller applies its own default.
///
/// # Examples
///
/// ```
/// use std::time::{Duration, SystemTime};
/// use xybrid_core::http::parse_retry_after;
///
/// let now = SystemTime::now();
/// assert_eq!(parse_retry_after("120", now), Some(Duration::from_secs(120)));
/// assert_eq!(
///     parse_retry_after("Thu, 01 Jan 1970 00:00:00 GMT", now),
///     Some(Duration::ZERO)
/// );
/// assert_eq!(parse_retry_after("soon", now), None);
/// ```
pub fn parse_retry_after(value: &str, now: std::time::SystemTime) -> Option<Duration> {
    let value = value.trim();
    if let Some(secs) = parse_delay_seconds(value) {
        return Some(Duration::from_secs(secs));
    }

    use chrono::{DateTime, NaiveDateTime, Utc};
    let date = DateTime::parse_from_rfc2822(value)
        .map(|date| date.with_timezone(&Utc))
        .or_else(|_| {
            NaiveDateTime::parse_from_str(value, "%a, %d %b %Y %H:%M:%S GMT")
                .map(|date| date.and_utc())
        })
        .ok()?;
    // Negative (a date already past) is no wait at all.
    let ahead = date
        .signed_duration_since(DateTime::<Utc>::from(now))
        .to_std()
        .unwrap_or_default();
    let secs = ahead
        .as_secs()
        .saturating_add(u64::from(ahead.subsec_nanos() > 0));
    Some(Duration::from_secs(secs))
}

/// Delay-seconds, plus the out-of-spec forms [`parse_retry_after`] accepts:
/// digits that overflow `u64` saturate, and a fraction rounds up.
fn parse_delay_seconds(value: &str) -> Option<u64> {
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    let all_digits = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    if whole.is_empty() || !all_digits(whole) || !all_digits(fraction) {
        return None;
    }
    // All ASCII digits, so the only possible parse error is overflow.
    let whole = whole.parse::<u64>().unwrap_or(u64::MAX);
    let round_up = fraction.bytes().any(|b| b != b'0');
    Some(whole.saturating_add(u64::from(round_up)))
}

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
    /// All retry attempts exhausted, or the server asked to wait longer than
    /// the [`RetryAfterBudget`] allows.
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
    /// - 429 (Rate Limited), but not a spent quota such as OpenAI's
    ///   `insufficient_quota`, which waiting does not restore
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
    /// This is typically parsed from the `Retry-After` HTTP header. Report the
    /// server's value as is: [`with_retry`] gives up rather than wait past
    /// [`MAX_RETRY_AFTER`], which it can only do when it sees the real delay.
    fn retry_after(&self) -> Option<Duration>;

    /// Returns true if the error says the endpoint is unhealthy, so it counts
    /// against a circuit breaker.
    ///
    /// Defaults to `true`. Return `false` for the server's considered answer
    /// to the request itself (a spent quota, say): [`with_retry`] records that
    /// as a success, since the endpoint is up, so repeating it never opens the
    /// circuit and hides the server's message behind a circuit-open error.
    fn trips_circuit(&self) -> bool {
        true
    }

    /// Construct the error to report when a circuit breaker stayed open for the
    /// entire attempt budget, so no real attempt ever ran.
    ///
    /// [`with_retry`] returns this — rather than panicking — when every
    /// iteration is skipped because the circuit is open.
    fn circuit_open() -> Self
    where
        Self: Sized;
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
/// A server-directed wait ([`RetryableError::retry_after`]) replaces the
/// policy backoff for that retry, and is honored only while it fits in the
/// call's [`RetryAfterBudget`] ([`MAX_RETRY_AFTER`] in total). One that does
/// not fit returns [`RetryResult::Exhausted`] with that error at once, without
/// sleeping.
///
/// # Example
///
/// ```no_run
/// # fn _example() {
/// use std::time::Duration;
/// use xybrid_core::http::{RetryPolicy, with_retry, RetryableError};
///
/// #[derive(Debug)]
/// struct MyError;
/// impl RetryableError for MyError {
///     fn is_retryable(&self) -> bool { true }
///     fn retry_after(&self) -> Option<Duration> { None }
///     fn circuit_open() -> Self { MyError }
/// }
///
/// # fn make_http_request() -> Result<String, MyError> { Ok("ok".into()) }
/// let policy = RetryPolicy::default();
/// let result = with_retry(&policy, None, || {
///     make_http_request()
/// });
///
/// match result.into_result() {
///     Ok(response) => println!("Success: {:?}", response),
///     Err(e) => eprintln!("Failed: {:?}", e),
/// }
/// # }
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
    let mut retry_after_budget = RetryAfterBudget::new();
    let mut attempts_made: u32 = 0;

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

        // Calculate delay for this attempt (0 for first attempt). The previous
        // error is consumed here: the attempt below either returns or
        // replaces it.
        let delay = match last_error.take() {
            None => policy.delay_for_attempt(attempt),
            Some(err) => match err.retry_after() {
                // Use server-specified retry-after while the budget lasts
                Some(wait) if retry_after_budget.spend(wait) => wait,
                // The server wants longer than this call may wait: report its
                // error now instead of sleeping or retrying early.
                Some(_) => {
                    return RetryResult::Exhausted {
                        last_error: err,
                        attempts: attempts_made,
                    }
                }
                None => policy.delay_for_attempt(attempt),
            },
        };

        if !delay.is_zero() {
            std::thread::sleep(delay);
        }

        // Execute the operation
        attempts_made += 1;
        match operation() {
            Ok(result) => {
                // Record success with circuit breaker
                if let Some(cb) = circuit {
                    cb.record_success();
                }
                return RetryResult::Success(result);
            }
            Err(err) => {
                // Record failure with circuit breaker, unless the endpoint
                // answered for itself: that shows it is up (and must close a
                // half-open circuit, which would otherwise refuse every call)
                if let Some(cb) = circuit {
                    if err.trips_circuit() {
                        cb.record_failure();
                    } else {
                        cb.record_success();
                    }
                }

                // Check if error is retryable
                if !err.is_retryable() {
                    return RetryResult::Failure(err);
                }

                last_error = Some(err);
            }
        }
    }

    // All attempts exhausted. `last_error` is `None` only if every iteration
    // was skipped by an open circuit breaker (no operation ever ran) — report
    // that as a circuit-open error instead of panicking.
    RetryResult::Exhausted {
        last_error: last_error.unwrap_or_else(E::circuit_open),
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

        fn circuit_open() -> Self {
            TestError {
                retryable: false,
                retry_after: None,
            }
        }
    }

    #[test]
    fn circuit_open_for_all_attempts_returns_error_not_panic() {
        use crate::http::CircuitConfig;

        // Open the breaker by exceeding its failure threshold up front.
        let breaker = CircuitBreaker::new(CircuitConfig::default());
        for _ in 0..10 {
            breaker.record_failure();
        }
        assert!(breaker.is_open(), "breaker should be open before the call");

        // Short attempt budget so the per-skip 100ms wait stays quick.
        let policy = RetryPolicy {
            max_attempts: 2,
            ..RetryPolicy::default()
        };

        let mut calls = 0;
        let result: RetryResult<&str, TestError> = with_retry(&policy, Some(&breaker), || {
            calls += 1;
            Ok("unreachable while the circuit is open")
        });

        // The operation never ran, and we got a graceful error — previously
        // this path panicked via `last_error.expect(...)`.
        assert_eq!(calls, 0, "operation must not run while the circuit is open");
        assert!(matches!(result, RetryResult::Exhausted { .. }));
        assert!(result.into_result().is_err());
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

    /// A `Retry-After` beyond the budget ends the loop after the one attempt
    /// that produced it, without sleeping, and reports that error.
    #[test]
    fn retry_after_beyond_budget_gives_up_without_sleeping() {
        let policy = RetryPolicy {
            max_attempts: 5,
            initial_delay_ms: 1,
            max_delay_ms: 10,
            jitter_factor: 0.0,
        };
        let mut call_count = 0;
        let started = std::time::Instant::now();

        let result: RetryResult<&str, TestError> = with_retry(&policy, None, || {
            call_count += 1;
            Err(TestError {
                retryable: true,
                retry_after: Some(Duration::from_secs(86_400)),
            })
        });

        assert_eq!(call_count, 1, "no retry once the wait exceeds the budget");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "gave up after {:?}",
            started.elapsed()
        );
        match result {
            RetryResult::Exhausted {
                last_error,
                attempts,
            } => {
                assert_eq!(attempts, 1);
                assert_eq!(last_error.retry_after, Some(Duration::from_secs(86_400)));
            }
            other => panic!("expected Exhausted, got {other:?}"),
        }
    }

    /// A short `Retry-After` is honored (it replaces the policy backoff) and
    /// the operation is retried.
    #[test]
    fn short_retry_after_is_honored_and_retried() {
        let policy = RetryPolicy {
            max_attempts: 3,
            initial_delay_ms: 1,
            max_delay_ms: 10,
            jitter_factor: 0.0,
        };
        let wait = Duration::from_millis(50);
        let mut call_count = 0;
        let started = std::time::Instant::now();

        let result: RetryResult<&str, TestError> = with_retry(&policy, None, || {
            call_count += 1;
            if call_count < 3 {
                Err(TestError {
                    retryable: true,
                    retry_after: Some(wait),
                })
            } else {
                Ok("success")
            }
        });

        assert!(result.is_success());
        assert_eq!(call_count, 3);
        assert!(
            started.elapsed() >= wait * 2,
            "server wait skipped: {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn retry_after_budget_is_cumulative_per_call() {
        let mut budget = RetryAfterBudget::new();
        assert_eq!(budget.remaining(), MAX_RETRY_AFTER);

        // Waits that fit are charged; zero always fits.
        assert!(budget.spend(Duration::ZERO));
        assert!(budget.spend(MAX_RETRY_AFTER / 2));
        assert!(budget.spend(MAX_RETRY_AFTER / 2));
        assert_eq!(budget.remaining(), Duration::ZERO);

        // Nothing more fits, and a refusal leaves the budget unchanged.
        assert!(!budget.spend(Duration::from_millis(1)));
        assert_eq!(budget.remaining(), Duration::ZERO);

        // A single wait above the cap never fits a fresh budget.
        let mut fresh = RetryAfterBudget::default();
        assert!(!fresh.spend(MAX_RETRY_AFTER + Duration::from_millis(1)));
        assert!(!fresh.spend(Duration::from_secs(u64::MAX)));
        assert_eq!(fresh.remaining(), MAX_RETRY_AFTER);
        assert!(fresh.spend(MAX_RETRY_AFTER));
    }

    #[test]
    fn retry_after_budget_with_limit_never_exceeds_the_cap() {
        let tight = RetryAfterBudget::with_limit(Duration::from_secs(1));
        assert_eq!(tight.remaining(), Duration::from_secs(1));
        let mut tight = tight;
        assert!(!tight.spend(Duration::from_secs(2)));

        let generous = RetryAfterBudget::with_limit(Duration::from_secs(86_400));
        assert_eq!(generous.remaining(), MAX_RETRY_AFTER);
    }

    #[test]
    fn parse_retry_after_reads_every_form_a_server_sends() {
        use std::time::SystemTime;

        // 2026-10-21T07:27:00Z
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_792_567_620);
        let secs = |value: &str| parse_retry_after(value, now).map(|d| d.as_secs());

        // delay-seconds, surrounding whitespace tolerated
        assert_eq!(secs("0"), Some(0));
        assert_eq!(secs(" 120 "), Some(120));
        // HTTP-date: the time until then; one already past is no wait
        assert_eq!(secs("Wed, 21 Oct 2026 07:28:00 GMT"), Some(60));
        assert_eq!(secs("Wed, 21 Oct 2026 07:26:00 GMT"), Some(0));
        assert_eq!(secs("Wed, 21 Oct 2099 07:28:00 GMT"), Some(2_303_683_260));
        // Out of spec but unambiguous: saturate, or round up
        assert_eq!(secs("99999999999999999999999"), Some(u64::MAX));
        assert_eq!(secs("1.5"), Some(2));
        assert_eq!(secs("2.0"), Some(2));
        // Anything else is left to the caller's default
        for unreadable in ["", "soon", "-1", "1.5s", ".5", "1e3", "inf", "NaN"] {
            assert_eq!(secs(unreadable), None, "{unreadable:?}");
        }
    }

    /// An error the endpoint answered with on purpose (a spent quota, say).
    #[derive(Debug)]
    struct Refusal;

    impl RetryableError for Refusal {
        fn is_retryable(&self) -> bool {
            false
        }

        fn retry_after(&self) -> Option<Duration> {
            None
        }

        fn trips_circuit(&self) -> bool {
            false
        }

        fn circuit_open() -> Self {
            Refusal
        }
    }

    /// Repeating an answer that does not trip the circuit never opens it, so
    /// each caller keeps getting the server's error; and it closes a
    /// half-open circuit rather than leaving it refusing every call.
    #[test]
    fn errors_that_do_not_trip_the_circuit_count_as_the_endpoint_answering() {
        use crate::http::{CircuitConfig, CircuitState};

        let policy = RetryPolicy {
            max_attempts: 3,
            initial_delay_ms: 0,
            max_delay_ms: 0,
            jitter_factor: 0.0,
        };
        let breaker = CircuitBreaker::new(CircuitConfig {
            failure_threshold: 1,
            open_duration_ms: 0,
            half_open_max: 1,
        });

        let mut calls = 0;
        for _ in 0..5 {
            let result: RetryResult<(), Refusal> = with_retry(&policy, Some(&breaker), || {
                calls += 1;
                Err(Refusal)
            });
            assert!(matches!(result, RetryResult::Failure(Refusal)));
        }
        assert_eq!(calls, 5, "one request per call, none refused locally");
        assert!(breaker.is_closed());

        // Opened, with its zero open duration already over: the next call is
        // the half-open probe.
        breaker.record_failure();
        assert!(breaker.is_open());
        let mut state_during_probe = None;
        let result: RetryResult<(), Refusal> = with_retry(&policy, Some(&breaker), || {
            state_during_probe = Some(breaker.state());
            Err(Refusal)
        });
        assert!(matches!(result, RetryResult::Failure(Refusal)));
        assert_eq!(state_during_probe, Some(CircuitState::HalfOpen));
        assert!(breaker.is_closed());

        // Errors that do trip it still open it.
        let result: RetryResult<(), TestError> = with_retry(&policy, Some(&breaker), || {
            Err(TestError {
                retryable: false,
                retry_after: None,
            })
        });
        assert!(matches!(result, RetryResult::Failure(_)));
        assert!(breaker.is_open());
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
