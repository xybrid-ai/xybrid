//! Circuit breaker pattern for preventing cascading failures.
//!
//! The circuit breaker prevents hammering failing endpoints by tracking failures
//! and temporarily blocking requests when a threshold is exceeded.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Default circuit breaker configuration values.
const DEFAULT_FAILURE_THRESHOLD: u32 = 3;
const DEFAULT_OPEN_DURATION_MS: u64 = 30000;
const DEFAULT_HALF_OPEN_MAX: u32 = 1;

/// Circuit breaker states.
const STATE_CLOSED: u8 = 0;
const STATE_OPEN: u8 = 1;
const STATE_HALF_OPEN: u8 = 2;

/// Configuration for the circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitConfig {
    /// Number of consecutive failures before opening the circuit.
    pub failure_threshold: u32,
    /// Duration in milliseconds to keep the circuit open before transitioning to half-open.
    pub open_duration_ms: u64,
    /// Maximum number of test requests allowed in half-open state.
    pub half_open_max: u32,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            failure_threshold: DEFAULT_FAILURE_THRESHOLD,
            open_duration_ms: DEFAULT_OPEN_DURATION_MS,
            half_open_max: DEFAULT_HALF_OPEN_MAX,
        }
    }
}

impl CircuitConfig {
    /// Create a strict configuration that opens quickly and stays open longer.
    pub fn strict() -> Self {
        Self {
            failure_threshold: 2,
            open_duration_ms: 60000,
            half_open_max: 1,
        }
    }

    /// Create a lenient configuration that tolerates more failures.
    pub fn lenient() -> Self {
        Self {
            failure_threshold: 5,
            open_duration_ms: 15000,
            half_open_max: 2,
        }
    }
}

/// The current state of the circuit breaker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed (normal operation). All requests are allowed.
    Closed,
    /// Circuit is open. Requests are blocked until the timeout expires.
    Open {
        /// When the circuit will transition to half-open.
        until: Instant,
    },
    /// Circuit is half-open. A limited number of test requests are allowed.
    HalfOpen,
}

/// A circuit breaker that tracks failures and blocks requests to failing endpoints.
///
/// # State Machine
///
/// ```text
/// CLOSED (normal) ──[failure threshold]──► OPEN (blocking)
///                                               │
///                                         [timeout]
///                                               │
///                                               ▼
///                                         HALF-OPEN (testing)
///                                               │
///                               ┌───────────────┴───────────────┐
///                         [success]                        [failure]
///                               │                               │
///                               ▼                               ▼
///                           CLOSED                            OPEN
/// ```
///
/// # Thread Safety
///
/// The circuit breaker uses atomic operations and is safe to use from multiple threads.
///
/// # Example
///
/// ```rust
/// use xybrid_core::http::{CircuitBreaker, CircuitConfig};
///
/// let circuit = CircuitBreaker::new(CircuitConfig::default());
///
/// // Check if we can make a request
/// if circuit.can_execute() {
///     // Make the request...
///     let success = true; // or false
///
///     if success {
///         circuit.record_success();
///     } else {
///         circuit.record_failure();
///     }
/// } else {
///     // Circuit is open, fail fast
///     println!("Circuit breaker is open, skipping request");
/// }
/// ```
pub struct CircuitBreaker {
    /// Current state: 0=CLOSED, 1=OPEN, 2=HALF_OPEN
    state: AtomicU8,
    /// Number of consecutive failures
    failure_count: AtomicU32,
    /// Unix timestamp (ms) when the circuit was opened
    opened_at_ms: AtomicU64,
    /// Number of requests allowed through in half-open state
    half_open_requests: AtomicU32,
    /// Configuration
    config: CircuitConfig,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            state: AtomicU8::new(STATE_CLOSED),
            failure_count: AtomicU32::new(0),
            opened_at_ms: AtomicU64::new(0),
            half_open_requests: AtomicU32::new(0),
            config,
        }
    }

    /// Create a circuit breaker with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CircuitConfig::default())
    }

    /// Check if a request can be executed.
    ///
    /// Returns `true` if:
    /// - Circuit is CLOSED (normal operation)
    /// - Circuit is HALF-OPEN and hasn't exceeded max test requests
    /// - Circuit is OPEN but timeout has expired (transitions to HALF-OPEN)
    ///
    /// Returns `false` if:
    /// - Circuit is OPEN and timeout hasn't expired
    /// - Circuit is HALF-OPEN and max test requests exceeded
    pub fn can_execute(&self) -> bool {
        let current_state = self.state.load(Ordering::SeqCst);

        match current_state {
            STATE_CLOSED => true,
            STATE_OPEN => {
                // Check if we should transition to half-open
                let opened_at = self.opened_at_ms.load(Ordering::SeqCst);
                let now_ms = current_time_ms();
                let elapsed_ms = now_ms.saturating_sub(opened_at);

                if elapsed_ms >= self.config.open_duration_ms {
                    // Transition to half-open
                    if self
                        .state
                        .compare_exchange(
                            STATE_OPEN,
                            STATE_HALF_OPEN,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .is_ok()
                    {
                        self.half_open_requests.store(0, Ordering::SeqCst);
                    }
                    // Allow this request as the first half-open test
                    self.half_open_requests.fetch_add(1, Ordering::SeqCst)
                        < self.config.half_open_max
                } else {
                    false
                }
            }
            STATE_HALF_OPEN => {
                // Allow limited requests in half-open state
                self.half_open_requests.fetch_add(1, Ordering::SeqCst) < self.config.half_open_max
            }
            _ => false,
        }
    }

    /// Record a successful operation.
    ///
    /// If the circuit is HALF-OPEN, transitions to CLOSED.
    /// Resets the failure count.
    pub fn record_success(&self) {
        let current_state = self.state.load(Ordering::SeqCst);

        // Reset failure count on any success
        self.failure_count.store(0, Ordering::SeqCst);

        // If half-open, transition to closed
        if current_state == STATE_HALF_OPEN {
            self.state.store(STATE_CLOSED, Ordering::SeqCst);
        }
    }

    /// Record a failed operation.
    ///
    /// If the circuit is CLOSED and failure threshold is exceeded, transitions to OPEN.
    /// If the circuit is HALF-OPEN, immediately transitions back to OPEN.
    pub fn record_failure(&self) {
        let current_state = self.state.load(Ordering::SeqCst);

        match current_state {
            STATE_CLOSED => {
                let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;

                if failures >= self.config.failure_threshold {
                    self.open_circuit();
                }
            }
            STATE_HALF_OPEN => {
                // Any failure in half-open state reopens the circuit
                self.open_circuit();
            }
            STATE_OPEN => {
                // Already open, nothing to do
            }
            _ => {}
        }
    }

    /// Record a rate limit response (429).
    ///
    /// This immediately opens the circuit regardless of the current state.
    pub fn record_rate_limited(&self) {
        self.open_circuit();
    }

    /// Record a service unavailable response (503).
    ///
    /// This immediately opens the circuit regardless of the current state.
    pub fn record_service_unavailable(&self) {
        self.open_circuit();
    }

    /// Get the current state of the circuit breaker.
    pub fn state(&self) -> CircuitState {
        let current_state = self.state.load(Ordering::SeqCst);

        match current_state {
            STATE_CLOSED => CircuitState::Closed,
            STATE_OPEN => {
                let opened_at = self.opened_at_ms.load(Ordering::SeqCst);
                let now_ms = current_time_ms();
                let remaining_ms = self
                    .config
                    .open_duration_ms
                    .saturating_sub(now_ms.saturating_sub(opened_at));

                CircuitState::Open {
                    until: Instant::now() + Duration::from_millis(remaining_ms),
                }
            }
            STATE_HALF_OPEN => CircuitState::HalfOpen,
            _ => CircuitState::Closed, // Fallback
        }
    }

    /// Check if the circuit is currently open (blocking requests).
    pub fn is_open(&self) -> bool {
        self.state.load(Ordering::SeqCst) == STATE_OPEN
    }

    /// Check if the circuit is currently closed (normal operation).
    pub fn is_closed(&self) -> bool {
        self.state.load(Ordering::SeqCst) == STATE_CLOSED
    }

    /// Get the current failure count.
    pub fn failure_count(&self) -> u32 {
        self.failure_count.load(Ordering::SeqCst)
    }

    /// Manually reset the circuit breaker to closed state.
    pub fn reset(&self) {
        self.state.store(STATE_CLOSED, Ordering::SeqCst);
        self.failure_count.store(0, Ordering::SeqCst);
        self.half_open_requests.store(0, Ordering::SeqCst);
    }

    /// Open the circuit.
    fn open_circuit(&self) {
        self.state.store(STATE_OPEN, Ordering::SeqCst);
        self.opened_at_ms.store(current_time_ms(), Ordering::SeqCst);
        self.half_open_requests.store(0, Ordering::SeqCst);
    }
}

/// Get the current time in milliseconds since Unix epoch.
fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_default_config() {
        let config = CircuitConfig::default();
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.open_duration_ms, 30000);
        assert_eq!(config.half_open_max, 1);
    }

    #[test]
    fn test_initial_state_is_closed() {
        let circuit = CircuitBreaker::with_defaults();
        assert!(circuit.is_closed());
        assert!(!circuit.is_open());
        assert!(circuit.can_execute());
    }

    #[test]
    fn test_opens_after_failure_threshold() {
        let circuit = CircuitBreaker::new(CircuitConfig {
            failure_threshold: 3,
            open_duration_ms: 30000,
            half_open_max: 1,
        });

        // First two failures shouldn't open the circuit
        circuit.record_failure();
        assert!(circuit.is_closed());
        assert_eq!(circuit.failure_count(), 1);

        circuit.record_failure();
        assert!(circuit.is_closed());
        assert_eq!(circuit.failure_count(), 2);

        // Third failure should open it
        circuit.record_failure();
        assert!(circuit.is_open());
        assert!(!circuit.can_execute());
    }

    #[test]
    fn test_success_resets_failure_count() {
        let circuit = CircuitBreaker::new(CircuitConfig {
            failure_threshold: 3,
            open_duration_ms: 30000,
            half_open_max: 1,
        });

        circuit.record_failure();
        circuit.record_failure();
        assert_eq!(circuit.failure_count(), 2);

        circuit.record_success();
        assert_eq!(circuit.failure_count(), 0);
        assert!(circuit.is_closed());
    }

    #[test]
    fn test_rate_limited_immediately_opens() {
        let circuit = CircuitBreaker::with_defaults();
        assert!(circuit.is_closed());

        circuit.record_rate_limited();
        assert!(circuit.is_open());
    }

    #[test]
    fn test_service_unavailable_immediately_opens() {
        let circuit = CircuitBreaker::with_defaults();
        assert!(circuit.is_closed());

        circuit.record_service_unavailable();
        assert!(circuit.is_open());
    }

    #[test]
    fn test_transitions_to_half_open_after_timeout() {
        let circuit = CircuitBreaker::new(CircuitConfig {
            failure_threshold: 1,
            open_duration_ms: 50, // Very short for testing
            half_open_max: 1,
        });

        // Open the circuit
        circuit.record_failure();
        assert!(circuit.is_open());
        assert!(!circuit.can_execute());

        // Wait for timeout
        thread::sleep(Duration::from_millis(60));

        // Should be able to execute now (half-open)
        assert!(circuit.can_execute());

        // State should be half-open
        assert!(matches!(circuit.state(), CircuitState::HalfOpen));
    }

    #[test]
    fn test_half_open_success_closes_circuit() {
        let circuit = CircuitBreaker::new(CircuitConfig {
            failure_threshold: 1,
            open_duration_ms: 10,
            half_open_max: 1,
        });

        // Open and wait for half-open
        circuit.record_failure();
        thread::sleep(Duration::from_millis(20));

        // Transition to half-open by trying to execute
        assert!(circuit.can_execute());

        // Record success
        circuit.record_success();

        // Should be closed now
        assert!(circuit.is_closed());
        assert!(circuit.can_execute());
    }

    #[test]
    fn test_half_open_failure_reopens_circuit() {
        let circuit = CircuitBreaker::new(CircuitConfig {
            failure_threshold: 1,
            open_duration_ms: 10,
            half_open_max: 1,
        });

        // Open and wait for half-open
        circuit.record_failure();
        thread::sleep(Duration::from_millis(20));

        // Transition to half-open
        assert!(circuit.can_execute());

        // Record failure in half-open state
        circuit.record_failure();

        // Should be open again
        assert!(circuit.is_open());
    }

    #[test]
    fn test_reset() {
        let circuit = CircuitBreaker::with_defaults();

        // Open the circuit
        circuit.record_rate_limited();
        assert!(circuit.is_open());

        // Reset
        circuit.reset();
        assert!(circuit.is_closed());
        assert_eq!(circuit.failure_count(), 0);
        assert!(circuit.can_execute());
    }

    #[test]
    fn test_half_open_limits_requests() {
        let circuit = CircuitBreaker::new(CircuitConfig {
            failure_threshold: 1,
            open_duration_ms: 10,
            half_open_max: 2,
        });

        // Open and wait for half-open
        circuit.record_failure();
        thread::sleep(Duration::from_millis(20));

        // First two requests should be allowed
        assert!(circuit.can_execute());
        assert!(circuit.can_execute());

        // Third should be blocked
        assert!(!circuit.can_execute());
    }
}
