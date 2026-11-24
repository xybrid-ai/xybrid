//! Error types for cloud operations.

use std::time::Duration;
use thiserror::Error;

use crate::http::RetryableError;

/// Errors that can occur during cloud operations.
#[derive(Debug, Error)]
pub enum CloudError {
    /// Backend not configured or unavailable.
    #[error("Cloud backend not available: {0}")]
    BackendUnavailable(String),

    /// Gateway connection failed.
    #[error("Gateway connection failed: {0}")]
    GatewayError(String),

    /// Authentication failed (invalid or missing credentials).
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// HTTP/network error.
    #[error("Network error: {0}")]
    NetworkError(String),

    /// API returned an error response.
    ///
    /// `message` is the server's own. A 429 here is a spent quota
    /// (`insufficient_quota`), which is not retryable; a plain rate limit is
    /// [`CloudError::RateLimited`].
    #[error("API error ({status}): {message}")]
    ApiError { status: u16, message: String },

    /// Failed to parse response.
    #[error("Failed to parse response: {0}")]
    ParseError(String),

    /// Rate limit exceeded (HTTP 429 other than a spent quota).
    ///
    /// `retry_after_secs` is the server's `Retry-After`, read by
    /// [`parse_retry_after`](crate::http::parse_retry_after) (an HTTP-date
    /// counts the seconds until then), or 60 when it sent none that reads.
    /// It is never clamped; the retry loop gives up rather than wait past
    /// [`MAX_RETRY_AFTER`](crate::http::MAX_RETRY_AFTER).
    #[error("Rate limit exceeded. Retry after {retry_after_secs} seconds.")]
    RateLimited { retry_after_secs: u64 },

    /// Request timeout.
    #[error("Request timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u32 },

    /// Invalid request parameters.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Model not found or not supported.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Content blocked by safety filter.
    #[error("Content blocked: {reason}")]
    ContentBlocked { reason: String },

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Circuit breaker is open (too many recent failures).
    #[error("Circuit breaker open: {0}")]
    CircuitOpen(String),
}

impl CloudError {
    /// Low-cardinality class of this error, safe to export as `error.type`:
    /// a closed vocabulary that never carries a message, URL or credential.
    pub(crate) fn error_type(&self) -> &'static str {
        match self {
            CloudError::BackendUnavailable(_) => "backend_unavailable",
            CloudError::GatewayError(_) => "gateway_error",
            CloudError::AuthenticationError(_) => "authentication_error",
            CloudError::NetworkError(_) => "network_error",
            CloudError::ApiError { .. } => "api_error",
            CloudError::ParseError(_) => "parse_error",
            CloudError::RateLimited { .. } => "rate_limited",
            CloudError::Timeout { .. } => "timeout",
            CloudError::InvalidRequest(_) => "invalid_request",
            CloudError::ModelNotFound(_) => "model_not_found",
            CloudError::ContentBlocked { .. } => "content_blocked",
            CloudError::ConfigError(_) => "config_error",
            CloudError::CircuitOpen(_) => "circuit_open",
        }
    }
}

/// OpenAI-style error `code` / `type` for a quota that is spent, not
/// throttled.
///
/// OpenAI sends it with HTTP 429, and so does the Xybrid gateway for an
/// unclaimed agent workspace over its quota, whose message says how to claim
/// it. Waiting does not restore the quota, so such a 429 is reported as
/// [`CloudError::ApiError`] (not retryable) rather than
/// [`CloudError::RateLimited`], whatever `Retry-After` came with it.
pub(crate) const INSUFFICIENT_QUOTA: &str = "insufficient_quota";

/// Whether an OpenAI-style error body's `error.code` or `error.type` is
/// [`INSUFFICIENT_QUOTA`].
pub(crate) fn is_insufficient_quota(code: Option<&str>, error_type: Option<&str>) -> bool {
    code == Some(INSUFFICIENT_QUOTA) || error_type == Some(INSUFFICIENT_QUOTA)
}

impl From<std::io::Error> for CloudError {
    fn from(err: std::io::Error) -> Self {
        CloudError::NetworkError(err.to_string())
    }
}

// Allow conversion from cloud_llm errors
impl From<crate::cloud_llm::LlmError> for CloudError {
    fn from(err: crate::cloud_llm::LlmError) -> Self {
        match err {
            crate::cloud_llm::LlmError::ApiKeyMissing { provider, env_var } => {
                CloudError::AuthenticationError(format!(
                    "API key missing for {}. Set {} or use gateway.",
                    provider, env_var
                ))
            }
            crate::cloud_llm::LlmError::HttpError(msg) => CloudError::NetworkError(msg),
            crate::cloud_llm::LlmError::ApiError { status, message } => {
                CloudError::ApiError { status, message }
            }
            crate::cloud_llm::LlmError::ParseError(msg) => CloudError::ParseError(msg),
            crate::cloud_llm::LlmError::RateLimited { retry_after_secs } => {
                CloudError::RateLimited { retry_after_secs }
            }
            crate::cloud_llm::LlmError::Timeout { timeout_ms } => {
                CloudError::Timeout { timeout_ms }
            }
            crate::cloud_llm::LlmError::InvalidRequest(msg) => CloudError::InvalidRequest(msg),
            crate::cloud_llm::LlmError::UnsupportedProvider(msg) => {
                CloudError::BackendUnavailable(msg)
            }
            crate::cloud_llm::LlmError::UnsupportedModel { provider, model } => {
                CloudError::ModelNotFound(format!("{}/{}", provider, model))
            }
            crate::cloud_llm::LlmError::ContentBlocked { reason } => {
                CloudError::ContentBlocked { reason }
            }
            crate::cloud_llm::LlmError::IoError(msg) => CloudError::NetworkError(msg),
        }
    }
}

impl RetryableError for CloudError {
    fn is_retryable(&self) -> bool {
        match self {
            // Retryable errors (transient failures)
            CloudError::RateLimited { .. } => true,
            CloudError::Timeout { .. } => true,
            CloudError::NetworkError(_) => true,
            CloudError::GatewayError(_) => true,
            CloudError::ApiError { status, .. } => match status {
                // A 429 that waiting can lift arrives as `RateLimited`. One
                // left as `ApiError` is a spent quota (`insufficient_quota`,
                // see `is_insufficient_quota`): fail fast with its message.
                429 => false,
                502..=504 => true,
                _ => false,
            },

            // Non-retryable errors (permanent failures)
            CloudError::BackendUnavailable(_) => false,
            CloudError::AuthenticationError(_) => false,
            CloudError::ParseError(_) => false,
            CloudError::InvalidRequest(_) => false,
            CloudError::ModelNotFound(_) => false,
            CloudError::ContentBlocked { .. } => false,
            CloudError::ConfigError(_) => false,
            CloudError::CircuitOpen(_) => false,
        }
    }

    fn retry_after(&self) -> Option<Duration> {
        match self {
            CloudError::RateLimited { retry_after_secs } => {
                Some(Duration::from_secs(*retry_after_secs))
            }
            _ => None,
        }
    }

    fn trips_circuit(&self) -> bool {
        // A spent quota (the only 429 left as `ApiError`) is the gateway
        // answering, not failing. Counted, a few repeats would open the
        // circuit and swap the claim guidance for "try again later".
        !matches!(self, CloudError::ApiError { status: 429, .. })
    }

    fn circuit_open() -> Self {
        CloudError::CircuitOpen(
            "gateway circuit breaker stayed open for the entire retry window; no request was sent"
                .to_string(),
        )
    }
}
