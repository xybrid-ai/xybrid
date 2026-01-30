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
    #[error("API error ({status}): {message}")]
    ApiError { status: u16, message: String },

    /// Failed to parse response.
    #[error("Failed to parse response: {0}")]
    ParseError(String),

    /// Rate limit exceeded.
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
            CloudError::ApiError { status, .. } => {
                // Retry on server errors and specific client errors
                matches!(status, 429 | 502 | 503 | 504)
            }

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
}
