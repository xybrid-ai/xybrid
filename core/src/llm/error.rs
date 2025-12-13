//! Error types for LLM operations.

use thiserror::Error;

/// Errors that can occur during LLM operations.
#[derive(Debug, Error)]
pub enum LlmError {
    /// Backend not configured or unavailable.
    #[error("LLM backend not available: {0}")]
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
    ApiError {
        status: u16,
        message: String,
    },

    /// Failed to parse response.
    #[error("Failed to parse response: {0}")]
    ParseError(String),

    /// Rate limit exceeded.
    #[error("Rate limit exceeded. Retry after {retry_after_secs} seconds.")]
    RateLimited {
        retry_after_secs: u64,
    },

    /// Request timeout.
    #[error("Request timed out after {timeout_ms}ms")]
    Timeout {
        timeout_ms: u32,
    },

    /// Invalid request parameters.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Model not found or not supported.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Content blocked by safety filter.
    #[error("Content blocked: {reason}")]
    ContentBlocked {
        reason: String,
    },

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Local model inference error.
    #[error("Inference error: {0}")]
    InferenceError(String),
}

impl From<std::io::Error> for LlmError {
    fn from(err: std::io::Error) -> Self {
        LlmError::NetworkError(err.to_string())
    }
}

// Allow conversion from cloud_llm errors
impl From<crate::cloud_llm::LlmError> for LlmError {
    fn from(err: crate::cloud_llm::LlmError) -> Self {
        match err {
            crate::cloud_llm::LlmError::ApiKeyMissing { provider, env_var } => {
                LlmError::AuthenticationError(format!(
                    "API key missing for {}. Set {} or use gateway.",
                    provider, env_var
                ))
            }
            crate::cloud_llm::LlmError::HttpError(msg) => LlmError::NetworkError(msg),
            crate::cloud_llm::LlmError::ApiError { status, message } => {
                LlmError::ApiError { status, message }
            }
            crate::cloud_llm::LlmError::ParseError(msg) => LlmError::ParseError(msg),
            crate::cloud_llm::LlmError::RateLimited { retry_after_secs } => {
                LlmError::RateLimited { retry_after_secs }
            }
            crate::cloud_llm::LlmError::Timeout { timeout_ms } => {
                LlmError::Timeout { timeout_ms }
            }
            crate::cloud_llm::LlmError::InvalidRequest(msg) => LlmError::InvalidRequest(msg),
            crate::cloud_llm::LlmError::UnsupportedProvider(msg) => {
                LlmError::BackendUnavailable(msg)
            }
            crate::cloud_llm::LlmError::UnsupportedModel { provider, model } => {
                LlmError::ModelNotFound(format!("{}/{}", provider, model))
            }
            crate::cloud_llm::LlmError::ContentBlocked { reason } => {
                LlmError::ContentBlocked { reason }
            }
            crate::cloud_llm::LlmError::IoError(msg) => LlmError::NetworkError(msg),
        }
    }
}
