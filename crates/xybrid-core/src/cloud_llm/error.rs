//! Error types for cloud LLM operations.

use thiserror::Error;

/// Errors that can occur during LLM API operations.
#[derive(Debug, Error)]
pub enum LlmError {
    /// API key not found or invalid.
    #[error("API key not found for {provider}. Set {env_var} environment variable.")]
    ApiKeyMissing { provider: String, env_var: String },

    /// HTTP request failed.
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    /// API returned an error response.
    #[error("API error ({status}): {message}")]
    ApiError { status: u16, message: String },

    /// Failed to parse API response.
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

    /// Unsupported provider.
    #[error("Unsupported provider: {0}")]
    UnsupportedProvider(String),

    /// Model not supported by provider.
    #[error("Model '{model}' not supported by {provider}")]
    UnsupportedModel { provider: String, model: String },

    /// Content moderation blocked the request.
    #[error("Content blocked by safety filter: {reason}")]
    ContentBlocked { reason: String },

    /// Generic IO error.
    #[error("IO error: {0}")]
    IoError(String),
}

impl From<std::io::Error> for LlmError {
    fn from(err: std::io::Error) -> Self {
        LlmError::IoError(err.to_string())
    }
}
