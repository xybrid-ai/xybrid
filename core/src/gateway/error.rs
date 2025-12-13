//! Gateway error types.

use thiserror::Error;

/// Errors that can occur when using the Gateway.
#[derive(Debug, Error)]
pub enum GatewayError {
    /// Authentication failed (invalid or missing API key).
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// Rate limit exceeded.
    #[error("Rate limit exceeded. Retry after {retry_after_secs} seconds")]
    RateLimited { retry_after_secs: u64 },

    /// Request validation failed.
    #[error("Invalid request: {0}")]
    ValidationError(String),

    /// Model not found or not available.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Provider error (upstream API error).
    #[error("Provider error ({provider}): {message}")]
    ProviderError { provider: String, message: String },

    /// Network error.
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Request timeout.
    #[error("Request timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u32 },

    /// Content filtered by safety system.
    #[error("Content filtered: {reason}")]
    ContentFiltered { reason: String },

    /// Quota exceeded.
    #[error("Quota exceeded: {0}")]
    QuotaExceeded(String),

    /// Server error.
    #[error("Server error: {0}")]
    ServerError(String),

    /// Parse error (invalid response from provider).
    #[error("Failed to parse response: {0}")]
    ParseError(String),
}

impl GatewayError {
    /// Get the HTTP status code for this error.
    pub fn status_code(&self) -> u16 {
        match self {
            GatewayError::AuthenticationError(_) => 401,
            GatewayError::RateLimited { .. } => 429,
            GatewayError::ValidationError(_) => 400,
            GatewayError::ModelNotFound(_) => 404,
            GatewayError::ProviderError { .. } => 502,
            GatewayError::NetworkError(_) => 503,
            GatewayError::Timeout { .. } => 504,
            GatewayError::ContentFiltered { .. } => 400,
            GatewayError::QuotaExceeded(_) => 429,
            GatewayError::ServerError(_) => 500,
            GatewayError::ParseError(_) => 500,
        }
    }

    /// Get the error type string for OpenAI-compatible error response.
    pub fn error_type(&self) -> &'static str {
        match self {
            GatewayError::AuthenticationError(_) => "authentication_error",
            GatewayError::RateLimited { .. } => "rate_limit_error",
            GatewayError::ValidationError(_) => "invalid_request_error",
            GatewayError::ModelNotFound(_) => "invalid_request_error",
            GatewayError::ProviderError { .. } => "api_error",
            GatewayError::NetworkError(_) => "api_error",
            GatewayError::Timeout { .. } => "timeout_error",
            GatewayError::ContentFiltered { .. } => "content_filter_error",
            GatewayError::QuotaExceeded(_) => "quota_exceeded_error",
            GatewayError::ServerError(_) => "server_error",
            GatewayError::ParseError(_) => "server_error",
        }
    }

    /// Convert to OpenAI-compatible error response.
    pub fn to_error_response(&self) -> super::api::ErrorResponse {
        super::api::ErrorResponse {
            error: super::api::ErrorDetail {
                message: self.to_string(),
                error_type: self.error_type().to_string(),
                param: None,
                code: Some(self.error_code().to_string()),
            },
        }
    }

    /// Get the error code.
    fn error_code(&self) -> &'static str {
        match self {
            GatewayError::AuthenticationError(_) => "invalid_api_key",
            GatewayError::RateLimited { .. } => "rate_limit_exceeded",
            GatewayError::ValidationError(_) => "invalid_request",
            GatewayError::ModelNotFound(_) => "model_not_found",
            GatewayError::ProviderError { .. } => "provider_error",
            GatewayError::NetworkError(_) => "network_error",
            GatewayError::Timeout { .. } => "timeout",
            GatewayError::ContentFiltered { .. } => "content_filtered",
            GatewayError::QuotaExceeded(_) => "quota_exceeded",
            GatewayError::ServerError(_) => "server_error",
            GatewayError::ParseError(_) => "parse_error",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_status_codes() {
        assert_eq!(GatewayError::AuthenticationError("bad key".into()).status_code(), 401);
        assert_eq!(GatewayError::RateLimited { retry_after_secs: 60 }.status_code(), 429);
        assert_eq!(GatewayError::ValidationError("bad param".into()).status_code(), 400);
        assert_eq!(GatewayError::ModelNotFound("gpt-5".into()).status_code(), 404);
    }

    #[test]
    fn test_error_response() {
        let error = GatewayError::AuthenticationError("Invalid API key".into());
        let response = error.to_error_response();

        assert_eq!(response.error.error_type, "authentication_error");
        assert!(response.error.message.contains("Invalid API key"));
    }
}
