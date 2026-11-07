//! Canonical error type.

/// Errors returned by `xybrid-observability`.
///
/// Recording calls on operation handles never fail; only parsing,
/// configuration, and pipeline lifecycle return errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A `traceparent` value did not follow W3C Trace Context.
    #[error("invalid traceparent: {0:?}")]
    InvalidTraceparent(String),
    /// An observation reference did not contain valid hex trace/span ids.
    #[error("invalid observation reference")]
    InvalidObservationRef,
    /// Exporter or pipeline configuration was rejected.
    #[error("invalid observability configuration: {0}")]
    Config(String),
    /// Building an exporter failed.
    #[error("failed to build {signal} exporter: {message}")]
    ExporterBuild {
        /// `traces`, `metrics`, or `logs`.
        signal: &'static str,
        /// Exporter error description (never contains payloads).
        message: String,
    },
    /// A flush or shutdown did not complete within its timeout.
    #[error("{operation} did not complete within {timeout_ms} ms")]
    Timeout {
        /// `force_flush` or `shutdown`.
        operation: &'static str,
        /// The bound that was exceeded.
        timeout_ms: u64,
    },
    /// A provider reported a flush/shutdown failure.
    #[error("{operation} failed: {message}")]
    Lifecycle {
        /// `force_flush` or `shutdown`.
        operation: &'static str,
        /// Provider error description.
        message: String,
    },
}

impl Error {
    /// Whether this error is a lifecycle timeout.
    #[must_use]
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout { .. })
    }
}

/// Crate result alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;
