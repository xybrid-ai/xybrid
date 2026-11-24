//! Canonical error type.

/// Errors returned by `xybrid-observability`.
///
/// Recording calls on operation handles never fail; only parsing,
/// configuration, and pipeline lifecycle return errors. `Clone` so a
/// pipeline can return the same shutdown result to every caller.
#[derive(Clone, Debug, thiserror::Error)]
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
    /// A flush, shutdown or transport start-up failed for another reason.
    #[error("{operation} failed: {message}")]
    Lifecycle {
        /// `force_flush`, `shutdown`, or `build`.
        operation: &'static str,
        /// Description (never contains payloads or credentials).
        message: String,
    },
    /// A flush or shutdown completed in time, but covered items were lost.
    #[error("{operation} completed with losses: {failed} failed, {dropped} dropped")]
    Delivery {
        /// `force_flush` or `shutdown`.
        operation: &'static str,
        /// Items whose export ended without acknowledgement.
        failed: u64,
        /// Items discarded by local policy after admission.
        dropped: u64,
    },
}

impl Error {
    /// Whether this error is a lifecycle timeout.
    #[must_use]
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout { .. })
    }

    /// Whether this error reports items lost during a flush or shutdown.
    #[must_use]
    pub fn is_delivery(&self) -> bool {
        matches!(self, Self::Delivery { .. })
    }
}

/// Crate result alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;
