//! Error and status types.

use std::fmt;

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Canonical status code returned by a LiteRT-LM C API call.
///
/// The numeric values are upstream's and are frozen there (they mirror
/// `absl::StatusCode`). A value this crate does not know is kept as
/// [`StatusCode::Other`], so a newer library never makes a call look
/// successful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusCode {
    /// The operation was cancelled.
    Cancelled,
    /// Unknown error.
    Unknown,
    /// The caller passed an invalid argument.
    InvalidArgument,
    /// A deadline expired.
    DeadlineExceeded,
    /// A requested entity was not found.
    NotFound,
    /// The entity already exists.
    AlreadyExists,
    /// The caller lacks permission.
    PermissionDenied,
    /// A resource (memory, context window) is exhausted.
    ResourceExhausted,
    /// The system is not in a state the operation requires.
    FailedPrecondition,
    /// The operation was aborted.
    Aborted,
    /// A value is out of range.
    OutOfRange,
    /// The operation is not implemented by this runtime build.
    Unimplemented,
    /// An internal invariant of the runtime was broken.
    Internal,
    /// The runtime is unavailable.
    Unavailable,
    /// Unrecoverable data loss or corruption.
    DataLoss,
    /// The request lacks valid authentication.
    Unauthenticated,
    /// A code this crate does not recognize.
    Other(i32),
}

impl StatusCode {
    /// Map a raw non-zero status value. `0` (OK) is not an error and has no
    /// variant; it maps to [`StatusCode::Other`] so a caller that passes it by
    /// mistake still gets a visible value.
    pub fn from_raw(raw: i32) -> Self {
        match raw {
            1 => Self::Cancelled,
            2 => Self::Unknown,
            3 => Self::InvalidArgument,
            4 => Self::DeadlineExceeded,
            5 => Self::NotFound,
            6 => Self::AlreadyExists,
            7 => Self::PermissionDenied,
            8 => Self::ResourceExhausted,
            9 => Self::FailedPrecondition,
            10 => Self::Aborted,
            11 => Self::OutOfRange,
            12 => Self::Unimplemented,
            13 => Self::Internal,
            14 => Self::Unavailable,
            15 => Self::DataLoss,
            16 => Self::Unauthenticated,
            other => Self::Other(other),
        }
    }
}

impl fmt::Display for StatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Other(raw) => write!(f, "status {raw}"),
            known => write!(f, "{known:?}"),
        }
    }
}

/// Errors raised by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The runtime library could not be opened.
    #[error("the LiteRT-LM runtime library could not be opened ({tried}): {reason}")]
    LibraryUnavailable {
        /// The path or name that was tried.
        tried: String,
        /// The loader's reason.
        reason: String,
    },

    /// The library was opened but does not export a function this crate needs.
    #[error("the LiteRT-LM runtime library is missing `{symbol}`: {reason}")]
    MissingSymbol {
        /// The missing symbol.
        symbol: String,
        /// The loader's reason.
        reason: String,
    },

    /// The library's C API version is not one this crate can call.
    #[error(
        "the LiteRT-LM runtime library reports C API {major}.{minor}.{patch}; \
         this build needs {needed_major}.x"
    )]
    IncompatibleVersion {
        /// Reported major version.
        major: i32,
        /// Reported minor version.
        minor: i32,
        /// Reported patch version.
        patch: i32,
        /// The major version this crate was written against.
        needed_major: i32,
    },

    /// A C API call returned a failure status.
    #[error("{call} failed: {code}: {message}")]
    Call {
        /// The C function that failed.
        call: &'static str,
        /// Its status code.
        code: StatusCode,
        /// The runtime's last error message on the calling thread.
        message: String,
    },

    /// A string argument contained an interior NUL byte.
    #[error("{what} contains a NUL byte")]
    InteriorNul {
        /// Which argument.
        what: &'static str,
    },

    /// The runtime reported an error through a stream chunk.
    #[error("the LiteRT-LM stream failed: {0}")]
    Stream(String),

    /// The stream produced no event within the deadline and could not be
    /// stopped. The conversation and its engine were leaked instead of freed.
    #[error("the LiteRT-LM stream stalled; its engine is no longer usable")]
    Stalled,

    /// The engine had a stalled stream earlier and refuses further work.
    #[error("the LiteRT-LM engine is unusable after a stalled stream")]
    EngineUnusable,

    /// The runtime broke the streaming protocol this crate relies on.
    #[error("the LiteRT-LM runtime broke the stream protocol: {0}")]
    Protocol(&'static str),
}

impl Error {
    /// The status code, when the error came from a failed C API call.
    pub fn status(&self) -> Option<StatusCode> {
        match self {
            Self::Call { code, .. } => Some(*code),
            _ => None,
        }
    }

    /// Whether the runtime library itself is absent or unusable, as opposed
    /// to a request failing on a working library.
    pub fn is_runtime_unavailable(&self) -> bool {
        matches!(
            self,
            Self::LibraryUnavailable { .. }
                | Self::MissingSymbol { .. }
                | Self::IncompatibleVersion { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_status_values_are_kept() {
        assert_eq!(StatusCode::from_raw(12), StatusCode::Unimplemented);
        assert_eq!(StatusCode::from_raw(99), StatusCode::Other(99));
        assert_eq!(StatusCode::from_raw(99).to_string(), "status 99");
    }

    #[test]
    fn only_loader_failures_count_as_unavailable() {
        let missing = Error::LibraryUnavailable {
            tried: "liblitert-lm.so".into(),
            reason: "not found".into(),
        };
        assert!(missing.is_runtime_unavailable());
        let call = Error::Call {
            call: "litert_lm_engine_create",
            code: StatusCode::NotFound,
            message: "no such file".into(),
        };
        assert!(!call.is_runtime_unavailable());
        assert_eq!(call.status(), Some(StatusCode::NotFound));
    }
}
