//! Errors from reading a GGUF header and classifying it.

use crate::error::XybridError;
use crate::runtime_adapter::AdapterError;

/// A GGUF header that could not be read completely, or whose metadata
/// cannot be classified.
///
/// Messages name keys, types and sizes from the model file; they never
/// carry request content.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GgufError {
    /// Reading the file failed.
    #[error("cannot read the GGUF file: {0}")]
    Io(#[from] std::io::Error),

    /// The file does not start with the GGUF magic.
    #[error("not a GGUF file")]
    NotGguf,

    /// A GGUF version llama.cpp does not read: 0, 1, above 3, or a
    /// big-endian file.
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),

    /// The input ends inside the header, or a length in it points past the
    /// end of the file.
    #[error("the GGUF header is truncated or damaged: it ends inside {at}")]
    Truncated {
        /// The header field being read.
        at: String,
    },

    /// The header is larger than the scan's [`GgufLimits`](super::GgufLimits).
    #[error("the GGUF header is larger than the {limit}-byte scan limit")]
    LimitExceeded {
        /// The limit in bytes.
        limit: u64,
    },

    /// A header llama.cpp would refuse, or one this scanner cannot
    /// represent (a key or tensor name that is not UTF-8).
    #[error("malformed GGUF header: {0}")]
    Malformed(String),

    /// A key has a different type than its meaning requires.
    #[error("GGUF key {key:?} is {found}, not {expected}")]
    WrongType {
        /// The key.
        key: String,
        /// The type the key needs.
        expected: &'static str,
        /// The type the file has, such as `arr[str]`.
        found: String,
    },

    /// A key classification needs is absent.
    #[error("GGUF key {key:?} is missing")]
    MissingKey {
        /// The key.
        key: String,
    },

    /// Decision metadata that no decision model can carry, such as a
    /// temperature that is not positive or decision keys without a
    /// decision type.
    #[error("invalid decision metadata: {0}")]
    InvalidDecision(String),

    /// llama.cpp's view of a loaded model differs from the header read
    /// from its file.
    #[error("the loaded model's metadata differs from its GGUF header at {key:?}")]
    NativeMismatch {
        /// The first key that differs (sorted order).
        key: String,
    },

    /// llama.cpp could not report a loaded model's metadata.
    #[error("cannot read the loaded model's metadata: {0}")]
    Native(String),
}

impl From<GgufError> for AdapterError {
    /// An I/O failure stays one; every other problem is with the model
    /// file, which a load refuses as invalid input.
    fn from(error: GgufError) -> Self {
        match error {
            GgufError::Io(e) => Self::IOError(e),
            other => Self::InvalidInput(other.to_string()),
        }
    }
}

impl From<GgufError> for XybridError {
    fn from(error: GgufError) -> Self {
        AdapterError::from(error).into()
    }
}
