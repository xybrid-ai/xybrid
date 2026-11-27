//! The safe-wrapper error surface. No `unsafe`, no FFI imports — this module
//! compiles on every target, so downstream code can spell error variants
//! even in no-bindings builds.

use crate::dtype::MlxDtype;

/// Errors produced by the safe MLX wrappers.
///
/// All MLX-returning APIs in this crate yield [`MlxResult`]. Values here
/// map cleanly from the underlying C API's nonzero return codes, with two
/// xybrid-specific variants ([`MlxError::InvalidShape`] and
/// [`MlxError::InvalidDtype`]) that fire before any FFI call — they are
/// checked in the safe wrapper before dispatch.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MlxError {
    /// A tensor shape did not match what the caller or op expected.
    ///
    /// Both sides are captured as owned vectors so the error can be bubbled
    /// up through `?` without borrowing live mlx handles.
    #[error("invalid shape: expected {expected:?}, got {got:?}")]
    InvalidShape {
        /// The shape the caller/op demands.
        expected: Vec<i32>,
        /// The shape the offending array actually carries.
        got: Vec<i32>,
    },

    /// A tensor dtype did not match what the caller or op expected.
    #[error("invalid dtype: expected {expected:?}, got {got:?}")]
    InvalidDtype {
        /// The dtype the caller/op demands.
        expected: MlxDtype,
        /// The dtype the offending array actually carries.
        got: MlxDtype,
    },

    /// MLX reported an allocation failure. On iOS this often maps to hitting
    /// the per-process memory limit under backgrounding; on macOS it usually
    /// means the Metal heap is exhausted.
    #[error("MLX allocation failed: out of memory")]
    OutOfMemory,

    /// A Metal compute kernel failed to compile. Typically a toolchain
    /// problem — e.g. the vendored `.metallib` is missing or the Metal
    /// toolchain version does not match the MLX build.
    #[error("Metal kernel compilation failed: {0}")]
    MetalCompileFailure(String),

    /// Catch-all for other nonzero return codes from mlx-c. Carries whatever
    /// contextual string the wrapper could gather (usually via
    /// `mlx_*_tostring` on the handle involved, or a static description).
    #[error("internal MLX error: {0}")]
    Internal(String),
}

/// Result alias used throughout the crate.
pub type MlxResult<T> = Result<T, MlxError>;
