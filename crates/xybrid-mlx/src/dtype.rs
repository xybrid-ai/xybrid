//! Typed dtype enum.
//!
//! The C API (`mlx_dtype`) enumerates 14 variants covering every dtype MLX
//! supports (bool, unsigned/signed integers up to 64-bit, f16/f32/f64,
//! bfloat16, complex64). xybrid's runtime only uses a 7-variant subset —
//! the dtypes that appear in LLM / embedding inference. Restricting the
//! enum here keeps the public API small and lets downstream code
//! exhaustively match without dead arms.
//!
//! Values outside the supported set still round-trip on the C side; they
//! surface in Rust via [`MlxDtype::try_from`] returning
//! [`crate::MlxError::Internal`] rather than silently succeeding with the
//! wrong variant.

/// Supported MLX element types.
///
/// The variants match the PRD specification for US-005 exactly. Extension
/// happens via additional variants in future stories (e.g. adding bool for
/// mask tensors in US-007) — the enum is [`#[non_exhaustive]`][non_ex] so
/// doing so is not a breaking change for downstream matchers that use
/// catch-all arms.
///
/// [non_ex]: https://doc.rust-lang.org/reference/attributes/type_system.html#the-non_exhaustive-attribute
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MlxDtype {
    /// 32-bit IEEE float.
    F32,
    /// IEEE float16.
    F16,
    /// Brain-float 16 (8-bit exponent, 7-bit mantissa).
    Bf16,
    /// 32-bit signed integer.
    I32,
    /// 32-bit unsigned integer.
    U32,
    /// 8-bit signed integer.
    I8,
    /// 8-bit unsigned integer.
    U8,
}

impl MlxDtype {
    /// Size in bytes of a single element of this dtype.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }
}

// FFI conversion impls live behind the `bindings` feature — they are the
// only part of this module that pulls in mlx-c-sys.
#[cfg(all(feature = "bindings", any(target_os = "macos", target_os = "ios")))]
mod ffi_convert {
    use super::MlxDtype;
    use crate::error::{MlxError, MlxResult};
    use mlx_c_sys::bindings::{self as sys, mlx_dtype};

    impl From<MlxDtype> for mlx_dtype {
        fn from(value: MlxDtype) -> Self {
            match value {
                MlxDtype::F32 => sys::mlx_dtype__MLX_FLOAT32,
                MlxDtype::F16 => sys::mlx_dtype__MLX_FLOAT16,
                MlxDtype::Bf16 => sys::mlx_dtype__MLX_BFLOAT16,
                MlxDtype::I32 => sys::mlx_dtype__MLX_INT32,
                MlxDtype::U32 => sys::mlx_dtype__MLX_UINT32,
                MlxDtype::I8 => sys::mlx_dtype__MLX_INT8,
                MlxDtype::U8 => sys::mlx_dtype__MLX_UINT8,
            }
        }
    }

    impl TryFrom<mlx_dtype> for MlxDtype {
        type Error = MlxError;

        fn try_from(value: mlx_dtype) -> MlxResult<Self> {
            // The mlx_dtype__MLX_* constants are generated as `pub const`s by
            // bindgen, so we cannot `match` them directly — they are values,
            // not patterns. Fall through an if-else ladder instead.
            if value == sys::mlx_dtype__MLX_FLOAT32 {
                Ok(MlxDtype::F32)
            } else if value == sys::mlx_dtype__MLX_FLOAT16 {
                Ok(MlxDtype::F16)
            } else if value == sys::mlx_dtype__MLX_BFLOAT16 {
                Ok(MlxDtype::Bf16)
            } else if value == sys::mlx_dtype__MLX_INT32 {
                Ok(MlxDtype::I32)
            } else if value == sys::mlx_dtype__MLX_UINT32 {
                Ok(MlxDtype::U32)
            } else if value == sys::mlx_dtype__MLX_INT8 {
                Ok(MlxDtype::I8)
            } else if value == sys::mlx_dtype__MLX_UINT8 {
                Ok(MlxDtype::U8)
            } else {
                Err(MlxError::Internal(format!(
                    "mlx dtype {value} is not in the xybrid-supported subset \
                     (F32, F16, Bf16, I32, U32, I8, U8)"
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_bytes_matches_ieee() {
        assert_eq!(MlxDtype::F32.size_bytes(), 4);
        assert_eq!(MlxDtype::F16.size_bytes(), 2);
        assert_eq!(MlxDtype::Bf16.size_bytes(), 2);
        assert_eq!(MlxDtype::I32.size_bytes(), 4);
        assert_eq!(MlxDtype::U32.size_bytes(), 4);
        assert_eq!(MlxDtype::I8.size_bytes(), 1);
        assert_eq!(MlxDtype::U8.size_bytes(), 1);
    }
}
