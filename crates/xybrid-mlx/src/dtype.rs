//! Typed dtype enum.
//!
//! The C API (`mlx_dtype`) enumerates 14 variants covering every dtype MLX
//! supports (bool, unsigned/signed integers up to 64-bit, f16/f32/f64,
//! bfloat16, complex64). xybrid's runtime only uses a 9-variant subset —
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
/// The variants match the dtypes that appear in LLM / embedding inference.
/// Extension happens via additional variants in future stories (e.g. adding
/// bool for mask tensors) — the enum is [`#[non_exhaustive]`][non_ex] so doing
/// so is not a breaking change for downstream matchers that use catch-all arms.
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
    /// 64-bit signed integer.
    I64,
    /// 64-bit unsigned integer.
    U64,
    /// 8-bit signed integer.
    I8,
    /// 8-bit unsigned integer.
    U8,
}

/// Convert little-endian IEEE float16 bytes from SafeTensors into `f32`.
///
/// SafeTensors stores 16-bit float buffers as raw little-endian bytes. The MLX
/// adapter promotes those values through `f32` because the safe array wrapper
/// currently exposes `from_slice_f32` as the portable weight-loading
/// constructor.
pub fn f16_le_bytes_to_f32_vec(bytes: &[u8]) -> crate::MlxResult<Vec<f32>> {
    validate_16bit_byte_len(bytes, "f16")?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

/// Convert little-endian bfloat16 bytes from SafeTensors into `f32`.
///
/// bfloat16 keeps the high 16 bits of an IEEE `f32`; conversion is therefore a
/// lossless bit placement into the upper half of a `u32`.
pub fn bf16_le_bytes_to_f32_vec(bytes: &[u8]) -> crate::MlxResult<Vec<f32>> {
    validate_16bit_byte_len(bytes, "bf16")?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = (u32::from(u16::from_le_bytes([c[0], c[1]]))) << 16;
            f32::from_bits(bits)
        })
        .collect())
}

fn validate_16bit_byte_len(bytes: &[u8], dtype: &str) -> crate::MlxResult<()> {
    if bytes.len().is_multiple_of(2) {
        return Ok(());
    }

    Err(crate::MlxError::Internal(format!(
        "{dtype} SafeTensors payload length must be divisible by 2 bytes, got {}",
        bytes.len()
    )))
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = (bits & 0x03ff) as u32;

    let f32_bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mant = frac;
                let mut exponent = -14_i32;
                while mant & 0x0400 == 0 {
                    mant <<= 1;
                    exponent -= 1;
                }
                mant &= 0x03ff;
                let exp_bits = ((exponent + 127) as u32) << 23;
                sign | exp_bits | (mant << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp_bits = (u32::from(exp) + 112) << 23;
            sign | exp_bits | (frac << 13)
        }
    };

    f32::from_bits(f32_bits)
}

impl MlxDtype {
    /// Size in bytes of a single element of this dtype.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }
}

// FFI conversion impls live behind the `bindings` feature — they are the
// only part of this module that pulls in mlx-c-sys.
#[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
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
                MlxDtype::I64 => sys::mlx_dtype__MLX_INT64,
                MlxDtype::U64 => sys::mlx_dtype__MLX_UINT64,
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
            } else if value == sys::mlx_dtype__MLX_INT64 {
                Ok(MlxDtype::I64)
            } else if value == sys::mlx_dtype__MLX_UINT64 {
                Ok(MlxDtype::U64)
            } else if value == sys::mlx_dtype__MLX_INT8 {
                Ok(MlxDtype::I8)
            } else if value == sys::mlx_dtype__MLX_UINT8 {
                Ok(MlxDtype::U8)
            } else {
                Err(MlxError::Internal(format!(
                    "mlx dtype {value} is not in the xybrid-supported subset \
                     (F32, F16, Bf16, I32, U32, I64, U64, I8, U8)"
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
        assert_eq!(MlxDtype::I64.size_bytes(), 8);
        assert_eq!(MlxDtype::U64.size_bytes(), 8);
        assert_eq!(MlxDtype::I8.size_bytes(), 1);
        assert_eq!(MlxDtype::U8.size_bytes(), 1);
    }

    #[test]
    fn f16_and_bf16_le_bytes_convert_to_f32() {
        let f16 = f16_le_bytes_to_f32_vec(&[0x00, 0x3c, 0x00, 0xbc]).unwrap();
        assert_eq!(f16, vec![1.0, -1.0]);

        let bf16 = bf16_le_bytes_to_f32_vec(&[0x80, 0x3f]).unwrap();
        assert_eq!(bf16, vec![1.0]);
    }
}
