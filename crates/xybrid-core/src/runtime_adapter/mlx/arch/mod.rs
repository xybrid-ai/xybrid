//! Per-architecture transformer builders for the MLX LLM adapter.
//!
//! Each submodule ports one upstream MLX-LM architecture to Rust + xybrid-mlx
//! ops. The adapter in [`super::model`] dispatches on `config.json`'s
//! `model_type` field and hands off to the matching builder's `build` /
//! `forward` functions.
//!
//! Builders landing:
//! - [`qwen35`] — Qwen 3 family (US-011, `model_type = "qwen3"`).
//! - [`gemma4`] — Gemma 4 family (US-012, `model_type = "gemma4"`).
//! - [`lfm35`] — Liquid Foundation Model 2 / 2.5 / 3.5 family (US-013,
//!   `model_type = "lfm2"`, `"lfm"` or `"lfm3"`).
//! - [`bert`] — BERT-family encoders for embeddings (US-015,
//!   `model_type = "bert"` or `"nomic_bert"`).

use super::model::{MlxLlmError, MlxLlmResult};

pub mod bert;
pub mod gemma4;
pub mod lfm35;
#[cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub mod linear;
pub mod qwen35;

pub(super) fn validate_i32_dimensions(
    model_type: &str,
    dims: &[(&str, usize)],
) -> MlxLlmResult<()> {
    for &(field, value) in dims {
        shape_dim_i32(model_type, field, value)?;
    }
    Ok(())
}

pub(super) fn validate_i32_product(
    model_type: &str,
    field: &str,
    factors: &[usize],
) -> MlxLlmResult<()> {
    shape_product_i32(model_type, field, factors)?;
    Ok(())
}

pub(super) fn shape_dim_i32(model_type: &str, field: &str, value: usize) -> MlxLlmResult<i32> {
    i32::try_from(value).map_err(|_| {
        MlxLlmError::ConfigInvalid(format!(
            "{model_type} config dimension {field}={value} exceeds MLX i32 shape limit"
        ))
    })
}

pub(super) fn shape_product_i32(
    model_type: &str,
    field: &str,
    factors: &[usize],
) -> MlxLlmResult<i32> {
    let mut value = 1usize;
    for &factor in factors {
        value = value.checked_mul(factor).ok_or_else(|| {
            MlxLlmError::ConfigInvalid(format!(
                "{model_type} config dimension {field} overflows usize"
            ))
        })?;
    }
    shape_dim_i32(model_type, field, value)
}

pub(super) fn shape_dim_usize(model_type: &str, field: &str, value: i32) -> MlxLlmResult<usize> {
    usize::try_from(value).map_err(|_| {
        MlxLlmError::ConfigInvalid(format!(
            "{model_type} runtime shape dimension {field}={value} is negative"
        ))
    })
}

pub(super) fn shape_product_usize(
    model_type: &str,
    field: &str,
    factors: &[usize],
) -> MlxLlmResult<usize> {
    let mut value = 1usize;
    for &factor in factors {
        value = value.checked_mul(factor).ok_or_else(|| {
            MlxLlmError::ConfigInvalid(format!(
                "{model_type} runtime shape dimension {field} overflows usize"
            ))
        })?;
    }
    Ok(value)
}

pub(super) fn checked_i32_add(
    model_type: &str,
    field: &str,
    lhs: i32,
    rhs: i32,
) -> MlxLlmResult<i32> {
    lhs.checked_add(rhs).ok_or_else(|| {
        MlxLlmError::ConfigInvalid(format!(
            "{model_type} runtime shape dimension {field} overflows i32"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_dim_i32_rejects_values_over_mlx_limit() {
        match shape_dim_i32("test", "too_large", i32::MAX as usize + 1).unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("too_large"), "got: {msg}");
                assert!(msg.contains("exceeds MLX i32 shape limit"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn shape_product_i32_rejects_values_over_mlx_limit() {
        match shape_product_i32("test", "wide", &[2, i32::MAX as usize]).unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("wide"), "got: {msg}");
                assert!(msg.contains("exceeds MLX i32 shape limit"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn shape_dim_usize_rejects_negative_runtime_dims() {
        match shape_dim_usize("test", "seq", -1).unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("seq=-1"), "got: {msg}");
                assert!(msg.contains("negative"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    #[test]
    fn checked_i32_add_rejects_overflow() {
        match checked_i32_add("test", "offset", i32::MAX, 1).unwrap_err() {
            MlxLlmError::ConfigInvalid(msg) => {
                assert!(msg.contains("offset"), "got: {msg}");
                assert!(msg.contains("overflows i32"), "got: {msg}");
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }
}
