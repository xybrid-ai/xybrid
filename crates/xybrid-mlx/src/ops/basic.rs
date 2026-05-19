//! Basic tensor ops: matmul, quantized matmul, add, mul, softmax, norms, cast,
//! activations.
//!
//! Safe, non-panicking wrappers over the corresponding mlx-c entry points.
//! All ops take an optional `&MlxStream`; `None` dispatches to the
//! thread-local default CPU stream (see module docs in [`crate::ops`] for
//! why the default is CPU rather than GPU).

use crate::array::MlxArray;
use crate::dtype::MlxDtype;
use crate::error::{MlxError, MlxResult};
use crate::ffi;
use crate::stream::MlxStream;
use std::cell::OnceCell;
use std::ffi::CStr;
use std::thread::LocalKey;

/// Borrow the caller-provided stream or, if `None`, fetch a clone of the
/// thread-local default CPU stream. Returned value's `Drop` cleans up the
/// cloned handle in the default case; the `Either` split is modelled via a
/// small owned-or-borrowed wrapper to avoid a heap allocation.
enum StreamRef<'a> {
    Borrowed(&'a MlxStream),
    Owned(MlxStream),
}

impl StreamRef<'_> {
    fn as_stream(&self) -> &MlxStream {
        match self {
            StreamRef::Borrowed(s) => s,
            StreamRef::Owned(s) => s,
        }
    }
}

fn resolve_stream(opt: Option<&MlxStream>) -> MlxResult<StreamRef<'_>> {
    match opt {
        Some(s) => Ok(StreamRef::Borrowed(s)),
        None => Ok(StreamRef::Owned(MlxStream::default_cpu()?)),
    }
}

thread_local! {
    static SCALAR_HALF: OnceCell<MlxArray> = const { OnceCell::new() };
    static SCALAR_ONE: OnceCell<MlxArray> = const { OnceCell::new() };
    static SCALAR_INV_SQRT2: OnceCell<MlxArray> = const { OnceCell::new() };
    static SCALAR_GELU_CUBIC: OnceCell<MlxArray> = const { OnceCell::new() };
    static SCALAR_SQRT_2_OVER_PI: OnceCell<MlxArray> = const { OnceCell::new() };
}

fn scalar_f32(value: f32) -> MlxResult<MlxArray> {
    let bits = value.to_bits();
    if bits == 0.5f32.to_bits() {
        return cached_scalar(&SCALAR_HALF, value);
    }
    if bits == 1.0f32.to_bits() {
        return cached_scalar(&SCALAR_ONE, value);
    }
    if bits == std::f32::consts::FRAC_1_SQRT_2.to_bits() {
        return cached_scalar(&SCALAR_INV_SQRT2, value);
    }
    if bits == 0.044_715f32.to_bits() {
        return cached_scalar(&SCALAR_GELU_CUBIC, value);
    }
    if bits == 0.797_884_6f32.to_bits() {
        return cached_scalar(&SCALAR_SQRT_2_OVER_PI, value);
    }
    MlxArray::from_slice_f32(&[value], &[1])
}

fn cached_scalar(cell: &'static LocalKey<OnceCell<MlxArray>>, value: f32) -> MlxResult<MlxArray> {
    cell.with(|slot| {
        if let Some(array) = slot.get() {
            return Ok(array.clone());
        }
        let array = MlxArray::from_slice_f32(&[value], &[1])?;
        let _ = slot.set(array.clone());
        Ok(array)
    })
}

/// Matrix multiplication `a @ b`.
///
/// Follows NumPy broadcasting for batched matmul: the last two dims are
/// the matrix dims, preceding dims are broadcast batch dims. Returns an
/// error when mlx rejects the shapes (caller sees
/// [`crate::MlxError::Internal`] with the mlx-c rc).
pub fn matmul(a: &MlxArray, b: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: all three handles are live for the duration of the call.
    let raw = unsafe { ffi::op_matmul(a.as_raw(), b.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Matrix multiplication where `weight` is an MLX packed quantized matrix.
///
/// This wraps `mlx_quantized_matmul`. Xybrid currently enables only affine
/// quantization because Gemma/LFM 4-bit MLX bundles use
/// `{ bits: 4, group_size: 64, mode: "affine" }`; MXFP/NVFP modes need
/// separate fixtures before they are admitted.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul(
    x: &MlxArray,
    weight: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    mode: &str,
    stream: Option<&MlxStream>,
) -> MlxResult<MlxArray> {
    validate_quantized_args(x, weight, scales, biases, transpose, group_size, bits, mode)?;
    let mode = std::ffi::CString::new(mode)
        .map_err(|_| MlxError::Internal("MLX quantization mode contains NUL".into()))?;
    quantized_matmul_prechecked(
        x,
        weight,
        scales,
        biases,
        transpose,
        group_size,
        bits,
        mode.as_c_str(),
        stream,
    )
}

/// Quantized matrix multiplication for callers that validated static weight
/// metadata at load time.
///
/// This keeps model hot paths from repeatedly checking shapes/dtypes and
/// allocating a C string for the quantization mode on every token.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_prechecked(
    x: &MlxArray,
    weight: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    mode: &CStr,
    stream: Option<&MlxStream>,
) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    let group_size = ffi::optional_int(Some(group_size));
    let bits = ffi::optional_int(Some(bits));
    // SAFETY: all array/stream handles are live for the duration of the call.
    // `biases` is either live or the null sentinel accepted by mlx-c.
    let raw = unsafe {
        ffi::op_quantized_matmul(
            x.as_raw(),
            weight.as_raw(),
            scales.as_raw(),
            biases.map_or_else(|| ffi::array_null(), MlxArray::as_raw),
            transpose,
            group_size,
            bits,
            mode,
            s.as_stream().as_raw(),
        )?
    };
    Ok(MlxArray::from_raw(raw))
}

/// Dequantize an MLX packed quantized matrix.
///
/// Used for tensors consumed by operations that do not have row-wise
/// quantized variants yet, such as embedding lookup. Projection weights stay
/// packed and use [`quantized_matmul`].
#[allow(clippy::too_many_arguments)]
pub fn dequantize(
    weight: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
    output_dtype: Option<MlxDtype>,
    stream: Option<&MlxStream>,
) -> MlxResult<MlxArray> {
    validate_quantized_weight(weight, scales, biases, group_size, bits, mode)?;
    let mode = std::ffi::CString::new(mode)
        .map_err(|_| MlxError::Internal("MLX quantization mode contains NUL".into()))?;
    let s = resolve_stream(stream)?;
    let dtype = ffi::optional_dtype(output_dtype.map(Into::into));
    // SAFETY: all array/stream handles are live for the duration of the call.
    // `biases` and `global_scale` are null sentinels when absent.
    let raw = unsafe {
        ffi::op_dequantize(
            weight.as_raw(),
            scales.as_raw(),
            biases.map_or_else(|| ffi::array_null(), MlxArray::as_raw),
            ffi::optional_int(Some(group_size)),
            ffi::optional_int(Some(bits)),
            mode.as_c_str(),
            ffi::array_null(),
            dtype,
            s.as_stream().as_raw(),
        )?
    };
    Ok(MlxArray::from_raw(raw))
}

#[allow(clippy::too_many_arguments)]
fn validate_quantized_args(
    x: &MlxArray,
    weight: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> MlxResult<()> {
    validate_quantized_weight(weight, scales, biases, group_size, bits, mode)?;
    let x_dtype = x.dtype()?;
    if !is_float_dtype(x_dtype) {
        return Err(MlxError::Internal(format!(
            "quantized_matmul input dtype must be F32/F16/Bf16, got {x_dtype:?}"
        )));
    }
    let x_shape = x.shape();
    let Some(&x_inner) = x_shape.last() else {
        return Err(MlxError::InvalidShape {
            expected: vec![1],
            got: x_shape,
        });
    };
    let w_shape = weight.shape();
    let w_rank = w_shape.len();
    let w_inner = if transpose {
        w_shape
            .get(w_rank.saturating_sub(1))
            .and_then(|dim| dim.checked_mul(32))
            .map(|dim| dim / bits)
    } else {
        w_shape.get(w_rank.saturating_sub(2)).copied()
    }
    .ok_or_else(|| MlxError::InvalidShape {
        expected: vec![2],
        got: w_shape.clone(),
    })?;
    if w_inner != x_inner {
        return Err(MlxError::InvalidShape {
            expected: vec![w_inner],
            got: vec![x_inner],
        });
    }
    Ok(())
}

fn validate_quantized_weight(
    weight: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> MlxResult<()> {
    let weight_dtype = weight.dtype()?;
    if weight_dtype != MlxDtype::U32 {
        return Err(MlxError::InvalidDtype {
            expected: MlxDtype::U32,
            got: weight_dtype,
        });
    }
    let scales_dtype = scales.dtype()?;
    if !is_float_dtype(scales_dtype) {
        return Err(MlxError::Internal(format!(
            "quantized scales dtype must be F32/F16/Bf16, got {scales_dtype:?}"
        )));
    }
    if let Some(biases) = biases {
        let bias_dtype = biases.dtype()?;
        if !is_float_dtype(bias_dtype) {
            return Err(MlxError::Internal(format!(
                "quantized biases dtype must be F32/F16/Bf16, got {bias_dtype:?}"
            )));
        }
        if biases.shape() != scales.shape() {
            return Err(MlxError::InvalidShape {
                expected: scales.shape(),
                got: biases.shape(),
            });
        }
    }
    if mode != "affine" {
        return Err(MlxError::Internal(format!(
            "unsupported MLX quantization mode `{mode}`"
        )));
    }
    if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return Err(MlxError::Internal(format!(
            "unsupported MLX quantization bits {bits}"
        )));
    }
    if !matches!(group_size, 32 | 64 | 128) {
        return Err(MlxError::Internal(format!(
            "unsupported MLX quantization group_size {group_size}"
        )));
    }
    Ok(())
}

fn is_float_dtype(dtype: MlxDtype) -> bool {
    matches!(dtype, MlxDtype::F32 | MlxDtype::F16 | MlxDtype::Bf16)
}

/// Element-wise add with NumPy broadcasting.
pub fn add(a: &MlxArray, b: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: all three handles are live for the duration of the call.
    let raw = unsafe { ffi::op_add(a.as_raw(), b.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Element-wise multiply with NumPy broadcasting.
pub fn mul(a: &MlxArray, b: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: all three handles are live for the duration of the call.
    let raw = unsafe { ffi::op_mul(a.as_raw(), b.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Element-wise subtract with NumPy broadcasting.
pub fn sub(a: &MlxArray, b: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: all three handles are live for the duration of the call.
    let raw = unsafe { ffi::op_sub(a.as_raw(), b.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Element-wise divide with NumPy broadcasting.
pub fn div(a: &MlxArray, b: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: all three handles are live for the duration of the call.
    let raw = unsafe { ffi::op_div(a.as_raw(), b.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Softmax along a single axis. Uses MLX's `precise` path so f16 / bf16
/// inputs are computed in f32 intermediates (matches PyTorch default).
///
/// Negative axes are supported and follow NumPy indexing (e.g. `-1` is
/// the last axis).
pub fn softmax(a: &MlxArray, axis: i32, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: both handles are live for the duration of the call.
    let raw = unsafe { ffi::op_softmax_axis(a.as_raw(), axis, s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Root-mean-square normalisation (the flavour used by LLaMA / Qwen /
/// Gemma). Computes `x * weight / sqrt(mean(x^2) + eps)` along the last
/// axis of `x`. Passing `None` for `weight` skips the per-element scale
/// (equivalent to a weight of all ones).
///
/// Dispatches to `mlx_fast_rms_norm`, which uses the MLX-provided fused
/// Metal kernel when a GPU stream is supplied.
pub fn rms_norm(
    x: &MlxArray,
    weight: Option<&MlxArray>,
    eps: f32,
    stream: Option<&MlxStream>,
) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: all handles are live for the duration of the call. The null
    // sentinel from `ffi::array_null()` is only passed when `weight` is
    // None — mlx-c documents that the weight parameter "may be null" and
    // we never free the sentinel.
    let raw = unsafe {
        let w = match weight {
            Some(w) => w.as_raw(),
            None => ffi::array_null(),
        };
        ffi::op_fast_rms_norm(x.as_raw(), w, eps, s.as_stream().as_raw())?
    };
    Ok(MlxArray::from_raw(raw))
}

/// Layer normalisation along the last axis with affine weight and bias.
///
/// Computes `(x - mean(x)) / sqrt(var(x) + eps) * weight + bias`, where
/// `mean` and `var` are reduced over the last axis. BERT-family encoders
/// use this form rather than RMSNorm.
pub fn layer_norm(
    x: &MlxArray,
    weight: &MlxArray,
    bias: &MlxArray,
    eps: f32,
    stream: Option<&MlxStream>,
) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: all handles are live for the duration of the call.
    let raw = unsafe {
        ffi::op_fast_layer_norm(
            x.as_raw(),
            weight.as_raw(),
            bias.as_raw(),
            eps,
            s.as_stream().as_raw(),
        )?
    };
    Ok(MlxArray::from_raw(raw))
}

/// Cast an array to a different dtype.
///
/// Thin wrapper over `mlx_astype`. Casts to the same dtype are a no-op
/// (MLX short-circuits internally).
pub fn cast(a: &MlxArray, dtype: MlxDtype, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: both handles are live for the duration of the call.
    let raw = unsafe { ffi::op_astype(a.as_raw(), dtype.into(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Element-wise sigmoid `1 / (1 + exp(-x))`.
pub fn sigmoid(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: both handles are live for the duration of the call.
    let raw = unsafe { ffi::op_sigmoid(a.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Element-wise exponential `exp(x)`.
pub fn exp(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: both handles are live for the duration of the call.
    let raw = unsafe { ffi::op_exp(a.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Element-wise negation `-x`.
pub fn neg(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: both handles are live for the duration of the call.
    let raw = unsafe { ffi::op_neg(a.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Element-wise square `x * x`.
pub fn square(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: both handles are live for the duration of the call.
    let raw = unsafe { ffi::op_square(a.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Element-wise error function `erf(x)`.
pub fn erf(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: both handles are live for the duration of the call.
    let raw = unsafe { ffi::op_erf(a.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Element-wise square root `sqrt(x)`.
pub fn sqrt(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: both handles are live for the duration of the call.
    let raw = unsafe { ffi::op_sqrt(a.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Element-wise hyperbolic tangent `tanh(x)`.
pub fn tanh(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: both handles are live for the duration of the call.
    let raw = unsafe { ffi::op_tanh(a.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Element-wise reciprocal `1 / x`.
pub fn reciprocal(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: both handles are live for the duration of the call.
    let raw = unsafe { ffi::op_reciprocal(a.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// SiLU / Swish activation: `x * sigmoid(x)`.
///
/// Composed from [`sigmoid`] + [`mul`] rather than a dedicated mlx-c op;
/// the fused form is idiomatic in the MLX-LM reference and this matches
/// PyTorch's `nn.SiLU` numerically. Qwen 3's SwiGLU FFN reads this.
pub fn silu(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let sig = sigmoid(a, stream)?;
    mul(a, &sig, stream)
}

/// Exact Gaussian Error Linear Unit: `0.5 * x * (1 + erf(x / sqrt(2)))`.
///
/// This is the canonical BERT activation. Gemma's gated-GeLU block uses
/// the tanh approximation upstream; use [`gelu_tanh`] for that path.
pub fn gelu(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let inv_sqrt2 = scalar_f32(std::f32::consts::FRAC_1_SQRT_2)?;
    let half = scalar_f32(0.5)?;
    let one = scalar_f32(1.0)?;

    let scaled = mul(a, &inv_sqrt2, stream)?;
    let erf_scaled = erf(&scaled, stream)?;
    let cdf = add(&one, &erf_scaled, stream)?;
    let half_x = mul(a, &half, stream)?;
    mul(&half_x, &cdf, stream)
}

/// Tanh-approximate Gaussian Error Linear Unit.
///
/// Formula: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
/// Gemma-family gated-GeLU FFNs use this approximation in upstream MLX-LM.
pub fn gelu_tanh(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let half = scalar_f32(0.5)?;
    let one = scalar_f32(1.0)?;
    let cubic_coeff = scalar_f32(0.044_715)?;
    let sqrt_2_over_pi = scalar_f32(0.797_884_6)?;

    let x2 = mul(a, a, stream)?;
    let x3 = mul(&x2, a, stream)?;
    let cubic = mul(&x3, &cubic_coeff, stream)?;
    let inner = add(a, &cubic, stream)?;
    let scaled = mul(&inner, &sqrt_2_over_pi, stream)?;
    let activated = tanh(&scaled, stream)?;
    let cdf = add(&one, &activated, stream)?;
    let half_x = mul(a, &half, stream)?;
    mul(&half_x, &cdf, stream)
}

/// Argmax along `axis`, returning a `u32` index array with `keepdims=false`
/// (the argmax axis is reduced away). Used by the greedy sampler in
/// `runtime_adapter/mlx/sampler.rs` to pick the most-likely token from a
/// logits row.
pub fn argmax(a: &MlxArray, axis: i32, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: both handles are live for the duration of the call. `keepdims=false`
    // matches NumPy's `argmax` default.
    let raw = unsafe { ffi::op_argmax_axis(a.as_raw(), axis, false, s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

#[cfg(test)]
mod tests {
    //! Numerical-parity tests for the basic ops.
    //!
    //! Run with `cargo test -p xybrid-mlx --features bindings`. Each test
    //! compares against a reference value computed in plain Rust (for
    //! matmul / softmax / add / mul) or against a hand-verified PyTorch
    //! output (for rms_norm) — we keep the reference in-tree to avoid a
    //! dev-dep on ndarray or a captured-fixture step that would drift.
    //!
    //! All tests use the default CPU stream so they work on Apple Silicon macOS
    //! without a dedicated GPU in CI.
    use super::*;
    use crate::MlxDtype;

    /// Relative tolerance for matmul-style reductions. PRD says
    /// `f32::EPSILON * 10` which is ~1.19e-6; we use 2e-6 to leave a touch
    /// of headroom for the fused FMA path mlx-c takes on CPU.
    const MATMUL_TOL: f32 = 2.0e-6;

    fn assert_close(got: &[f32], expected: &[f32], tol: f32, ctx: &str) {
        assert_eq!(
            got.len(),
            expected.len(),
            "{ctx}: length mismatch (got {}, expected {})",
            got.len(),
            expected.len()
        );
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff <= tol,
                "{ctx}: element {i} differs: got {g}, expected {e}, diff {diff}"
            );
        }
    }

    #[test]
    fn matmul_matches_reference() {
        // a: 2x3, b: 3x2 — result: 2x2.
        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        // Reference: a @ b
        // [ [1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12] ]
        // [ [58, 64], [139, 154] ]
        let expected = [58.0f32, 64.0, 139.0, 154.0];

        let a = MlxArray::from_slice_f32(&a_data, &[2, 3]).unwrap();
        let b = MlxArray::from_slice_f32(&b_data, &[3, 2]).unwrap();
        let c = matmul(&a, &b, None).unwrap();

        assert_eq!(c.shape(), vec![2, 2]);
        let got = c.to_vec_f32().unwrap();
        assert_close(&got, &expected, MATMUL_TOL, "matmul 2x3 @ 3x2");
    }

    #[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
    fn quantize_for_test(
        weight: &MlxArray,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> MlxResult<(MlxArray, MlxArray, MlxArray)> {
        let mode = std::ffi::CString::new(mode)
            .map_err(|_| MlxError::Internal("MLX quantization mode contains NUL".into()))?;
        let stream = crate::MlxStream::default_cpu()?;
        // SAFETY: the dense weight and stream handles are live for the call;
        // mlx_quantize returns three owned array handles for affine mode.
        let (weight, scales, biases) = unsafe {
            crate::ffi::op_quantize(
                weight.as_raw(),
                crate::ffi::optional_int(Some(group_size)),
                crate::ffi::optional_int(Some(bits)),
                mode.as_c_str(),
                crate::ffi::array_null(),
                stream.as_raw(),
            )?
        };
        Ok((
            MlxArray::from_raw(weight),
            MlxArray::from_raw(scales),
            MlxArray::from_raw(biases),
        ))
    }

    #[test]
    #[cfg(all(feature = "bindings", target_os = "macos", target_arch = "aarch64"))]
    fn quantized_matmul_matches_dequantize_then_matmul() {
        let weight_data: Vec<f32> = (0..128).map(|i| ((i as f32 % 17.0) - 8.0) / 5.0).collect();
        let x_data: Vec<f32> = (0..64).map(|i| ((i as f32 % 11.0) - 5.0) / 7.0).collect();

        let dense_weight = MlxArray::from_slice_f32(&weight_data, &[2, 64]).unwrap();
        let x = MlxArray::from_slice_f32(&x_data, &[1, 64]).unwrap();
        let (packed_weight, scales, biases) =
            quantize_for_test(&dense_weight, 64, 4, "affine").unwrap();

        assert_eq!(packed_weight.dtype().unwrap(), MlxDtype::U32);
        let got = quantized_matmul(
            &x,
            &packed_weight,
            &scales,
            Some(&biases),
            true,
            64,
            4,
            "affine",
            None,
        )
        .unwrap()
        .to_vec_f32()
        .unwrap();

        let dequantized = dequantize(
            &packed_weight,
            &scales,
            Some(&biases),
            64,
            4,
            "affine",
            Some(MlxDtype::F32),
            None,
        )
        .unwrap();
        let expected = matmul(
            &x,
            &crate::ops::transpose(&dequantized, &[1, 0], None).unwrap(),
            None,
        )
        .unwrap()
        .to_vec_f32()
        .unwrap();

        assert_close(
            &got,
            &expected,
            1.0e-3,
            "quantized_matmul vs dequantize + matmul",
        );
    }

    #[test]
    fn add_mul_elementwise() {
        let a = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = MlxArray::from_slice_f32(&[10.0, 20.0, 30.0, 40.0], &[4]).unwrap();

        let sum = add(&a, &b, None).unwrap();
        assert_close(
            &sum.to_vec_f32().unwrap(),
            &[11.0, 22.0, 33.0, 44.0],
            MATMUL_TOL,
            "add",
        );

        let prod = mul(&a, &b, None).unwrap();
        assert_close(
            &prod.to_vec_f32().unwrap(),
            &[10.0, 40.0, 90.0, 160.0],
            MATMUL_TOL,
            "mul",
        );
    }

    #[test]
    fn sub_div_elementwise() {
        let a = MlxArray::from_slice_f32(&[8.0, 9.0, 10.0], &[3]).unwrap();
        let b = MlxArray::from_slice_f32(&[2.0, 3.0, 5.0], &[3]).unwrap();

        let diff = sub(&a, &b, None).unwrap();
        assert_close(
            &diff.to_vec_f32().unwrap(),
            &[6.0, 6.0, 5.0],
            MATMUL_TOL,
            "sub",
        );

        let quotient = div(&a, &b, None).unwrap();
        assert_close(
            &quotient.to_vec_f32().unwrap(),
            &[4.0, 3.0, 2.0],
            MATMUL_TOL,
            "div",
        );
    }

    #[test]
    fn softmax_sums_to_one_on_axis() {
        // 2x3 — softmax along axis=1 means each row sums to 1.0.
        let data = [1.0f32, 2.0, 3.0, -1.0, 0.0, 4.0];
        let a = MlxArray::from_slice_f32(&data, &[2, 3]).unwrap();
        let out = softmax(&a, -1, None).unwrap();
        assert_eq!(out.shape(), vec![2, 3]);

        let got = out.to_vec_f32().unwrap();
        for (row_idx, chunk) in got.chunks_exact(3).enumerate() {
            let s: f32 = chunk.iter().sum();
            let diff = (s - 1.0).abs();
            assert!(
                diff < 1.0e-5,
                "row {row_idx} softmax sum = {s}, expected 1.0 (diff {diff})"
            );
        }

        // Spot-check first row: softmax([1,2,3]) ≈ [0.09003057, 0.24472848, 0.66524094]
        let expected_row0 = [0.090_030_57f32, 0.244_728_48, 0.665_240_94];
        assert_close(&got[..3], &expected_row0, 1.0e-5, "softmax row 0");
    }

    #[test]
    fn rms_norm_matches_pytorch_reference() {
        // Reference computed by:
        //   import torch
        //   x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        //   w = torch.tensor([1.0, 1.0, 1.0, 1.0])
        //   # rms = sqrt(mean(x*x) + 1e-5) = sqrt(7.50001) = 2.7386140...
        //   # y = x * w / rms
        //   print(x * w / torch.sqrt((x*x).mean(-1, keepdim=True) + 1e-5))
        //   # -> [[0.3651482, 0.7302964, 1.0954446, 1.4605929]]
        let x = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let w = MlxArray::from_slice_f32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let y = rms_norm(&x, Some(&w), 1.0e-5, None).unwrap();
        let got = y.to_vec_f32().unwrap();

        let expected = [0.365_148_2f32, 0.730_296_4, 1.095_444_6, 1.460_592_9];
        // rms_norm involves a sqrt + mean reduction; the fused kernel has
        // slightly looser tolerances than pure add/mul.
        assert_close(&got, &expected, 1.0e-5, "rms_norm vs torch reference");
    }

    #[test]
    fn layer_norm_matches_reference() {
        // Reference:
        // x = [1,2,3,4], mean = 2.5, var = 1.25, eps = 1e-5.
        // affine weight/bias then applied elementwise.
        let x = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let w = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = MlxArray::from_slice_f32(&[0.5, 0.25, -0.25, -0.5], &[4]).unwrap();
        let y = layer_norm(&x, &w, &b, 1.0e-5, None).unwrap();
        let got = y.to_vec_f32().unwrap();
        let expected = [-0.841_635_4f32, -0.644_423_6, 1.091_635_5, 4.866_542];
        assert_close(&got, &expected, 1.0e-5, "layer_norm reference");
    }

    #[test]
    fn silu_matches_reference() {
        // silu(x) = x * sigmoid(x)
        //   x=-1 -> -0.2689414...
        //   x=0  ->  0.0
        //   x=1  ->  0.7310585...
        //   x=2  ->  1.761594...
        let x = MlxArray::from_slice_f32(&[-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        let y = silu(&x, None).unwrap();
        let got = y.to_vec_f32().unwrap();
        // sigmoid(x) = 1 / (1 + exp(-x)); tolerance matches softmax row-sum tests.
        let expected = [-0.268_941_43f32, 0.0, 0.731_058_6, 1.761_594_2];
        assert_close(&got, &expected, 1.0e-5, "silu reference");
    }

    #[test]
    fn erf_matches_reference() {
        let x = MlxArray::from_slice_f32(&[-1.0, 0.0, 1.0], &[3]).unwrap();
        let y = erf(&x, None).unwrap();
        let got = y.to_vec_f32().unwrap();
        let expected = [-0.842_700_8f32, 0.0, 0.842_700_8];
        assert_close(&got, &expected, 1.0e-5, "erf reference");
    }

    #[test]
    fn neg_square_sqrt_elementwise() {
        let x = MlxArray::from_slice_f32(&[-2.0, 3.0, 4.0], &[3]).unwrap();
        let n = neg(&x, None).unwrap();
        assert_close(
            &n.to_vec_f32().unwrap(),
            &[2.0, -3.0, -4.0],
            MATMUL_TOL,
            "neg",
        );

        let sq = square(&x, None).unwrap();
        assert_close(
            &sq.to_vec_f32().unwrap(),
            &[4.0, 9.0, 16.0],
            MATMUL_TOL,
            "square",
        );

        let root = sqrt(&sq, None).unwrap();
        assert_close(
            &root.to_vec_f32().unwrap(),
            &[2.0, 3.0, 4.0],
            MATMUL_TOL,
            "sqrt",
        );
    }

    #[test]
    fn tanh_matches_reference() {
        let x = MlxArray::from_slice_f32(&[-1.0, 0.0, 1.0], &[3]).unwrap();
        let y = tanh(&x, None).unwrap();
        let got = y.to_vec_f32().unwrap();
        let expected = [-0.761_594_2f32, 0.0, 0.761_594_2];
        assert_close(&got, &expected, 1.0e-5, "tanh reference");
    }

    #[test]
    fn gelu_matches_exact_reference() {
        let x = MlxArray::from_slice_f32(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let y = gelu(&x, None).unwrap();
        let got = y.to_vec_f32().unwrap();
        let expected = [
            -0.045_500_264f32,
            -0.158_655_26,
            0.0,
            0.841_344_7,
            1.954_499_7,
        ];
        assert_close(&got, &expected, 2.0e-5, "gelu exact reference");
    }

    #[test]
    fn gelu_tanh_matches_reference() {
        let x = MlxArray::from_slice_f32(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let y = gelu_tanh(&x, None).unwrap();
        let got = y.to_vec_f32().unwrap();
        let expected = [
            -0.045_402_307f32,
            -0.158_808_01,
            0.0,
            0.841_192,
            1.954_597_7,
        ];
        assert_close(&got, &expected, 2.0e-5, "gelu tanh reference");
    }

    #[test]
    fn exp_reciprocal_elementwise() {
        let x = MlxArray::from_slice_f32(&[0.0, 1.0, 2.0], &[3]).unwrap();
        let e = exp(&x, None).unwrap().to_vec_f32().unwrap();
        let expected = [
            1.0f32,
            std::f32::consts::E,
            std::f32::consts::E * std::f32::consts::E,
        ];
        assert_close(&e, &expected, 1.0e-5, "exp");

        let r = reciprocal(
            &MlxArray::from_slice_f32(&[1.0, 2.0, 4.0], &[3]).unwrap(),
            None,
        )
        .unwrap()
        .to_vec_f32()
        .unwrap();
        assert_close(&r, &[1.0f32, 0.5, 0.25], MATMUL_TOL, "reciprocal");
    }

    #[test]
    fn argmax_picks_max_along_axis() {
        // 2x3: row 0 max at col 2, row 1 max at col 0.
        let x = MlxArray::from_slice_f32(&[0.1, 0.3, 0.9, 5.0, 1.0, -2.0], &[2, 3]).unwrap();
        let idx = argmax(&x, -1, None).unwrap();
        assert_eq!(idx.shape(), vec![2]);
        assert_eq!(idx.dtype().unwrap(), MlxDtype::U32);
        assert_eq!(idx.to_vec_u32().unwrap(), vec![2, 0]);
    }

    #[test]
    fn cast_changes_dtype() {
        let a = MlxArray::from_slice_f32(&[1.5, 2.5, -3.25], &[3]).unwrap();
        let b = cast(&a, MlxDtype::I32, None).unwrap();
        assert_eq!(b.dtype().unwrap(), MlxDtype::I32);

        // Round-trip back to f32 so we can inspect the values (mlx-c's
        // cast from f32 to i32 truncates toward zero, matching NumPy).
        let c = cast(&b, MlxDtype::F32, None).unwrap();
        assert_close(
            &c.to_vec_f32().unwrap(),
            &[1.0, 2.0, -3.0],
            MATMUL_TOL,
            "cast f32->i32->f32 truncates",
        );
    }
}
