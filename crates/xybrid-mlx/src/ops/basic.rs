//! Basic tensor ops: matmul, add, mul, softmax, rms_norm, cast.
//!
//! Safe, non-panicking wrappers over the corresponding mlx-c entry points.
//! All ops take an optional `&MlxStream`; `None` dispatches to the
//! thread-local default CPU stream (see module docs in [`crate::ops`] for
//! why the default is CPU rather than GPU).

use crate::array::MlxArray;
use crate::dtype::MlxDtype;
use crate::error::MlxResult;
use crate::ffi;
use crate::stream::MlxStream;

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
    //! All tests use the default CPU stream so they work on any Apple host
    //! (no GPU required in CI).
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
        // We can't read the u32 data back through our f32 helpers without a
        // cast; exercise cast-to-f32 as a round-trip sanity check (the
        // index values round-trip exactly because they fit in f32).
        let as_f32 = cast(&idx, MlxDtype::F32, None)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(as_f32, vec![2.0, 0.0]);
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
