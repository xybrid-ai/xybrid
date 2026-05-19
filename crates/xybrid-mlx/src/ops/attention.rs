//! Attention-flavoured tensor ops: rope, scaled_dot_product_attention,
//! gather, concat, reshape, transpose.
//!
//! These are the per-layer building blocks the xybrid-core MLX LLM adapter
//! (US-010 onwards) will compose into transformer blocks. Ops live here
//! rather than in [`super::basic`] because they either (a) target a fused
//! mlx-c "fast" kernel (rope, SDPA) or (b) manipulate shape/layout rather
//! than pure element-wise arithmetic (gather, concat, reshape, transpose).
//!
//! All ops follow the same default-stream convention documented in
//! [`super`] — pass `Some(&stream)` to pin a device, `None` to fall through
//! to the thread-local default CPU stream.

use crate::array::MlxArray;
use crate::error::{MlxError, MlxResult};
use crate::ffi;
use crate::stream::MlxStream;

/// Borrow the caller-provided stream or, if `None`, fetch a clone of the
/// thread-local default CPU stream.
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

/// Reshape an array to a new shape (without copying — mlx produces a view
/// when the strides admit one).
///
/// `shape` follows NumPy/`torch.reshape` semantics; a single `-1` entry may
/// be used to infer one dimension from the others, subject to mlx-c's own
/// validation. Returns [`MlxError::Internal`] with the mlx-c return code on
/// rank / size mismatch.
pub fn reshape(a: &MlxArray, shape: &[i32], stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: `a` and the stream are live for the duration of the call.
    // `shape.as_ptr()` is valid for the call because the slice is borrowed.
    let raw = unsafe { ffi::op_reshape(a.as_raw(), shape, s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Permute the dimensions of `a` according to `axes`.
///
/// Pass `None` (via `&[]` semantics) is not supported here — callers who
/// want the default "reverse all axes" transpose should call
/// [`transpose_default`] instead. We keep the two APIs separate so the
/// typed path is unambiguous.
pub fn transpose(a: &MlxArray, axes: &[i32], stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    if axes.is_empty() {
        return Err(MlxError::Internal(
            "transpose(axes=[]) is not supported; use transpose_default() for the reverse-all \
             variant"
                .into(),
        ));
    }
    let s = resolve_stream(stream)?;
    // SAFETY: `a` and the stream are live for the duration of the call;
    // `axes.as_ptr()` points at borrowed memory for the same scope.
    let raw = unsafe { ffi::op_transpose_axes(a.as_raw(), axes, s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Default transpose: reverse all axes (equivalent to `a.T` on a 2-D array).
///
/// Exposed as a separate entry point so the common rank-2 case does not
/// require callers to materialise the `[1, 0]` axis permutation.
pub fn transpose_default(a: &MlxArray, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: `a` and the stream are live for the duration of the call.
    let raw = unsafe { ffi::op_transpose_default(a.as_raw(), s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Gather rows / slices of `a` along `axis` using `indices`.
///
/// Thin wrapper over `mlx_take_axis` — equivalent to `torch.index_select` or
/// `numpy.take` with `axis=axis`. The most common use case is embedding
/// lookup: `gather(tok_embeddings, input_ids, 0, stream)`.
///
/// `indices` must have an integer dtype (i32 / i64 / u32); mlx-c validates
/// this internally and returns a nonzero rc on mismatch.
pub fn gather(
    a: &MlxArray,
    indices: &MlxArray,
    axis: i32,
    stream: Option<&MlxStream>,
) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    // SAFETY: all handles are live for the duration of the call.
    let raw =
        unsafe { ffi::op_take_axis(a.as_raw(), indices.as_raw(), axis, s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Static slice using MLX's `[start, stop, stride]` transform.
pub fn slice(
    a: &MlxArray,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    stream: Option<&MlxStream>,
) -> MlxResult<MlxArray> {
    validate_slice_args(start, stop, strides)?;
    let s = resolve_stream(stream)?;
    // SAFETY: all handles and slice argument buffers are live for the call.
    let raw = unsafe { ffi::op_slice(a.as_raw(), start, stop, strides, s.as_stream().as_raw())? };
    Ok(MlxArray::from_raw(raw))
}

/// Static slice update using MLX's `[start, stop, stride]` transform.
pub fn slice_update(
    src: &MlxArray,
    update: &MlxArray,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    stream: Option<&MlxStream>,
) -> MlxResult<MlxArray> {
    validate_slice_args(start, stop, strides)?;
    let s = resolve_stream(stream)?;
    // SAFETY: all handles and slice argument buffers are live for the call.
    let raw = unsafe {
        ffi::op_slice_update(
            src.as_raw(),
            update.as_raw(),
            start,
            stop,
            strides,
            s.as_stream().as_raw(),
        )?
    };
    Ok(MlxArray::from_raw(raw))
}

fn validate_slice_args(start: &[i32], stop: &[i32], strides: &[i32]) -> MlxResult<()> {
    if start.len() != stop.len() || start.len() != strides.len() {
        return Err(MlxError::InvalidShape {
            expected: vec![start.len() as i32],
            got: vec![stop.len() as i32, strides.len() as i32],
        });
    }
    if start.is_empty() {
        return Err(MlxError::InvalidShape {
            expected: vec![1],
            got: Vec::new(),
        });
    }
    Ok(())
}

/// Concatenate an array of arrays along `axis`.
///
/// All input arrays must have identical shapes except along `axis`. Returns
/// [`MlxError::InvalidShape`] with `expected=[]` and `got=[]` when the input
/// list is empty — mlx-c rejects zero-array concat, so we short-circuit
/// before building the vector handle.
pub fn concat(arrays: &[&MlxArray], axis: i32, stream: Option<&MlxStream>) -> MlxResult<MlxArray> {
    if arrays.is_empty() {
        return Err(MlxError::InvalidShape {
            expected: Vec::new(),
            got: Vec::new(),
        });
    }
    let s = resolve_stream(stream)?;
    // Build the mlx_vector_array, dispatch, and free the vector even on the
    // error path. We collect raw handles into a small stack-allocated Vec —
    // mlx-c copies them into its own std::vector<array>, so we don't need
    // to keep this Vec alive past the FFI call.
    let raws: Vec<_> = arrays.iter().map(|a| a.as_raw()).collect();

    // SAFETY: every handle in `raws` is live for the duration of the call
    // (backed by a live `&MlxArray`). The vector returned by
    // `vector_array_from` is owned by us and freed unconditionally via the
    // scope guard below.
    let vec = unsafe { ffi::vector_array_from(&raws) };

    let result = unsafe { ffi::op_concat_axis(vec, axis, s.as_stream().as_raw()) };

    // SAFETY: `vec` was produced by `vector_array_from` above and has not
    // been freed elsewhere.
    unsafe { ffi::vector_array_free(vec) };

    result.map(MlxArray::from_raw)
}

/// Apply Rotary Position Embedding (RoPE) to `x`.
///
/// - `dims` — head dimension (the trailing axis; must be even for the
///   canonical interleaved/non-interleaved rotation).
/// - `traditional` — when `true`, use the "traditional" interleaved
///   rotation (real part in even indices, imaginary in odd). When `false`,
///   use the "half-split" variant used by LLaMA / Qwen / Gemma (first half
///   / second half split).
/// - `theta_base` — the RoPE base frequency. Pass `None` to let mlx fall
///   through to its internal default (10000 in upstream).
/// - `scale` — RoPE scaling factor applied to positions (1.0 for no
///   scaling; used for YaRN / linear position scaling variants).
/// - `offset` — starting position index; typically 0 for a fresh prompt and
///   `seq_len` for incremental decoding inside a KV-cache loop.
/// - `freqs` — pre-computed cos/sin table. `None` tells mlx to compute its
///   own default table from `theta_base` + `scale`; this is the usual
///   choice. Providing an explicit table is for parity with trained YaRN
///   variants that override the base frequencies.
///
/// Dispatches to mlx-c's `mlx_fast_rope` kernel, which is fused on Metal.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    x: &MlxArray,
    dims: i32,
    traditional: bool,
    theta_base: Option<f32>,
    scale: f32,
    offset: i32,
    freqs: Option<&MlxArray>,
    stream: Option<&MlxStream>,
) -> MlxResult<MlxArray> {
    let s = resolve_stream(stream)?;
    let base = ffi::optional_float(theta_base);
    // SAFETY: `x` and the stream are live for the duration of the call.
    // `freqs` is either a live handle (the `Some` arm) or the null sentinel
    // — mlx-c documents that freqs "may be null" and checks ctx internally.
    // The null sentinel is never freed.
    let raw = unsafe {
        let f = match freqs {
            Some(f) => f.as_raw(),
            None => ffi::array_null(),
        };
        ffi::op_fast_rope(
            x.as_raw(),
            dims,
            traditional,
            base,
            scale,
            offset,
            f,
            s.as_stream().as_raw(),
        )?
    };
    Ok(MlxArray::from_raw(raw))
}

/// Scaled dot-product attention: `softmax(Q @ K^T / sqrt(d_k)) @ V`, with
/// optional causal masking and additive mask.
///
/// - `queries`, `keys`, `values` — the usual rank-4 `[batch, heads, seq, d]`
///   tensors. mlx-c validates matching batch / head / dim shapes.
/// - `scale` — typically `1.0 / sqrt(head_dim)`. Callers compute this
///   themselves so we don't need to read `d_k` back from the tensor.
/// - `causal` — when `true`, apply the standard upper-triangular causal
///   mask internally (no explicit `mask` needed). Callers that want a
///   custom mask should pass `causal = false` and a `mask` tensor.
/// - `mask` — optional additive attention bias broadcastable to
///   `[batch, heads, q_len, k_len]`. Mutually exclusive with `causal =
///   true`; mlx-c rejects the combination.
///
/// Dispatches to `mlx_fast_scaled_dot_product_attention`, which picks the
/// fused MPS kernel when a GPU stream is supplied and falls back to the
/// unfused CPU reference otherwise.
pub fn scaled_dot_product_attention(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    causal: bool,
    mask: Option<&MlxArray>,
    stream: Option<&MlxStream>,
) -> MlxResult<MlxArray> {
    if causal && mask.is_some() {
        return Err(MlxError::Internal(
            "scaled_dot_product_attention: causal=true is mutually exclusive with an \
             explicit mask"
                .into(),
        ));
    }
    let s = resolve_stream(stream)?;
    // mlx-c expects a C string for `mask_mode`. "causal" selects the
    // upper-triangular mask; empty string selects "no mode" (use the
    // provided mask_arr, or no mask if both are null).
    let mode_cstr: &[u8] = if causal { b"causal\0" } else { b"\0" };
    // SAFETY: all array handles are live; the null sentinel is never freed;
    // `mode_cstr.as_ptr()` points at a `&'static [u8]` with explicit nul
    // terminator, so it is a valid C string for the duration of the call.
    let raw = unsafe {
        let mask_raw = match mask {
            Some(m) => m.as_raw(),
            None => ffi::array_null(),
        };
        let sinks_raw = ffi::array_null();
        ffi::op_sdpa(
            queries.as_raw(),
            keys.as_raw(),
            values.as_raw(),
            scale,
            mode_cstr.as_ptr().cast::<std::os::raw::c_char>(),
            mask_raw,
            sinks_raw,
            s.as_stream().as_raw(),
        )?
    };
    Ok(MlxArray::from_raw(raw))
}

#[cfg(test)]
mod tests {
    //! Numerical-parity tests for the attention ops.
    //!
    //! Run with `cargo test -p xybrid-mlx --features bindings` on an Apple
    //! host (gated on `bindings` via the `ops` module cfg in `lib.rs`).
    //! Reference values are computed in plain Rust where feasible and
    //! captured from a Python session where the derivation is long (RoPE).
    use super::*;
    use crate::ops::basic::{matmul, softmax};
    use crate::MlxDtype;

    /// Shared with `ops::basic::tests::MATMUL_TOL`. Kept separate to avoid a
    /// cross-module `pub(crate)` export just for tests.
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
    fn reshape_preserves_data() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let a = MlxArray::from_slice_f32(&data, &[3, 4]).unwrap();
        let b = reshape(&a, &[4, 3], None).unwrap();
        assert_eq!(b.shape(), vec![4, 3]);
        // Row-major layout is preserved — flattening must match the input.
        let got = b.to_vec_f32().unwrap();
        assert_close(&got, &data, MATMUL_TOL, "reshape preserves flat order");
    }

    #[test]
    fn transpose_swaps_axes() {
        // 2x3 matrix:
        //   [[0, 1, 2],
        //    [3, 4, 5]]
        // transpose -> 3x2:
        //   [[0, 3],
        //    [1, 4],
        //    [2, 5]]
        let data = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let a = MlxArray::from_slice_f32(&data, &[2, 3]).unwrap();
        let b = transpose(&a, &[1, 0], None).unwrap();
        assert_eq!(b.shape(), vec![3, 2]);
        let got = b.to_vec_f32().unwrap();
        assert_close(
            &got,
            &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
            MATMUL_TOL,
            "2x3 -> 3x2 transpose",
        );
    }

    #[test]
    fn transpose_default_reverses_all_axes() {
        let data = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let a = MlxArray::from_slice_f32(&data, &[2, 3]).unwrap();
        let b = transpose_default(&a, None).unwrap();
        assert_eq!(b.shape(), vec![3, 2]);
        let got = b.to_vec_f32().unwrap();
        assert_close(
            &got,
            &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
            MATMUL_TOL,
            "default transpose reverses axes",
        );
    }

    #[test]
    fn transpose_rejects_empty_axes() {
        let a = MlxArray::from_slice_f32(&[1.0, 2.0], &[2]).unwrap();
        let err = transpose(&a, &[], None).unwrap_err();
        assert!(matches!(err, MlxError::Internal(_)));
    }

    #[test]
    fn gather_is_embedding_lookup() {
        // Table: 4 rows of 3 cols — an embedding table of vocab=4, dim=3.
        //   row 0: [10, 20, 30]
        //   row 1: [40, 50, 60]
        //   row 2: [70, 80, 90]
        //   row 3: [100, 110, 120]
        let table = MlxArray::from_slice_f32(
            &[
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
            ],
            &[4, 3],
        )
        .unwrap();
        let idx = MlxArray::from_slice_i32(&[2, 0, 3], &[3]).unwrap();
        let out = gather(&table, &idx, 0, None).unwrap();
        assert_eq!(out.shape(), vec![3, 3]);
        let got = out.to_vec_f32().unwrap();
        assert_close(
            &got,
            &[70.0, 80.0, 90.0, 10.0, 20.0, 30.0, 100.0, 110.0, 120.0],
            MATMUL_TOL,
            "gather rows [2, 0, 3] from a 4x3 table",
        );
    }

    #[test]
    fn slice_selects_static_range() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let a = MlxArray::from_slice_f32(&data, &[1, 3, 4]).unwrap();

        let out = slice(&a, &[0, 1, 0], &[1, 3, 4], &[1, 1, 1], None).unwrap();

        assert_eq!(out.shape(), vec![1, 2, 4]);
        let got = out.to_vec_f32().unwrap();
        assert_close(
            &got,
            &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            MATMUL_TOL,
            "slice keeps the requested middle-axis range",
        );
    }

    #[test]
    fn slice_update_overwrites_static_range() {
        let src = MlxArray::from_slice_f32(&[0.0; 6], &[1, 3, 2]).unwrap();
        let update = MlxArray::from_slice_f32(&[9.0, 8.0], &[1, 1, 2]).unwrap();

        let out = slice_update(&src, &update, &[0, 1, 0], &[1, 2, 2], &[1, 1, 1], None).unwrap();

        assert_eq!(out.shape(), vec![1, 3, 2]);
        let got = out.to_vec_f32().unwrap();
        assert_close(
            &got,
            &[0.0, 0.0, 9.0, 8.0, 0.0, 0.0],
            MATMUL_TOL,
            "slice_update writes only the requested region",
        );
    }

    #[test]
    fn concat_along_axis_0() {
        let a = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = MlxArray::from_slice_f32(&[5.0, 6.0], &[1, 2]).unwrap();
        let out = concat(&[&a, &b], 0, None).unwrap();
        assert_eq!(out.shape(), vec![3, 2]);
        let got = out.to_vec_f32().unwrap();
        assert_close(
            &got,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            MATMUL_TOL,
            "concat([2x2, 1x2], axis=0)",
        );
    }

    #[test]
    fn concat_rejects_empty() {
        let err = concat(&[], 0, None).unwrap_err();
        assert!(matches!(err, MlxError::InvalidShape { .. }));
    }

    /// RoPE reference (non-traditional / half-split variant, as used by
    /// LLaMA / Qwen / Gemma):
    ///
    /// ```text
    /// Given head_dim=4, dims=4, traditional=false, base=10000, scale=1,
    /// offset=0, we split each head-vector in half as (x0..x_{d/2-1},
    /// x_{d/2}..x_{d-1}) and apply
    ///   x'_{i}     = x_i * cos(m * inv_freq[i])     - x_{i+d/2} * sin(m * inv_freq[i])
    ///   x'_{i+d/2} = x_i * sin(m * inv_freq[i])     + x_{i+d/2} * cos(m * inv_freq[i])
    /// where m is the token position and inv_freq[i] = base^{-2i/d}.
    /// ```
    ///
    /// We verify the first two positions (m=0 and m=1) for d=4 and base=10000.
    #[test]
    fn rope_matches_formula_reference() {
        // Batch=1, heads=1, seq=2, dim=4 — shape [1, 1, 2, 4].
        // Position 0:       x0 = [1, 2, 3, 4]
        // Position 1:       x1 = [5, 6, 7, 8]
        let x_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = MlxArray::from_slice_f32(&x_data, &[1, 1, 2, 4]).unwrap();

        // Reference:
        //   dim = 4, so half = 2. inv_freq = [1.0, 10000^{-2/4}] = [1.0, 0.01].
        //   m=0: cos=[1,1], sin=[0,0] -> x unchanged = [1, 2, 3, 4]
        //   m=1: c = [cos(1), cos(0.01)] = [0.5403023, 0.9999500]
        //        s = [sin(1), sin(0.01)] = [0.8414710, 0.0099998]
        //     x1' = [x0*c0 - x2*s0,  x1*c1 - x3*s1,
        //            x0*s0 + x2*c0,  x1*s1 + x3*c1]
        //         = [5*0.5403023 - 7*0.8414710,
        //            6*0.9999500 - 8*0.0099998,
        //            5*0.8414710 + 7*0.5403023,
        //            6*0.0099998 + 8*0.9999500]
        //         = [-3.188812,  5.919703, 7.989408,  8.059617]
        let expected = [
            1.0f32, 2.0, 3.0, 4.0, // m=0 unchanged
            -3.188_812, 5.919_703, 7.989_408, 8.059_617, // m=1
        ];

        let y = rope(&x, 4, false, Some(10_000.0), 1.0, 0, None, None).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 2, 4]);
        let got = y.to_vec_f32().unwrap();
        // PRD asks for 1e-4 tolerance against the PyTorch reference.
        assert_close(&got, &expected, 1.0e-4, "rope vs formula reference");
    }

    /// Sanity SDPA test: reproduce `softmax(Q @ K^T / sqrt(d)) @ V` by hand
    /// on a tiny shape and verify our fused path agrees.
    ///
    /// Shape: batch=1, heads=1, seq=2, d=2 (head_dim).
    #[test]
    fn sdpa_matches_naive_reference() {
        // Q, K, V all [1, 1, 2, 2].
        let q_data = [1.0f32, 0.0, 0.0, 1.0];
        let k_data = [1.0f32, 0.0, 0.0, 1.0];
        let v_data = [1.0f32, 2.0, 3.0, 4.0];

        let q = MlxArray::from_slice_f32(&q_data, &[1, 1, 2, 2]).unwrap();
        let k = MlxArray::from_slice_f32(&k_data, &[1, 1, 2, 2]).unwrap();
        let v = MlxArray::from_slice_f32(&v_data, &[1, 1, 2, 2]).unwrap();

        let scale = 1.0f32 / (2.0f32.sqrt());

        // Naive reference path: Q @ K^T -> softmax(axis=-1) -> @ V.
        // Reshape Q, K, V down to [2, 2] for matmul (single batch, single
        // head). We use the xybrid-mlx ops themselves as the reference so
        // we catch drift in our own matmul/softmax alongside SDPA.
        let q2 = reshape(&q, &[2, 2], None).unwrap();
        let k2 = reshape(&k, &[2, 2], None).unwrap();
        let v2 = reshape(&v, &[2, 2], None).unwrap();
        let k2t = transpose(&k2, &[1, 0], None).unwrap();
        let qk = matmul(&q2, &k2t, None).unwrap();
        let scale_arr = MlxArray::from_slice_f32(&[scale], &[1]).unwrap();
        let qk_scaled = crate::ops::mul(&qk, &scale_arr, None).unwrap();
        let probs = softmax(&qk_scaled, -1, None).unwrap();
        let expected_arr = matmul(&probs, &v2, None).unwrap();
        let expected = expected_arr.to_vec_f32().unwrap();

        let out = scaled_dot_product_attention(&q, &k, &v, scale, false, None, None).unwrap();
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);
        let got = out.to_vec_f32().unwrap();

        // SDPA's fused kernel and the unfused reference path both run on
        // f32 CPU here, so the agreement should be tight — use
        // `MATMUL_TOL * 2` for the extra softmax accumulation.
        assert_close(
            &got,
            &expected,
            MATMUL_TOL * 2.0,
            "sdpa vs naive softmax(QK^T/sqrt(d))@V",
        );
    }

    #[test]
    fn sdpa_rejects_causal_with_mask() {
        let q = MlxArray::from_slice_f32(&[0.0; 4], &[1, 1, 2, 2]).unwrap();
        let k = MlxArray::from_slice_f32(&[0.0; 4], &[1, 1, 2, 2]).unwrap();
        let v = MlxArray::from_slice_f32(&[0.0; 4], &[1, 1, 2, 2]).unwrap();
        let mask = MlxArray::from_slice_f32(&[0.0; 4], &[1, 1, 2, 2]).unwrap();
        let err =
            scaled_dot_product_attention(&q, &k, &v, 1.0, true, Some(&mask), None).unwrap_err();
        assert!(matches!(err, MlxError::Internal(_)));
    }

    /// PRD acceptance: `eval()` on the resulting array must be Send — i.e.
    /// MlxArray can be moved into a `tokio::task::spawn_blocking` closure.
    /// We verify this statically at compile time without depending on
    /// tokio: the static-assertion helper below fails to compile if
    /// MlxArray is not Send.
    #[test]
    fn mlx_array_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MlxArray>();
        // Cast to i32 so the dtype-tagging path is exercised on the value
        // being moved. No functional check — we only need Drop to run on a
        // different thread to prove real Send semantics.
        let a = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = reshape(&a, &[4], None).unwrap();
        // Move `b` into a spawned thread and drop it there.
        std::thread::spawn(move || {
            let _ = b.dtype().unwrap();
            assert_eq!(b.dtype().unwrap(), MlxDtype::F32);
            drop(b);
        })
        .join()
        .unwrap();
    }
}
