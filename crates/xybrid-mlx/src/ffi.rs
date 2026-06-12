//! `pub(crate)` FFI helpers over `mlx_c_sys::bindings`.
//!
//! Every `unsafe` block in `xybrid-mlx` lives in this module. The public
//! surface in `lib.rs` exposes only safe wrappers. Each helper here carries
//! a `# Safety` doc comment describing the invariants the caller must
//! uphold; the safe wrappers in [`crate::array`] and [`crate::stream`] are
//! responsible for proving those invariants hold.
//!
//! We keep the helpers small and purpose-specific rather than re-exporting
//! the whole binding set: callers should not need to reach into
//! `mlx_c_sys::bindings` directly.

use mlx_c_sys::bindings::{
    self as sys, mlx_array, mlx_dtype, mlx_optional_dtype, mlx_optional_float, mlx_optional_int,
    mlx_stream, mlx_vector_array,
};

use crate::error::MlxError;

/// Translate an mlx-c nonzero return into an [`MlxError`].
///
/// mlx-c's convention is "0 = success, nonzero = error". The underlying
/// detail is fetched via the companion error surface in later stories; this
/// wrapper preserves the nonzero code so callers get a debuggable signal.
pub(crate) fn check_rc(rc: i32, ctx: &'static str) -> Result<(), MlxError> {
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Internal(format!(
            "mlx-c `{ctx}` returned rc={rc}"
        )))
    }
}

/// Create a fresh default CPU stream handle.
///
/// # Safety
///
/// Safe to call from any thread at any time. The returned handle owns
/// exactly one reference to the underlying stream and must be freed via
/// [`stream_free`] exactly once.
pub(crate) unsafe fn default_cpu_stream() -> mlx_stream {
    sys::mlx_default_cpu_stream_new()
}

/// Create a fresh default GPU stream handle.
///
/// # Safety
///
/// Same contract as [`default_cpu_stream`]. Requires the Metal runtime to
/// be reachable at call time — on Apple hardware that is always true.
pub(crate) unsafe fn default_gpu_stream() -> mlx_stream {
    sys::mlx_default_gpu_stream_new()
}

/// Retain-copy a stream handle.
///
/// Implements the `Clone` semantics for [`crate::MlxStream`] in terms of
/// mlx-c's `mlx_stream_set`: construct a fresh empty handle, then bump the
/// underlying refcount via `set`.
///
/// # Safety
///
/// `src` must be a live stream handle. The returned handle is owned by the
/// caller and must be freed via [`stream_free`].
pub(crate) unsafe fn stream_clone(src: mlx_stream) -> Result<mlx_stream, MlxError> {
    let mut dst = sys::mlx_stream_new();
    let rc = sys::mlx_stream_set(&mut dst, src);
    if rc != 0 {
        // The empty handle returned by `mlx_stream_new` is still safe to
        // free on failure — it owns nothing beyond the empty descriptor.
        let _ = sys::mlx_stream_free(dst);
        return Err(MlxError::Internal(format!(
            "mlx-c `mlx_stream_set` returned rc={rc}"
        )));
    }
    Ok(dst)
}

/// Drop one reference from a stream handle.
///
/// # Safety
///
/// `stream` must have been produced by one of the mlx-c stream
/// constructors (or by [`stream_clone`]). It must not be used after this
/// call. Any nonzero return is tracing-logged and ignored — drop glue
/// cannot surface a `Result`.
pub(crate) unsafe fn stream_free(stream: mlx_stream) {
    let rc = sys::mlx_stream_free(stream);
    if rc != 0 {
        tracing::warn!(rc, "mlx_stream_free returned nonzero");
    }
}

/// Retain-copy an array handle.
///
/// # Safety
///
/// `src` must be a live array handle. The returned handle is owned by the
/// caller and must be freed via [`array_free`].
pub(crate) unsafe fn array_clone(src: mlx_array) -> Result<mlx_array, MlxError> {
    let mut dst = sys::mlx_array_new();
    let rc = sys::mlx_array_set(&mut dst, src);
    if rc != 0 {
        let _ = sys::mlx_array_free(dst);
        return Err(MlxError::Internal(format!(
            "mlx-c `mlx_array_set` returned rc={rc}"
        )));
    }
    Ok(dst)
}

/// Drop one reference from an array handle.
///
/// # Safety
///
/// `arr` must have been produced by an mlx-c array constructor (or by
/// [`array_clone`]). It must not be used after this call.
pub(crate) unsafe fn array_free(arr: mlx_array) {
    let rc = sys::mlx_array_free(arr);
    if rc != 0 {
        tracing::warn!(rc, "mlx_array_free returned nonzero");
    }
}

/// Read the dtype tag of an array.
///
/// # Safety
///
/// `arr` must be a live array handle.
pub(crate) unsafe fn array_dtype(arr: mlx_array) -> mlx_dtype {
    sys::mlx_array_dtype(arr)
}

/// Read the rank (number of dimensions) of an array.
///
/// # Safety
///
/// `arr` must be a live array handle.
pub(crate) unsafe fn array_ndim(arr: mlx_array) -> usize {
    sys::mlx_array_ndim(arr)
}

/// Copy an array's shape into an owned vector.
///
/// # Safety
///
/// `arr` must be a live array handle. We immediately copy the returned C
/// pointer into a `Vec<i32>` so the caller never holds a dangling slice
/// across further FFI calls.
pub(crate) unsafe fn array_shape(arr: mlx_array) -> Vec<i32> {
    let ndim = sys::mlx_array_ndim(arr);
    if ndim == 0 {
        return Vec::new();
    }
    let raw = sys::mlx_array_shape(arr);
    if raw.is_null() {
        return Vec::new();
    }
    let slice = std::slice::from_raw_parts(raw, ndim);
    slice.to_vec()
}

/// Read the total element count of an array (product of shape).
///
/// # Safety
///
/// `arr` must be a live array handle.
pub(crate) unsafe fn array_size(arr: mlx_array) -> usize {
    sys::mlx_array_size(arr)
}

fn shape_rank(shape: &[i32]) -> i32 {
    i32::try_from(shape.len()).expect("MLX shape rank exceeds i32::MAX")
}

/// Construct an array from a host-side f32 slice via `mlx_array_new_data`.
/// The underlying C call copies the buffer, so the caller does not need to
/// keep `data` alive beyond the call.
///
/// # Safety
///
/// - `data` must point to at least `product(shape) * sizeof(f32)` valid f32
///   values.
/// - `shape` must be a valid slice of i32 dimension sizes and its rank must
///   fit in mlx-c's i32 rank parameter.
/// - Returned handle is owned by the caller and must be freed via
///   [`array_free`] exactly once.
pub(crate) unsafe fn array_new_data_f32(data: *const f32, shape: &[i32]) -> mlx_array {
    sys::mlx_array_new_data(
        data.cast::<std::ffi::c_void>(),
        shape.as_ptr(),
        shape_rank(shape),
        sys::mlx_dtype__MLX_FLOAT32,
    )
}

/// Construct an array from a host-side i32 slice via `mlx_array_new_data`.
///
/// # Safety
///
/// Same contract as [`array_new_data_f32`] but for `i32` elements.
pub(crate) unsafe fn array_new_data_i32(data: *const i32, shape: &[i32]) -> mlx_array {
    sys::mlx_array_new_data(
        data.cast::<std::ffi::c_void>(),
        shape.as_ptr(),
        shape_rank(shape),
        sys::mlx_dtype__MLX_INT32,
    )
}

/// Construct an array from a host-side i64 slice via `mlx_array_new_data`.
///
/// # Safety
///
/// Same contract as [`array_new_data_f32`] but for `i64` elements.
pub(crate) unsafe fn array_new_data_i64(data: *const i64, shape: &[i32]) -> mlx_array {
    sys::mlx_array_new_data(
        data.cast::<std::ffi::c_void>(),
        shape.as_ptr(),
        shape_rank(shape),
        sys::mlx_dtype__MLX_INT64,
    )
}

/// Construct an array from a host-side u64 slice via `mlx_array_new_data`.
///
/// # Safety
///
/// Same contract as [`array_new_data_f32`] but for `u64` elements.
pub(crate) unsafe fn array_new_data_u64(data: *const u64, shape: &[i32]) -> mlx_array {
    sys::mlx_array_new_data(
        data.cast::<std::ffi::c_void>(),
        shape.as_ptr(),
        shape_rank(shape),
        sys::mlx_dtype__MLX_UINT64,
    )
}

/// Construct an array from raw bytes interpreted as `dtype`.
///
/// # Safety
///
/// Same contract as [`array_new_data_f32`]: `data` must point to at least
/// `product(shape) * dtype.size()` valid bytes. The underlying C call copies
/// the buffer, so the caller does not need to keep it alive beyond the call.
pub(crate) unsafe fn array_new_data_raw(
    data: *const u8,
    shape: &[i32],
    dtype: mlx_dtype,
) -> mlx_array {
    sys::mlx_array_new_data(
        data.cast::<std::ffi::c_void>(),
        shape.as_ptr(),
        shape_rank(shape),
        dtype,
    )
}

/// Evaluate an array (materialise any pending compute graph).
///
/// # Safety
///
/// `arr` must be a live array handle.
pub(crate) unsafe fn array_eval(arr: mlx_array) -> Result<(), MlxError> {
    check_rc(sys::mlx_array_eval(arr), "mlx_array_eval")
}

/// Asynchronously schedule evaluation of a vector of arrays.
///
/// # Safety
///
/// `outputs` must be a live vector handle whose arrays remain live for the
/// duration of the call. Ownership of the vector is not consumed.
pub(crate) unsafe fn async_eval(outputs: mlx_vector_array) -> Result<(), MlxError> {
    check_rc(sys::mlx_async_eval(outputs), "mlx_async_eval")
}

/// Clear MLX's process-local allocation/cache pool.
///
/// # Safety
///
/// Safe to call when no caller relies on cached allocations remaining alive.
pub(crate) unsafe fn clear_cache() -> Result<(), MlxError> {
    check_rc(sys::mlx_clear_cache(), "mlx_clear_cache")
}

/// Set MLX's process-local allocation cache limit.
///
/// Returns the previous limit reported by MLX.
///
/// # Safety
///
/// This mutates MLX process-local allocator state. Callers must coordinate
/// policy at the application/runtime level.
pub(crate) unsafe fn set_cache_limit(limit: usize) -> Result<usize, MlxError> {
    let mut previous = 0usize;
    check_rc(
        sys::mlx_set_cache_limit(&mut previous, limit),
        "mlx_set_cache_limit",
    )?;
    Ok(previous)
}

/// Read the linked MLX library's version string (e.g. `"0.31.1"`).
///
/// # Safety
///
/// Safe to call from any thread at any time. The temporary `mlx_string`
/// handle is created and freed within this function; the returned `String`
/// owns a copy of the data.
pub(crate) unsafe fn version() -> Result<String, MlxError> {
    let mut handle = sys::mlx_string_new();
    let result = check_rc(sys::mlx_version(&mut handle), "mlx_version").and_then(|()| {
        let data = sys::mlx_string_data(handle);
        if data.is_null() {
            Err(MlxError::Internal(
                "mlx_string_data returned NULL for mlx_version".into(),
            ))
        } else {
            Ok(std::ffi::CStr::from_ptr(data)
                .to_string_lossy()
                .into_owned())
        }
    });
    sys::mlx_string_free(handle);
    result
}

/// Copy an evaluated array's f32 data into an owned `Vec<f32>`.
///
/// # Safety
///
/// `arr` must be a live array handle. The array must have dtype f32 and
/// must have been evaluated; `mlx_array_data_float32` returns NULL otherwise.
pub(crate) unsafe fn array_data_f32(arr: mlx_array) -> Result<Vec<f32>, MlxError> {
    let ptr = sys::mlx_array_data_float32(arr);
    if ptr.is_null() {
        return Err(MlxError::Internal(
            "mlx_array_data_float32 returned null (array unevaluated or wrong dtype)".into(),
        ));
    }
    let n = sys::mlx_array_size(arr);
    Ok(std::slice::from_raw_parts(ptr, n).to_vec())
}

/// Copy an evaluated array's u8 data into an owned `Vec<u8>`.
///
/// # Safety
///
/// `arr` must be a live array handle. The array must have dtype u8 and
/// must have been evaluated; `mlx_array_data_uint8` returns NULL otherwise.
pub(crate) unsafe fn array_data_u8(arr: mlx_array) -> Result<Vec<u8>, MlxError> {
    let ptr = sys::mlx_array_data_uint8(arr);
    if ptr.is_null() {
        return Err(MlxError::Internal(
            "mlx_array_data_uint8 returned null (array unevaluated or wrong dtype)".into(),
        ));
    }
    let n = sys::mlx_array_size(arr);
    Ok(std::slice::from_raw_parts(ptr, n).to_vec())
}

/// Construct an `mlx_array` that reuses an externally-owned buffer, with a
/// caller-provided destructor run exactly once when MLX drops the last
/// reference.
///
/// This is the primitive behind [`crate::array::MlxArray::from_shared_buffer`]
/// — MLX gets the buffer's data pointer, and the destructor is responsible
/// for releasing the backing allocation (typically a [`crate::SharedBuffer`]
/// wrapped in a `Box`).
///
/// # Safety
///
/// - `data` must point to at least `product(shape) * dtype.size()` valid
///   bytes for the duration of the array's lifetime — which, since MLX
///   drives the lifetime, means "until `dtor(payload)` fires".
/// - `shape`'s rank must fit in mlx-c's i32 rank parameter.
/// - `payload` is forwarded to `dtor` verbatim. Common use: the Rust
///   `Box::into_raw(Box::new(shared_buffer))` pointer, with `dtor`
///   performing `drop(Box::from_raw(payload as *mut SharedBuffer))`.
/// - `dtor` must be safe to call exactly once, from any thread, after
///   MLX has finished using the array.
/// - Returned handle is owned by the caller and must be freed via
///   [`array_free`] exactly once.
pub(crate) unsafe fn array_new_data_managed_payload(
    data: *mut std::ffi::c_void,
    shape: &[i32],
    dtype: mlx_dtype,
    payload: *mut std::ffi::c_void,
    dtor: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
) -> mlx_array {
    sys::mlx_array_new_data_managed_payload(
        data,
        shape.as_ptr(),
        shape_rank(shape),
        dtype,
        payload,
        dtor,
    )
}

/// Dispatch `mlx_matmul(res, a, b, s)`.
///
/// # Safety
///
/// `a`, `b`, `s` must be live handles. The returned handle is owned by the
/// caller and must be freed via [`array_free`] exactly once.
pub(crate) unsafe fn op_matmul(
    a: mlx_array,
    b: mlx_array,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_matmul(&mut res, a, b, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_matmul rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_add(res, a, b, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_add(
    a: mlx_array,
    b: mlx_array,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_add(&mut res, a, b, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_add rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_subtract(res, a, b, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_sub(
    a: mlx_array,
    b: mlx_array,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_subtract(&mut res, a, b, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_subtract rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_multiply(res, a, b, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_mul(
    a: mlx_array,
    b: mlx_array,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_multiply(&mut res, a, b, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_multiply rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_divide(res, a, b, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_div(
    a: mlx_array,
    b: mlx_array,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_divide(&mut res, a, b, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_divide rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_zeros(res, shape, shape_num, dtype, s)`.
///
/// # Safety
///
/// `s` must be a live stream handle. `shape` must describe a valid MLX shape.
/// The returned handle is owned by the caller and must be freed via
/// [`array_free`] exactly once.
pub(crate) unsafe fn op_zeros(
    shape: &[i32],
    dtype: mlx_dtype,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_zeros(&mut res, shape.as_ptr(), shape.len(), dtype, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_zeros rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_negative(res, a, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_neg(a: mlx_array, s: mlx_stream) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_negative(&mut res, a, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_negative rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_square(res, a, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_square(a: mlx_array, s: mlx_stream) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_square(&mut res, a, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_square rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_erf(res, a, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_erf(a: mlx_array, s: mlx_stream) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_erf(&mut res, a, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_erf rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_sqrt(res, a, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_sqrt(a: mlx_array, s: mlx_stream) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_sqrt(&mut res, a, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_sqrt rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_tanh(res, a, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_tanh(a: mlx_array, s: mlx_stream) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_tanh(&mut res, a, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_tanh rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_softmax_axis(res, a, axis, precise=true, s)`.
///
/// `precise=true` matches PyTorch's default numerical behaviour (promote to
/// f32 for the intermediate max/exp/sum even when the input is f16/bf16).
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_softmax_axis(
    a: mlx_array,
    axis: i32,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_softmax_axis(&mut res, a, axis, /* precise */ true, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_softmax_axis rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_fast_rms_norm(res, x, weight_or_null, eps, s)`.
///
/// Pass a zero-initialised `mlx_array` (`mlx_array_new()` style, but without
/// ownership) as `weight` to request the unweighted variant — mlx-c checks
/// for a null `ctx`. The safe wrapper in `ops/basic.rs` constructs that
/// sentinel and is responsible for never freeing it.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_fast_rms_norm(
    x: mlx_array,
    weight: mlx_array,
    eps: f32,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_fast_rms_norm(&mut res, x, weight, eps, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_fast_rms_norm rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_fast_layer_norm(res, x, weight, bias, eps, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`]. `weight` and `bias` must be live array
/// handles with shapes broadcastable to the last dimension of `x`.
pub(crate) unsafe fn op_fast_layer_norm(
    x: mlx_array,
    weight: mlx_array,
    bias: mlx_array,
    eps: f32,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_fast_layer_norm(&mut res, x, weight, bias, eps, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_fast_layer_norm rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_astype(res, a, dtype, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_astype(
    a: mlx_array,
    dtype: mlx_dtype,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_astype(&mut res, a, dtype, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_astype rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_contiguous(res, a, allow_col_major=false, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_contiguous(
    a: mlx_array,
    allow_col_major: bool,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_contiguous(&mut res, a, allow_col_major, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_contiguous rc={rc}")));
    }
    Ok(res)
}

/// Construct a sentinel null `mlx_array` (ctx = null) without allocating.
///
/// Used to pass the "no weight" option to [`op_fast_rms_norm`] — mlx-c
/// detects the null ctx and skips the weight multiply. The returned value
/// must **not** be freed (there is no underlying handle to free).
///
/// # Safety
///
/// The caller must not call `mlx_array_free` on the returned value. Using
/// it as input to any op other than ones that explicitly accept a null
/// array (see mlx-c comments `/* may be null */`) is undefined.
pub(crate) unsafe fn array_null() -> mlx_array {
    mlx_array {
        ctx: std::ptr::null_mut(),
    }
}

/// Dispatch `mlx_reshape(res, a, shape, shape_num, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`]. `shape.as_ptr()` must remain valid for the
/// duration of the call (we pass the Rust slice's pointer through).
pub(crate) unsafe fn op_reshape(
    a: mlx_array,
    shape: &[i32],
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_reshape(&mut res, a, shape.as_ptr(), shape.len(), s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_reshape rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_transpose_axes(res, a, axes, axes_num, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_transpose_axes(
    a: mlx_array,
    axes: &[i32],
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_transpose_axes(&mut res, a, axes.as_ptr(), axes.len(), s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_transpose_axes rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_transpose(res, a, s)` — the default "reverse all axes"
/// variant used when no explicit permutation is supplied.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_transpose_default(
    a: mlx_array,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_transpose(&mut res, a, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_transpose rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_take_axis(res, a, indices, axis, s)`.
///
/// This is mlx-c's single-axis gather — equivalent to `torch.index_select` or
/// numpy's `take`. Returns rows / slices of `a` picked by `indices` along
/// `axis`, with `indices.shape` broadcasting into the gathered dim.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_take_axis(
    a: mlx_array,
    indices: mlx_array,
    axis: i32,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_take_axis(&mut res, a, indices, axis, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_take_axis rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_slice(res, a, start, stop, strides, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`]. The start/stop/stride pointers must stay
/// valid for the duration of the call.
pub(crate) unsafe fn op_slice(
    a: mlx_array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_slice(
        &mut res,
        a,
        start.as_ptr(),
        start.len(),
        stop.as_ptr(),
        stop.len(),
        strides.as_ptr(),
        strides.len(),
        s,
    );
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_slice rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_slice_update(res, src, update, start, stop, strides, s)`.
///
/// # Safety
///
/// Same contract as [`op_slice`]. `src` and `update` must be live arrays.
pub(crate) unsafe fn op_slice_update(
    src: mlx_array,
    update: mlx_array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_slice_update(
        &mut res,
        src,
        update,
        start.as_ptr(),
        start.len(),
        stop.as_ptr(),
        stop.len(),
        strides.as_ptr(),
        strides.len(),
        s,
    );
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_slice_update rc={rc}")));
    }
    Ok(res)
}

/// Build an owned `mlx_vector_array` from an array of `mlx_array` handles.
///
/// The vector bumps the refcount of each underlying array (mlx-c's
/// `mlx_vector_array_new_data` copies handles into a new `std::vector<array>`
/// which retains each element). The returned vector is owned by the caller
/// and must be freed via [`vector_array_free`] exactly once.
///
/// # Safety
///
/// `arrays` must be a slice of live `mlx_array` handles. Each handle remains
/// owned by its original `MlxArray`; this helper does not consume ownership.
pub(crate) unsafe fn vector_array_from(arrays: &[mlx_array]) -> mlx_vector_array {
    sys::mlx_vector_array_new_data(arrays.as_ptr(), arrays.len())
}

/// Drop one reference from a vector-of-array handle.
///
/// # Safety
///
/// `vec` must have been produced by [`vector_array_from`] or another mlx-c
/// vector constructor.
pub(crate) unsafe fn vector_array_free(vec: mlx_vector_array) {
    let rc = sys::mlx_vector_array_free(vec);
    if rc != 0 {
        tracing::warn!(rc, "mlx_vector_array_free returned nonzero");
    }
}

/// Return the number of arrays held by an mlx-c vector.
///
/// # Safety
///
/// `vec` must be a live vector handle.
#[cfg(test)]
pub(crate) unsafe fn vector_array_len(vec: mlx_vector_array) -> usize {
    sys::mlx_vector_array_size(vec)
}

/// Retain-copy one array from an mlx-c vector.
///
/// # Safety
///
/// `vec` must be a live vector handle and `idx` must be in bounds. The
/// returned array handle is owned by the caller and must be freed via
/// [`array_free`].
#[cfg(test)]
unsafe fn vector_array_get_owned(vec: mlx_vector_array, idx: usize) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_vector_array_get(&mut res, vec, idx);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!(
            "mlx_vector_array_get idx={idx} rc={rc}"
        )));
    }
    Ok(res)
}

/// Dispatch `mlx_concatenate_axis(res, arrays, axis, s)`.
///
/// Concatenates a list of arrays along `axis`. All arrays must have matching
/// shapes in every dim except `axis`.
///
/// # Safety
///
/// Same contract as [`op_matmul`]. `arrays` is a borrowed vector handle —
/// freeing it is the caller's responsibility (we do not consume it).
pub(crate) unsafe fn op_concat_axis(
    arrays: mlx_vector_array,
    axis: i32,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_concatenate_axis(&mut res, arrays, axis, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_concatenate_axis rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_conv1d(res, input, weight, stride, padding, dilation, groups, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_conv1d(
    input: mlx_array,
    weight: mlx_array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_conv1d(
        &mut res, input, weight, stride, padding, dilation, groups, s,
    );
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_conv1d rc={rc}")));
    }
    Ok(res)
}

/// Construct an `mlx_optional_float` (the mlx-c "maybe-float" struct used by
/// [`op_fast_rope`] for the base frequency).
pub(crate) fn optional_float(v: Option<f32>) -> mlx_optional_float {
    match v {
        Some(value) => mlx_optional_float {
            value,
            has_value: true,
        },
        None => mlx_optional_float {
            value: 0.0,
            has_value: false,
        },
    }
}

/// Construct an `mlx_optional_int`.
pub(crate) fn optional_int(v: Option<i32>) -> mlx_optional_int {
    match v {
        Some(value) => mlx_optional_int {
            value,
            has_value: true,
        },
        None => mlx_optional_int {
            value: 0,
            has_value: false,
        },
    }
}

/// Construct an `mlx_optional_dtype`.
pub(crate) fn optional_dtype(v: Option<mlx_dtype>) -> mlx_optional_dtype {
    match v {
        Some(value) => mlx_optional_dtype {
            value,
            has_value: true,
        },
        None => mlx_optional_dtype {
            value: 0,
            has_value: false,
        },
    }
}

/// Dispatch `mlx_quantized_matmul(res, x, w, scales, biases, transpose,
/// group_size, bits, mode, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`]. `mode` must be a valid nul-terminated
/// string for the duration of the call. `biases` may be the null sentinel
/// from [`array_null`].
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn op_quantized_matmul(
    x: mlx_array,
    w: mlx_array,
    scales: mlx_array,
    biases: mlx_array,
    transpose: bool,
    group_size: mlx_optional_int,
    bits: mlx_optional_int,
    mode: &std::ffi::CStr,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_quantized_matmul(
        &mut res,
        x,
        w,
        scales,
        biases,
        transpose,
        group_size,
        bits,
        mode.as_ptr(),
        s,
    );
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_quantized_matmul rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_dequantize(res, w, scales, biases, group_size, bits, mode,
/// global_scale, dtype, s)`.
///
/// # Safety
///
/// Same contract as [`op_quantized_matmul`]. `global_scale` and `biases` may
/// be null sentinels when absent.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn op_dequantize(
    w: mlx_array,
    scales: mlx_array,
    biases: mlx_array,
    group_size: mlx_optional_int,
    bits: mlx_optional_int,
    mode: &std::ffi::CStr,
    global_scale: mlx_array,
    dtype: mlx_optional_dtype,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_dequantize(
        &mut res,
        w,
        scales,
        biases,
        group_size,
        bits,
        mode.as_ptr(),
        global_scale,
        dtype,
        s,
    );
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_dequantize rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_quantize(res, w, group_size, bits, mode, global_scale, s)`.
///
/// # Safety
///
/// Same contract as [`op_dequantize`]. This helper is test-only because the
/// runtime loads existing packed SafeTensors and does not quantize model
/// weights itself.
#[cfg(test)]
pub(crate) unsafe fn op_quantize(
    w: mlx_array,
    group_size: mlx_optional_int,
    bits: mlx_optional_int,
    mode: &std::ffi::CStr,
    global_scale: mlx_array,
    s: mlx_stream,
) -> Result<(mlx_array, mlx_array, mlx_array), MlxError> {
    let mut outputs = sys::mlx_vector_array_new();
    let rc = sys::mlx_quantize(
        &mut outputs,
        w,
        group_size,
        bits,
        mode.as_ptr(),
        global_scale,
        s,
    );
    if rc != 0 {
        vector_array_free(outputs);
        return Err(MlxError::Internal(format!("mlx_quantize rc={rc}")));
    }

    let len = vector_array_len(outputs);
    if len != 3 {
        vector_array_free(outputs);
        return Err(MlxError::Internal(format!(
            "mlx_quantize returned {len} arrays, expected 3"
        )));
    }

    let weight = match vector_array_get_owned(outputs, 0) {
        Ok(array) => array,
        Err(err) => {
            vector_array_free(outputs);
            return Err(err);
        }
    };
    let scales = match vector_array_get_owned(outputs, 1) {
        Ok(array) => array,
        Err(err) => {
            array_free(weight);
            vector_array_free(outputs);
            return Err(err);
        }
    };
    let biases = match vector_array_get_owned(outputs, 2) {
        Ok(array) => array,
        Err(err) => {
            array_free(weight);
            array_free(scales);
            vector_array_free(outputs);
            return Err(err);
        }
    };
    vector_array_free(outputs);
    Ok((weight, scales, biases))
}

/// Dispatch `mlx_fast_rope(res, x, dims, traditional, base, scale, offset,
/// freqs_or_null, s)`.
///
/// Applies Rotary Position Embedding to `x`. `freqs` may be null (pass the
/// sentinel from [`array_null`]) to let mlx compute the default cos/sin
/// table from `base` + `scale`.
///
/// # Safety
///
/// Same contract as [`op_matmul`]. The `freqs` null-sentinel must **not** be
/// freed — it owns no underlying handle.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn op_fast_rope(
    x: mlx_array,
    dims: i32,
    traditional: bool,
    base: mlx_optional_float,
    scale: f32,
    offset: i32,
    freqs: mlx_array,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_fast_rope(
        &mut res,
        x,
        dims,
        traditional,
        base,
        scale,
        offset,
        freqs,
        s,
    );
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_fast_rope rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_sigmoid(res, a, s)` — element-wise `1 / (1 + exp(-x))`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_sigmoid(a: mlx_array, s: mlx_stream) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_sigmoid(&mut res, a, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_sigmoid rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_exp(res, a, s)` — element-wise exponential.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_exp(a: mlx_array, s: mlx_stream) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_exp(&mut res, a, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_exp rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_reciprocal(res, a, s)` — element-wise `1 / x`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_reciprocal(a: mlx_array, s: mlx_stream) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_reciprocal(&mut res, a, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_reciprocal rc={rc}")));
    }
    Ok(res)
}

/// Dispatch `mlx_argmax_axis(res, a, axis, keepdims, s)`.
///
/// # Safety
///
/// Same contract as [`op_matmul`].
pub(crate) unsafe fn op_argmax_axis(
    a: mlx_array,
    axis: i32,
    keepdims: bool,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_argmax_axis(&mut res, a, axis, keepdims, s);
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!("mlx_argmax_axis rc={rc}")));
    }
    Ok(res)
}

/// Copy an evaluated array's u32 data into an owned `Vec<u32>`. Used to read
/// back `mlx_argmax_axis` results (mlx returns `u32` indices).
///
/// # Safety
///
/// `arr` must be a live array handle with dtype u32 that has been
/// evaluated; `mlx_array_data_uint32` returns NULL otherwise.
pub(crate) unsafe fn array_data_u32(arr: mlx_array) -> Result<Vec<u32>, MlxError> {
    let ptr = sys::mlx_array_data_uint32(arr);
    if ptr.is_null() {
        return Err(MlxError::Internal(
            "mlx_array_data_uint32 returned null (array unevaluated or wrong dtype)".into(),
        ));
    }
    let n = sys::mlx_array_size(arr);
    Ok(std::slice::from_raw_parts(ptr, n).to_vec())
}

/// Copy an evaluated array's i64 data into an owned `Vec<i64>`.
///
/// # Safety
///
/// `arr` must be a live array handle with dtype i64 that has been
/// evaluated; `mlx_array_data_int64` returns NULL otherwise.
pub(crate) unsafe fn array_data_i64(arr: mlx_array) -> Result<Vec<i64>, MlxError> {
    let ptr = sys::mlx_array_data_int64(arr);
    if ptr.is_null() {
        return Err(MlxError::Internal(
            "mlx_array_data_int64 returned null (array unevaluated or wrong dtype)".into(),
        ));
    }
    let n = sys::mlx_array_size(arr);
    Ok(std::slice::from_raw_parts(ptr, n).to_vec())
}

/// Copy an evaluated array's u64 data into an owned `Vec<u64>`.
///
/// # Safety
///
/// `arr` must be a live array handle with dtype u64 that has been
/// evaluated; `mlx_array_data_uint64` returns NULL otherwise.
pub(crate) unsafe fn array_data_u64(arr: mlx_array) -> Result<Vec<u64>, MlxError> {
    let ptr = sys::mlx_array_data_uint64(arr);
    if ptr.is_null() {
        return Err(MlxError::Internal(
            "mlx_array_data_uint64 returned null (array unevaluated or wrong dtype)".into(),
        ));
    }
    let n = sys::mlx_array_size(arr);
    Ok(std::slice::from_raw_parts(ptr, n).to_vec())
}

/// Dispatch `mlx_fast_scaled_dot_product_attention(res, q, k, v, scale,
/// mask_mode, mask_or_null, sinks_or_null, s)`.
///
/// `mask_mode` is a C string — callers pass `"causal"` for the causal mask or
/// `""` (empty) for no mode. When supplying an explicit additive mask, pass
/// `mode = ""` and a non-null `mask`. `sinks` is unused by the safe wrapper
/// today and always forwarded as a null sentinel.
///
/// # Safety
///
/// Same contract as [`op_matmul`]. `mask_mode` must be a valid nul-terminated
/// C string pointer; the caller is responsible for its lifetime through the
/// call. Null sentinels for `mask_arr` / `sinks` must **not** be freed.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn op_sdpa(
    q: mlx_array,
    k: mlx_array,
    v: mlx_array,
    scale: f32,
    mask_mode: *const std::os::raw::c_char,
    mask_arr: mlx_array,
    sinks: mlx_array,
    s: mlx_stream,
) -> Result<mlx_array, MlxError> {
    let mut res = sys::mlx_array_new();
    let rc = sys::mlx_fast_scaled_dot_product_attention(
        &mut res, q, k, v, scale, mask_mode, mask_arr, sinks, s,
    );
    if rc != 0 {
        let _ = sys::mlx_array_free(res);
        return Err(MlxError::Internal(format!(
            "mlx_fast_scaled_dot_product_attention rc={rc}"
        )));
    }
    Ok(res)
}
