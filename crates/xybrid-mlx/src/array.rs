//! Safe wrapper over `mlx_array`.
//!
//! An `MlxArray` is the owning handle for a tensor living in MLX's unified
//! memory space. US-005 delivers the *shell*: RAII lifetime management,
//! shared-ownership clone, dtype/shape introspection. The actual tensor
//! constructors (from slices, from MTLBuffer, from Envelope) land in
//! US-006 / US-008 — leaving them out of this base layer keeps the review
//! surface small and means we can change the constructor API without a
//! breaking cascade.
//!
//! All `unsafe` here flows through [`crate::ffi`]; this module just enforces
//! invariants and maps return codes to [`crate::MlxError`].

use std::ffi::c_void;
use std::fmt;

use crate::buffer::SharedBuffer;
use crate::dtype::MlxDtype;
use crate::envelope::{EnvelopePayload, EnvelopeSource, OwnedEnvelopePayload};
use crate::error::{MlxError, MlxResult};
use crate::ffi;
use crate::resources::ensure_metallib_resource;
use crate::stream::MlxStream;
use mlx_c_sys::bindings::{mlx_array, mlx_dtype};

/// Owning handle to an `mlx_array`.
///
/// # Ownership
///
/// Each `MlxArray` owns exactly one reference to the underlying tensor.
/// [`Clone`] bumps the MLX-internal refcount (via `mlx_array_set`); [`Drop`]
/// releases one reference (via `mlx_array_free`).
///
/// # Invariants
///
/// - `self.raw.ctx` is non-null for the entire lifetime of the handle.
///   Constructors return [`MlxError::Internal`] rather than a handle with a
///   null `ctx`.
/// - The handle is never aliased across threads concurrently — see the
///   `Send` / `Sync` discussion below.
pub struct MlxArray {
    raw: mlx_array,
}

fn len_as_i32_saturating(len: usize) -> i32 {
    i32::try_from(len).unwrap_or(i32::MAX)
}

fn len_as_shape_dim(len: usize) -> MlxResult<i32> {
    i32::try_from(len).map_err(|_| MlxError::InvalidShape {
        expected: vec![i32::MAX],
        got: vec![i32::MAX],
    })
}

fn rank_len_as_i32(len: usize) -> MlxResult<i32> {
    i32::try_from(len).map_err(|_| MlxError::InvalidShape {
        expected: vec![i32::MAX],
        got: vec![i32::MAX],
    })
}

fn shape_element_count(shape: &[i32]) -> MlxResult<usize> {
    let _rank = rank_len_as_i32(shape.len())?;
    let mut elems = 1usize;
    for &dim in shape {
        if dim < 0 {
            return Err(MlxError::InvalidShape {
                expected: shape.to_vec(),
                got: vec![dim],
            });
        }
        elems = elems
            .checked_mul(dim as usize)
            .ok_or_else(|| MlxError::InvalidShape {
                expected: shape.to_vec(),
                got: vec![i32::MAX],
            })?;
    }
    Ok(elems)
}

impl MlxArray {
    /// Dtype of the array's elements.
    ///
    /// Returns [`MlxError::Internal`] when MLX reports a dtype outside the
    /// xybrid-supported subset (see [`MlxDtype`]). Callers that never
    /// handle exotic dtypes (bool, f64, complex64, …) can `.expect(…)` the
    /// result with confidence.
    pub fn dtype(&self) -> MlxResult<MlxDtype> {
        // SAFETY: self.raw is live per struct invariant.
        let raw = unsafe { ffi::array_dtype(self.raw) };
        MlxDtype::try_from(raw)
    }

    /// Shape of the array. Empty vector for 0-d (scalar) arrays.
    ///
    /// The returned vector is a fresh copy of MLX's internal shape buffer,
    /// so the caller can safely outlive any further FFI calls that mutate
    /// the array.
    #[must_use]
    pub fn shape(&self) -> Vec<i32> {
        // SAFETY: self.raw is live per struct invariant. `array_shape`
        // copies the shape into an owned Vec before returning.
        unsafe { ffi::array_shape(self.raw) }
    }

    /// Rank (number of dimensions). Returns 0 for scalar arrays.
    #[must_use]
    pub fn ndim(&self) -> usize {
        // SAFETY: self.raw is live per struct invariant.
        unsafe { ffi::array_ndim(self.raw) }
    }

    /// Total number of elements across all dimensions (product of shape).
    #[must_use]
    pub fn size(&self) -> usize {
        // SAFETY: self.raw is live per struct invariant.
        unsafe { ffi::array_size(self.raw) }
    }

    /// Construct an `MlxArray` from a host-side f32 slice.
    ///
    /// The data is copied into MLX-managed storage via `mlx_array_new_data`,
    /// so the input slice does not need to outlive the call. This is the
    /// intentionally-simple constructor used by tests and small examples;
    /// zero-copy Envelope ingestion lands in US-008.
    ///
    /// Returns [`MlxError::InvalidShape`] if the product of `shape` does
    /// not match `data.len()`.
    pub fn from_slice_f32(data: &[f32], shape: &[i32]) -> MlxResult<Self> {
        let expected = shape_element_count(shape)?;
        if expected != data.len() {
            return Err(MlxError::InvalidShape {
                expected: shape.to_vec(),
                got: vec![len_as_i32_saturating(data.len())],
            });
        }
        // SAFETY: we just validated `data.len() == product(shape)`, so mlx-c
        // reads exactly `product(shape)` f32 values from `data.as_ptr()`.
        // `mlx_array_new_data` copies the buffer, so `data`'s lifetime can
        // end at the end of this call.
        ensure_metallib_resource()?;
        let raw = unsafe { ffi::array_new_data_f32(data.as_ptr(), shape) };
        if raw.ctx.is_null() {
            return Err(MlxError::Internal(
                "mlx_array_new_data returned a null handle".into(),
            ));
        }
        Ok(Self { raw })
    }

    /// Construct an `MlxArray` from a host-side i32 slice.
    ///
    /// Same shape-validation semantics as [`Self::from_slice_f32`]; intended
    /// for index / position tensors (e.g. the `indices` argument to
    /// [`crate::ops::gather`] or token-id inputs).
    pub fn from_slice_i32(data: &[i32], shape: &[i32]) -> MlxResult<Self> {
        let expected = shape_element_count(shape)?;
        if expected != data.len() {
            return Err(MlxError::InvalidShape {
                expected: shape.to_vec(),
                got: vec![len_as_i32_saturating(data.len())],
            });
        }
        // SAFETY: we just validated `data.len() == product(shape)`. MLX copies
        // the buffer out of `data`, so the slice's lifetime ends at the end
        // of this call.
        ensure_metallib_resource()?;
        let raw = unsafe { ffi::array_new_data_i32(data.as_ptr(), shape) };
        if raw.ctx.is_null() {
            return Err(MlxError::Internal(
                "mlx_array_new_data returned a null handle".into(),
            ));
        }
        Ok(Self { raw })
    }

    /// Construct an `MlxArray` from a host-side i64 slice.
    ///
    /// This is the lossless token-id constructor for HuggingFace tokenizers
    /// and SafeTensors metadata paths that represent token IDs as 64-bit
    /// integers. Use [`Self::from_slice_i32`] only when the caller has already
    /// range-checked the IDs.
    pub fn from_slice_i64(data: &[i64], shape: &[i32]) -> MlxResult<Self> {
        let expected = shape_element_count(shape)?;
        if expected != data.len() {
            return Err(MlxError::InvalidShape {
                expected: shape.to_vec(),
                got: vec![len_as_i32_saturating(data.len())],
            });
        }
        // SAFETY: we just validated `data.len() == product(shape)`. MLX copies
        // the buffer out of `data`, so the slice's lifetime ends at the end
        // of this call.
        ensure_metallib_resource()?;
        let raw = unsafe { ffi::array_new_data_i64(data.as_ptr(), shape) };
        if raw.ctx.is_null() {
            return Err(MlxError::Internal(
                "mlx_array_new_data returned a null handle".into(),
            ));
        }
        Ok(Self { raw })
    }

    /// Construct an `MlxArray` from a host-side u64 slice.
    ///
    /// MLX supports uint64 arrays natively. Xybrid keeps this constructor
    /// alongside i64 so registry/tokenizer code can preserve upstream token-id
    /// widths instead of narrowing through `i32`.
    pub fn from_slice_u64(data: &[u64], shape: &[i32]) -> MlxResult<Self> {
        let expected = shape_element_count(shape)?;
        if expected != data.len() {
            return Err(MlxError::InvalidShape {
                expected: shape.to_vec(),
                got: vec![len_as_i32_saturating(data.len())],
            });
        }
        // SAFETY: we just validated `data.len() == product(shape)`. MLX copies
        // the buffer out of `data`, so the slice's lifetime ends at the end
        // of this call.
        ensure_metallib_resource()?;
        let raw = unsafe { ffi::array_new_data_u64(data.as_ptr(), shape) };
        if raw.ctx.is_null() {
            return Err(MlxError::Internal(
                "mlx_array_new_data returned a null handle".into(),
            ));
        }
        Ok(Self { raw })
    }

    /// Construct an `MlxArray` from raw bytes interpreted as `dtype`.
    ///
    /// The bytes are copied into MLX-managed storage via
    /// `mlx_array_new_data`, so the input slice does not need to outlive the
    /// call. This is primarily for file formats such as SafeTensors, where
    /// F16/BF16 payloads already exist as little-endian raw element bytes.
    pub fn from_dtype_bytes(data: &[u8], shape: &[i32], dtype: MlxDtype) -> MlxResult<Self> {
        let elems = shape_element_count(shape)?;
        if elems == 0 {
            return Err(MlxError::InvalidShape {
                expected: shape.to_vec(),
                got: vec![len_as_i32_saturating(data.len())],
            });
        }
        let expected_bytes =
            elems
                .checked_mul(dtype.size_bytes())
                .ok_or_else(|| MlxError::InvalidShape {
                    expected: shape.to_vec(),
                    got: vec![len_as_i32_saturating(data.len())],
                })?;
        if data.len() != expected_bytes {
            return Err(MlxError::InvalidShape {
                expected: shape.to_vec(),
                got: vec![len_as_i32_saturating(data.len())],
            });
        }
        // SAFETY: we just validated `data.len() == product(shape) *
        // dtype.size_bytes()`. MLX copies the buffer, so `data`'s lifetime
        // can end at the end of this call.
        ensure_metallib_resource()?;
        let raw = unsafe { ffi::array_new_data_raw(data.as_ptr(), shape, dtype.into()) };
        if raw.ctx.is_null() {
            return Err(MlxError::Internal(
                "mlx_array_new_data returned a null handle".into(),
            ));
        }
        Ok(Self { raw })
    }

    /// Force evaluation of any pending compute graph backing this array.
    ///
    /// MLX operations are lazy — `ops::matmul` etc. return arrays whose
    /// payload is not materialised until `eval` is called (or an
    /// `mlx_array_item_*` / `mlx_array_data_*` read forces it). Calling
    /// `eval` is idempotent: subsequent calls are cheap no-ops.
    pub fn eval(&self) -> MlxResult<()> {
        // SAFETY: self.raw is live per struct invariant.
        unsafe { ffi::array_eval(self.raw) }
    }

    /// Materialize this array into row-contiguous storage.
    ///
    /// MLX arrays can be lazy expressions or views with non-row-major
    /// strides. This returns a fresh contiguous array while keeping the data
    /// in MLX-managed storage.
    fn contiguous(&self, stream: Option<&MlxStream>) -> MlxResult<Self> {
        let default_stream;
        let stream = match stream {
            Some(stream) => stream,
            None => {
                default_stream = MlxStream::default_cpu()?;
                &default_stream
            }
        };
        let raw = unsafe { ffi::op_contiguous(self.raw, false, stream.as_raw())? };
        Ok(Self::from_raw(raw))
    }

    /// Materialize this array into row-contiguous storage on the provided
    /// stream.
    ///
    /// This is useful for runtime weight preprocessing, where a transposed
    /// view should become a resident matmul-ready tensor once at load time
    /// instead of being normalised on every decode step.
    pub fn contiguous_on_stream(&self, stream: Option<&MlxStream>) -> MlxResult<Self> {
        self.contiguous(stream)
    }

    /// Read the array's contents as an owned `Vec<f32>`.
    ///
    /// Evaluates the array first (so callers do not need to remember to
    /// call [`Self::eval`]). Returns [`MlxError::InvalidDtype`] if the
    /// underlying dtype is not `F32`.
    pub fn to_vec_f32(&self) -> MlxResult<Vec<f32>> {
        self.to_vec_f32_on_stream(None)
    }

    /// Read the array's contents as an owned `Vec<f32>` after making it
    /// contiguous on the caller-provided stream.
    ///
    /// Runtime model code should pass the same stream that produced the
    /// array, otherwise the final readback may force a device handoff before
    /// evaluation.
    pub fn to_vec_f32_on_stream(&self, stream: Option<&MlxStream>) -> MlxResult<Vec<f32>> {
        let dt = self.dtype()?;
        if dt != MlxDtype::F32 {
            return Err(MlxError::InvalidDtype {
                expected: MlxDtype::F32,
                got: dt,
            });
        }
        let materialized = self.contiguous(stream)?;
        materialized.eval()?;
        // SAFETY: materialized.raw is live per struct invariant; we just
        // checked dtype and evaluated so `mlx_array_data_float32` is
        // guaranteed to return a valid pointer.
        unsafe { ffi::array_data_f32(materialized.raw) }
    }

    /// Read the array's contents as an owned `Vec<u32>`.
    ///
    /// Companion to [`Self::to_vec_f32`] for reading back `mlx_argmax`
    /// results, which are `u32` indices. Returns [`MlxError::InvalidDtype`]
    /// if the underlying dtype is not `U32`.
    pub fn to_vec_u32(&self) -> MlxResult<Vec<u32>> {
        let dt = self.dtype()?;
        if dt != MlxDtype::U32 {
            return Err(MlxError::InvalidDtype {
                expected: MlxDtype::U32,
                got: dt,
            });
        }
        let materialized = self.contiguous_for_readback()?;
        materialized.eval()?;
        // SAFETY: materialized.raw is live per struct invariant; we just
        // checked dtype and evaluated so `mlx_array_data_uint32` is
        // guaranteed to return a valid pointer.
        unsafe { ffi::array_data_u32(materialized.raw) }
    }

    /// Read the array's contents as an owned `Vec<i64>`.
    ///
    /// Companion to [`Self::from_slice_i64`]. Returns
    /// [`MlxError::InvalidDtype`] if the underlying dtype is not `I64`.
    pub fn to_vec_i64(&self) -> MlxResult<Vec<i64>> {
        let dt = self.dtype()?;
        if dt != MlxDtype::I64 {
            return Err(MlxError::InvalidDtype {
                expected: MlxDtype::I64,
                got: dt,
            });
        }
        let materialized = self.contiguous_for_readback()?;
        materialized.eval()?;
        // SAFETY: materialized.raw is live per struct invariant; we just
        // checked dtype and evaluated so `mlx_array_data_int64` is
        // guaranteed to return a valid pointer.
        unsafe { ffi::array_data_i64(materialized.raw) }
    }

    /// Read the array's contents as an owned `Vec<u64>`.
    ///
    /// Companion to [`Self::from_slice_u64`]. Returns
    /// [`MlxError::InvalidDtype`] if the underlying dtype is not `U64`.
    pub fn to_vec_u64(&self) -> MlxResult<Vec<u64>> {
        let dt = self.dtype()?;
        if dt != MlxDtype::U64 {
            return Err(MlxError::InvalidDtype {
                expected: MlxDtype::U64,
                got: dt,
            });
        }
        let materialized = self.contiguous_for_readback()?;
        materialized.eval()?;
        // SAFETY: materialized.raw is live per struct invariant; we just
        // checked dtype and evaluated so `mlx_array_data_uint64` is
        // guaranteed to return a valid pointer.
        unsafe { ffi::array_data_u64(materialized.raw) }
    }

    /// Fallible clone. See the note on [`Clone`] for the panic contract of
    /// the infallible variant.
    pub fn try_clone(&self) -> MlxResult<Self> {
        // SAFETY: self.raw is live for as long as &self is borrowed.
        let raw = unsafe { ffi::array_clone(self.raw)? };
        Ok(Self { raw })
    }

    /// Construct an `MlxArray` backed by a [`SharedBuffer`] — the zero-copy
    /// path (US-008).
    ///
    /// The MTLBuffer's contents pointer is handed to MLX directly via
    /// `mlx_array_new_data_managed_payload`; the `SharedBuffer` itself is
    /// parked as the array's payload so that when MLX drops the last
    /// reference, the custom destructor reclaims the SharedBuffer (which
    /// in turn releases the MTLBuffer and frees the Rust-side heap
    /// allocation).
    ///
    /// `shape`'s product, scaled by `dtype.size_bytes()`, must equal
    /// `buf.len_bytes()`. Otherwise an [`MlxError::InvalidShape`] is
    /// returned and the SharedBuffer is dropped (freeing its backing).
    pub fn from_shared_buffer(
        buf: SharedBuffer,
        shape: &[i32],
        dtype: MlxDtype,
    ) -> MlxResult<Self> {
        // Validate geometry up front. Any nonsense here would typically
        // crash MLX with a cryptic shape mismatch inside the first op.
        let elems = shape_element_count(shape)?;
        if elems == 0 {
            return Err(MlxError::InvalidShape {
                expected: shape.to_vec(),
                got: vec![len_as_i32_saturating(buf.len_bytes())],
            });
        }
        let expected_bytes =
            elems
                .checked_mul(dtype.size_bytes())
                .ok_or_else(|| MlxError::InvalidShape {
                    expected: shape.to_vec(),
                    got: vec![len_as_i32_saturating(buf.len_bytes())],
                })?;
        if buf.len_bytes() != expected_bytes {
            // shape's observed byte count — reported as a single-element
            // "shape" for consistency with the rest of the error surface.
            return Err(MlxError::InvalidShape {
                expected: shape.to_vec(),
                got: vec![len_as_i32_saturating(buf.len_bytes())],
            });
        }
        ensure_metallib_resource()?;

        // Park the SharedBuffer behind a Box so MLX can hold a stable
        // pointer to it. `data_ptr` aliases `SharedBuffer::as_ptr()`,
        // which in turn aliases the MTLBuffer's contents().
        let data_ptr = buf.as_mut_ptr() as *mut c_void;
        let payload = Box::into_raw(Box::new(buf)) as *mut c_void;

        // SAFETY: `data_ptr` points into the SharedBuffer we parked above;
        // it stays live until `drop_shared_buffer_payload` fires, which
        // drops the Box and therefore the SharedBuffer. The dtor matches
        // the `extern "C" fn(*mut c_void)` signature mlx-c expects.
        let raw = unsafe {
            ffi::array_new_data_managed_payload(
                data_ptr,
                shape,
                mlx_dtype::from(dtype),
                payload,
                Some(drop_shared_buffer_payload),
            )
        };
        if raw.ctx.is_null() {
            // Reclaim the SharedBuffer so its backing is freed instead of
            // leaking — MLX never took ownership of the payload.
            // SAFETY: we just produced `payload` via `Box::into_raw` above
            // and the dtor has not fired.
            unsafe {
                drop(Box::from_raw(payload as *mut SharedBuffer));
            }
            return Err(MlxError::Internal(
                "mlx_array_new_data_managed_payload returned a null handle".into(),
            ));
        }
        Ok(Self { raw })
    }

    /// Build an `MlxArray` from any [`EnvelopeSource`] — typically an
    /// `xybrid_core::ir::Envelope` once the downstream impl lands.
    ///
    /// The payload slice is cloned into a `Box<[T]>` matching the element
    /// type, wrapped in a [`SharedBuffer`] with `.storageModeShared`, and
    /// then handed to MLX via [`Self::from_shared_buffer`]. The resulting
    /// `MlxArray`'s storage pointer aliases the Box the SharedBuffer took
    /// ownership of — "the original allocation" in the PRD's wording.
    ///
    /// # Returns
    ///
    /// - [`EnvelopePayload::Audio`] → 1-D `u8` tensor of length
    ///   `payload.len()`.
    /// - [`EnvelopePayload::Embedding`] → 1-D `f32` tensor of length
    ///   `payload.len()`.
    /// - [`EnvelopePayload::TokenIds`] → 1-D `i64` tensor of length
    ///   `payload.len()`.
    pub fn from_envelope<E: EnvelopeSource>(envelope: &E) -> MlxResult<Self> {
        match envelope.payload() {
            EnvelopePayload::Embedding(data) => {
                let boxed: Box<[f32]> = data.to_vec().into_boxed_slice();
                let len = len_as_shape_dim(boxed.len())?;
                let buf = SharedBuffer::from_box(boxed)?;
                Self::from_shared_buffer(buf, &[len], MlxDtype::F32)
            }
            EnvelopePayload::Audio(data) => {
                let boxed: Box<[u8]> = data.to_vec().into_boxed_slice();
                let len = len_as_shape_dim(boxed.len())?;
                let buf = SharedBuffer::from_box(boxed)?;
                Self::from_shared_buffer(buf, &[len], MlxDtype::U8)
            }
            EnvelopePayload::TokenIds(data) => {
                let len = len_as_shape_dim(data.len())?;
                Self::from_slice_i64(data, &[len])
            }
        }
    }

    /// Consume the array and return an [`OwnedEnvelopePayload`] suitable
    /// for reconstituting an `xybrid_core::ir::Envelope` upstream.
    ///
    /// # Reuse vs. copy
    ///
    /// MLX does not expose the managed payload back to the caller, so we
    /// cannot reclaim the original [`SharedBuffer`] that
    /// [`Self::from_envelope`] / [`Self::from_shared_buffer`] handed off.
    /// Every readback therefore falls through to MLX's typed data-copy
    /// accessor (`mlx_array_data_*`) — we log at `tracing::debug!` so
    /// consumers can tell when the fast path was unreachable.
    ///
    /// Supported dtypes today: `F32` → `Embedding`, `U8` → `Audio`,
    /// `I64` → `TokenIds`. Other dtypes return [`MlxError::InvalidDtype`].
    pub fn to_envelope_payload(self) -> MlxResult<OwnedEnvelopePayload> {
        let dtype = self.dtype()?;
        let materialized = self.contiguous_for_readback()?;
        materialized.eval()?;
        match dtype {
            MlxDtype::F32 => {
                // SAFETY: dtype checked F32 and eval() completed.
                let data = unsafe { ffi::array_data_f32(materialized.raw)? };
                tracing::debug!(
                    bytes = data.len() * std::mem::size_of::<f32>(),
                    "MlxArray::to_envelope_payload: copying f32 back — SharedBuffer reuse \
                     unavailable (MLX does not expose managed payloads)"
                );
                Ok(OwnedEnvelopePayload::Embedding(data))
            }
            MlxDtype::U8 => {
                // SAFETY: dtype checked U8 and eval() completed.
                let data = unsafe { ffi::array_data_u8(materialized.raw)? };
                tracing::debug!(
                    bytes = data.len(),
                    "MlxArray::to_envelope_payload: copying u8 back — SharedBuffer reuse \
                     unavailable (MLX does not expose managed payloads)"
                );
                Ok(OwnedEnvelopePayload::Audio(data))
            }
            MlxDtype::I64 => {
                // SAFETY: dtype checked I64 and eval() completed.
                let data = unsafe { ffi::array_data_i64(materialized.raw)? };
                tracing::debug!(
                    bytes = data.len() * std::mem::size_of::<i64>(),
                    "MlxArray::to_envelope_payload: copying i64 token IDs back — SharedBuffer \
                     reuse unavailable (MLX does not expose managed payloads)"
                );
                Ok(OwnedEnvelopePayload::TokenIds(data))
            }
            other => Err(MlxError::InvalidDtype {
                expected: MlxDtype::F32,
                got: other,
            }),
        }
    }

    /// Construct an `MlxArray` from a raw, already-owned mlx-c handle.
    ///
    /// This is the integration point for ops (US-006+) and conversion
    /// paths (US-008). Kept `pub(crate)` so downstream consumers cannot
    /// fabricate handles of unknown provenance.
    pub(crate) fn from_raw(raw: mlx_array) -> Self {
        debug_assert!(!raw.ctx.is_null(), "MlxArray::from_raw received null ctx");
        Self { raw }
    }

    /// Borrow the raw handle for dispatch to mlx-c ops. `pub(crate)` for
    /// the same reason as [`Self::from_raw`].
    pub(crate) fn as_raw(&self) -> mlx_array {
        self.raw
    }

    /// Materialize this array into row-contiguous storage before typed host
    /// readback. MLX view ops such as transpose can be evaluated but still
    /// point at non-contiguous backing storage; copying their raw data pointer
    /// would return storage order instead of logical row-major order.
    fn contiguous_for_readback(&self) -> MlxResult<Self> {
        self.contiguous(None)
    }
}

impl Clone for MlxArray {
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("mlx_array_set failed during MlxArray::clone — MLX runtime is in a bad state")
    }
}

impl Drop for MlxArray {
    fn drop(&mut self) {
        // SAFETY: self.raw was produced by a constructor or try_clone and
        // is not aliased — each MlxArray owns one reference.
        unsafe { ffi::array_free(self.raw) }
    }
}

impl fmt::Debug for MlxArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // We format the dtype through a helper so arrays carrying an
        // unsupported dtype (e.g. bool, f64) still Debug-print cleanly
        // without panicking inside logging macros.
        //
        // SAFETY: self.raw is live per struct invariant.
        let raw_dtype = unsafe { ffi::array_dtype(self.raw) };
        f.debug_struct("MlxArray")
            .field("dtype", &DtypeDbg(raw_dtype))
            .field("shape", &self.shape())
            .finish()
    }
}

/// Destructor that `mlx_array_new_data_managed_payload` invokes exactly
/// once when MLX drops its last reference to an array built by
/// [`MlxArray::from_shared_buffer`]. Reclaims the `Box<SharedBuffer>` so
/// its Drop impl releases the MTLBuffer and frees the heap region.
///
/// # Safety
///
/// - `payload` must be a pointer produced by
///   `Box::into_raw(Box::new(SharedBuffer))` inside
///   [`MlxArray::from_shared_buffer`]. MLX guarantees this dtor is called
///   exactly once — calling it zero or two times breaks the pair with
///   `Box::from_raw`.
/// - Called from an arbitrary thread inside MLX's dropper; `SharedBuffer`
///   is marked `Send`, which is why this is sound.
unsafe extern "C" fn drop_shared_buffer_payload(payload: *mut c_void) {
    if payload.is_null() {
        // Shouldn't happen — MLX always forwards the payload we handed it
        // — but guard so a defective MLX build doesn't segfault us.
        return;
    }
    drop(Box::from_raw(payload as *mut SharedBuffer));
}

/// Debug shim that renders a raw mlx dtype as the MlxDtype variant name
/// when supported, or `mlx_dtype(<value>)` otherwise.
struct DtypeDbg(mlx_dtype);

impl fmt::Debug for DtypeDbg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match MlxDtype::try_from(self.0) {
            Ok(dt) => write!(f, "{dt:?}"),
            Err(_) => write!(f, "mlx_dtype({})", self.0),
        }
    }
}

// MLX's tensor core is documented thread-safe: ops can be dispatched from
// any thread as long as each handle is not concurrently mutated from two
// places. We therefore mark `MlxArray: Send` but deliberately NOT `Sync` —
// `&MlxArray` only exposes inherently-const methods today, but mutable
// eval paths (US-007) will need mutable access, and we do not want a
// downstream consumer to assume `&MlxArray` is safe to share across
// threads without re-evaluating that claim.
//
// SAFETY: `mlx_array` wraps a `void*` pointing at reference-counted state
// managed by MLX itself. Ownership transfer across threads does not
// introduce aliasing beyond what mlx's internal locking already handles.
unsafe impl Send for MlxArray {}

#[cfg(test)]
mod tests {
    //! Refcount-behaviour test for [`MlxArray`].
    //!
    //! The PRD (US-005) asks us to verify that `MlxArray::drop` decrements
    //! the underlying mlx_array refcount by calling `mlx_array_get_count`.
    //! That symbol does **not** exist in mlx-c v0.6.0 — there is no public
    //! refcount getter. The equivalent observable invariant is "the
    //! payload's destructor fires exactly once, and only after the last
    //! wrapper is dropped". We verify that directly by constructing the
    //! array via `mlx_array_new_data_managed_payload` with a destructor
    //! that increments an atomic counter.
    //!
    //! This proves the two claims we actually care about:
    //!   1. `MlxArray::clone()` does **not** free the payload.
    //!   2. Dropping the last clone **does** free it, exactly once.
    //!
    //! The module is gated on the crate's `bindings` feature (via the
    //! enclosing file-level cfg in `lib.rs`), so default workspace test
    //! runs do not try to link MLX.
    use super::*;
    use mlx_c_sys::bindings as sys;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Payload handed to MLX: a single atomic counter address.
    ///
    /// We use a `&'static AtomicUsize` (cast to a raw pointer for FFI)
    /// rather than leaking a `Box<AtomicUsize>` because the counter has to
    /// outlive any thread MLX might run the dtor on, and statics in the
    /// test's scope are the simplest way to guarantee that.
    static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// Destructor that mlx-c calls when the final reference to the array
    /// is released.
    ///
    /// # Safety
    ///
    /// Called by MLX with the payload pointer passed to
    /// `mlx_array_new_data_managed_payload`. We registered `&DROP_COUNTER`
    /// as that payload, so the pointer is valid for 'static.
    extern "C" fn drop_payload(_payload: *mut std::ffi::c_void) {
        DROP_COUNTER.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn shape_element_count_rejects_negative_dim() {
        let err = shape_element_count(&[2, -1, 4]).unwrap_err();

        assert!(matches!(err, MlxError::InvalidShape { .. }));
    }

    #[test]
    fn shape_element_count_rejects_overflow() {
        let err = shape_element_count(&[i32::MAX, i32::MAX, i32::MAX]).unwrap_err();

        assert!(matches!(err, MlxError::InvalidShape { .. }));
    }

    #[test]
    fn len_as_shape_dim_rejects_lengths_over_mlx_i32_limit() {
        let err = len_as_shape_dim(i32::MAX as usize + 1).unwrap_err();

        assert!(matches!(err, MlxError::InvalidShape { .. }));
    }

    #[test]
    fn rank_len_as_i32_rejects_ranks_over_mlx_i32_limit() {
        let err = rank_len_as_i32(i32::MAX as usize + 1).unwrap_err();

        assert!(matches!(err, MlxError::InvalidShape { .. }));
    }

    #[test]
    fn from_dtype_bytes_rejects_overflow_shape_before_ffi() {
        let err = MlxArray::from_dtype_bytes(&[], &[i32::MAX, i32::MAX, i32::MAX], MlxDtype::F32)
            .unwrap_err();

        assert!(matches!(err, MlxError::InvalidShape { .. }));
    }

    #[test]
    fn clone_and_drop_track_refcount() {
        // Fresh counter — the atomic is process-global but each test run
        // gets its own instance (cargo-test runs tests in separate procs
        // by default for integration tests, but unit tests share a process,
        // so we reset explicitly).
        DROP_COUNTER.store(0, Ordering::SeqCst);

        // Allocate a tiny backing buffer on the heap. MLX will own the
        // *payload* pointer; the data pointer is the actual buffer.
        let mut data: Box<[f32]> = vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice();
        let data_ptr = data.as_mut_ptr();
        let shape: [i32; 1] = [4];

        // SAFETY: we keep `data` alive for the full scope of the test, so
        // MLX's internal pointer stays valid. The payload is
        // `&DROP_COUNTER` which is 'static. The dtor is `extern "C"` and
        // matches the signature mlx-c expects.
        let raw = unsafe {
            sys::mlx_array_new_data_managed_payload(
                data_ptr.cast::<std::ffi::c_void>(),
                shape.as_ptr(),
                rank_len_as_i32(shape.len()).unwrap(),
                sys::mlx_dtype__MLX_FLOAT32,
                std::ptr::addr_of!(DROP_COUNTER) as *mut std::ffi::c_void,
                Some(drop_payload),
            )
        };
        assert!(
            !raw.ctx.is_null(),
            "mlx_array_new_data_managed_payload returned null ctx"
        );

        let array = MlxArray::from_raw(raw);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.shape(), vec![4]);
        assert_eq!(array.size(), 4);
        assert_eq!(
            DROP_COUNTER.load(Ordering::SeqCst),
            0,
            "dtor must not fire while a reference is live"
        );

        let clone = array.clone();
        assert_eq!(
            DROP_COUNTER.load(Ordering::SeqCst),
            0,
            "Clone must not fire the payload dtor"
        );

        drop(array);
        assert_eq!(
            DROP_COUNTER.load(Ordering::SeqCst),
            0,
            "Dropping one of two references must not fire the dtor"
        );

        drop(clone);
        assert_eq!(
            DROP_COUNTER.load(Ordering::SeqCst),
            1,
            "Dropping the last reference must fire the dtor exactly once"
        );

        // `data` is dropped here; MLX already released its reference, so
        // this is a no-op on the MLX side.
        drop(data);
    }

    // ========================================================================
    // US-008: Envelope ↔ MlxArray zero-copy tests.
    // ========================================================================
    //
    // The key invariants exercised below:
    //   1. `MlxArray::from_shared_buffer` preserves the backing pointer —
    //      MLX reads from the same address the caller wrote to.
    //   2. A mutation written through the SharedBuffer's `as_mut_ptr` is
    //      visible in a subsequent MLX op (proving
    //      `.storageModeShared` semantics).
    //   3. `MlxArray::from_envelope` round-trips through
    //      `to_envelope_payload` with byte-exact values.
    //   4. `MlxArray::from_envelope` maps `EnvelopePayload::TokenIds` to
    //      an i64 MLX array without narrowing through i32.

    /// A minimal `EnvelopeSource` impl for tests — avoids pulling xybrid-core
    /// in as a dev-dep just to get the real `Envelope` type.
    struct TestEnvelope<'a> {
        payload: EnvelopePayload<'a>,
    }
    impl EnvelopeSource for TestEnvelope<'_> {
        fn payload(&self) -> EnvelopePayload<'_> {
            self.payload
        }
    }

    /// Pointer parity: the MTLBuffer-backed array reads from the same
    /// pointer the caller Box::into_raw'd.
    ///
    /// We build a Box<[f32]>, remember its element pointer, wrap it in a
    /// SharedBuffer (this consumes the box but preserves the allocation),
    /// and confirm `SharedBuffer::as_ptr` — which is what MLX will see —
    /// matches the original.
    #[test]
    fn from_shared_buffer_pointer_parity() {
        let data: Box<[f32]> = vec![1.0f32, 2.0, 3.0, 4.0].into_boxed_slice();
        let original_ptr = data.as_ptr();
        let buf = SharedBuffer::from_box(data).expect("shared buffer");
        assert_eq!(
            buf.as_ptr() as *const f32,
            original_ptr,
            "SharedBuffer::as_ptr must alias the original Box allocation — any reallocation \
             here breaks the zero-copy guarantee"
        );
        let array = MlxArray::from_shared_buffer(buf, &[4], MlxDtype::F32).expect("from_sb");
        assert_eq!(array.shape(), vec![4]);
        assert_eq!(array.dtype().expect("dtype"), MlxDtype::F32);
        // Reading back through MLX's typed accessor confirms MLX sees our
        // values (proves the data pointer is wired correctly, independent
        // of whether it was zero-copy).
        let readback = array.to_vec_f32().expect("readback");
        assert_eq!(readback, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// Shape-mismatch rejection: `from_shared_buffer` validates that the
    /// shape * dtype.size matches the buffer's byte length before handing
    /// off to MLX. On mismatch, the SharedBuffer is dropped (its backing
    /// is freed).
    #[test]
    fn from_shared_buffer_rejects_shape_mismatch() {
        let data: Box<[f32]> = vec![0.0f32; 4].into_boxed_slice();
        let buf = SharedBuffer::from_box(data).expect("shared buffer");
        // f32 × shape [2, 3] = 24 bytes, but the buffer is 16.
        let err = MlxArray::from_shared_buffer(buf, &[2, 3], MlxDtype::F32).unwrap_err();
        match err {
            MlxError::InvalidShape { expected, got } => {
                assert_eq!(expected, vec![2, 3]);
                assert_eq!(got, vec![16]);
            }
            other => panic!("expected InvalidShape, got {other:?}"),
        }
    }

    #[test]
    fn i64_token_id_constructor_round_trips() {
        let ids = [0_i64, 42, i64::from(i32::MAX) + 1];
        let array = MlxArray::from_slice_i64(&ids, &[1, 3]).expect("from i64 ids");

        assert_eq!(array.shape(), vec![1, 3]);
        assert_eq!(array.dtype().expect("dtype"), MlxDtype::I64);
        assert_eq!(array.to_vec_i64().expect("readback"), ids);
    }

    #[test]
    fn u64_token_id_constructor_round_trips() {
        let ids = [0_u64, 42, u64::from(u32::MAX) + 1];
        let array = MlxArray::from_slice_u64(&ids, &[1, 3]).expect("from u64 ids");

        assert_eq!(array.shape(), vec![1, 3]);
        assert_eq!(array.dtype().expect("dtype"), MlxDtype::U64);
        assert_eq!(array.to_vec_u64().expect("readback"), ids);
    }

    #[test]
    fn from_dtype_bytes_loads_bf16_payload() {
        let bytes = [0x80, 0x3f, 0x00, 0x40]; // 1.0, 2.0 in BF16 LE.
        let array =
            MlxArray::from_dtype_bytes(&bytes, &[2], MlxDtype::Bf16).expect("from bf16 bytes");

        assert_eq!(array.shape(), vec![2]);
        assert_eq!(array.dtype().expect("dtype"), MlxDtype::Bf16);

        let as_f32 = crate::ops::cast(&array, MlxDtype::F32, None).expect("cast bf16 to f32");
        assert_eq!(as_f32.to_vec_f32().expect("readback"), vec![1.0, 2.0]);
    }

    /// Shared-storage semantics: mutate the SharedBuffer via its pointer
    /// (simulating an external pipeline stage writing into the
    /// envelope-owned memory), then dispatch an MLX op and confirm the
    /// op reads the mutated values.
    ///
    /// This is the PRD's "mutation visible via the original Vec pointer"
    /// acceptance — because SharedBuffer's pointer IS the backing Vec's
    /// pointer after `from_box`.
    #[test]
    fn mlx_sees_shared_storage_mutations() {
        // Start with zeroes.
        let data: Box<[f32]> = vec![0.0f32; 4].into_boxed_slice();
        let buf = SharedBuffer::from_box(data).expect("shared buffer");
        let ptr = buf.as_mut_ptr() as *mut f32;

        // Mutate via the shared pointer BEFORE building the array.
        // SAFETY: we own `buf`; no other thread / MLX task has touched it.
        unsafe {
            for i in 0..4 {
                std::ptr::write(ptr.add(i), (i + 1) as f32);
            }
        }

        let array = MlxArray::from_shared_buffer(buf, &[4], MlxDtype::F32).expect("from_sb");

        // MLX reads through the same .storageModeShared region, so our
        // mutations must be visible.
        let readback = array.to_vec_f32().expect("readback");
        assert_eq!(
            readback,
            vec![1.0, 2.0, 3.0, 4.0],
            "MLX must see CPU-side writes through .storageModeShared"
        );
    }

    /// End-to-end Envelope round-trip: a borrowed-slice payload goes
    /// through `from_envelope` → `to_envelope_payload` and the bytes
    /// survive intact. The readback is a copy (MLX doesn't expose managed
    /// payloads, see the to_envelope_payload doc), but the data should
    /// round-trip exactly.
    #[test]
    fn envelope_round_trip_embedding() {
        let original = vec![1.5f32, -2.5, 3.75, 0.0, 42.0];
        let env = TestEnvelope {
            payload: EnvelopePayload::Embedding(&original),
        };
        let array = MlxArray::from_envelope(&env).expect("from_envelope");
        assert_eq!(array.shape(), vec![original.len() as i32]);
        assert_eq!(array.dtype().expect("dtype"), MlxDtype::F32);
        let back = array.to_envelope_payload().expect("to_envelope_payload");
        match back {
            OwnedEnvelopePayload::Embedding(v) => assert_eq!(v, original),
            other => panic!("expected Embedding, got {other:?}"),
        }
    }

    /// Audio round-trip: u8 payload through the envelope path.
    #[test]
    fn envelope_round_trip_audio() {
        let original: Vec<u8> = (0..=31u8).collect();
        let env = TestEnvelope {
            payload: EnvelopePayload::Audio(&original),
        };
        let array = MlxArray::from_envelope(&env).expect("from_envelope");
        assert_eq!(array.shape(), vec![original.len() as i32]);
        assert_eq!(array.dtype().expect("dtype"), MlxDtype::U8);
        let back = array.to_envelope_payload().expect("to_envelope_payload");
        match back {
            OwnedEnvelopePayload::Audio(v) => assert_eq!(v, original),
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn envelope_round_trip_token_ids() {
        let original: Vec<i64> = vec![10, 20, i64::from(i32::MAX) + 1];
        let env = TestEnvelope {
            payload: EnvelopePayload::TokenIds(&original),
        };
        let array = MlxArray::from_envelope(&env).expect("from_envelope");
        assert_eq!(array.shape(), vec![original.len() as i32]);
        assert_eq!(array.dtype().expect("dtype"), MlxDtype::I64);
        let back = array.to_envelope_payload().expect("to_envelope_payload");
        match back {
            OwnedEnvelopePayload::TokenIds(v) => assert_eq!(v, original),
            other => panic!("expected TokenIds, got {other:?}"),
        }
    }

    /// TokenIds mapping: decoder models keep their i64 token IDs intact.
    #[test]
    fn envelope_maps_token_ids_to_i64_array() {
        let ids: Vec<i64> = vec![10, 20, 30];
        let env = TestEnvelope {
            payload: EnvelopePayload::TokenIds(&ids),
        };
        let array = MlxArray::from_envelope(&env).expect("from token ids");

        assert_eq!(array.shape(), vec![ids.len() as i32]);
        assert_eq!(array.dtype().expect("dtype"), MlxDtype::I64);
        assert_eq!(array.to_vec_i64().expect("readback"), ids);
    }

    /// Dropping the MlxArray built via `from_shared_buffer` must run our
    /// custom dtor exactly once, which in turn releases the SharedBuffer
    /// (MTLBuffer + heap allocation). We can't observe the MTLBuffer
    /// release directly from Rust, but the test runs under whatever
    /// sanitisers the CI enables; a double-drop would crash here.
    #[test]
    fn from_shared_buffer_drop_is_clean() {
        for _ in 0..16 {
            let data: Box<[f32]> = vec![1.0; 128].into_boxed_slice();
            let buf = SharedBuffer::from_box(data).expect("shared buffer");
            let array = MlxArray::from_shared_buffer(buf, &[128], MlxDtype::F32).expect("array");
            // Force an eval so MLX has actually touched the memory.
            array.eval().expect("eval");
            drop(array);
        }
    }
}
