//! Safe wrapper over `mlx_stream`.
//!
//! An `MlxStream` is the queueing point for MLX ops — every op takes a
//! stream argument that determines CPU vs GPU dispatch.
//!
//! There is one stream per device for the whole process. MLX's own default
//! streams are per thread, and an array built on one thread's stream cannot
//! be evaluated on another thread; xybrid loads a model on one thread and
//! runs it on others. Every `MlxStream` is therefore a handle to a stream
//! that MLX registers for all threads and leaves unsynchronized. The
//! process-wide lock in [`crate::exclusive`] is what makes that sound.

use std::fmt;

use crate::error::{MlxError, MlxResult};
use crate::ffi;
use crate::resources::ensure_metallib_resource;
use mlx_c_sys::bindings::{self as sys, mlx_stream};

/// MLX device placement.
///
/// Matches the subset of `mlx_device_type` we care about. The ordering
/// mirrors "GPU first" because that is the default dispatch target on
/// Apple Silicon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// Dispatches via Metal.
    Gpu,
    /// Dispatches via the CPU backend (Accelerate on Apple).
    Cpu,
}

/// A handle to an MLX stream. Dispatches ops to a device via the MLX
/// scheduler.
///
/// Ownership mirrors the C API: each `MlxStream` owns exactly one
/// reference to the underlying stream. [`Clone`] bumps the refcount via
/// `mlx_stream_set`; [`Drop`] calls `mlx_stream_free`.
///
/// # One stream per device
///
/// [`MlxStream::new`], [`MlxStream::default_cpu`] and
/// [`MlxStream::default_gpu`] return handles to the same two process-wide
/// streams, whichever thread asks. See the [module docs](self).
pub struct MlxStream {
    raw: mlx_stream,
    device: Device,
}

impl MlxStream {
    /// A handle to the process-wide stream of the given device.
    ///
    /// Two calls, on any two threads, yield two handles to the same
    /// underlying stream.
    pub fn new(device: Device) -> MlxResult<Self> {
        ensure_metallib_resource()?;
        // SAFETY: `ensure_metallib_resource` installed the error handler and
        // placed the Metal library. The shared handle stays alive for the
        // life of the process, so cloning it is sound.
        let raw = unsafe { ffi::stream_clone(ffi::shared_stream(device_as_type(device))?)? };
        if raw.ctx.is_null() {
            return Err(MlxError::Internal(format!(
                "mlx {device:?} stream returned a null handle — the MLX runtime \
                 failed to initialise"
            )));
        }
        Ok(Self { raw, device })
    }

    /// The process-wide CPU stream.
    pub fn default_cpu() -> MlxResult<Self> {
        Self::new(Device::Cpu)
    }

    /// The process-wide GPU stream.
    pub fn default_gpu() -> MlxResult<Self> {
        Self::new(Device::Gpu)
    }

    /// Device this stream dispatches to.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Fallible clone.
    ///
    /// The implementation of [`Clone`] panics on the (extremely rare) case
    /// where `mlx_stream_set` fails — which only happens on allocation
    /// failure during refcount bump. Callers that need to handle the
    /// failure explicitly should use `try_clone` instead.
    pub fn try_clone(&self) -> MlxResult<Self> {
        // SAFETY: self.raw is a live stream for as long as &self is borrowed.
        let raw = unsafe { ffi::stream_clone(self.raw)? };
        Ok(Self {
            raw,
            device: self.device,
        })
    }

    /// Borrow the raw handle for dispatch to mlx-c ops.
    ///
    /// This is intentionally `pub(crate)` — downstream code never needs
    /// a raw `mlx_stream`; op wrappers in the same crate (landing in
    /// US-006 / US-007) pass it through.
    pub(crate) fn as_raw(&self) -> mlx_stream {
        self.raw
    }
}

impl Clone for MlxStream {
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("mlx_stream_set failed during MlxStream::clone — MLX runtime is in a bad state")
    }
}

impl Drop for MlxStream {
    fn drop(&mut self) {
        // SAFETY: self.raw was produced by a constructor or try_clone and
        // is not aliased — each MlxStream owns one reference.
        unsafe { ffi::stream_free(self.raw) }
    }
}

impl fmt::Debug for MlxStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlxStream")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

// SAFETY: `mlx_stream` wraps a pointer to an `{index, device}` pair that this
// handle owns and nothing mutates. The stream it names is registered for
// every thread (see the module docs), so a handle moved to another thread
// still names a stream that thread can evaluate on. MLX does not lock that
// stream; every evaluating call of this crate takes `crate::exclusive`.
unsafe impl Send for MlxStream {}

/// The mlx-c device type of a [`Device`].
pub(crate) fn device_as_type(device: Device) -> sys::mlx_device_type {
    match device {
        Device::Cpu => sys::mlx_device_type__MLX_CPU,
        Device::Gpu => sys::mlx_device_type__MLX_GPU,
    }
}
