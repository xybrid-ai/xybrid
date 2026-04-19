//! Safe wrapper over `mlx_stream`.
//!
//! An `MlxStream` is the queueing point for MLX ops — every op takes a
//! stream argument that determines CPU vs GPU dispatch. Default streams
//! (one per device) are cached per-thread so callers do not pay the
//! construction cost on every op.

use std::fmt;

use crate::error::{MlxError, MlxResult};
use crate::ffi;
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
/// # Thread-local defaults
///
/// [`MlxStream::default_cpu`] and [`MlxStream::default_gpu`] return a
/// clone of a thread-local cached default stream. This avoids hitting
/// the mlx-c constructor on every op while still not relying on
/// process-wide mutable state.
pub struct MlxStream {
    raw: mlx_stream,
    device: Device,
}

impl MlxStream {
    /// Construct a fresh stream bound to the given device.
    ///
    /// This calls `mlx_default_{cpu,gpu}_stream_new` under the hood,
    /// which returns the MLX default stream for that device rather than
    /// a freshly-allocated queue. Two calls yield two handles pointing
    /// at the same underlying stream — which is the behaviour we want
    /// for xybrid (one scheduler per device per process).
    pub fn new(device: Device) -> MlxResult<Self> {
        // SAFETY: the mlx-c default-stream constructors take no arguments
        // and are safe to call from any thread. We immediately check the
        // returned handle's ctx pointer below.
        let raw = unsafe {
            match device {
                Device::Cpu => ffi::default_cpu_stream(),
                Device::Gpu => ffi::default_gpu_stream(),
            }
        };
        if raw.ctx.is_null() {
            return Err(MlxError::Internal(format!(
                "mlx default {device:?} stream returned a null handle — the MLX runtime \
                 failed to initialise"
            )));
        }
        Ok(Self { raw, device })
    }

    /// Return the thread-local default CPU stream.
    pub fn default_cpu() -> MlxResult<Self> {
        default_for(Device::Cpu)
    }

    /// Return the thread-local default GPU stream.
    pub fn default_gpu() -> MlxResult<Self> {
        default_for(Device::Gpu)
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

// MLX's scheduler is itself thread-safe (see the mlx-c README), so handing
// a stream handle off to another thread is fine as long as no aliased
// reference is used concurrently. We therefore mark MlxStream Send but not
// Sync — concurrent dispatch on the same handle needs an external lock.
//
// SAFETY: `mlx_stream` wraps a `void*` that mlx itself owns. The C++ core
// performs locking on the underlying scheduler. Moving the handle across
// threads therefore does not introduce a data race beyond what mlx itself
// already handles.
unsafe impl Send for MlxStream {}

/// Fetch (or lazily initialise) the thread-local default stream for a
/// device and return a cloned handle.
fn default_for(device: Device) -> MlxResult<MlxStream> {
    thread_local! {
        static DEFAULT_CPU: std::cell::OnceCell<MlxStream> = const { std::cell::OnceCell::new() };
        static DEFAULT_GPU: std::cell::OnceCell<MlxStream> = const { std::cell::OnceCell::new() };
    }
    match device {
        Device::Cpu => DEFAULT_CPU.with(|cell| {
            let stream = match cell.get() {
                Some(s) => s,
                None => {
                    let created = MlxStream::new(Device::Cpu)?;
                    // OnceCell::set can only fail if another set raced in, which
                    // is impossible inside a thread-local closure. Fall back to
                    // the returned borrow regardless of set's result.
                    let _ = cell.set(created);
                    cell.get().expect("OnceCell just set")
                }
            };
            stream.try_clone()
        }),
        Device::Gpu => DEFAULT_GPU.with(|cell| {
            let stream = match cell.get() {
                Some(s) => s,
                None => {
                    let created = MlxStream::new(Device::Gpu)?;
                    let _ = cell.set(created);
                    cell.get().expect("OnceCell just set")
                }
            };
            stream.try_clone()
        }),
    }
}

/// Placeholder to use a device handle from mlx-c when future ops need one
/// (e.g. selecting a GPU index other than zero). Kept here so the stream
/// module stays the single owner of device concepts. Used only by
/// internal paths today.
#[allow(dead_code)]
pub(crate) fn device_as_type(device: Device) -> sys::mlx_device_type {
    match device {
        Device::Cpu => sys::mlx_device_type__MLX_CPU,
        Device::Gpu => sys::mlx_device_type__MLX_GPU,
    }
}
