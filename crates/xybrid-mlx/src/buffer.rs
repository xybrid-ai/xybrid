//! Zero-copy storage backing an `MlxArray`.
//!
//! [`SharedBuffer`] is the low-level primitive behind US-008's zero-copy
//! Envelope ↔ MlxArray conversion. It owns a page-aligned heap allocation
//! and an `MTLBuffer` that wraps the same bytes via
//! `newBufferWithBytesNoCopy:length:options:deallocator:`, allocated with
//! [`MTLResourceOptions::StorageModeShared`]. MLX is handed the buffer's
//! `.contents()` pointer, so GPU kernels that execute on an Apple Silicon
//! unified-memory system read and write the same physical bytes as the CPU.
//! [`SharedBuffer::from_box`] performs one CPU copy into Metal-compatible
//! storage; the MLX handoff itself remains no-copy.
//!
//! # MTLStorageMode tradeoffs
//!
//! Metal exposes three storage modes for buffers ([`MTLStorageMode`][mt]).
//! Only `.storageModeShared` meets xybrid's zero-copy requirement for
//! Envelope payloads:
//!
//! | Mode | CPU-reachable? | GPU-reachable? | Sync discipline | Verdict for Envelope payloads |
//! | --- | --- | --- | --- | --- |
//! | `.storageModeShared` | yes | yes | unified-memory; no explicit flush | **chosen** — host writes immediately visible to GPU on Apple Silicon |
//! | `.storageModeManaged` | yes | yes (discrete) | needs `didModifyRange:` on every CPU write | wrong tool: extra syncs for no upside on the unified-memory targets we care about, and Managed is unsupported on iOS |
//! | `.storageModePrivate` | no | yes | requires a staging buffer + blit for every CPU interaction | defeats the whole zero-copy point — every Envelope read or write would force a GPU round-trip |
//!
//! `.storageModeShared` is also the only mode that lets CPU-side writes to
//! the shared buffer become immediately visible to MLX without explicit
//! staging or synchronization.
//!
//! [mt]: https://developer.apple.com/documentation/metal/mtlstoragemode
//!
//! # Ownership and drop order
//!
//! `SharedBuffer` owns:
//! - an `MTLBuffer` handle (retained via `metal::Buffer`) constructed with
//!   `deallocator: None` — meaning Metal does *not* attempt to
//!   `free()` the backing memory itself; that is the Rust side's job;
//! - a raw pointer + [`std::alloc::Layout`] recording the heap allocation
//!   we took ownership of.
//!
//! Drop order matters: the MTLBuffer must be released *before* we free the
//! heap memory, because while the buffer is live MLX might still reference
//! it. We use [`std::mem::ManuallyDrop`] on the metal handle so the drop
//! glue is explicit and we can sequence the two steps in a single [`Drop`]
//! impl.
//!
//! # Allocator compatibility
//!
//! The heap allocation has to be reclaimable via
//! [`std::alloc::dealloc`] with the same [`Layout`] we recorded at
//! construction time. We accept two entry points:
//!
//! - [`SharedBuffer::from_box`] — the typed convenience, for callers
//!   holding a `Box<[T]>`. Internally copies the box contents into a
//!   Metal-compatible page-aligned allocation before handing that pointer
//!   to `newBufferWithBytesNoCopy`.
//! - [`SharedBuffer::from_raw_parts`] — for callers that already have a
//!   raw page-aligned pointer + matching layout (e.g. the benchmark wants
//!   to reuse a single allocation across repeated cycles without
//!   re-routing through `Box<[T]>`).
//!
//! Both paths keep the backing allocation on Rust's default allocator; we
//! never mix Metal-allocated buffers with Rust-dealloc'd memory.

use std::alloc::Layout;
use std::ffi::c_void;
use std::mem::ManuallyDrop;

use metal::foreign_types::ForeignType;
use metal::{Buffer, Device, MTLResourceOptions};

use crate::error::{MlxError, MlxResult};

/// A heap region wrapped in an `MTLBuffer` with `.storageModeShared`.
///
/// See the [module docs](self) for the storage-mode tradeoffs and drop-order
/// rationale. Most callers want [`SharedBuffer::from_box`]; the raw-parts
/// constructor is provided for advanced paths (benchmarks, custom
/// allocators) that hand-roll the layout.
pub struct SharedBuffer {
    // Order matters for Drop: we release the MTLBuffer first, then free the
    // backing memory in `Drop::drop` below. `ManuallyDrop` keeps us in
    // control of the sequence.
    buffer: ManuallyDrop<Buffer>,
    storage_ptr: *mut u8,
    storage_layout: Layout,
    logical_len_bytes: usize,
}

pub(crate) fn metal_page_size() -> usize {
    extern "C" {
        fn getpagesize() -> std::os::raw::c_int;
    }

    // SAFETY: getpagesize has no preconditions and returns the host VM page
    // size used by Metal's no-copy buffer alignment contract.
    let page_size = unsafe { getpagesize() };
    if page_size > 0 {
        page_size as usize
    } else {
        16 * 1024
    }
}

fn round_up_to_multiple(value: usize, multiple: usize) -> Option<usize> {
    if multiple == 0 {
        return None;
    }
    let remainder = value % multiple;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(multiple - remainder)
    }
}

pub(crate) fn metal_aligned_layout(logical_size: usize) -> MlxResult<Layout> {
    if logical_size == 0 {
        return Err(MlxError::Internal(
            "SharedBuffer requires at least one logical byte".into(),
        ));
    }

    let page_size = metal_page_size();
    let storage_size = round_up_to_multiple(logical_size, page_size).ok_or_else(|| {
        MlxError::Internal(format!(
            "SharedBuffer page-size rounding overflowed for {logical_size} bytes"
        ))
    })?;
    Layout::from_size_align(storage_size, page_size).map_err(|e| {
        MlxError::Internal(format!(
            "SharedBuffer page-aligned layout failed for {storage_size} bytes: {e}"
        ))
    })
}

impl SharedBuffer {
    /// Wrap an existing heap allocation in a shared-storage `MTLBuffer`.
    ///
    /// Ownership of the allocation transfers to the returned `SharedBuffer`.
    /// `layout` must match the layout the allocation was made with and must
    /// be page-aligned with a page-multiple size. Most callers should prefer
    /// [`Self::from_box`], which builds such an allocation automatically.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError::Internal`] when the Metal system default device
    /// is unreachable (no Metal-capable GPU — typical of a CI VM without a
    /// virtualised Metal stack). Returns [`MlxError::OutOfMemory`] if the
    /// MTLBuffer construction fails.
    ///
    /// # Safety
    ///
    /// `ptr` must be a live allocation from Rust's global allocator with the
    /// exact `layout` passed here. The allocation must not be aliased by any
    /// other owner after this call because successful construction transfers
    /// ownership to the returned [`SharedBuffer`], whose [`Drop`] implementation
    /// releases it with [`std::alloc::dealloc`].
    pub unsafe fn from_raw_parts(ptr: *mut u8, layout: Layout) -> MlxResult<Self> {
        // SAFETY: forwarded from this function's caller.
        unsafe { Self::from_raw_parts_with_len(ptr, layout, layout.size()) }
    }

    unsafe fn from_raw_parts_with_len(
        ptr: *mut u8,
        layout: Layout,
        logical_len_bytes: usize,
    ) -> MlxResult<Self> {
        if ptr.is_null() {
            return Err(MlxError::Internal(
                "SharedBuffer::from_raw_parts received a null pointer".into(),
            ));
        }
        if layout.size() == 0 {
            return Err(MlxError::Internal(
                "SharedBuffer::from_raw_parts requires a non-empty layout (size > 0)".into(),
            ));
        }
        if logical_len_bytes == 0 || logical_len_bytes > layout.size() {
            return Err(MlxError::Internal(format!(
                "SharedBuffer logical length {logical_len_bytes} must be in 1..={} bytes",
                layout.size()
            )));
        }
        let page_size = metal_page_size();
        if !(ptr as usize).is_multiple_of(page_size) || !layout.size().is_multiple_of(page_size) {
            return Err(MlxError::Internal(format!(
                "SharedBuffer backing allocation must be page-aligned with a page-multiple \
                 length for Metal no-copy buffers (page_size={page_size}, ptr={ptr:p}, \
                 layout_size={})",
                layout.size()
            )));
        }
        let device = Device::system_default().ok_or_else(|| {
            MlxError::Internal(
                "MTLCreateSystemDefaultDevice returned nil — no Metal-capable device found".into(),
            )
        })?;
        // `deallocator: None` means Metal will NOT call a block to free the
        // underlying bytes; the app (us) is responsible for deallocation.
        // This is required because our allocation came from Rust's default
        // allocator, whose `dealloc` call is layout-aware — not compatible
        // with Metal's internal `free()` path.
        let buffer = ManuallyDrop::new(device.new_buffer_with_bytes_no_copy(
            ptr as *const c_void,
            layout.size() as u64,
            MTLResourceOptions::StorageModeShared,
            /* deallocator */ None,
        ));
        if buffer.as_ptr().is_null() {
            return Err(MlxError::OutOfMemory);
        }
        // Invariant we rely on everywhere: MTLBuffer.contents() must alias
        // the no-copy backing pointer when the buffer was created via
        // `newBufferWithBytesNoCopy`. If Metal ever hands us a different
        // address, the zero-copy guarantee is broken and every downstream
        // claim in this module is a lie — so assert even in release.
        assert_eq!(
            buffer.contents() as *mut u8,
            ptr,
            "MTLBuffer.contents() must alias the no-copy backing pointer; Metal returned a \
             different address — zero-copy invariant violated"
        );
        Ok(Self {
            buffer,
            storage_ptr: ptr,
            storage_layout: layout,
            logical_len_bytes,
        })
    }

    /// Wrap a `Box<[T]>` in a shared-storage `MTLBuffer`.
    ///
    /// The box contents are copied into a Metal-compatible page-aligned
    /// allocation. After construction, `SharedBuffer::as_ptr()` aliases
    /// `MTLBuffer.contents()` and is the pointer handed to MLX.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError::Internal`] for a zero-length input (MLX and
    /// Metal both reject empty buffers) or for layout-overflow corner
    /// cases (`Layout::array` returns `LayoutError`). Propagates the
    /// device-unreachable / OOM errors from [`Self::from_raw_parts`].
    pub fn from_box<T: Copy>(data: Box<[T]>) -> MlxResult<Self> {
        let len = data.len();
        if len == 0 {
            return Err(MlxError::Internal(
                "SharedBuffer::from_box received a zero-length box — MTLBuffer requires at \
                 least one byte"
                    .into(),
            ));
        }
        let logical_layout = Layout::array::<T>(len).map_err(|e| {
            MlxError::Internal(format!("Layout::array::<T>({len}) overflowed: {e}"))
        })?;
        let logical_len_bytes = logical_layout.size();
        if logical_len_bytes == 0 {
            return Err(MlxError::Internal(
                "SharedBuffer::from_box received a zero-sized element allocation".into(),
            ));
        }
        let storage_layout = metal_aligned_layout(logical_len_bytes)?;

        // SAFETY: alloc is called with a non-zero Layout returned by
        // metal_aligned_layout.
        let ptr = unsafe { std::alloc::alloc(storage_layout) };
        if ptr.is_null() {
            return Err(MlxError::OutOfMemory);
        }

        // SAFETY: ptr is valid for storage_layout.size() bytes. Zero the
        // page padding before copying the logical payload so the full
        // MTLBuffer range is initialized even though MLX only reads the
        // logical shape.
        unsafe {
            std::ptr::write_bytes(ptr, 0, storage_layout.size());
        }

        // SAFETY: both regions are valid for logical_len_bytes bytes, do
        // not overlap, and T: Copy means byte-copying the initialized
        // elements preserves the value semantics.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), ptr, logical_len_bytes);
        }
        drop(data);

        // SAFETY: ptr came from alloc(storage_layout) above, is still live,
        // and ownership transfers to SharedBuffer only on successful return.
        match unsafe { Self::from_raw_parts_with_len(ptr, storage_layout, logical_len_bytes) } {
            Ok(buf) => Ok(buf),
            Err(err) => {
                // SAFETY: ptr came from alloc(storage_layout) above and
                // SharedBuffer construction failed before taking ownership.
                unsafe { std::alloc::dealloc(ptr, storage_layout) };
                Err(err)
            }
        }
    }

    /// Pointer to the start of the shared region.
    ///
    /// This pointer aliases `MTLBuffer.contents()` and stays valid for as
    /// long as `self` is live. Safe to cast to `*const T` when the caller
    /// knows the original element type.
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.storage_ptr
    }

    /// Mutable pointer to the start of the shared region.
    ///
    /// Same validity guarantee as [`Self::as_ptr`]. Because we document the
    /// buffer as `.storageModeShared`, mutations through this pointer are
    /// visible to any MTLBuffer-aware MLX op on Apple Silicon unified
    /// memory — subject to the usual "evaluate before reading the MLX-side
    /// result" discipline.
    ///
    /// # Safety
    ///
    /// The caller must not mutate the region concurrently with an MLX op
    /// that reads from it. There is no cross-thread synchronisation here.
    #[must_use]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.storage_ptr
    }

    /// Total size of the shared region in bytes.
    #[must_use]
    pub fn len_bytes(&self) -> usize {
        self.logical_len_bytes
    }

    /// Borrow the MTLBuffer handle for passing to Metal / MLX APIs.
    ///
    /// `pub(crate)` because downstream code never needs a raw Metal handle:
    /// the [`crate::array::MlxArray::from_shared_buffer`] entry point is
    /// the sanctioned integration point.
    #[allow(dead_code)]
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        // Step 1: release the MTLBuffer. The retain count on the Metal side
        // drops by one; if this was the last retain, Metal frees its own
        // bookkeeping (but NOT our backing bytes, because we registered
        // `deallocator: None` at construction).
        //
        // SAFETY: `buffer` is ManuallyDrop and has not been dropped before.
        unsafe { ManuallyDrop::drop(&mut self.buffer) };

        // Step 2: free the backing allocation. After the MTLBuffer release
        // above, no Metal-side code holds a reference to this region, so
        // reclaiming the memory is safe.
        //
        // SAFETY: `storage_ptr` / `storage_layout` were recorded at
        // construction time from a live allocation on Rust's default
        // allocator and have not been freed elsewhere.
        if !self.storage_ptr.is_null() && self.storage_layout.size() > 0 {
            unsafe { std::alloc::dealloc(self.storage_ptr, self.storage_layout) };
        }
    }
}

// SharedBuffer contains raw pointers but they are opaque to Rust — external
// access goes through `as_ptr` / the MTLBuffer. Sending across threads is
// fine (Metal handles its own refcount atomics); concurrent mutable aliasing
// through `as_mut_ptr` is the caller's responsibility.
//
// SAFETY: the MTLBuffer's refcount is atomic; our raw ptr is immutable
// metadata. Moving across threads does not introduce aliasing.
unsafe impl Send for SharedBuffer {}

#[cfg(test)]
mod tests {
    //! SharedBuffer tests.
    //!
    //! Gated on `bindings` + Apple via the enclosing `lib.rs` cfg, so a
    //! non-Apple `cargo test -p xybrid-mlx` default run does not pull
    //! Metal. These tests exercise the MTLBuffer round-trip: allocate a
    //! Box, hand it to SharedBuffer, read back via the raw pointer, mutate
    //! via the raw pointer, and ensure the Metal side sees the mutation.

    use super::*;

    /// Constructing from a Box<[f32]> preserves contents in shared storage.
    ///
    /// Metal requires no-copy buffers to use page-aligned backing storage,
    /// so `from_box` copies the typed payload into such a region before
    /// creating the MTLBuffer. From that point on, MTLBuffer.contents()
    /// aliases SharedBuffer::as_ptr().
    #[test]
    fn from_box_preserves_contents_in_page_aligned_storage() {
        let data: Box<[f32]> = vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice();
        let buf = SharedBuffer::from_box(data).expect("from_box");
        assert_eq!(
            (buf.as_ptr() as usize) % metal_page_size(),
            0,
            "SharedBuffer backing pointer must be page-aligned for Metal"
        );
        assert_eq!(buf.as_ptr(), buf.buffer().contents() as *const u8);
        assert_eq!(buf.len_bytes(), 4 * std::mem::size_of::<f32>());
        // SAFETY: `as_ptr` is live for `buf`, and len_bytes proves there
        // are four f32 values in the logical payload.
        let values = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, 4) };
        assert_eq!(values, &[1.0, 2.0, 3.0, 4.0]);
    }

    /// MTLBuffer contents() aliases the backing pointer — this is what
    /// makes MLX's kernel reads zero-copy. Writing through the raw pointer
    /// must be visible via MTLBuffer.contents() and vice-versa.
    #[test]
    fn mtl_buffer_contents_aliases_backing() {
        let data: Box<[f32]> = vec![0.0; 8].into_boxed_slice();
        let buf = SharedBuffer::from_box(data).expect("from_box");

        let raw_ptr = buf.as_mut_ptr() as *mut f32;
        let mtl_ptr = buf.buffer().contents() as *mut f32;
        assert_eq!(raw_ptr, mtl_ptr, "MTLBuffer must alias the backing pointer");

        // Write through the raw pointer, read through the MTLBuffer view.
        // SAFETY: we own the buffer and no other thread holds a reference.
        unsafe {
            std::ptr::write(raw_ptr.add(3), 42.0);
        }
        // SAFETY: mtl_ptr is a live MTLBuffer contents pointer with
        // .storageModeShared, aliased to raw_ptr.
        let observed = unsafe { std::ptr::read(mtl_ptr.add(3)) };
        assert_eq!(observed, 42.0, "MTLBuffer must see CPU-side writes");
    }

    /// A zero-length box is rejected — MTLBuffer needs at least one byte,
    /// and we prefer a clean error to an opaque Metal failure.
    #[test]
    fn from_box_rejects_zero_length() {
        let empty: Box<[f32]> = Vec::<f32>::new().into_boxed_slice();
        let err = match SharedBuffer::from_box(empty) {
            Ok(_) => panic!("zero-length box should be rejected"),
            Err(err) => err,
        };
        match err {
            MlxError::Internal(msg) => assert!(
                msg.contains("zero-length"),
                "error message should mention zero-length, got: {msg}"
            ),
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    /// Dropping a SharedBuffer must release the MTLBuffer AND free the
    /// backing memory without tripping the allocator's sanity checks.
    /// We can't directly observe the free, but the test will abort under
    /// a sanitiser if the order is wrong, and Miri (when we eventually
    /// run it here) will flag UAFs.
    #[test]
    fn drop_round_trip() {
        for _ in 0..16 {
            let data: Box<[f32]> = vec![2.5; 256].into_boxed_slice();
            let buf = SharedBuffer::from_box(data).expect("from_box");
            // A harmless read to prove the buffer is usable.
            // SAFETY: `as_ptr` is live for the lifetime of `buf`.
            let first = unsafe { std::ptr::read(buf.as_ptr() as *const f32) };
            assert_eq!(first, 2.5);
            drop(buf);
        }
    }
}
