//! Safe RAII wrapper for a llama.cpp inference context.
//!
//! Owns `llama_context*` via [`Drop`], and shares the native handle of the
//! model it was created from, so that model stays loaded while the context
//! reads it. Inherent methods cover the KV-cache manipulation surface the
//! multi-turn prefix-reuse path needs.
//!
//! # Threading
//!
//! [`LlamaContext`] is `Send` but **not** `Sync`. `llama_decode_c` (the
//! inner loop of every generation path) mutates the KV cache and scratch
//! buffers; concurrent access from multiple threads is UB. Callers that
//! need shared access — including
//! `xybrid-core::runtime_adapter::llama_cpp::LlamaCppBackend` behind
//! `&self` — must serialize through a [`std::sync::Mutex`].

use std::ffi::{c_int, c_void};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::error::{LlamaError, LlamaResult};
use crate::ffi;
use crate::model::{LlamaModel, ModelHandle};

/// Source of [`LlamaContext::id`]: identifies the context a snapshot came
/// from without trusting a pointer that could be reused after a free.
static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(0);

/// The wrapper's batch size when the caller passes `n_batch = 0`
/// (llama.cpp's own default is 2048).
const DEFAULT_N_BATCH: usize = 512;

/// Opaque handle to a llama.cpp inference context.
pub struct LlamaContext {
    ptr: *mut c_void,
    id: u64,
    /// The batch size the context was created with (`0` mapped to 512).
    n_batch: usize,
    /// The most tokens one decode call may carry (see [`decode_limit`]).
    decode_limit: usize,
    /// The model this context was created from. Fields drop after
    /// [`Drop::drop`] frees the native context, so the model outlives it.
    model: Arc<ModelHandle>,
}

/// A saved copy of one sequence's state, from
/// [`LlamaContext::state_seq_save`].
///
/// Opaque on purpose. llama.cpp does not validate snapshot bytes, and a
/// damaged snapshot or one from another model can hit one of its asserts
/// and abort the process, so a snapshot can only be restored into the
/// context that saved it.
pub struct LlamaSeqSnapshot {
    context_id: u64,
    bytes: Vec<u8>,
}

impl LlamaSeqSnapshot {
    /// Size of the saved state in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bytes.len()
    }
}

impl fmt::Debug for LlamaSeqSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlamaSeqSnapshot")
            .field("context_id", &self.context_id)
            .field("size_bytes", &self.bytes.len())
            .finish()
    }
}

impl LlamaContext {
    /// Create a new context bound to `model`.
    ///
    /// The context keeps `model` loaded until it is dropped too, so the two
    /// may be dropped in either order.
    ///
    /// `n_threads = 0` means "auto-detect"; `n_batch = 0` means "use the
    /// 512-token llama.cpp default". `flash_attn` enables Flash Attention
    /// (2-4× speedup on longer contexts) where supported.
    pub fn new(
        model: &LlamaModel,
        n_ctx: usize,
        n_threads: usize,
        n_batch: usize,
        flash_attn: bool,
    ) -> LlamaResult<Self> {
        // llama.cpp receives the sizes as C ints; the decode limit below is
        // derived from these same values.
        let (n_ctx_c, n_batch_c) = (n_ctx as c_int, n_batch as c_int);
        // SAFETY: model.as_ptr() is non-null (LlamaModel's ctor guarantees
        // it). Null return surfaces as ContextCreationFailed.
        let ptr = unsafe {
            ffi::new_context_with_model(
                model.as_ptr() as *mut c_void,
                n_ctx_c,
                n_threads as c_int,
                n_batch_c,
                flash_attn,
            )
        };
        if ptr.is_null() {
            return Err(LlamaError::ContextCreationFailed(format!(
                "llama_new_context_with_model returned null (n_ctx={n_ctx}, n_threads={n_threads}, n_batch={n_batch}, flash_attn={flash_attn})"
            )));
        }
        // SAFETY: ptr is the live context created above.
        let native_n_ctx = usize::try_from(unsafe { ffi::n_ctx(ptr) }).unwrap_or(0);
        Ok(Self {
            ptr,
            id: NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed),
            n_batch: if n_batch == 0 {
                DEFAULT_N_BATCH
            } else {
                n_batch
            },
            decode_limit: decode_limit(n_ctx_c, n_batch_c, native_n_ctx),
            model: Arc::clone(model.handle()),
        })
    }

    /// A context with no native handle, for tests. Only native wrappers that
    /// reject a null context may receive it.
    #[cfg(test)]
    pub(crate) fn test_stub(model: &LlamaModel) -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            id: NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed),
            n_batch: DEFAULT_N_BATCH,
            decode_limit: DEFAULT_N_BATCH,
            model: Arc::clone(model.handle()),
        }
    }

    /// Raw pointer for the in-crate generation paths.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// The native handle of the model this context was created from.
    #[inline]
    pub(crate) fn model_handle(&self) -> &Arc<ModelHandle> {
        &self.model
    }

    /// Errors unless this context was created from `model`.
    ///
    /// Generation reads `model`'s vocabulary against this context's
    /// buffers, so it must run before any native call that takes both.
    pub(crate) fn ensure_created_from(&self, model: &LlamaModel) -> LlamaResult<()> {
        if Arc::ptr_eq(self.model_handle(), model.handle()) {
            Ok(())
        } else {
            Err(LlamaError::InvalidInput(
                "the context was created from a different model".to_string(),
            ))
        }
    }

    /// Context length (tokens).
    pub fn n_ctx(&self) -> usize {
        // SAFETY: self.ptr is a live context pointer.
        unsafe { ffi::n_ctx(self.ptr) as usize }
    }

    /// The batch size this context was created with, `0` mapped to the
    /// wrapper's default of 512.
    ///
    /// llama.cpp's own limit per decode call can be lower: for a causal model
    /// it caps the batch at the context size requested at creation, before
    /// padding it, and a longer call aborts the process.
    pub fn n_batch(&self) -> usize {
        self.n_batch
    }

    /// The most tokens one `llama_decode` call may carry on this context.
    pub(crate) fn decode_limit(&self) -> usize {
        self.decode_limit
    }

    /// Vocabulary size of the model this context was created from, which
    /// is also the width of each row of its logits.
    pub(crate) fn n_vocab(&self) -> usize {
        // SAFETY: the shared handle keeps the model pointer live.
        let n = unsafe { ffi::n_vocab(self.model.as_ptr()) };
        usize::try_from(n).unwrap_or(0)
    }

    /// Fully clear the KV cache, resetting context state for a new
    /// conversation. Cheap; used as the fallback when prefix-reuse is
    /// not viable.
    pub fn kv_cache_clear(&self) {
        // SAFETY: self.ptr is a live context pointer.
        unsafe { ffi::kv_cache_clear(self.ptr) };
    }

    /// Truncate the KV cache for `seq_id` to a prefix length `p_keep`,
    /// dropping tokens at positions `[p_keep, ∞)`.
    ///
    /// Pairs with the `n_past_in` parameter on
    /// [`crate::generate_streaming`]: caller computes the longest common
    /// prefix between the new prompt and the previously-tokenized prompt,
    /// truncates the cache here to drop the diverged tail, then
    /// re-prefills only the new tail at position `p_keep`.
    ///
    /// On recurrent / hybrid models this is **unsafe at the semantic
    /// level** even though the call itself is memory-safe — the residual
    /// recurrent state remains keyed to the original prefix and
    /// `llama_decode` fails on the diverging tail. Gate calls on
    /// [`LlamaModel::has_recurrent_state`] = false.
    pub fn kv_cache_seq_rm(&self, seq_id: i32, p_keep: usize) {
        // SAFETY: self.ptr is a live context pointer.
        unsafe { ffi::kv_cache_seq_rm(self.ptr, seq_id, p_keep) };
    }

    /// Snapshot the full state of `seq_id`: its KV cache and, on
    /// recurrent / hybrid models, its recurrent state.
    ///
    /// This is the prefix-reuse path that also works where
    /// [`Self::kv_cache_seq_rm`] cannot (see
    /// [`LlamaModel::has_recurrent_state`]): prefill a shared prefix
    /// (system prompt, tool definitions) once and save it, then before each
    /// request [`Self::state_seq_restore`] it and prefill only the new tail
    /// via [`crate::generate_streaming`] with `n_past_in` = the prefix
    /// length. The snapshot can only be restored into this context, and
    /// holds a full copy of the sequence's state.
    ///
    /// Do not call this from a generation callback on the same context.
    ///
    /// # Errors
    ///
    /// [`LlamaError::Internal`] if `seq_id` is not one of this context's
    /// sequences (`0` for contexts from [`Self::new`]), the snapshot buffer
    /// cannot be allocated, or llama.cpp fails to serialize the sequence.
    pub fn state_seq_save(&self, seq_id: i32) -> LlamaResult<LlamaSeqSnapshot> {
        // SAFETY: self.ptr is a live context pointer; the shim rejects a
        // seq_id outside this context's sequences.
        let size = unsafe { ffi::state_seq_get_size(self.ptr, seq_id) };
        if size == 0 {
            return Err(LlamaError::Internal(format!(
                "llama_state_seq_get_size failed for seq_id {seq_id} \
                 (not a sequence of this context, or not serializable)"
            )));
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(size).map_err(|e| {
            LlamaError::Internal(format!("cannot allocate a {size}-byte snapshot: {e}"))
        })?;
        bytes.resize(size, 0);
        // SAFETY: self.ptr is a live context pointer; the buffer is sized
        // by llama.cpp's own probe above.
        let written = unsafe { ffi::state_seq_get_data(self.ptr, &mut bytes, seq_id) };
        if written == 0 {
            return Err(LlamaError::Internal(format!(
                "llama_state_seq_get_data failed for seq_id {seq_id}"
            )));
        }
        bytes.truncate(written);
        Ok(LlamaSeqSnapshot {
            context_id: self.id,
            bytes,
        })
    }

    /// Restore a snapshot from [`Self::state_seq_save`] into `seq_id`,
    /// replacing whatever that sequence held.
    ///
    /// Only the cache comes back, not the logits, so the next call must
    /// decode a non-empty tail ([`crate::generate_streaming`] with
    /// `n_past_in` = the snapshot's length) rather than sample from the
    /// current logits. Do not call this from a generation callback on the
    /// same context.
    ///
    /// # Errors
    ///
    /// [`LlamaError::InvalidInput`] if `snapshot` was saved by another
    /// context; [`LlamaError::Internal`] if `seq_id` is not one of this
    /// context's sequences or llama.cpp fails to load the snapshot (for
    /// example, an allocation failure inside it). A failed load leaves
    /// `seq_id` empty, so it can be prefilled again from position 0.
    pub fn state_seq_restore(&self, snapshot: &LlamaSeqSnapshot, seq_id: i32) -> LlamaResult<()> {
        if snapshot.context_id != self.id {
            return Err(LlamaError::InvalidInput(
                "state_seq_restore: the snapshot was saved by another context".to_string(),
            ));
        }
        // SAFETY: self.ptr is a live context pointer; the bytes are an
        // unmodified llama.cpp snapshot of this context, and the shim
        // rejects a seq_id outside this context's sequences.
        let read = unsafe { ffi::state_seq_set_data(self.ptr, &snapshot.bytes, seq_id) };
        if read == 0 {
            return Err(LlamaError::Internal(format!(
                "llama_state_seq_set_data rejected a {}-byte snapshot for seq_id {seq_id}",
                snapshot.bytes.len()
            )));
        }
        Ok(())
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from `ffi::new_context_with_model`,
            // checked non-null on construction. Drop runs at most once,
            // and `self.model` drops after it, so the model outlives the
            // context.
            unsafe { ffi::free_context(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

// SAFETY: LlamaContext is Send because llama.cpp accepts handoff across
// threads as long as no two threads call into the same context
// concurrently. NOT Sync — see the type-level rationale at the top of
// this file. The caller is responsible for serializing access.
unsafe impl Send for LlamaContext {}

/// The most tokens llama.cpp accepts in one decode call on a context created
/// with these sizes, `native_n_ctx` being the context size it then reports.
///
/// For a causal model llama.cpp caps the batch at the requested context size
/// and only then pads that size up to a multiple of 256, the size
/// `llama_n_ctx` reports; a longer call aborts the process. `n_ctx = 0` asks
/// for the model's training size, which is unknown here, but padding added
/// less than 256 to it. A batch size of zero or less is the wrapper's 512.
fn decode_limit(n_ctx: c_int, n_batch: c_int, native_n_ctx: usize) -> usize {
    let positive = |n: c_int| usize::try_from(n).ok().filter(|&n| n > 0);
    let n_batch = positive(n_batch).unwrap_or(DEFAULT_N_BATCH);
    let n_ctx = positive(n_ctx).unwrap_or_else(|| native_n_ctx.saturating_sub(255));
    n_batch.min(n_ctx).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_decode_limit_is_llama_cpps_own_batch_size() {
        assert_eq!(decode_limit(4096, 512, 4096), 512);
        assert_eq!(
            decode_limit(64, 512, 256),
            64,
            "the requested context caps it, not the padded one"
        );
        assert_eq!(decode_limit(4096, 0, 4096), 512, "0 is the wrapper's 512");
        assert_eq!(decode_limit(4096, -1, 4096), 512, "so is a negative size");
        assert_eq!(
            decode_limit(0, 4096, 1024),
            769,
            "an unknown context size is more than its padded size minus 256"
        );
        assert_eq!(decode_limit(0, 512, 0), 1, "never zero");
    }

    #[test]
    fn a_context_keeps_its_model_loaded_until_both_are_dropped() {
        let model = LlamaModel::test_stub();
        let handle = Arc::downgrade(model.handle());
        let ctx = LlamaContext::test_stub(&model);

        drop(model);
        assert!(
            handle.upgrade().is_some(),
            "the context must keep the model"
        );

        drop(ctx);
        assert!(handle.upgrade().is_none(), "the last owner frees the model");
    }

    #[test]
    fn a_context_knows_which_model_created_it() {
        let model = LlamaModel::test_stub();
        let other = LlamaModel::test_stub();
        let ctx = LlamaContext::test_stub(&model);

        assert!(ctx.ensure_created_from(&model).is_ok());
        assert!(matches!(
            ctx.ensure_created_from(&other),
            Err(LlamaError::InvalidInput(_))
        ));
    }
}
