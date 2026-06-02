//! Safe RAII wrapper for a llama.cpp inference context.
//!
//! Owns `llama_context*` via [`Drop`]. Inherent methods cover the
//! KV-cache manipulation surface the multi-turn prefix-reuse path needs.
//!
//! # Threading
//!
//! [`LlamaContext`] is `Send` but **not** `Sync`. `llama_decode_c` (the
//! inner loop of every generation path) mutates the KV cache and scratch
//! buffers; concurrent access from multiple threads is UB. Callers that
//! need shared access — including
//! `xybrid-core::runtime_adapter::llama_cpp::LlamaCppBackend` behind
//! `&self` — must serialize through a [`std::sync::Mutex`].

use std::ffi::c_void;

use crate::error::{LlamaError, LlamaResult};
use crate::ffi;
use crate::model::LlamaModel;

/// Opaque handle to a llama.cpp inference context.
pub struct LlamaContext {
    ptr: *mut c_void,
}

impl LlamaContext {
    /// Create a new context bound to `model`.
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
        // SAFETY: model.as_ptr() is non-null (LlamaModel's ctor guarantees
        // it). Null return surfaces as ContextCreationFailed.
        let ptr = unsafe {
            ffi::new_context_with_model(
                model.as_ptr() as *mut c_void,
                n_ctx,
                n_threads,
                n_batch,
                flash_attn,
            )
        };
        if ptr.is_null() {
            return Err(LlamaError::ContextCreationFailed(format!(
                "llama_new_context_with_model returned null (n_ctx={n_ctx}, n_threads={n_threads}, n_batch={n_batch}, flash_attn={flash_attn})"
            )));
        }
        Ok(Self { ptr })
    }

    /// Raw pointer for the in-crate generation paths.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Context length (tokens).
    pub fn n_ctx(&self) -> usize {
        // SAFETY: self.ptr is a live context pointer.
        unsafe { ffi::n_ctx(self.ptr) as usize }
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
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from `ffi::new_context_with_model`,
            // checked non-null on construction. Drop runs at most once.
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
