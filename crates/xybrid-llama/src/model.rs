//! Safe RAII wrapper for a loaded llama.cpp model.
//!
//! Owns `llama_model*` via [`Drop`] glue. Inherent methods over the model
//! pointer cover tokenization, detokenization, special-token lookup, chat
//! template extraction, and the recurrent / hybrid architecture
//! predicates the KV-cache prefix-reuse path gates on.

use std::ffi::{c_void, CStr};

use crate::error::{LlamaError, LlamaResult};
use crate::ffi;

/// Opaque handle to a loaded llama.cpp model.
pub struct LlamaModel {
    ptr: *mut c_void,
}

impl LlamaModel {
    /// Load a model from a GGUF file.
    ///
    /// `n_gpu_layers` is forwarded verbatim to llama.cpp's loader (a
    /// negative value means "all layers on the GPU"; zero forces CPU
    /// inference).
    pub fn load(path: &str, n_gpu_layers: i32) -> LlamaResult<Self> {
        let c_path = ffi::cstring(path, "model path")?;
        // SAFETY: `c_path` outlives the call; `n_gpu_layers` is plain
        // data. A null return surfaces as `LoadFailed`.
        let ptr = unsafe { ffi::load_model_from_file(&c_path, n_gpu_layers) };
        if ptr.is_null() {
            return Err(LlamaError::LoadFailed(path.to_string()));
        }
        Ok(Self { ptr })
    }

    /// Returns the raw pointer for the `xybrid-llama` internals.
    ///
    /// `pub(crate)` so neither downstream nor docs.rs renders it as
    /// callable. Consumers stay on the safe surface.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const c_void {
        self.ptr
    }

    /// Tokenize `text` without special-token parsing.
    ///
    /// Returns the model's preferred token ID sequence. `add_special`
    /// controls whether the BOS token is prepended (varies per model
    /// family).
    pub fn tokenize(&self, text: &str, add_special: bool) -> LlamaResult<Vec<i32>> {
        self.tokenize_internal(text, add_special, false)
    }

    /// Tokenize `text` with special-token parsing enabled.
    ///
    /// Special tokens such as `<|im_end|>`, `<start_of_turn>`,
    /// `<end_of_turn>` are recognized and emitted as their dedicated
    /// token IDs instead of being broken into characters. Use this for
    /// chat-templated prompts and for stop sequences that reference
    /// special tokens.
    pub fn tokenize_special(&self, text: &str, add_special: bool) -> LlamaResult<Vec<i32>> {
        self.tokenize_internal(text, add_special, true)
    }

    fn tokenize_internal(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> LlamaResult<Vec<i32>> {
        let c_text = ffi::cstring(text, "tokenize text")?;
        // SAFETY: `c_text` outlives both calls. Probe + real call follow
        // llama.cpp's documented two-pass tokenization protocol.
        let probe = unsafe {
            ffi::tokenize_probe(self.ptr, &c_text, text.len(), add_special, parse_special)
        };
        let required = if probe < 0 { -probe } else { probe };
        if required <= 0 {
            return Ok(Vec::new());
        }
        let capacity = required as usize + 16; // padding for safety
        let mut tokens = vec![0i32; capacity];
        // SAFETY: `tokens` is writable for `capacity` i32 elements.
        let result = unsafe {
            ffi::tokenize_into(
                self.ptr,
                &c_text,
                text.len(),
                tokens.as_mut_ptr(),
                capacity,
                add_special,
                parse_special,
            )
        };
        if result < 0 {
            return Err(LlamaError::TokenizationFailed);
        }
        tokens.truncate(result as usize);
        Ok(tokens)
    }

    /// Detokenize `tokens` to a UTF-8 string, rendering special tokens
    /// (e.g. `<|im_end|>`) verbatim in their text form.
    pub fn detokenize(&self, tokens: &[i32]) -> LlamaResult<String> {
        let mut result = String::new();
        let mut buf = vec![0u8; 256];
        for &token in tokens {
            // SAFETY: `buf` writable for `buf.len()` bytes. The C wrapper
            // returns the number of bytes written (or required); we
            // re-allocate and retry on overflow exactly once, matching
            // the pre-refactor behavior.
            let len = unsafe {
                ffi::token_to_piece(
                    self.ptr,
                    token,
                    buf.as_mut_ptr() as *mut std::os::raw::c_char,
                    buf.len(),
                    0,
                    true,
                )
            };
            if len > 0 {
                let len_usize = len as usize;
                if len_usize >= buf.len() {
                    buf.resize(len_usize + 1, 0);
                    // SAFETY: buf resized above.
                    let retry_len = unsafe {
                        ffi::token_to_piece(
                            self.ptr,
                            token,
                            buf.as_mut_ptr() as *mut std::os::raw::c_char,
                            buf.len(),
                            0,
                            true,
                        )
                    };
                    if retry_len > 0 {
                        if let Ok(piece) = std::str::from_utf8(&buf[..retry_len as usize]) {
                            result.push_str(piece);
                        }
                    }
                } else if let Ok(piece) = std::str::from_utf8(&buf[..len_usize]) {
                    result.push_str(piece);
                }
            }
        }
        Ok(result)
    }

    /// BOS (beginning-of-sequence) token.
    pub fn token_bos(&self) -> i32 {
        // SAFETY: self.ptr is a live model pointer.
        unsafe { ffi::token_bos(self.ptr) }
    }

    /// Primary EOS token. For end-of-generation checks across modern
    /// multi-EOG vocabularies, prefer [`Self::vocab_is_eog`].
    pub fn token_eos(&self) -> i32 {
        // SAFETY: self.ptr is a live model pointer.
        unsafe { ffi::token_eos(self.ptr) }
    }

    /// True when `token` is registered as an end-of-generation token in
    /// the model vocabulary. Covers ALL EOG tokens (Llama 3:
    /// `<|eot_id|>` + `<|end_of_text|>`; Gemma: `<end_of_turn>`;
    /// Qwen: `<|im_end|>` + `<|endoftext|>`).
    pub fn vocab_is_eog(&self, token: i32) -> bool {
        // SAFETY: self.ptr is a live model pointer.
        unsafe { ffi::vocab_is_eog(self.ptr, token) }
    }

    /// Vocabulary size.
    pub fn n_vocab(&self) -> usize {
        // SAFETY: self.ptr is a live model pointer.
        unsafe { ffi::n_vocab(self.ptr) as usize }
    }

    /// Returns `true` for fully recurrent architectures (Mamba, RWKV).
    /// Most callers want [`Self::has_recurrent_state`] instead — it
    /// additionally covers hybrid models (LFM2, Qwen35, Granite-hybrid,
    /// …) that share the same KV-cache truncation hazard.
    pub fn is_recurrent(&self) -> bool {
        // SAFETY: self.ptr is a live model pointer.
        unsafe { ffi::model_is_recurrent(self.ptr) }
    }

    /// Returns `true` for any model with recurrent state — fully
    /// recurrent (Mamba, RWKV) or hybrid (LFM2 / LFM2MOE, Qwen35 /
    /// Qwen35MOE, Granite-hybrid, …). Callers that manipulate the KV
    /// cache by position — in particular the multi-turn prefix-reuse
    /// path in `LlamaCppBackend::prepare_kv_cache_and_get_tail` — must
    /// skip those optimisations on these models and full-clear the
    /// cache between turns instead. Truncating recurrent state
    /// mid-sequence leaves the residual state inconsistent with the new
    /// prefix length and `llama_decode` fails on the diverging tail
    /// (wrapper error code -3 — see [`LlamaError::DecodeFailed`]).
    pub fn has_recurrent_state(&self) -> bool {
        // SAFETY: self.ptr is a live model pointer.
        unsafe { ffi::model_has_recurrent_state(self.ptr) }
    }

    /// The model's built-in chat template, if any, from GGUF metadata.
    /// Returns `None` when the model doesn't embed a template — callers
    /// fall back to their preferred default (commonly ChatML).
    pub fn chat_template(&self) -> Option<String> {
        // SAFETY: self.ptr is a live model pointer. The returned C
        // string is owned by llama.cpp; we copy it into an owned String
        // before any further C-side mutation could invalidate it.
        let ptr = unsafe { ffi::model_chat_template(self.ptr) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: ptr is a non-null, NUL-terminated C string owned by
        // llama.cpp, valid for the model's lifetime.
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .ok()
            .map(|s| s.to_string())
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from `ffi::load_model_from_file` (or
            // equivalent), checked non-null on construction. Drop runs
            // at most once per handle.
            unsafe { ffi::free_model(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

// Send + Sync audit (epic Open decision §3 / brief Phase 2.9).
//
// Classification of every `llama_model_*_c` symbol consumed via
// `ffi::*` and called above:
//
//   - llama_load_model_from_file_c  : constructor, never aliased
//   - llama_free_model_c            : Drop only, never aliased
//   - llama_tokenize_c              : read-only over model vocab; the
//                                     mutable write target is the caller's
//                                     out-buffer, not model state
//   - llama_token_to_piece_c        : read-only over model vocab + writes
//                                     to caller's out-buffer
//   - llama_token_bos_c             : read-only
//   - llama_token_eos_c             : read-only
//   - llama_vocab_is_eog_c          : read-only
//   - llama_model_chat_template_c   : read-only (returns interior C string)
//   - llama_n_vocab_c               : read-only
//   - llama_model_is_recurrent_c    : read-only
//   - llama_model_has_recurrent_state_c : read-only
//   - llama_format_chat_with_model_c    : read-only over model template;
//                                         writes to caller's out-buffer
//
// All non-constructor / non-Drop symbols are read-only against the model
// handle. There is no quantization or reload path exposed in this surface
// that would mutate the model in-place. `Send + Sync` is therefore sound.
//
// Quantization paths in llama.cpp DO mutate model state, but they are not
// reachable through this safe surface (no symbol for them is in
// `llama-cpp-sys::bindings`). If a future binding adds one, this audit
// must be re-run before the symbol joins the allowlist.
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}
