//! Safe RAII wrapper for a loaded llama.cpp model.
//!
//! Owns `llama_model*` through a shared [`ModelHandle`]: the native model is
//! freed when the last of the [`LlamaModel`] and the contexts created from it
//! is dropped. Inherent methods over the model pointer cover tokenization,
//! detokenization, special-token lookup, chat template extraction (default
//! and named), the GGUF metadata llama.cpp loaded, vocabulary entries, and
//! the recurrent / hybrid architecture predicates the KV-cache prefix-reuse
//! path gates on.
//!
//! A model from [`LlamaModel::load_vocab_only`] has all of that but no
//! tensors: it tokenizes and answers lookups, and no context runs it.

use std::ffi::{c_void, CStr};
use std::sync::Arc;

use crate::error::{LlamaError, LlamaResult};
use crate::ffi;

/// Opaque handle to a loaded llama.cpp model.
///
/// A [`LlamaContext`](crate::LlamaContext) or `MtmdContext` created from the
/// model shares its native handle, so dropping the `LlamaModel` first is safe:
/// the native model stays loaded until the last of them is dropped.
pub struct LlamaModel {
    handle: Arc<ModelHandle>,
    /// Loaded by [`Self::load_vocab_only`]: no context may be created.
    vocab_only: bool,
}

/// The native `llama_model*`, freed when its last owner drops.
///
/// [`LlamaModel`] and every context created from it hold one, because native
/// contexts keep reading the model they were created from.
pub(crate) struct ModelHandle {
    ptr: *mut c_void,
}

impl ModelHandle {
    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for ModelHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from `ffi::load_model_from_file` or
            // `ffi::load_model_vocab_only`, checked non-null on
            // construction. Drop runs once, after every `LlamaModel` and
            // context sharing this handle is gone.
            unsafe { ffi::free_model(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

impl LlamaModel {
    /// Load a model from a GGUF file.
    ///
    /// `n_gpu_layers` is forwarded verbatim to llama.cpp's loader (a
    /// negative value means "all layers on the GPU"; zero forces CPU
    /// inference).
    ///
    /// # Errors
    ///
    /// [`LlamaError::LoadFailed`] when llama.cpp cannot load the file (it
    /// reports its own exceptions, an allocation failure included, this
    /// way), [`LlamaError::Internal`] when an exception escaped llama.cpp,
    /// and [`LlamaError::InvalidInput`] when `path` contains a NUL byte.
    pub fn load(path: &str, n_gpu_layers: i32) -> LlamaResult<Self> {
        let c_path = ffi::cstring(path, "model path")?;
        // SAFETY: `c_path` outlives the call; `n_gpu_layers` is plain
        // data. A null model surfaces as `LoadFailed`.
        let ptr = unsafe { ffi::load_model_from_file(&c_path, n_gpu_layers) }?;
        if ptr.is_null() {
            return Err(LlamaError::LoadFailed(path.to_string()));
        }
        Ok(Self {
            handle: Arc::new(ModelHandle { ptr }),
            vocab_only: false,
        })
    }

    /// Load only the vocabulary and metadata of a GGUF file, on the CPU.
    ///
    /// llama.cpp parses the whole header, tensor descriptions included (the
    /// file may have none), builds the vocabulary, and reads no tensor data:
    /// nothing is mapped and no device is used. It still checks that each
    /// described tensor's data lies inside the file, so a header cut from a
    /// model is refused: drop its tensor entries too. The model tokenizes and
    /// answers the vocabulary, metadata and named chat template lookups as
    /// one from [`Self::load`] does, but cannot run: every context
    /// constructor refuses it with [`LlamaError::VocabOnly`]. One answer
    /// differs: [`Self::chat_template`] loses llama.cpp's one built-in
    /// fallback that counts the model's layers (for Mistral-Small-2503), so
    /// it can be `None` where a full load has a template.
    ///
    /// llama.cpp needs no backend for this load, but logs it, so xybrid's
    /// log policy is applied first ([`crate::backend_init`]).
    ///
    /// # Errors
    ///
    /// [`LlamaError::LoadFailed`] when llama.cpp cannot load the file (it
    /// reports its own exceptions, an allocation failure included, this
    /// way), [`LlamaError::Internal`] when an exception escaped llama.cpp,
    /// and [`LlamaError::InvalidInput`] when `path` contains a NUL byte.
    pub fn load_vocab_only(path: &str) -> LlamaResult<Self> {
        let c_path = ffi::cstring(path, "model path")?;
        crate::backend_init();
        // SAFETY: `c_path` outlives the call. A null model surfaces as
        // `LoadFailed`.
        let ptr = unsafe { ffi::load_model_vocab_only(&c_path) }?;
        if ptr.is_null() {
            return Err(LlamaError::LoadFailed(path.to_string()));
        }
        Ok(Self {
            handle: Arc::new(ModelHandle { ptr }),
            vocab_only: true,
        })
    }

    /// Whether the model came from [`Self::load_vocab_only`], so that no
    /// context can run it.
    pub fn is_vocab_only(&self) -> bool {
        self.vocab_only
    }

    /// Refuses a model from [`Self::load_vocab_only`]: context constructors
    /// call this before any native call.
    pub(crate) fn ensure_runnable(&self) -> LlamaResult<()> {
        if self.vocab_only {
            return Err(LlamaError::VocabOnly);
        }
        Ok(())
    }

    /// Returns the raw pointer for the `xybrid-llama` internals.
    ///
    /// `pub(crate)` so neither downstream nor docs.rs renders it as
    /// callable. Consumers stay on the safe surface.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const c_void {
        self.handle.as_ptr()
    }

    /// The shared native handle, for contexts that must keep the model loaded.
    #[inline]
    pub(crate) fn handle(&self) -> &Arc<ModelHandle> {
        &self.handle
    }

    /// A handle with no native model, for tests. Only native wrappers that
    /// reject a null model may receive it.
    #[cfg(test)]
    pub(crate) fn test_stub() -> Self {
        Self {
            handle: Arc::new(ModelHandle {
                ptr: std::ptr::null_mut(),
            }),
            vocab_only: false,
        }
    }

    /// [`Self::test_stub`] marked vocabulary-only, as from
    /// [`Self::load_vocab_only`].
    #[cfg(test)]
    pub(crate) fn test_stub_vocab_only() -> Self {
        Self {
            vocab_only: true,
            ..Self::test_stub()
        }
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
            ffi::tokenize_probe(
                self.as_ptr(),
                &c_text,
                text.len(),
                add_special,
                parse_special,
            )
        };
        // i32::MIN is a failure: a result too large for llama.cpp, or an
        // exception the shim caught. Any other negative probe is the size.
        if probe == i32::MIN {
            return Err(LlamaError::TokenizationFailed);
        }
        let required = probe.unsigned_abs();
        if required == 0 {
            return Ok(Vec::new());
        }
        let capacity = required as usize + 16; // padding for safety
        let mut tokens = vec![0i32; capacity];
        // SAFETY: `tokens` is writable for `capacity` i32 elements.
        let result = unsafe {
            ffi::tokenize_into(
                self.as_ptr(),
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
                    self.as_ptr(),
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
                            self.as_ptr(),
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
        // SAFETY: the handle keeps the model pointer live.
        unsafe { ffi::token_bos(self.as_ptr()) }
    }

    /// Primary EOS token. For end-of-generation checks across modern
    /// multi-EOG vocabularies, prefer [`Self::vocab_is_eog`].
    pub fn token_eos(&self) -> i32 {
        // SAFETY: the handle keeps the model pointer live.
        unsafe { ffi::token_eos(self.as_ptr()) }
    }

    /// True when `token` is registered as an end-of-generation token in
    /// the model vocabulary. Covers ALL EOG tokens (Llama 3:
    /// `<|eot_id|>` + `<|end_of_text|>`; Gemma: `<end_of_turn>`;
    /// Qwen: `<|im_end|>` + `<|endoftext|>`).
    pub fn vocab_is_eog(&self, token: i32) -> bool {
        // SAFETY: the handle keeps the model pointer live.
        unsafe { ffi::vocab_is_eog(self.as_ptr(), token) }
    }

    /// True when tokenizing with `add_special` prepends the BOS token.
    ///
    /// Reads the GGUF `tokenizer.ggml.add_bos_token` flag. When it is
    /// unset (LFM3), the BOS token only reaches the model if the prompt
    /// text carries it, which is how the model's chat template renders
    /// `bos_token`.
    pub fn adds_bos(&self) -> bool {
        // SAFETY: the handle keeps the model pointer live.
        unsafe { ffi::vocab_get_add_bos(self.as_ptr()) }
    }

    /// Vocabulary size.
    pub fn n_vocab(&self) -> usize {
        // SAFETY: the handle keeps the model pointer live.
        unsafe { ffi::n_vocab(self.as_ptr()) as usize }
    }

    /// Returns `true` for fully recurrent architectures (Mamba, RWKV).
    /// Most callers want [`Self::has_recurrent_state`] instead — it
    /// additionally covers hybrid models (LFM2, Qwen35, Granite-hybrid,
    /// …) that share the same KV-cache truncation hazard.
    pub fn is_recurrent(&self) -> bool {
        // SAFETY: the handle keeps the model pointer live.
        unsafe { ffi::model_is_recurrent(self.as_ptr()) }
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
        // SAFETY: the handle keeps the model pointer live.
        unsafe { ffi::model_has_recurrent_state(self.as_ptr()) }
    }

    /// The model's built-in chat template, if any, from GGUF metadata.
    /// Returns `None` when the model doesn't embed a template — callers
    /// fall back to their preferred default (commonly ChatML).
    pub fn chat_template(&self) -> Option<String> {
        // SAFETY: the handle keeps the model pointer live. The returned C
        // string is owned by llama.cpp; we copy it into an owned String
        // before any further C-side mutation could invalidate it.
        let ptr = unsafe { ffi::model_chat_template(self.as_ptr()) };
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

    /// The chat template stored under `tokenizer.chat_template.<name>`,
    /// such as a decision model's `"systemone"`.
    ///
    /// Unlike [`Self::chat_template`], a named template gets no built-in
    /// fallback from llama.cpp: `None` means the GGUF has no such key.
    ///
    /// # Errors
    ///
    /// [`LlamaError::InvalidInput`] when `name` contains a NUL byte, and
    /// [`LlamaError::Internal`] when the template is not UTF-8 or llama.cpp
    /// failed while looking it up (never reported as a missing template).
    pub fn named_chat_template(&self, name: &str) -> LlamaResult<Option<String>> {
        let c_name = ffi::cstring(name, "chat template name")?;
        // SAFETY: the handle keeps the model pointer live and `c_name`
        // outlives the call; the bytes are copied before it returns.
        let bytes = unsafe { ffi::model_chat_template_named(self.as_ptr(), &c_name) }?;
        bytes
            .map(|bytes| {
                String::from_utf8(bytes).map_err(|_| {
                    LlamaError::Internal(format!("the {name:?} chat template is not UTF-8"))
                })
            })
            .transpose()
    }

    /// The number of scalar GGUF metadata keys llama.cpp loaded with the
    /// model. Array-valued keys are not counted: llama.cpp skips them when
    /// it loads the model.
    ///
    /// # Errors
    ///
    /// [`LlamaError::Internal`] when llama.cpp failed to count them.
    pub fn meta_count(&self) -> LlamaResult<usize> {
        // SAFETY: the handle keeps the model pointer live.
        let count = unsafe { ffi::model_meta_count(self.as_ptr()) };
        usize::try_from(count).map_err(|_| ffi::native_failure("counting model metadata"))
    }

    /// The values of `keys` as llama.cpp's own string accessor returns them:
    /// one `None` per key the model does not have as a scalar, in order.
    ///
    /// These are the *effective* values: llama.cpp writes integers in
    /// decimal, booleans as `true`/`false`, floats through `std::to_string`,
    /// so with six decimals (an `f32` of 1.3373984 reads `1.337398`), and
    /// strings up to their first NUL byte. Array-valued keys read `None`:
    /// llama.cpp skips them when it loads the model, so this cannot prove a
    /// key absent from the file (read the GGUF header for that).
    ///
    /// The keys come from the caller, typically the GGUF header: llama.cpp
    /// enumerates its metadata only by index, and each index lookup walks
    /// its map from the start. This makes two native calls, whatever the
    /// number of keys, each a hash lookup per key.
    ///
    /// # Errors
    ///
    /// [`LlamaError::InvalidInput`] when a key contains a NUL byte, and
    /// [`LlamaError::Internal`] when llama.cpp failed while reading them
    /// (never reported as missing keys).
    pub fn meta_values<K: AsRef<str>>(&self, keys: &[K]) -> LlamaResult<Vec<Option<Vec<u8>>>> {
        let keys = keys
            .iter()
            .map(|key| ffi::cstring(key.as_ref(), "metadata key"))
            .collect::<LlamaResult<Vec<_>>>()?;
        // SAFETY: the handle keeps the model pointer live; the keys outlive
        // both calls and the values are copied.
        unsafe { ffi::model_meta_lookup(self.as_ptr(), &keys) }
    }

    /// The value of the scalar metadata key `key`, as
    /// [`Self::meta_values`] reports it, as text, or `None` when the model
    /// has no such scalar key. An array-valued key also reads `None`.
    ///
    /// # Errors
    ///
    /// [`LlamaError::InvalidInput`] when `key` contains a NUL byte, and
    /// [`LlamaError::Internal`] when the value is not UTF-8 or llama.cpp
    /// failed while looking it up (never reported as a missing key).
    pub fn meta_str(&self, key: &str) -> LlamaResult<Option<String>> {
        let c_key = ffi::cstring(key, "metadata key")?;
        let value = ffi::read_meta_string(|buf, len| {
            // SAFETY: the handle keeps the model pointer live; `c_key`
            // outlives the call and `buf` is writable for `len` bytes, or
            // null with `len == 0`.
            unsafe { ffi::model_meta_val(self.as_ptr(), &c_key, buf, len) }
        })?;
        value
            .map(|bytes| {
                String::from_utf8(bytes).map_err(|_| {
                    LlamaError::Internal(format!("the value of metadata key {key:?} is not UTF-8"))
                })
            })
            .transpose()
    }

    /// The attributes of vocabulary entry `token`, or `None` when it is
    /// outside the vocabulary.
    ///
    /// # Errors
    ///
    /// [`LlamaError::Internal`] when llama.cpp failed while reading the
    /// entry (never reported as an id outside the vocabulary).
    pub fn token_attr(&self, token: i32) -> LlamaResult<Option<TokenAttr>> {
        // SAFETY: the handle keeps the model pointer live; the shim checks
        // the id against the vocabulary.
        let bits = unsafe { ffi::vocab_get_attr(self.as_ptr(), token) }?;
        Ok(bits.map(TokenAttr::from_bits))
    }

    /// The text vocabulary entry `token` stores, such as `<|im_start|>` for
    /// a control token, or `None` when it is outside the vocabulary.
    ///
    /// This is the raw entry, not the rendered piece
    /// [`Self::detokenize`] produces: byte-level BPE entries keep their
    /// byte-to-character mapping (a leading space reads `Ġ`).
    ///
    /// # Errors
    ///
    /// As [`Self::token_attr`].
    pub fn token_text(&self, token: i32) -> LlamaResult<Option<Vec<u8>>> {
        // SAFETY: the handle keeps the model pointer live; the shim checks
        // the id against the vocabulary and the bytes are copied.
        unsafe { ffi::vocab_get_text(self.as_ptr(), token) }
    }

    /// Whether the model has an encoder (T5-style encoder-decoder models).
    ///
    /// llama.cpp ignores a context's output cap for these and reserves a
    /// full batch of output rows instead.
    ///
    /// # Errors
    ///
    /// [`LlamaError::Internal`] when llama.cpp failed to answer.
    pub fn has_encoder(&self) -> LlamaResult<bool> {
        // SAFETY: the handle keeps the model pointer live.
        unsafe { ffi::model_has_encoder(self.as_ptr()) }
            .ok_or_else(|| ffi::native_failure("checking the model for an encoder"))
    }
}

/// The attributes llama.cpp's vocabulary records for a token: a set of the
/// `LLAMA_TOKEN_ATTR_*` bits.
///
/// [`TokenAttr::CONTROL`] marks the special tokens a chat or decision
/// template spells out (`<|im_start|>`), which special-token parsing turns
/// back into single tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenAttr(u32);

impl TokenAttr {
    /// No attributes.
    pub const UNDEFINED: Self = Self(0);
    /// The unknown token.
    pub const UNKNOWN: Self = Self(1 << 0);
    /// An unused vocabulary slot.
    pub const UNUSED: Self = Self(1 << 1);
    /// An ordinary token.
    pub const NORMAL: Self = Self(1 << 2);
    /// A control (special) token.
    pub const CONTROL: Self = Self(1 << 3);
    /// A user-defined token, matched as text before tokenization.
    pub const USER_DEFINED: Self = Self(1 << 4);
    /// A byte-fallback token.
    pub const BYTE: Self = Self(1 << 5);
    /// A token stored normalized.
    pub const NORMALIZED: Self = Self(1 << 6);
    /// Strips whitespace on its left.
    pub const LSTRIP: Self = Self(1 << 7);
    /// Strips whitespace on its right.
    pub const RSTRIP: Self = Self(1 << 8);
    /// Matches only as a single word.
    pub const SINGLE_WORD: Self = Self(1 << 9);

    /// The attributes with these `LLAMA_TOKEN_ATTR_*` bits.
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// The `LLAMA_TOKEN_ATTR_*` bits.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether every bit of `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether any bit of `other` is set.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

// Send + Sync audit (epic Open decision §3 / brief Phase 2.9).
//
// Classification of every `llama_model_*_c` symbol consumed via
// `ffi::*` and called above:
//
//   - llama_load_model_from_file_c  : constructor, never aliased
//   - llama_load_model_vocab_only_c : constructor, never aliased
//   - llama_free_model_c            : Drop only, never aliased
//   - llama_tokenize_c              : read-only over model vocab; the
//                                     mutable write target is the caller's
//                                     out-buffer, not model state
//   - llama_token_to_piece_c        : read-only over model vocab + writes
//                                     to caller's out-buffer
//   - llama_token_bos_c             : read-only
//   - llama_token_eos_c             : read-only
//   - llama_vocab_is_eog_c          : read-only
//   - llama_vocab_get_add_bos_c     : read-only
//   - llama_model_chat_template_c   : read-only (returns interior C string)
//   - llama_model_chat_template_named_c : read-only (returns interior C string)
//   - llama_model_meta_count_c / _val_str_c / _lookup_c : read-only over
//                                     the loaded metadata map (concurrent
//                                     finds on a const unordered_map); write
//                                     to caller's out-buffers
//   - llama_vocab_get_attr_c        : read-only (range-checked by the shim)
//   - llama_vocab_get_text_c        : read-only (returns interior C string)
//   - llama_n_vocab_c               : read-only
//   - llama_model_is_recurrent_c    : read-only
//   - llama_model_has_recurrent_state_c : read-only
//   - llama_model_has_encoder_c     : read-only
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
//
// The audit applies to `ModelHandle`, which owns the pointer. `LlamaModel`
// is `Send + Sync` through its `Arc`, which contexts on other threads share.
unsafe impl Send for ModelHandle {}
unsafe impl Sync for ModelHandle {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_models::{run_test_in_child, staged_d1};

    #[test]
    fn a_vocab_only_load_reports_a_file_it_cannot_load() {
        let path = "/nonexistent/xybrid/vocab-only.gguf";
        match LlamaModel::load_vocab_only(path) {
            Err(LlamaError::LoadFailed(failed)) => assert_eq!(failed, path),
            other => panic!("{:?}", other.map(|_| ())),
        }
        assert!(matches!(
            LlamaModel::load_vocab_only("nul\0path").map(|_| ()),
            Err(LlamaError::InvalidInput(_))
        ));
    }

    /// The stub has no native model, so a constructor that reached
    /// llama.cpp would fail some other way.
    #[test]
    fn every_context_constructor_refuses_a_vocab_only_model() {
        let model = LlamaModel::test_stub_vocab_only();
        assert!(model.is_vocab_only());
        assert!(!LlamaModel::test_stub().is_vocab_only());
        let calls = ffi::native_calls();
        assert!(matches!(
            crate::LlamaContext::new(&model, 512, 1, 512, false),
            Err(LlamaError::VocabOnly)
        ));
        // Before its parameters are checked, too.
        assert!(matches!(
            crate::MemorylessContext::new(&model, crate::MemorylessParams::new(0, 0, false)),
            Err(LlamaError::VocabOnly)
        ));
        #[cfg(feature = "vision")]
        assert!(matches!(
            crate::MtmdContext::from_file(
                "/nonexistent/mmproj.gguf",
                &model,
                false,
                false,
                1,
                false,
                None
            ),
            Err(LlamaError::VocabOnly)
        ));
        assert_eq!(ffi::native_calls(), calls);
    }

    #[cfg(target_os = "linux")]
    const FILE_USE_ENV: &str = "XYBRID_VOCAB_ONLY_FILE_USE";

    /// Whether this process maps `path`, and whether it holds it open.
    #[cfg(target_os = "linux")]
    fn file_use(path: &std::path::Path) -> (bool, bool) {
        let path = std::fs::canonicalize(path).expect("a canonical path");
        let name = path.to_str().expect("a UTF-8 path");
        let maps = std::fs::read_to_string("/proc/self/maps").expect("read /proc/self/maps");
        let mapped = maps.lines().any(|line| line.ends_with(name));
        let open = std::fs::read_dir("/proc/self/fd")
            .expect("list /proc/self/fd")
            .filter_map(Result::ok)
            .any(|fd| std::fs::read_link(fd.path()).is_ok_and(|target| target == path));
        (mapped, open)
    }

    /// A vocabulary-only load reads no tensor: llama.cpp neither maps the
    /// file nor keeps it open, where a full load maps it. In a child
    /// process, where no other test holds the model.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_vocab_only_load_neither_maps_nor_keeps_its_file() {
        if staged_d1().is_none() {
            return;
        }
        let output = run_test_in_child("model::tests::vocab_only_file_use", &[(FILE_USE_ENV, "1")]);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report = stdout
            .lines()
            .find_map(|line| {
                line.split_once("FILE USE ")
                    .map(|(_, rest)| rest.to_string())
            })
            .unwrap_or_else(|| panic!("no report in {stdout}"));
        assert!(
            report.starts_with("vocab-only (false, false), full (true, "),
            "(mapped, open): {report}"
        );
    }

    /// The child of [`a_vocab_only_load_neither_maps_nor_keeps_its_file`]
    /// (returns at once in a normal test run).
    #[cfg(target_os = "linux")]
    #[test]
    fn vocab_only_file_use() {
        if std::env::var_os(FILE_USE_ENV).is_none() {
            return;
        }
        let path = staged_d1().expect("the parent checked the model is staged");
        let name = path.to_str().expect("a UTF-8 path");
        let rss_kb = || {
            std::fs::read_to_string("/proc/self/status")
                .expect("read /proc/self/status")
                .lines()
                .find_map(|line| line.strip_prefix("VmRSS:"))
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|kb| kb.parse::<i64>().ok())
                .expect("VmRSS")
        };
        let (rss, start) = (rss_kb(), std::time::Instant::now());
        let vocab = LlamaModel::load_vocab_only(name).expect("a vocabulary-only load");
        let vocab_cost = (start.elapsed(), (rss_kb() - rss) / 1024);
        let vocab_use = file_use(&path);
        let (rss, start) = (rss_kb(), std::time::Instant::now());
        let full = LlamaModel::load(name, 0).expect("a full load");
        let full_cost = (start.elapsed(), (rss_kb() - rss) / 1024);
        let full_use = file_use(&path);
        drop((vocab, full));
        println!("\nFILE USE vocab-only {vocab_use:?}, full {full_use:?}");
        eprintln!("vocab-only load: {vocab_cost:?} (time, +MiB RSS); full load: {full_cost:?}");
    }

    #[test]
    fn metadata_takes_two_native_calls_whatever_the_number_of_keys() {
        let Some(path) = staged_d1() else { return };
        crate::backend_init();
        let model = LlamaModel::load(path.to_str().unwrap(), 0).expect("load d1-2B");
        let mut keys: Vec<String> = (0..10_000).map(|i| format!("missing.{i}")).collect();
        keys.push("lfm3.decision.type".to_string());

        let calls = ffi::native_calls();
        let values = model.meta_values(&keys).expect("one bulk lookup");
        assert_eq!(ffi::native_calls() - calls, 2, "size, then copy");
        assert!(values[..10_000].iter().all(Option::is_none));
        assert_eq!(values[10_000].as_deref(), Some(&b"lfm3-d1"[..]));
        assert!(model.meta_values::<&str>(&[]).unwrap().is_empty());
    }
}
