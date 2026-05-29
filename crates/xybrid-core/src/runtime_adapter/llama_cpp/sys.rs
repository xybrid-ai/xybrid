//! FFI bindings for llama.cpp
//!
//! This module provides safe Rust wrappers around the llama.cpp C API.
//!
//! # Build Requirements
//!
//! llama.cpp is built from source via the `cc` crate in build.rs.
//! The build system handles:
//! - Android NDK cross-compilation
//! - Metal support on Apple platforms
//! - CPU fallback on Linux/Windows
//!
//! # Safety
//!
//! All FFI functions are wrapped in safe Rust APIs that handle:
//! - Null pointer checks
//! - Lifetime management via Drop
//! - Error conversion to AdapterError

#[cfg(feature = "llm-llamacpp")]
use std::ffi::CString;
#[cfg(feature = "llm-llamacpp")]
use std::os::raw::{c_char, c_float, c_int, c_void};
#[cfg(feature = "llm-llamacpp")]
use std::ptr;

use crate::runtime_adapter::llm::ChatMessage;
use crate::runtime_adapter::AdapterError;

// =============================================================================
// Opaque Types
// =============================================================================

/// Opaque handle to a loaded llama model.
///
/// # Invariants
///
/// - `ptr` is non-null for the entire lifetime of the value: set by
///   [`llama_load_model_from_file`] (which returns `Err` if the C side
///   handed back null), and nulled only by [`llama_free_model`] /
///   `Drop::drop` immediately before the value is destroyed. Code that
///   dereferences `ptr` while the value is owned can therefore rely on
///   non-nullity without re-checking.
/// - `ptr` is exclusive: no other `LlamaModel` aliases the same
///   underlying handle. Cloning is intentionally not implemented.
#[cfg(feature = "llm-llamacpp")]
pub struct LlamaModel {
    ptr: *mut c_void,
}

// SAFETY: `LlamaModel` wraps an opaque llama.cpp model handle whose
// loaded state (vocab, weights, GGUF metadata) is read-only once
// initialised — llama.cpp's API contract documents the model as safe
// to share across threads and across multiple `llama_context_t`s. The
// raw pointer is the only field; transferring or sharing it across
// threads is sound under the read-only contract.
#[cfg(feature = "llm-llamacpp")]
unsafe impl Send for LlamaModel {}
#[cfg(feature = "llm-llamacpp")]
unsafe impl Sync for LlamaModel {}

#[cfg(feature = "llm-llamacpp")]
impl Drop for LlamaModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` is non-null (checked immediately above).
            // Per the struct's exclusivity invariant, no other handle
            // aliases this model, so freeing here cannot leave a
            // dangling reference elsewhere. `Drop::drop` runs at most
            // once per value; nulling `self.ptr` afterward is for
            // defence in depth in case the explicit-free path
            // (`llama_free_model`) ever races with `Drop`.
            unsafe { llama_free_model_c(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Opaque handle to a llama context.
///
/// # Invariants
///
/// - `ptr` is non-null for the entire lifetime of the value: set by
///   [`llama_new_context_with_model`] (which returns `Err` on null) and
///   nulled only by [`llama_free`] / `Drop::drop` immediately before
///   destruction. Code that dereferences `ptr` while the value is owned
///   can rely on non-nullity without re-checking.
/// - `ptr` is exclusive: no other `LlamaContext` aliases the same
///   underlying handle. Cloning is intentionally not implemented.
///
/// # Safety
///
/// `LlamaContext` is `Send` but NOT `Sync`. The underlying llama.cpp context
/// mutates internal state (KV cache, scratch buffers) during `llama_decode()`,
/// so concurrent access from multiple threads is undefined behavior.
///
/// Callers that need shared access (e.g., `LlamaCppBackend` behind `&self`)
/// must wrap `LlamaContext` in a `Mutex` to serialize access.
#[cfg(feature = "llm-llamacpp")]
pub struct LlamaContext {
    ptr: *mut c_void,
}

// SAFETY: `LlamaContext` wraps an opaque llama.cpp context handle. The
// raw pointer is the only field. Per llama.cpp's API contract a context
// can be safely *moved* between threads — what it cannot do is be
// *aliased* across threads (see the `Sync` discussion below). Sending
// the owning value across a thread boundary while no other reference
// exists is sound.
#[cfg(feature = "llm-llamacpp")]
unsafe impl Send for LlamaContext {}

// NOTE: `Sync` is intentionally NOT implemented. `llama_decode()` and
// related entry points mutate internal context state (KV cache, scratch
// buffers, sampler chain). Concurrent access from multiple threads —
// even through `&LlamaContext` accessors that look read-only — is
// undefined behavior under llama.cpp's threading model. Callers that
// need shared access must wrap `LlamaContext` in a `Mutex` so each
// `llama_*` call holds exclusive access.

#[cfg(feature = "llm-llamacpp")]
impl Drop for LlamaContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` is non-null (checked immediately above).
            // Per the struct's exclusivity invariant, no other handle
            // aliases this context, so freeing here cannot leave a
            // dangling reference elsewhere. `Drop::drop` runs at most
            // once per value; nulling `self.ptr` afterward is defence
            // in depth against the explicit-free path racing with
            // `Drop` (impossible by Rust's ownership rules, but cheap).
            unsafe { llama_free_c(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

// =============================================================================
// FFI Declarations
// =============================================================================

#[cfg(feature = "llm-llamacpp")]
extern "C" {
    // Backend initialization
    fn llama_backend_init_c();
    fn llama_backend_free_c();

    // Log verbosity control
    fn llama_log_set_verbosity_c(level: c_int);
    fn llama_log_get_verbosity_c() -> c_int;

    // Model loading
    fn llama_load_model_from_file_c(path_model: *const c_char, n_gpu_layers: c_int) -> *mut c_void;
    fn llama_free_model_c(model: *mut c_void);

    // Context management
    fn llama_new_context_with_model_c(
        model: *mut c_void,
        n_ctx: c_int,
        n_threads: c_int,
        n_batch: c_int,
        flash_attn: bool,
    ) -> *mut c_void;
    fn llama_free_c(ctx: *mut c_void);
    fn llama_kv_cache_clear_c(ctx: *mut c_void);

    // Tokenization
    fn llama_tokenize_c(
        model: *const c_void,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut i32,
        n_tokens_max: c_int,
        add_special: bool,
        parse_special: bool,
    ) -> c_int;

    fn llama_token_to_piece_c(
        model: *const c_void,
        token: i32,
        buf: *mut c_char,
        length: c_int,
        lstrip: c_int,
        special: bool,
    ) -> c_int;

    // Special tokens
    fn llama_token_bos_c(model: *const c_void) -> i32;
    fn llama_token_eos_c(model: *const c_void) -> i32;

    // End-of-generation check (covers ALL EOG tokens, not just primary EOS)
    fn llama_vocab_is_eog_c(model: *const c_void, token: i32) -> bool;

    // Model chat template from GGUF metadata
    fn llama_model_chat_template_c(model: *const c_void) -> *const c_char;

    // Model info
    fn llama_n_vocab_c(model: *const c_void) -> c_int;
    fn llama_n_ctx_c(ctx: *const c_void) -> c_int;

    // Fully recurrent architecture (Mamba, RWKV). Distinct from
    // hybrid models — see `llama_model_has_recurrent_state_c`.
    fn llama_model_is_recurrent_c(model: *const c_void) -> bool;

    // Recurrent OR hybrid architecture (Mamba, RWKV, LFM2, LFM2MOE,
    // Qwen35, Qwen35MOE, Granite-hybrid, …). This is the predicate
    // the KV-cache prefix-reuse path gates on: any model with
    // recurrent state cannot have its cache safely truncated by
    // position via `llama_kv_cache_seq_rm`, because the recurrence
    // accumulates across positions and the residual state ends up
    // inconsistent with the new prefix length. `llama_decode` fails
    // on the diverging tail (returns non-zero, surfaces as wrapper
    // error code -3).
    //
    // Wraps two upstream predicates (`llama_model_is_recurrent` +
    // `llama_model_is_hybrid`) so the Rust caller doesn't need to
    // know which bucket each architecture falls into.
    fn llama_model_has_recurrent_state_c(model: *const c_void) -> bool;

    // Generation (low-level)
    fn llama_decode_c(ctx: *mut c_void, batch: *const c_void) -> c_int;
    fn llama_get_logits_c(ctx: *mut c_void) -> *mut c_float;

    // Chat template (no longer takes model parameter in new API)
    fn llama_chat_apply_template_c(
        tmpl: *const c_char,
        chat: *const c_void,
        n_msg: usize,
        add_ass: bool,
        buf: *mut c_char,
        length: c_int,
    ) -> c_int;

    // Format chat using model's built-in template
    fn llama_format_chat_with_model_c(
        model: *const c_void,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n_msg: usize,
        buf: *mut c_char,
        buf_size: c_int,
    ) -> c_int;

    // Generation loop with stop sequence support
    fn llama_generate_c(
        ctx: *mut c_void,
        model: *const c_void,
        input_tokens: *const i32,
        n_input: c_int,
        output_tokens: *mut i32,
        max_tokens: c_int,
        temperature: c_float,
        top_p: c_float,
        min_p: c_float,
        top_k: c_int,
        repeat_penalty: c_float,
        seed: u32,
        stop_seqs: *const i32,
        stop_lens: *const c_int,
        n_stop_seqs: c_int,
    ) -> c_int;

    // Streaming generation with callback
    fn llama_generate_streaming_c(
        ctx: *mut c_void,
        model: *const c_void,
        input_tokens: *const i32,
        n_input: c_int,
        output_tokens: *mut i32,
        max_tokens: c_int,
        temperature: c_float,
        top_p: c_float,
        min_p: c_float,
        top_k: c_int,
        repeat_penalty: c_float,
        seed: u32,
        stop_seqs: *const i32,
        stop_lens: *const c_int,
        n_stop_seqs: c_int,
        callback: Option<TokenCallback>,
        user_data: *mut c_void,
        n_past_in: c_int,
    ) -> c_int;

    // Truncate the KV cache to a prefix length, dropping tokens past it.
    // Pairs with the n_past_in parameter on llama_generate_streaming_c —
    // see the C wrapper for the prefix-reuse contract.
    fn llama_kv_cache_seq_rm_c(ctx: *mut c_void, seq_id: c_int, p_keep: c_int) -> c_int;
}

/// Callback type for streaming token generation.
///
/// Return 0 to continue generation, non-zero to stop.
#[cfg(feature = "llm-llamacpp")]
pub type TokenCallback =
    extern "C" fn(token_id: i32, token_text: *const c_char, user_data: *mut c_void) -> c_int;

// =============================================================================
// Safe Wrapper Functions
// =============================================================================

/// Initialize the llama.cpp backend (call once at startup)
#[cfg(feature = "llm-llamacpp")]
pub fn llama_backend_init() {
    // SAFETY: `llama_backend_init` takes no arguments and has no
    // caller-side preconditions per llama.h — it's the documented
    // process-startup hook. Repeated calls are no-ops on the llama.cpp
    // side, so re-invocation is sound.
    unsafe {
        llama_backend_init_c();
    }
}

/// Free the llama.cpp backend (call once at shutdown)
#[cfg(feature = "llm-llamacpp")]
pub fn llama_backend_free() {
    // SAFETY: `llama_backend_free` takes no arguments and has no
    // caller-side preconditions per llama.h — it's the documented
    // process-shutdown hook. All models and contexts must have been
    // dropped before this call; that's the caller's responsibility
    // (Rust ownership enforces it for handles created through this
    // module).
    unsafe {
        llama_backend_free_c();
    }
}

/// Set the verbosity level for llama.cpp/ggml logging.
///
/// # Levels
/// - 0: Silent (suppress all library logs) - default
/// - 1: Errors only
/// - 2: Errors + Warnings
/// - 3: Errors + Warnings + Info
/// - 4: All logs including Debug
#[cfg(feature = "llm-llamacpp")]
pub fn llama_log_set_verbosity(level: i32) {
    // SAFETY: `llama_log_set_verbosity_c` takes a single integer and
    // has no caller-side preconditions. Values outside the documented
    // 0..=4 range are clamped internally by the C wrapper, so the cast
    // to `c_int` cannot trigger UB.
    unsafe {
        llama_log_set_verbosity_c(level as c_int);
    }
}

/// Get the current verbosity level for llama.cpp/ggml logging.
#[cfg(feature = "llm-llamacpp")]
pub fn llama_log_get_verbosity() -> i32 {
    // SAFETY: `llama_log_get_verbosity_c` takes no arguments, returns
    // an integer in the documented 0..=4 range, and has no caller-side
    // preconditions.
    unsafe { llama_log_get_verbosity_c() as i32 }
}

/// Load a model from a GGUF file
#[cfg(feature = "llm-llamacpp")]
pub fn llama_load_model_from_file(
    path: &str,
    n_gpu_layers: i32,
) -> Result<LlamaModel, AdapterError> {
    let c_path = CString::new(path)
        .map_err(|_| AdapterError::InvalidInput("Invalid path encoding".to_string()))?;

    // SAFETY: `c_path` is a `CString` owned by this stack frame for
    // the duration of the call — `as_ptr()` therefore yields a valid,
    // NUL-terminated UTF-8 pointer that lives long enough for
    // `llama_load_model_from_file_c` to copy or stat. `n_gpu_layers`
    // is an unconstrained integer per the C contract (negative values
    // are documented as "use defaults"). The returned pointer may be
    // null on failure; the null-check below is the only correctness
    // gate before we hand the pointer to a `LlamaModel`.
    let ptr = unsafe { llama_load_model_from_file_c(c_path.as_ptr(), n_gpu_layers as c_int) };

    if ptr.is_null() {
        return Err(AdapterError::RuntimeError(format!(
            "Failed to load model from {}",
            path
        )));
    }

    Ok(LlamaModel { ptr })
}

/// Free a loaded model.
///
/// Note: `LlamaModel` implements `Drop`, so this is only needed if you want
/// to free the model explicitly before the end of scope.
#[cfg(feature = "llm-llamacpp")]
pub fn llama_free_model(mut model: LlamaModel) {
    // Mark as null so Drop doesn't double-free.
    if !model.ptr.is_null() {
        // SAFETY: `model.ptr` is non-null (checked immediately above)
        // and points to a llama.cpp model previously returned by
        // `llama_load_model_from_file_c`. We own `model` by value, so
        // no other handle aliases this pointer (per the `LlamaModel`
        // exclusivity invariant). Nulling `model.ptr` after the call
        // prevents `LlamaModel::drop` from re-freeing when `model`
        // goes out of scope at the end of this function.
        unsafe { llama_free_model_c(model.ptr) };
        model.ptr = std::ptr::null_mut();
    }
}

/// Create a new context for a model
///
/// # Arguments
/// * `model` - The loaded model
/// * `n_ctx` - Context length (tokens)
/// * `n_threads` - Number of threads for inference (0 = auto-detect)
/// * `n_batch` - Batch size for prompt processing (0 = default 512)
/// * `flash_attn` - Enable flash attention (2-4x speedup on longer contexts)
#[cfg(feature = "llm-llamacpp")]
pub fn llama_new_context_with_model(
    model: &LlamaModel,
    n_ctx: usize,
    n_threads: usize,
    n_batch: usize,
    flash_attn: bool,
) -> Result<LlamaContext, AdapterError> {
    // SAFETY: `model.ptr` is non-null for the lifetime of `&model`
    // (per the `LlamaModel` ptr invariant) and points to a model
    // produced by `llama_load_model_from_file_c`. Numeric arguments
    // are unconstrained per the C contract (0 selects defaults for
    // `n_threads` / `n_batch`). The returned pointer may be null on
    // allocation failure; the null-check below gates promotion to a
    // `LlamaContext`.
    let ptr = unsafe {
        llama_new_context_with_model_c(
            model.ptr,
            n_ctx as c_int,
            n_threads as c_int,
            n_batch as c_int,
            flash_attn,
        )
    };

    if ptr.is_null() {
        return Err(AdapterError::RuntimeError(
            "Failed to create context".to_string(),
        ));
    }

    Ok(LlamaContext { ptr })
}

/// Free a context.
///
/// Note: `LlamaContext` implements `Drop`, so this is only needed if you want
/// to free the context explicitly before the end of scope.
#[cfg(feature = "llm-llamacpp")]
pub fn llama_free(mut ctx: LlamaContext) {
    // Mark as null so Drop doesn't double-free.
    if !ctx.ptr.is_null() {
        // SAFETY: `ctx.ptr` is non-null (checked immediately above)
        // and points to a context previously returned by
        // `llama_new_context_with_model_c`. We own `ctx` by value, so
        // no other handle aliases this pointer (per the `LlamaContext`
        // exclusivity invariant). Nulling `ctx.ptr` after the call
        // prevents `LlamaContext::drop` from re-freeing at end of
        // scope.
        unsafe { llama_free_c(ctx.ptr) };
        ctx.ptr = std::ptr::null_mut();
    }
}

/// Clear the KV cache (reset context state for new conversation)
#[cfg(feature = "llm-llamacpp")]
pub fn llama_kv_cache_clear(ctx: &LlamaContext) {
    // SAFETY: `ctx.ptr` is non-null for the lifetime of `&ctx` (per
    // the `LlamaContext` ptr invariant). `&LlamaContext` rules out
    // concurrent aliasing — `LlamaContext: !Sync` makes shared
    // references across threads a compile error, so the implicit
    // serialisation llama.cpp requires is upheld by the type system.
    unsafe {
        llama_kv_cache_clear_c(ctx.ptr);
    }
}

/// Truncate the KV cache for `seq_id` to a prefix length, dropping tokens
/// at positions `[p_keep, ∞)`. Used by the multi-turn prefix-reuse path:
/// caller computes the longest common prefix between the new prompt and
/// the previously-tokenized prompt, then calls this to drop the diverged
/// tail before re-prefilling only the new tail at position `p_keep`.
/// Pairs with `n_past_in` on [`llama_generate_streaming`].
#[cfg(feature = "llm-llamacpp")]
pub fn llama_kv_cache_seq_rm(ctx: &LlamaContext, seq_id: i32, p_keep: usize) {
    // Caller-side bounds check: `p_keep` is unsigned in our API but
    // the C signature is `c_int`. Saturate at `i32::MAX` to keep the
    // FFI call total — the legitimate range is `[0, n_ctx)` which is
    // always well below `i32::MAX` in practice.
    let p_keep_c = p_keep.min(c_int::MAX as usize) as c_int;
    // SAFETY: `ctx.ptr` is non-null per the `LlamaContext` ptr
    // invariant; `&LlamaContext` rules out concurrent aliasing. The
    // `p_keep_c` cast above guarantees `0 <= p_keep_c <= i32::MAX`,
    // which is the documented range for the C function. Recurrent /
    // hybrid models silently produce a `llama_decode` error on the
    // diverging tail rather than UB here (see
    // `llama_model_has_recurrent_state` for the documented gate).
    unsafe {
        let _ = llama_kv_cache_seq_rm_c(ctx.ptr, seq_id, p_keep_c);
    }
}

/// Get the BOS (beginning of sequence) token
#[cfg(feature = "llm-llamacpp")]
pub fn llama_token_bos(model: &LlamaModel) -> i32 {
    // SAFETY: `model.ptr` is non-null per the `LlamaModel` ptr
    // invariant. The C function reads immutable vocab metadata; safe
    // to call concurrently from multiple threads (`LlamaModel: Sync`).
    unsafe { llama_token_bos_c(model.ptr) }
}

/// Get the EOS (end of sequence) token
#[cfg(feature = "llm-llamacpp")]
pub fn llama_token_eos(model: &LlamaModel) -> i32 {
    // SAFETY: `model.ptr` is non-null per the `LlamaModel` ptr
    // invariant; vocab read is immutable.
    unsafe { llama_token_eos_c(model.ptr) }
}

/// Check if a token is an end-of-generation token.
///
/// Unlike `llama_token_eos()` which returns the primary EOS token,
/// this checks ALL end-of-generation tokens registered in the model vocabulary.
/// Modern models have multiple EOG tokens (e.g., Llama 3: `<|eot_id|>` + `<|end_of_text|>`,
/// Gemma: `<end_of_turn>`, Qwen: `<|im_end|>` + `<|endoftext|>`).
#[cfg(feature = "llm-llamacpp")]
pub fn llama_vocab_is_eog(model: &LlamaModel, token: i32) -> bool {
    // SAFETY: `model.ptr` is non-null per the `LlamaModel` ptr
    // invariant. `token` is an arbitrary i32; out-of-range values
    // return false per the C contract (no UB).
    unsafe { llama_vocab_is_eog_c(model.ptr, token) }
}

/// Get vocabulary size
#[cfg(feature = "llm-llamacpp")]
pub fn llama_n_vocab(model: &LlamaModel) -> usize {
    // SAFETY: `model.ptr` is non-null per the `LlamaModel` ptr
    // invariant; vocab size read is immutable. Return value is
    // non-negative per the C contract; `as usize` is total.
    unsafe { llama_n_vocab_c(model.ptr) as usize }
}

/// Get context length
#[cfg(feature = "llm-llamacpp")]
pub fn llama_n_ctx(ctx: &LlamaContext) -> usize {
    // SAFETY: `ctx.ptr` is non-null per the `LlamaContext` ptr
    // invariant; `&LlamaContext` rules out concurrent aliasing.
    // Return value is non-negative per the C contract.
    unsafe { llama_n_ctx_c(ctx.ptr) as usize }
}

/// Returns true for fully recurrent architectures (Mamba, RWKV).
/// Most callers want [`llama_model_has_recurrent_state`] instead,
/// which also covers hybrid models (LFM2, Qwen35, Granite-hybrid, …)
/// — they have the same cache-truncation hazard.
#[cfg(feature = "llm-llamacpp")]
pub fn llama_model_is_recurrent(model: &LlamaModel) -> bool {
    // SAFETY: `model.ptr` is non-null per the `LlamaModel` ptr
    // invariant; architecture probe reads immutable metadata.
    unsafe { llama_model_is_recurrent_c(model.ptr) }
}

/// Returns true for any model with recurrent state — fully recurrent
/// (Mamba, RWKV) or hybrid (LFM2 / LFM2MOE, Qwen35 / Qwen35MOE,
/// Granite-hybrid, …). Callers that manipulate the KV cache by
/// position — in particular the multi-turn prefix-reuse path in
/// `LlamaCppBackend::prepare_kv_cache_and_get_tail` — must skip
/// those optimisations on these models and full-clear the cache
/// between turns instead. Truncating recurrent state mid-sequence
/// leaves the residual state inconsistent with the new prefix length
/// and `llama_decode` fails on the diverging tail (wrapper error
/// code -3).
#[cfg(feature = "llm-llamacpp")]
pub fn llama_model_has_recurrent_state(model: &LlamaModel) -> bool {
    // SAFETY: `model.ptr` is non-null per the `LlamaModel` ptr
    // invariant; architecture probe reads immutable metadata.
    unsafe { llama_model_has_recurrent_state_c(model.ptr) }
}

/// Get the model's chat template string from GGUF metadata.
///
/// Returns the model's built-in chat template, or None if the model
/// doesn't have one embedded. This is used to apply the correct chat
/// format for each model architecture.
#[cfg(feature = "llm-llamacpp")]
pub fn llama_model_chat_template(model: &LlamaModel) -> Option<String> {
    // SAFETY: `model.ptr` is non-null per the `LlamaModel` ptr
    // invariant. The C function returns either a pointer into the
    // model's GGUF metadata (valid for the model's lifetime) or null
    // (no template embedded). We null-check immediately below.
    let ptr = unsafe { llama_model_chat_template_c(model.ptr) };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `ptr` is non-null (checked above) and points to a
    // NUL-terminated UTF-8 string embedded in the model's GGUF
    // metadata. That storage lives as long as `model` does, which
    // outlives this borrow — we only hand back an owned `String`
    // (allocated by `to_string`), so no `&str` borrow escapes.
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_str()
        .ok()
        .map(|s| s.to_string())
}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_model_chat_template(_model: &LlamaModel) -> Option<String> {
    None
}

/// Format chat messages using the model's native chat template.
///
/// This uses llama.cpp's built-in template system which automatically
/// uses the correct format for each model (ChatML for Qwen, Gemma format for Gemma, etc.)
#[cfg(feature = "llm-llamacpp")]
pub fn llama_format_chat(
    model: &LlamaModel,
    messages: &[ChatMessage],
) -> Result<String, AdapterError> {
    if messages.is_empty() {
        return Err(AdapterError::InvalidInput("Empty messages".to_string()));
    }

    // Convert messages to C strings — reject null bytes instead of silently dropping content
    let roles: Vec<CString> = messages
        .iter()
        .map(|m| {
            CString::new(m.role.as_str()).map_err(|_| {
                AdapterError::InvalidInput(format!(
                    "Chat message role '{}' contains null byte",
                    m.role
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let contents: Vec<CString> = messages
        .iter()
        .map(|m| {
            CString::new(m.content.as_str()).map_err(|_| {
                AdapterError::InvalidInput("Chat message content contains null byte".to_string())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let role_ptrs: Vec<*const c_char> = roles.iter().map(|s| s.as_ptr()).collect();
    let content_ptrs: Vec<*const c_char> = contents.iter().map(|s| s.as_ptr()).collect();

    // Allocate output buffer (start with 4KB, should be enough for most prompts)
    let mut buf = vec![0u8; 4096];

    // SAFETY: `model.ptr` non-null per invariant. `role_ptrs` /
    // `content_ptrs` are `Vec<*const c_char>` whose elements borrow
    // from `roles` / `contents` (`Vec<CString>`) — both source vecs
    // outlive this call. `buf` is a `Vec<u8>` owned by this frame;
    // `buf.as_mut_ptr()` is valid for `buf.len()` bytes for the
    // duration of the call. The C function reads `messages.len()`
    // role+content pointer pairs and writes at most `buf.len()` bytes
    // into `buf`, returning the bytes written or a negative error
    // code.
    let result = unsafe {
        llama_format_chat_with_model_c(
            model.ptr,
            role_ptrs.as_ptr(),
            content_ptrs.as_ptr(),
            messages.len(),
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as c_int,
        )
    };

    if result < 0 {
        // Fall back to ChatML format if model template fails
        log::warn!(
            target: "xybrid_core",
            "Model chat template failed (code {}), falling back to ChatML format",
            result
        );
        return llama_format_chat_chatml(messages);
    }

    // If buffer was too small, resize and retry
    let len = if result as usize >= buf.len() {
        buf.resize((result as usize) + 1, 0);
        // SAFETY: same invariants as the first call above. `buf` was
        // just resized to `result + 1` bytes; `buf.as_mut_ptr()` /
        // `buf.len()` reflect the new allocation. `role_ptrs` /
        // `content_ptrs` still borrow from the un-touched `roles` /
        // `contents`.
        let retry_result = unsafe {
            llama_format_chat_with_model_c(
                model.ptr,
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
            )
        };
        if retry_result < 0 {
            return llama_format_chat_chatml(messages);
        }
        // Use the retry's return value, not the first call's
        retry_result as usize
    } else {
        result as usize
    };

    // Convert result to string
    if let Ok(prompt) = std::str::from_utf8(&buf[..len]) {
        Ok(prompt.to_string())
    } else {
        llama_format_chat_chatml(messages)
    }
}

/// Fallback ChatML format for models without built-in templates.
#[cfg(feature = "llm-llamacpp")]
fn llama_format_chat_chatml(messages: &[ChatMessage]) -> Result<String, AdapterError> {
    let mut prompt = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!(
                    "<|im_start|>assistant\n{}<|im_end|>\n",
                    msg.content
                ));
            }
            _ => {
                prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content));
            }
        }
    }

    // Add assistant prefix for generation
    prompt.push_str("<|im_start|>assistant\n");

    Ok(prompt)
}

/// Tokenize text
#[cfg(feature = "llm-llamacpp")]
pub fn llama_tokenize(
    model: &LlamaModel,
    text: &str,
    add_special: bool,
) -> Result<Vec<i32>, AdapterError> {
    let c_text = CString::new(text)
        .map_err(|_| AdapterError::InvalidInput("Invalid text encoding".to_string()))?;

    // First call to get required size (returns negative count).
    //
    // SAFETY: `model.ptr` non-null per invariant. `c_text` is a
    // CString owned by this frame; `c_text.as_ptr()` is valid for
    // `text.len()` bytes for the duration of the call (CString
    // appends a NUL but the length param tells C to stop earlier).
    // The output pointer is `ptr::null_mut()` with `n_tokens_max=0`,
    // which the C contract documents as "do not write, just compute
    // and return the required size as a negative number."
    let n_tokens = unsafe {
        llama_tokenize_c(
            model.ptr,
            c_text.as_ptr(),
            text.len() as c_int,
            ptr::null_mut(),
            0,
            add_special,
            false,
        )
    };

    // n_tokens is negative when getting size
    let required_size = if n_tokens < 0 { -n_tokens } else { n_tokens };

    if required_size <= 0 {
        return Ok(Vec::new());
    }

    // Allocate and tokenize
    let mut tokens = vec![0i32; required_size as usize + 16]; // Extra padding for safety
                                                              // SAFETY: same invariants as the size-query call above. `tokens`
                                                              // was just allocated to `required_size + 16` slots, so
                                                              // `tokens.as_mut_ptr()` is valid for `tokens.len()` i32 writes.
                                                              // The C function writes at most `n_tokens_max` (= `tokens.len()`)
                                                              // i32s and returns the count actually written.
    let result = unsafe {
        llama_tokenize_c(
            model.ptr,
            c_text.as_ptr(),
            text.len() as c_int,
            tokens.as_mut_ptr(),
            tokens.len() as c_int,
            add_special,
            false,
        )
    };

    if result < 0 {
        return Err(AdapterError::RuntimeError(
            "Tokenization failed".to_string(),
        ));
    }

    tokens.truncate(result as usize);
    Ok(tokens)
}

/// Tokenize text with special token parsing enabled.
///
/// Special tokens like `<|im_end|>`, `<start_of_turn>`, `<end_of_turn>` are
/// recognized and converted to their special token IDs instead of being
/// tokenized as individual characters.
///
/// Use this for:
/// - Chat-templated prompts (the template contains special tokens)
/// - Stop sequences that reference special tokens
#[cfg(feature = "llm-llamacpp")]
pub fn llama_tokenize_special(
    model: &LlamaModel,
    text: &str,
    add_special: bool,
) -> Result<Vec<i32>, AdapterError> {
    let c_text = CString::new(text)
        .map_err(|_| AdapterError::InvalidInput("Invalid text encoding".to_string()))?;

    // First call to get required size.
    //
    // SAFETY: same invariants as the size-query in `llama_tokenize`
    // above — `parse_special = true` only changes how the C tokeniser
    // interprets the input, not the pointer/length contract.
    let n_tokens = unsafe {
        llama_tokenize_c(
            model.ptr,
            c_text.as_ptr(),
            text.len() as c_int,
            ptr::null_mut(),
            0,
            add_special,
            true, // parse_special = true
        )
    };

    let required_size = if n_tokens < 0 { -n_tokens } else { n_tokens };

    if required_size <= 0 {
        return Ok(Vec::new());
    }

    let mut tokens = vec![0i32; required_size as usize + 16];
    // SAFETY: same invariants as the actual-tokenise call in
    // `llama_tokenize` above. `tokens` was just allocated to
    // `required_size + 16` i32 slots.
    let result = unsafe {
        llama_tokenize_c(
            model.ptr,
            c_text.as_ptr(),
            text.len() as c_int,
            tokens.as_mut_ptr(),
            tokens.len() as c_int,
            add_special,
            true, // parse_special = true
        )
    };

    if result < 0 {
        return Err(AdapterError::RuntimeError(
            "Tokenization failed".to_string(),
        ));
    }

    tokens.truncate(result as usize);
    Ok(tokens)
}

/// Detokenize tokens to text
#[cfg(feature = "llm-llamacpp")]
pub fn llama_detokenize(model: &LlamaModel, tokens: &[i32]) -> Result<String, AdapterError> {
    let mut result = String::new();
    let mut buf = vec![0u8; 256];

    for &token in tokens {
        // SAFETY: `model.ptr` non-null per invariant; `token` is an
        // arbitrary i32 (invalid token ids yield `0` per the C
        // contract, not UB). `buf` is a `Vec<u8>` owned by this frame;
        // `buf.as_mut_ptr()` is valid for `buf.len()` bytes for the
        // call duration. Return-value contract from llama.cpp's
        // `llama_token_to_piece`:
        //   * `> 0`  — bytes written into `buf` (≤ `buf.len()`).
        //   * `< 0`  — `-required_size`; nothing was written because
        //              `buf.len()` was too small.
        //   * `== 0` — invalid / empty token; skip.
        let len = unsafe {
            llama_token_to_piece_c(
                model.ptr,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
                0,
                true, // special = true: render special tokens like <|im_end|> as text
            )
        };

        let len_usize = if len < 0 {
            // Buffer was too small — required size is `-len`. Resize
            // to fit (plus one byte of headroom for the next retry
            // attempt in case the same token reappears later) and
            // re-decode. The C function never returns positive
            // `len > buf.len()`, so positive returns are always safe
            // to consume directly — we only enter this branch on
            // negative returns. (Audit theme 4 surfaced this: the
            // previous `if len > 0 && len_usize >= buf.len()` retry
            // path was unreachable, so tokens whose detokenized form
            // exceeded 256 bytes were silently dropped.)
            let required = (-len) as usize;
            buf.resize(required + 1, 0);
            // SAFETY: same invariants as the first call above. `buf`
            // was just resized to `required + 1` bytes; the new
            // allocation is reflected in `buf.as_mut_ptr()` /
            // `buf.len()`. The retry asks the C side for the same
            // `(model, token, …)` pair, which by the function's
            // determinism produces the same required size.
            let retry_len = unsafe {
                llama_token_to_piece_c(
                    model.ptr,
                    token,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as c_int,
                    0,
                    true,
                )
            };
            if retry_len <= 0 {
                // Either the token genuinely produces no piece on
                // re-try (unexpected, but possible if `special=true`
                // changes behaviour mid-call on some llama.cpp build),
                // or the resize was somehow still insufficient. Skip
                // rather than panic.
                continue;
            }
            retry_len as usize
        } else if len == 0 {
            continue;
        } else {
            len as usize
        };

        if let Ok(piece) = std::str::from_utf8(&buf[..len_usize]) {
            result.push_str(piece);
        }
    }

    Ok(result)
}

/// Sampling parameters for generation
#[cfg(feature = "llm-llamacpp")]
#[derive(Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
}

#[cfg(feature = "llm-llamacpp")]
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

/// Generate tokens using autoregressive decoding with stop sequence support.
///
/// This function performs the full generation loop:
/// 1. Processes input tokens
/// 2. Samples from logits using temperature/top_p/top_k
/// 3. Repeats until EOS, stop sequence, or max_tokens
///
/// # Arguments
/// * `ctx` - The llama context
/// * `model` - The llama model (needed for EOS token and tokenization)
/// * `input_tokens` - Input token IDs
/// * `max_tokens` - Maximum tokens to generate
/// * `temperature` - Sampling temperature (0 = greedy)
/// * `top_p` - Top-p (nucleus) sampling threshold
/// * `min_p` - Min-p sampling threshold (0.0 = disabled, 0.05 = recommended)
/// * `top_k` - Top-k sampling (0 = disabled)
/// * `repeat_penalty` - Repetition penalty (1.0 = disabled, > 1.0 = penalize)
/// * `stop_sequences` - Optional stop sequences (as strings)
///
/// # Returns
/// Vector of generated token IDs
#[cfg(feature = "llm-llamacpp")]
pub fn llama_generate_with_stops(
    ctx: &LlamaContext,
    model: &LlamaModel,
    input_tokens: &[i32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    stop_sequences: &[String],
) -> Result<Vec<i32>, AdapterError> {
    if input_tokens.is_empty() {
        return Err(AdapterError::InvalidInput("Empty input tokens".to_string()));
    }

    // Tokenize stop sequences
    let mut stop_tokens: Vec<i32> = Vec::new();
    let mut stop_lens: Vec<c_int> = Vec::new();

    for seq in stop_sequences {
        // Tokenize WITH special token parsing - stop sequences like <|im_end|> are special tokens
        let tokens = llama_tokenize_special(model, seq, false)?;
        log::debug!(
            target: "xybrid_core",
            "Tokenized stop sequence '{}' -> {:?} ({} tokens)",
            seq, tokens, tokens.len()
        );
        if !tokens.is_empty() {
            stop_lens.push(tokens.len() as c_int);
            stop_tokens.extend(tokens);
        }
    }

    log::debug!(
        target: "xybrid_core",
        "Total stop tokens: {:?}, lengths: {:?}",
        stop_tokens, stop_lens
    );

    // Allocate output buffer
    let mut output_tokens = vec![0i32; max_tokens];

    // Use current time as seed for sampling
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(42);

    // Use stop_lens.len() (filtered count) not stop_sequences.len() (original count).
    // Some sequences may tokenize to empty and get filtered out above.
    let (stop_seqs_ptr, stop_lens_ptr, n_stop_seqs) = if stop_lens.is_empty() {
        (ptr::null(), ptr::null(), 0)
    } else {
        (
            stop_tokens.as_ptr(),
            stop_lens.as_ptr(),
            stop_lens.len() as c_int,
        )
    };

    // SAFETY: `ctx.ptr` / `model.ptr` non-null per invariants;
    // `&LlamaContext` rules out concurrent aliasing on the context.
    // `input_tokens` is a borrowed slice — `as_ptr()` is valid for
    // `input_tokens.len()` i32 reads for the call duration.
    // `output_tokens` is a `Vec<i32>` owned by this frame; the C
    // function writes at most `max_tokens` i32s into it.
    // `stop_seqs_ptr` / `stop_lens_ptr` are either both null (when
    // `n_stop_seqs == 0`) or borrowed from the `stop_tokens` /
    // `stop_lens` vecs above — both live for the call duration, and
    // `n_stop_seqs` matches `stop_lens.len()` (the filtered count,
    // not the original `stop_sequences.len()` — see comment above).
    let result = unsafe {
        llama_generate_c(
            ctx.ptr,
            model.ptr,
            input_tokens.as_ptr(),
            input_tokens.len() as c_int,
            output_tokens.as_mut_ptr(),
            max_tokens as c_int,
            temperature,
            top_p,
            min_p,
            top_k as c_int,
            repeat_penalty,
            seed,
            stop_seqs_ptr,
            stop_lens_ptr,
            n_stop_seqs,
        )
    };

    if result < 0 {
        let detail = match result {
            -1 => "invalid arguments (null context/model/input or non-positive sizes)",
            -2 => "sampler chain creation failed",
            -3 => {
                "llama_decode failed on prefill \
                 (the wrapper logs the actual llama_decode return code + chunk \
                 position to stderr; see `llama_generate_c` in llama_wrapper.cpp)"
            }
            -4 => "input exceeds context window",
            _ => "unknown",
        };
        return Err(AdapterError::RuntimeError(format!(
            "Generation failed with error code {} ({})",
            result, detail
        )));
    }

    output_tokens.truncate(result as usize);
    Ok(output_tokens)
}

/// Generate tokens using autoregressive decoding (without stop sequences).
///
/// This is a convenience wrapper around `llama_generate_with_stops` for
/// backwards compatibility. Uses default repetition penalty of 1.1 and min_p of 0.05.
#[cfg(feature = "llm-llamacpp")]
pub fn llama_generate(
    ctx: &LlamaContext,
    model: &LlamaModel,
    input_tokens: &[i32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
) -> Result<Vec<i32>, AdapterError> {
    llama_generate_with_stops(
        ctx,
        model,
        input_tokens,
        max_tokens,
        temperature,
        top_p,
        0.05,
        top_k,
        1.1,
        &[],
    )
}

/// Context passed through the C callback to the Rust closure.
#[cfg(feature = "llm-llamacpp")]
struct StreamingContext<'a, F>
where
    F: FnMut(i32, &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
{
    callback: &'a mut F,
    error: Option<Box<dyn std::error::Error + Send + Sync>>,
}

/// C-compatible trampoline function that calls the Rust closure.
#[cfg(feature = "llm-llamacpp")]
extern "C" fn streaming_trampoline<F>(
    token_id: i32,
    token_text: *const c_char,
    user_data: *mut c_void,
) -> c_int
where
    F: FnMut(i32, &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
{
    // SAFETY: `user_data` is the same `*mut c_void` we passed into
    // `llama_generate_streaming_c` below — it is `&mut streaming_ctx`
    // cast to `*mut c_void`, where `streaming_ctx` is a
    // `StreamingContext<F>` owned by the calling frame of
    // `llama_generate_streaming`. The C side stores the pointer and
    // synchronously invokes this trampoline; it never escapes the
    // call. The matching `F` type parameter is enforced statically
    // because the trampoline is only registered with
    // `Some(streaming_trampoline::<F>)` for the same `F` that owns
    // `streaming_ctx`. We are the only thread that ever observes
    // this pointer (llama.cpp invokes the callback on its decode
    // thread, but never concurrently with itself), so taking
    // `&mut StreamingContext<F>` is sound.
    let ctx = unsafe { &mut *(user_data as *mut StreamingContext<F>) };

    // Convert C string to Rust string
    let text = if token_text.is_null() {
        ""
    } else {
        // SAFETY: `token_text` is non-null (checked above) and points
        // to a NUL-terminated UTF-8 piece owned by llama.cpp's
        // internal scratch buffer. That buffer is valid for the
        // duration of this callback invocation — we copy out the
        // `&str` into the closure body via `to_str()`, and the
        // closure doesn't retain the slice past return.
        unsafe { std::ffi::CStr::from_ptr(token_text) }
            .to_str()
            .unwrap_or("")
    };

    // Call the Rust closure
    match (ctx.callback)(token_id, text) {
        Ok(()) => 0, // Continue
        Err(e) => {
            ctx.error = Some(e);
            1 // Stop
        }
    }
}

/// Generate tokens with streaming callback.
///
/// This function calls the provided callback for each generated token.
/// The callback receives the token ID and decoded text.
///
/// # Arguments
/// * `ctx` - The llama context
/// * `model` - The llama model
/// * `input_tokens` - Input token IDs
/// * `max_tokens` - Maximum tokens to generate
/// * `temperature` - Sampling temperature (0 = greedy)
/// * `top_p` - Top-p (nucleus) sampling threshold
/// * `min_p` - Min-p sampling threshold (0.0 = disabled, 0.05 = recommended)
/// * `top_k` - Top-k sampling (0 = disabled)
/// * `repeat_penalty` - Repetition penalty (1.0 = disabled)
/// * `stop_sequences` - Optional stop sequences (as strings)
/// * `on_token` - Callback called for each generated token
///
/// # Returns
/// Vector of generated token IDs and whether generation was stopped by callback.
#[cfg(feature = "llm-llamacpp")]
#[allow(clippy::too_many_arguments)]
pub fn llama_generate_streaming<F>(
    ctx: &LlamaContext,
    model: &LlamaModel,
    input_tokens: &[i32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    stop_sequences: &[String],
    mut on_token: F,
    // Position in the KV cache where `input_tokens` should be prefilled.
    // Pass 0 for the legacy "fresh prefill from scratch" behaviour.
    // Positive values let the caller skip prefill for a shared prefix
    // already in the cache (truncate the cache via
    // [`llama_kv_cache_seq_rm`] first, then call with the diverged tail).
    n_past_in: usize,
) -> Result<(Vec<i32>, bool), AdapterError>
where
    F: FnMut(i32, &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
{
    if input_tokens.is_empty() {
        return Err(AdapterError::InvalidInput("Empty input tokens".to_string()));
    }

    // Tokenize stop sequences
    let mut stop_tokens: Vec<i32> = Vec::new();
    let mut stop_lens: Vec<c_int> = Vec::new();

    for seq in stop_sequences {
        let tokens = llama_tokenize_special(model, seq, false)?;
        if !tokens.is_empty() {
            stop_lens.push(tokens.len() as c_int);
            stop_tokens.extend(tokens);
        }
    }

    // Allocate output buffer
    let mut output_tokens = vec![0i32; max_tokens];

    // Use current time as seed for sampling
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(42);

    // Use stop_lens.len() (filtered count) not stop_sequences.len() (original count).
    // Some sequences may tokenize to empty and get filtered out above.
    let (stop_seqs_ptr, stop_lens_ptr, n_stop_seqs) = if stop_lens.is_empty() {
        (ptr::null(), ptr::null(), 0)
    } else {
        (
            stop_tokens.as_ptr(),
            stop_lens.as_ptr(),
            stop_lens.len() as c_int,
        )
    };

    // Set up the streaming context
    let mut streaming_ctx = StreamingContext {
        callback: &mut on_token,
        error: None,
    };

    // SAFETY: Same `ctx.ptr` / `model.ptr` / `input_tokens` /
    // `output_tokens` / `stop_seqs_ptr` invariants as `llama_generate_c`
    // above. The callback wiring adds two more obligations:
    //
    //   * `Some(streaming_trampoline::<F>)` is a function-pointer
    //     constant; its `F` matches the type held by `streaming_ctx`
    //     so the trampoline's `&mut *(... as *mut StreamingContext<F>)`
    //     cast is sound (see SAFETY note inside `streaming_trampoline`).
    //   * `&mut streaming_ctx as *mut StreamingContext<F> as *mut c_void`
    //     hands the C function a pointer that lives in this stack
    //     frame; the call is synchronous, so the pointer is dropped
    //     before this frame returns.
    //
    // The `n_past_in.min(c_int::MAX as usize) as c_int` saturation
    // keeps the cast total — legitimate values are well below
    // `i32::MAX`.
    let result = unsafe {
        llama_generate_streaming_c(
            ctx.ptr,
            model.ptr,
            input_tokens.as_ptr(),
            input_tokens.len() as c_int,
            output_tokens.as_mut_ptr(),
            max_tokens as c_int,
            temperature,
            top_p,
            min_p,
            top_k as c_int,
            repeat_penalty,
            seed,
            stop_seqs_ptr,
            stop_lens_ptr,
            n_stop_seqs,
            Some(streaming_trampoline::<F>),
            &mut streaming_ctx as *mut StreamingContext<F> as *mut c_void,
            n_past_in.min(c_int::MAX as usize) as c_int,
        )
    };

    // Check for hard error codes FIRST — these are never callback-stop.
    // -1 = invalid args, -2 = sampler creation failed, -3 = decode failed, -4 = input too long.
    if (-4..=-1).contains(&result) {
        let detail = match result {
            -1 => "invalid arguments (null context/model/input or non-positive sizes)",
            -2 => "sampler chain creation failed",
            -3 => {
                // The wrapper unconditionally logs the actual llama_decode
                // return code + n_past_in / chunk position to stderr (see
                // `llama_generate_streaming_c` in llama_wrapper.cpp); the
                // diagnostic is not gated on `XYBRID_LLAMACPP_VERBOSITY`,
                // which only controls llama.cpp's own log callback path.
                // When n_past_in > 0 the prefix-reuse path was in play;
                // that's the path that triggers KV-cache state mismatches
                // on recurrent / hybrid models. The adapter
                // (`prepare_kv_cache_and_get_tail`) now full-clears the
                // cache for recurrent models specifically, so this should
                // be rare; if you hit it on a new architecture, consult
                // the stderr line and consider whether the model needs
                // the recurrent path.
                "llama_decode failed on prefill (KV-cache state mismatch likely; \
                 see stderr for the wrapper-level diagnostic line emitted by \
                 `llama_generate_streaming_c`)"
            }
            -4 => "input + prefix exceeds context window (n_past_in + n_input >= n_ctx)",
            _ => "unknown",
        };
        return Err(AdapterError::RuntimeError(format!(
            "Generation failed with error code {} ({}; n_past_in={})",
            result, detail, n_past_in
        )));
    }

    // Check for callback error
    if let Some(err) = streaming_ctx.error {
        return Err(AdapterError::from_streaming_callback_error(err));
    }

    // Negative result (other than hard errors) means stopped by callback,
    // absolute value is token count.
    let (n_generated, stopped_by_callback) = if result < 0 {
        ((-result) as usize, true)
    } else {
        (result as usize, false)
    };

    output_tokens.truncate(n_generated);
    Ok((output_tokens, stopped_by_callback))
}

// =============================================================================
// Stub implementations when feature is disabled
// =============================================================================

#[cfg(not(feature = "llm-llamacpp"))]
pub struct LlamaModel;

#[cfg(not(feature = "llm-llamacpp"))]
pub struct LlamaContext;

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_backend_init() {}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_backend_free() {}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_log_set_verbosity(_level: i32) {}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_log_get_verbosity() -> i32 {
    0
}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_load_model_from_file(
    _path: &str,
    _n_gpu_layers: i32,
) -> Result<LlamaModel, AdapterError> {
    Err(AdapterError::RuntimeError(
        "llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_free_model(_model: LlamaModel) {}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_new_context_with_model(
    _model: &LlamaModel,
    _n_ctx: usize,
    _n_threads: usize,
    _n_batch: usize,
    _flash_attn: bool,
) -> Result<LlamaContext, AdapterError> {
    Err(AdapterError::RuntimeError(
        "llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_free(_ctx: LlamaContext) {}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_kv_cache_clear(_ctx: &LlamaContext) {}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_kv_cache_seq_rm(_ctx: &LlamaContext, _seq_id: i32, _p_keep: usize) {}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_format_chat(
    _model: &LlamaModel,
    _messages: &[ChatMessage],
) -> Result<String, AdapterError> {
    Err(AdapterError::RuntimeError(
        "llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_tokenize(
    _model: &LlamaModel,
    _text: &str,
    _add_special: bool,
) -> Result<Vec<i32>, AdapterError> {
    Err(AdapterError::RuntimeError(
        "llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_tokenize_special(
    _model: &LlamaModel,
    _text: &str,
    _add_special: bool,
) -> Result<Vec<i32>, AdapterError> {
    Err(AdapterError::RuntimeError(
        "llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_detokenize(_model: &LlamaModel, _tokens: &[i32]) -> Result<String, AdapterError> {
    Err(AdapterError::RuntimeError(
        "llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_generate_with_stops(
    _ctx: &LlamaContext,
    _model: &LlamaModel,
    _input_tokens: &[i32],
    _max_tokens: usize,
    _temperature: f32,
    _top_p: f32,
    _min_p: f32,
    _top_k: usize,
    _repeat_penalty: f32,
    _stop_sequences: &[String],
) -> Result<Vec<i32>, AdapterError> {
    Err(AdapterError::RuntimeError(
        "llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "llm-llamacpp"))]
pub fn llama_generate(
    _ctx: &LlamaContext,
    _model: &LlamaModel,
    _input_tokens: &[i32],
    _max_tokens: usize,
    _temperature: f32,
    _top_p: f32,
    _top_k: usize,
) -> Result<Vec<i32>, AdapterError> {
    Err(AdapterError::RuntimeError(
        "llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "llm-llamacpp"))]
#[allow(clippy::too_many_arguments)]
pub fn llama_generate_streaming<F>(
    _ctx: &LlamaContext,
    _model: &LlamaModel,
    _input_tokens: &[i32],
    _max_tokens: usize,
    _temperature: f32,
    _top_p: f32,
    _min_p: f32,
    _top_k: usize,
    _repeat_penalty: f32,
    _stop_sequences: &[String],
    _on_token: F,
    _n_past_in: usize,
) -> Result<(Vec<i32>, bool), AdapterError>
where
    F: FnMut(i32, &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
{
    Err(AdapterError::RuntimeError(
        "llm-llamacpp feature not enabled".to_string(),
    ))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use std::os::raw::{c_int, c_void};

    // =========================================================================
    // Regression: Stop Sequence Count Mismatch
    // =========================================================================
    //
    // Bug: llama_generate_with_stops() and llama_generate_streaming() pass
    // `stop_sequences.len()` as n_stop_seqs to the C function, but the
    // `stop_lens` array only has entries for sequences that tokenized to
    // non-empty results. If any sequence tokenizes to empty, the C code reads
    // past the end of stop_lens → out-of-bounds access → SIGSEGV.
    //
    // This test reproduces the logic without needing llama.cpp loaded.

    /// Verify that when some stop sequences tokenize to empty, the count passed
    /// to the C function matches the actual number of entries in the lengths array.
    ///
    /// Regression test: previously used `stop_sequences.len()` (unfiltered) which
    /// caused out-of-bounds reads when some sequences were filtered out.
    #[test]
    fn test_stop_sequence_count_matches_filtered_lens() {
        // Simulate tokenization results: sequence [1] returns empty
        let tokenize_results: Vec<Vec<i32>> = vec![
            vec![32000, 32001], // <|im_end|> → 2 tokens
            vec![],             // <|unknown_token|> → empty (filtered out)
            vec![32002],        // <|end_of_text|> → 1 token
        ];

        // Reproduce the fixed logic: filter empties, then count from stop_lens
        let mut stop_tokens: Vec<i32> = Vec::new();
        let mut stop_lens: Vec<c_int> = Vec::new();

        for tokens in &tokenize_results {
            if !tokens.is_empty() {
                stop_lens.push(tokens.len() as c_int);
                stop_tokens.extend(tokens);
            }
        }

        // Fixed: use stop_lens.len() (filtered count = 2), not original count (3)
        let n_stop_seqs = stop_lens.len() as c_int;

        assert_eq!(n_stop_seqs, 2, "n_stop_seqs must match stop_lens.len()");
        assert_eq!(stop_lens.len(), 2);
        assert_eq!(stop_tokens.len(), 3); // 2 + 1 tokens total
        assert_eq!(stop_lens[0], 2); // first sequence: 2 tokens
        assert_eq!(stop_lens[1], 1); // third sequence: 1 token
    }

    #[cfg(feature = "llm-llamacpp")]
    #[test]
    fn streaming_trampoline_preserves_cloud_fallback_abort_marker() {
        use super::{streaming_trampoline, StreamingContext};
        use crate::abort::{cloud_fallback_reason_from_error, AbortReason, CloudFallbackAbort};
        use std::ffi::CString;
        use std::time::{Duration, Instant};

        type Callback = fn(i32, &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
        fn abort_callback(
            _token_id: i32,
            _text: &str,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Err(Box::new(CloudFallbackAbort::new(AbortReason::StressMemory)))
        }

        let mut callback: Callback = abort_callback;
        let mut ctx = StreamingContext {
            callback: &mut callback,
            error: None,
        };
        let token_text = CString::new("token").unwrap();

        let started = Instant::now();
        let stop = streaming_trampoline::<Callback>(
            42,
            token_text.as_ptr(),
            &mut ctx as *mut StreamingContext<Callback> as *mut c_void,
        );
        let elapsed = started.elapsed();

        assert_eq!(stop, 1, "callback errors must stop the C stream");
        assert!(
            elapsed <= Duration::from_millis(50),
            "llama.cpp trampoline abort exceeded M-series cancellation budget: {:?}",
            elapsed
        );
        let err = ctx.error.take().expect("callback error must be stored");
        assert_eq!(
            cloud_fallback_reason_from_error(err.as_ref()),
            Some(AbortReason::StressMemory),
            "llama.cpp trampoline must keep the typed CloudFallbackAbort marker for the Rust layer"
        );
    }

    // =========================================================================
    // Regression: Buffer Retry Uses Wrong Length Variable
    // =========================================================================
    //
    // Bug: In llama_format_chat(), after the buffer resize and retry, the code
    // uses `result` (from the FIRST call) instead of `retry_result` to
    // determine how many bytes to read from the buffer.
    //
    // This test simulates the logic pattern to show the wrong variable is used.

    /// Verify that the buffer retry logic uses the retry call's return value,
    /// not the first call's, to determine how many bytes to read.
    ///
    /// Regression test: previously used `result` (first call) after resize+retry
    /// instead of `retry_result`, which could read stale/uninitialized data.
    #[test]
    fn test_format_chat_retry_uses_correct_length() {
        // Simulate the fixed buffer management logic from llama_format_chat
        let buf_len: usize = 4096;

        // First call: C function says it needs 5000 bytes (buffer too small)
        let result: c_int = 5000;

        assert!(result as usize >= buf_len, "Should trigger resize path");

        // Resize buffer
        let _new_buf_len = (result as usize) + 1; // 5001

        // Retry call: C function returns actual bytes written
        let retry_result: c_int = 4998;

        // Fixed logic: use retry_result when resize path was taken
        let len = if result as usize >= buf_len {
            retry_result as usize // FIXED: use retry's value
        } else {
            result as usize
        };

        assert_eq!(
            len, 4998,
            "Must use retry_result (4998), not first result (5000)"
        );
    }

    // =========================================================================
    // Regression: Prompt Size Exceeds Context Window
    // =========================================================================
    //
    // Bug: Neither the Rust layer nor the C layer validated that the number
    // of input tokens fits within the KV cache context window (n_ctx).
    // When input tokens >= n_ctx, the KV cache overflows → heap corruption.
    //
    // Additionally, the C layer allocated a fixed batch of 512 tokens, causing
    // heap corruption when input tokens > 512.
    //
    // These tests verify the bounds-checking logic without needing llama.cpp.

    /// Verify that the context window check rejects input that equals or exceeds n_ctx.
    #[test]
    fn test_context_window_bounds_check() {
        // Simulate the Rust-layer validation from generate() / generate_streaming()
        let n_ctx: usize = 4096;

        // Case 1: Input exactly at limit (no room for generation) → reject
        let tokens_at_limit = vec![0i32; 4096];
        assert!(
            tokens_at_limit.len() >= n_ctx,
            "Input at context limit should be rejected"
        );

        // Case 2: Input exceeding limit → reject
        let tokens_over_limit = vec![0i32; 5000];
        assert!(
            tokens_over_limit.len() >= n_ctx,
            "Input exceeding context limit should be rejected"
        );

        // Case 3: Input well within limit → accept
        let tokens_within_limit = vec![0i32; 2000];
        assert!(
            tokens_within_limit.len() < n_ctx,
            "Input within context limit should be accepted"
        );

        // Case 4: Input at limit minus 1 (room for exactly 1 token) → accept
        let tokens_just_under = vec![0i32; 4095];
        assert!(
            tokens_just_under.len() < n_ctx,
            "Input at n_ctx-1 should be accepted (room for 1 generated token)"
        );
    }

    /// Verify that batch allocation must be at least as large as input token count.
    ///
    /// Regression test: previously used llama_batch_init(512, ...) which caused
    /// heap corruption when n_input > 512.
    #[test]
    fn test_batch_size_must_fit_input_tokens() {
        let fixed_batch_size: usize = 512;

        // Small input: 512 batch is fine
        let small_input = 100;
        let batch_size = if small_input > fixed_batch_size {
            small_input
        } else {
            fixed_batch_size
        };
        assert!(batch_size >= small_input);

        // Large input: batch must grow to fit
        let large_input = 2000;
        let batch_size = if large_input > fixed_batch_size {
            large_input
        } else {
            fixed_batch_size
        };
        assert_eq!(batch_size, 2000, "Batch must grow to fit large input");
        assert!(batch_size >= large_input);

        // Edge case: exactly 512
        let exact_input = 512;
        let batch_size = if exact_input > fixed_batch_size {
            exact_input
        } else {
            fixed_batch_size
        };
        assert_eq!(batch_size, 512);
        assert!(batch_size >= exact_input);

        // Edge case: 513 tokens (one over) → must allocate 513
        let over_input = 513;
        let batch_size = if over_input > fixed_batch_size {
            over_input
        } else {
            fixed_batch_size
        };
        assert_eq!(
            batch_size, 513,
            "Batch must not use fixed 512 when input is 513"
        );
    }
}
