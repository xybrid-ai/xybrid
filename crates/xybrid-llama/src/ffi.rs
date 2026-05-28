//! `pub(crate)` FFI helpers over `llama_cpp_sys::bindings`.
//!
//! Every `unsafe` block in `xybrid-llama` lives in this module or in
//! callsites that this module explicitly delegates to. The public surface
//! in [`crate::lib`] exposes only safe wrappers — `# Safety` comments here
//! describe the invariants the caller must uphold.
//!
//! Mirrors `xybrid-mlx::ffi`'s discipline. Keep helpers small and
//! purpose-specific rather than re-exporting the whole binding set.

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

use llama_cpp_sys::bindings as sys;

use crate::error::LlamaError;

/// Allocate a [`CString`] from a Rust string, mapping the rejection of
/// interior null bytes to a typed [`LlamaError::InvalidInput`].
pub(crate) fn cstring(s: &str, context: &'static str) -> Result<CString, LlamaError> {
    CString::new(s).map_err(|_| LlamaError::InvalidInput(format!("{context}: contains null byte")))
}

/// Load a GGUF model from a filesystem path.
///
/// # Safety
///
/// `path` is a NUL-terminated UTF-8 C string the caller owns for the
/// duration of this call. The returned pointer is owned by the caller and
/// must be freed via [`free_model`] exactly once. A null return indicates
/// failure.
pub(crate) unsafe fn load_model_from_file(path: &CString, n_gpu_layers: i32) -> *mut c_void {
    sys::llama_load_model_from_file_c(path.as_ptr(), n_gpu_layers as c_int)
}

/// Free a model handle previously returned by [`load_model_from_file`].
///
/// # Safety
///
/// `ptr` must have come from [`load_model_from_file`] and not yet been
/// freed. Null is tolerated (no-op).
pub(crate) unsafe fn free_model(ptr: *mut c_void) {
    if !ptr.is_null() {
        sys::llama_free_model_c(ptr);
    }
}

/// Create a new context bound to `model`.
///
/// # Safety
///
/// `model` must be a live, non-null pointer returned by
/// [`load_model_from_file`]. Returns null on failure.
pub(crate) unsafe fn new_context_with_model(
    model: *mut c_void,
    n_ctx: usize,
    n_threads: usize,
    n_batch: usize,
    flash_attn: bool,
) -> *mut c_void {
    sys::llama_new_context_with_model_c(
        model,
        n_ctx as c_int,
        n_threads as c_int,
        n_batch as c_int,
        flash_attn,
    )
}

/// Free a context handle.
///
/// # Safety
///
/// `ptr` must have come from [`new_context_with_model`] and not yet been
/// freed. Null is tolerated.
pub(crate) unsafe fn free_context(ptr: *mut c_void) {
    if !ptr.is_null() {
        sys::llama_free_c(ptr);
    }
}

/// Clear the KV cache for `ctx`.
///
/// # Safety
///
/// `ctx` must be a live, non-null context pointer.
pub(crate) unsafe fn kv_cache_clear(ctx: *mut c_void) {
    sys::llama_kv_cache_clear_c(ctx);
}

/// Truncate the KV cache for `seq_id` to a prefix length, dropping
/// positions `[p_keep, ∞)`.
///
/// # Safety
///
/// `ctx` must be a live, non-null context pointer. `p_keep` is saturated
/// at `i32::MAX` at the call site to honor the C signature.
pub(crate) unsafe fn kv_cache_seq_rm(ctx: *mut c_void, seq_id: i32, p_keep: usize) {
    let p_keep_c = p_keep.min(c_int::MAX as usize) as c_int;
    let _ = sys::llama_kv_cache_seq_rm_c(ctx, seq_id, p_keep_c);
}

/// Sizing-probe pass of tokenization.
///
/// # Safety
///
/// `model` and `text` must be valid for the duration of the call. Returns
/// a negative count carrying the required buffer length.
pub(crate) unsafe fn tokenize_probe(
    model: *const c_void,
    text: &CString,
    text_len: usize,
    add_special: bool,
    parse_special: bool,
) -> c_int {
    sys::llama_tokenize_c(
        model,
        text.as_ptr(),
        text_len as c_int,
        std::ptr::null_mut(),
        0,
        add_special,
        parse_special,
    )
}

/// Real tokenization pass.
///
/// # Safety
///
/// Same as [`tokenize_probe`], plus `out_tokens` must be writable for
/// `capacity` `i32` elements.
pub(crate) unsafe fn tokenize_into(
    model: *const c_void,
    text: &CString,
    text_len: usize,
    out_tokens: *mut i32,
    capacity: usize,
    add_special: bool,
    parse_special: bool,
) -> c_int {
    sys::llama_tokenize_c(
        model,
        text.as_ptr(),
        text_len as c_int,
        out_tokens,
        capacity as c_int,
        add_special,
        parse_special,
    )
}

/// Detokenize a single token into `buf`.
///
/// # Safety
///
/// `model` non-null. `buf` writable for `buf_len` bytes.
pub(crate) unsafe fn token_to_piece(
    model: *const c_void,
    token: i32,
    buf: *mut c_char,
    buf_len: usize,
    lstrip: i32,
    special: bool,
) -> c_int {
    sys::llama_token_to_piece_c(
        model,
        token,
        buf,
        buf_len as c_int,
        lstrip as c_int,
        special,
    )
}

/// # Safety: `model` must be a live, non-null model pointer.
pub(crate) unsafe fn token_bos(model: *const c_void) -> i32 {
    sys::llama_token_bos_c(model)
}

/// # Safety: `model` must be a live, non-null model pointer.
pub(crate) unsafe fn token_eos(model: *const c_void) -> i32 {
    sys::llama_token_eos_c(model)
}

/// # Safety: `model` must be a live, non-null model pointer.
pub(crate) unsafe fn vocab_is_eog(model: *const c_void, token: i32) -> bool {
    sys::llama_vocab_is_eog_c(model, token)
}

/// # Safety: `model` must be a live, non-null model pointer. The returned
/// C string is owned by llama.cpp and remains valid for the model's
/// lifetime; the caller must not free it.
pub(crate) unsafe fn model_chat_template(model: *const c_void) -> *const c_char {
    sys::llama_model_chat_template_c(model)
}

/// # Safety: `model` must be a live, non-null model pointer.
pub(crate) unsafe fn n_vocab(model: *const c_void) -> i32 {
    sys::llama_n_vocab_c(model)
}

/// # Safety: `ctx` must be a live, non-null context pointer.
pub(crate) unsafe fn n_ctx(ctx: *const c_void) -> i32 {
    sys::llama_n_ctx_c(ctx)
}

/// # Safety: `model` must be a live, non-null model pointer.
pub(crate) unsafe fn model_is_recurrent(model: *const c_void) -> bool {
    sys::llama_model_is_recurrent_c(model)
}

/// # Safety: `model` must be a live, non-null model pointer.
pub(crate) unsafe fn model_has_recurrent_state(model: *const c_void) -> bool {
    sys::llama_model_has_recurrent_state_c(model)
}

/// Render a chat conversation through the model's built-in template into
/// `buf`.
///
/// # Safety
///
/// `model` non-null. `roles` / `contents` must be parallel arrays of
/// `n_msg` C-string pointers, each valid for the call duration. `buf`
/// writable for `buf_len` bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn format_chat_with_model(
    model: *const c_void,
    roles: *const *const c_char,
    contents: *const *const c_char,
    n_msg: usize,
    buf: *mut c_char,
    buf_len: usize,
) -> c_int {
    sys::llama_format_chat_with_model_c(model, roles, contents, n_msg, buf, buf_len as c_int)
}

/// Autoregressive generation without streaming.
///
/// # Safety
///
/// `ctx` and `model` must be live, non-null pointers. `input_tokens`
/// valid for `n_input` elements. `output_tokens` writable for `max_tokens`
/// elements. `stop_seqs` and `stop_lens` either both null with
/// `n_stop_seqs == 0`, or both valid.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn generate(
    ctx: *mut c_void,
    model: *const c_void,
    input_tokens: *const i32,
    n_input: usize,
    output_tokens: *mut i32,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    seed: u32,
    stop_seqs: *const i32,
    stop_lens: *const c_int,
    n_stop_seqs: c_int,
) -> c_int {
    sys::llama_generate_c(
        ctx,
        model,
        input_tokens,
        n_input as c_int,
        output_tokens,
        max_tokens as c_int,
        temperature,
        top_p,
        min_p,
        top_k as c_int,
        repeat_penalty,
        seed,
        stop_seqs,
        stop_lens,
        n_stop_seqs,
    )
}

/// Autoregressive generation with a streaming callback.
///
/// # Safety
///
/// Same as [`generate`], plus `callback` + `user_data` form a matched
/// pair: the trampoline must safely interpret `user_data` as a
/// `*mut StreamingContext<F>`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn generate_streaming(
    ctx: *mut c_void,
    model: *const c_void,
    input_tokens: *const i32,
    n_input: usize,
    output_tokens: *mut i32,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    seed: u32,
    stop_seqs: *const i32,
    stop_lens: *const c_int,
    n_stop_seqs: c_int,
    callback: Option<sys::TokenCallback>,
    user_data: *mut c_void,
    n_past_in: usize,
) -> c_int {
    sys::llama_generate_streaming_c(
        ctx,
        model,
        input_tokens,
        n_input as c_int,
        output_tokens,
        max_tokens as c_int,
        temperature,
        top_p,
        min_p,
        top_k as c_int,
        repeat_penalty,
        seed,
        stop_seqs,
        stop_lens,
        n_stop_seqs,
        callback,
        user_data,
        n_past_in.min(c_int::MAX as usize) as c_int,
    )
}

/// Load an mtmd multimodal projector.
///
/// # Safety
///
/// `path` must be a valid C string for the duration of the call. `model`
/// must be a live llama model pointer. Returns null on failure.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_init_from_file(
    path: &CString,
    model: *const c_void,
    use_gpu: bool,
    warmup: bool,
    n_threads: usize,
    flash_attn: bool,
) -> *mut c_void {
    sys::mtmd_init_from_file_c(
        path.as_ptr(),
        model,
        use_gpu,
        warmup,
        n_threads.min(c_int::MAX as usize) as c_int,
        flash_attn,
    )
}

/// # Safety: `ptr` must be a live mtmd context pointer or null.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        sys::mtmd_free_c(ptr);
    }
}

/// # Safety: `ctx` may be null only for image-only decode. `bytes` must
/// point to `len` encoded image bytes.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_init_from_buf(
    ctx: *mut c_void,
    bytes: *const u8,
    len: usize,
) -> *mut c_void {
    sys::mtmd_bitmap_init_from_buf_c(ctx, bytes, len)
}

/// # Safety: `ptr` must be a live mtmd bitmap pointer or null.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        sys::mtmd_bitmap_free_c(ptr);
    }
}

/// # Safety: `ptr` must be a live bitmap pointer.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_width(ptr: *const c_void) -> u32 {
    sys::mtmd_bitmap_get_nx_c(ptr)
}

/// # Safety: `ptr` must be a live bitmap pointer.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_height(ptr: *const c_void) -> u32 {
    sys::mtmd_bitmap_get_ny_c(ptr)
}

/// # Safety: `ptr` must be a live bitmap pointer.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_n_bytes(ptr: *const c_void) -> usize {
    sys::mtmd_bitmap_get_n_bytes_c(ptr)
}

/// # Safety: `ptr` must be a live bitmap pointer. Returned C string is
/// owned by mtmd and must not be freed.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_id(ptr: *const c_void) -> *const c_char {
    sys::mtmd_bitmap_get_id_c(ptr)
}

/// # Safety: `ptr` must be a live bitmap pointer and `id` must be valid for
/// the call duration.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_set_id(ptr: *mut c_void, id: &CString) {
    sys::mtmd_bitmap_set_id_c(ptr, id.as_ptr());
}

/// # Safety: returns an owned chunk list pointer or null.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunks_init() -> *mut c_void {
    sys::mtmd_input_chunks_init_c()
}

/// # Safety: `ptr` must be a live chunk list pointer or null.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunks_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        sys::mtmd_input_chunks_free_c(ptr);
    }
}

/// # Safety: `ptr` must be a live chunk list pointer.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunks_size(ptr: *const c_void) -> usize {
    sys::mtmd_input_chunks_size_c(ptr)
}

/// # Safety: `ptr` must be a live chunk list pointer and `idx` in range.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunks_get(ptr: *const c_void, idx: usize) -> *const c_void {
    sys::mtmd_input_chunks_get_c(ptr, idx)
}

/// # Safety: `ptr` must be a live chunk pointer.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunk_type(ptr: *const c_void) -> c_int {
    sys::mtmd_input_chunk_get_type_c(ptr)
}

/// # Safety: `ptr` must be a live text chunk pointer and `n_tokens_output`
/// must be writable.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunk_tokens_text(
    ptr: *const c_void,
    n_tokens_output: *mut usize,
) -> *const i32 {
    sys::mtmd_input_chunk_get_tokens_text_c(ptr, n_tokens_output)
}

/// # Safety: `ptr` must be a live chunk pointer.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunk_tokens_image(ptr: *const c_void) -> *const c_void {
    sys::mtmd_input_chunk_get_tokens_image_c(ptr)
}

/// # Safety: `ptr` must be a live image-token pointer.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_image_tokens_n_tokens(ptr: *const c_void) -> usize {
    sys::mtmd_image_tokens_get_n_tokens_c(ptr)
}

/// # Safety: `ptr` must be a live image-token pointer.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_image_tokens_n_pos(ptr: *const c_void) -> i32 {
    sys::mtmd_image_tokens_get_n_pos_c(ptr)
}

/// # Safety: `ptr` must be a live chunk list pointer.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_helper_n_tokens(ptr: *const c_void) -> usize {
    sys::mtmd_helper_get_n_tokens_c(ptr)
}

/// # Safety: `ptr` must be a live chunk list pointer.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_helper_n_pos(ptr: *const c_void) -> i32 {
    sys::mtmd_helper_get_n_pos_c(ptr)
}

/// # Safety: all pointers must be live for the duration of the call.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_tokenize(
    ctx: *mut c_void,
    output: *mut c_void,
    text: &CString,
    add_special: bool,
    parse_special: bool,
    bitmaps: *mut *const c_void,
    n_bitmaps: usize,
) -> c_int {
    sys::mtmd_tokenize_c(
        ctx,
        output,
        text.as_ptr(),
        add_special,
        parse_special,
        bitmaps,
        n_bitmaps,
    )
}

/// # Safety: all pointers must be live and serialized by the caller.
#[cfg(feature = "vision")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mtmd_helper_eval_chunks(
    ctx: *mut c_void,
    lctx: *mut c_void,
    chunks: *const c_void,
    n_past: i32,
    seq_id: i32,
    n_batch: usize,
    logits_last: bool,
    new_n_past: *mut i32,
) -> c_int {
    sys::mtmd_helper_eval_chunks_c(
        ctx,
        lctx,
        chunks,
        n_past,
        seq_id,
        n_batch.min(c_int::MAX as usize) as c_int,
        logits_last,
        new_n_past,
    )
}

/// Continue generation from logits already present in `ctx`.
///
/// # Safety
///
/// Same pointer/lifetime requirements as [`generate_streaming`], except
/// the caller must already have prefilled the context with `logits_last`.
#[cfg(feature = "vision")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn generate_from_current_logits(
    ctx: *mut c_void,
    model: *const c_void,
    output_tokens: *mut i32,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    seed: u32,
    stop_seqs: *const i32,
    stop_lens: *const c_int,
    n_stop_seqs: c_int,
    callback: Option<sys::TokenCallback>,
    user_data: *mut c_void,
    n_past: usize,
) -> c_int {
    sys::llama_generate_from_current_logits_c(
        ctx,
        model,
        output_tokens,
        max_tokens as c_int,
        temperature,
        top_p,
        min_p,
        top_k as c_int,
        repeat_penalty,
        seed,
        stop_seqs,
        stop_lens,
        n_stop_seqs,
        callback,
        user_data,
        n_past.min(c_int::MAX as usize) as c_int,
    )
}

/// # Safety: callable from any thread at any time.
pub(crate) unsafe fn log_set_verbosity(level: i32) {
    sys::llama_log_set_verbosity_c(level as c_int);
}

/// # Safety: callable from any thread at any time.
pub(crate) unsafe fn log_get_verbosity() -> i32 {
    sys::llama_log_get_verbosity_c() as i32
}
