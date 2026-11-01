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

// Counts this thread's calls into the wrappers that the generation, mtmd
// evaluation and readout entry points use, so tests can prove their input
// checks run before any native call.
#[cfg(test)]
thread_local! {
    static NATIVE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_native_call() {
    NATIVE_CALLS.with(|count| count.set(count.get() + 1));
}

/// This thread's calls into the counted wrappers so far.
#[cfg(test)]
pub(crate) fn native_calls() -> usize {
    NATIVE_CALLS.with(std::cell::Cell::get)
}

#[cfg(feature = "vision")]
pub(crate) type MtmdContextRaw = sys::mtmd_context;
#[cfg(feature = "vision")]
pub(crate) type MtmdBitmapRaw = sys::mtmd_bitmap;
#[cfg(feature = "vision")]
pub(crate) type MtmdInputChunksRaw = sys::mtmd_input_chunks;
#[cfg(feature = "vision")]
pub(crate) type MtmdInputChunkRaw = sys::mtmd_input_chunk;
#[cfg(feature = "vision")]
pub(crate) type MtmdImageTokensRaw = sys::mtmd_image_tokens;
#[cfg(feature = "vision")]
pub(crate) type MtmdDecoderPosRaw = sys::mtmd_decoder_pos;

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
/// [`load_model_from_file`], and must stay live until the returned context
/// is freed: the context keeps reading it. Returns null on failure.
pub(crate) unsafe fn new_context_with_model(
    model: *mut c_void,
    n_ctx: c_int,
    n_threads: c_int,
    n_batch: c_int,
    flash_attn: bool,
) -> *mut c_void {
    sys::llama_new_context_with_model_c(model, n_ctx, n_threads, n_batch, flash_attn)
}

/// Create a one-shot readout context bound to `model`: one sequence, these
/// sizes, at most `n_outputs_max` outputs (`0` = `n_batch`), no embeddings,
/// and the model's own attention type.
///
/// # Safety
///
/// `model` must be a live, non-null pointer returned by
/// [`load_model_from_file`], and must stay live until the returned context
/// is freed with [`free_context`]. Returns null on failure.
pub(crate) unsafe fn new_readout_context(
    model: *mut c_void,
    sizes: ReadoutSizes,
    n_threads: c_int,
    flash_attn: bool,
) -> *mut c_void {
    #[cfg(test)]
    record_native_call();
    sys::llama_new_readout_context_c(
        model,
        sizes.n_ctx,
        sizes.n_batch,
        sizes.n_ubatch,
        sizes.n_outputs_max,
        n_threads,
        flash_attn,
    )
}

/// The sizes [`new_readout_context`] asks llama.cpp for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadoutSizes {
    pub(crate) n_ctx: u32,
    pub(crate) n_batch: u32,
    pub(crate) n_ubatch: u32,
    /// `0` lets llama.cpp reserve `n_batch` output rows.
    pub(crate) n_outputs_max: u32,
}

/// The status the guarded shims return for a missing key, template or
/// token; a failure is [`FAILED`], never this.
pub(crate) const ABSENT: i32 = sys::LLAMA_XYBRID_ABSENT;
/// The status the guarded shims return when llama.cpp raised an exception
/// (an allocation failure) or got a null argument.
pub(crate) const FAILED: i32 = sys::LLAMA_XYBRID_FAILED;
/// `llama_decode_c`'s status when llama.cpp raised an exception.
pub(crate) const DECODE_EXCEPTION: i32 = sys::LLAMA_XYBRID_DECODE_EXCEPTION;

/// The error for a guarded shim that failed: llama.cpp raised an exception,
/// in practice an allocation failure. Names the operation only.
pub(crate) fn native_failure(what: &str) -> LlamaError {
    LlamaError::Internal(format!(
        "llama.cpp failed while {what} (an exception, such as running out of memory)"
    ))
}

/// The sizes, memory and attention llama.cpp gave `ctx`, or `None` when the
/// guarded query failed.
///
/// # Safety: `ctx` must be a live context pointer or null (`None`).
pub(crate) unsafe fn context_facts(ctx: *const c_void) -> Option<sys::llama_xybrid_context_facts> {
    #[cfg(test)]
    record_native_call();
    let mut facts = sys::llama_xybrid_context_facts {
        n_ctx: 0,
        n_batch: 0,
        n_ubatch: 0,
        n_seq_max: 0,
        has_memory: false,
        causal_attn: false,
    };
    (sys::llama_context_facts_c(ctx, &mut facts) == 0).then_some(facts)
}

/// Makes the next call of the guarded shim `shim` throw `std::bad_alloc`
/// inside its exception boundary, as an allocation failure in llama.cpp
/// would.
#[cfg(test)]
pub(crate) fn fail_next_native_call(shim: &str) {
    let shim = CString::new(shim).expect("a shim name has no NUL");
    // SAFETY: the shim copies the name before returning.
    unsafe { sys::llama_xybrid_fail_next_c(shim.as_ptr()) };
}

/// The logits row of batch entry `i` from the last decode.
///
/// # Safety
///
/// `ctx` must be a live context whose last decode succeeded and flagged
/// entry `i` for output: llama.cpp aborts the process on any other entry
/// when built without `NDEBUG`. The row holds `n_vocab` floats and stays
/// valid until the next decode or the context's destruction.
pub(crate) unsafe fn logits_ith(ctx: *mut c_void, i: i32) -> *const f32 {
    #[cfg(test)]
    record_native_call();
    sys::llama_get_logits_ith_c(ctx, i)
}

/// Free a context handle.
///
/// # Safety
///
/// `ptr` must have come from [`new_context_with_model`] or
/// [`new_readout_context`] and not yet been freed. Null is tolerated.
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

/// Clear the whole memory of `ctx`: its KV cache and any recurrent state.
///
/// Returns `Ok(false)`, clearing nothing, when the context has no memory:
/// llama.cpp creates none for encoder-only and diffusion models, and decodes
/// their batches with the encoder. A failure inside llama.cpp is an error,
/// never "no memory".
///
/// # Safety
///
/// `ctx` must be a live, non-null context pointer.
pub(crate) unsafe fn clear_memory(ctx: *mut c_void) -> Result<bool, LlamaError> {
    // The shim's seq_rm returns -1 when the context has no memory. Dropping
    // sequence 0 first costs nothing: the full clear empties every sequence.
    match sys::llama_kv_cache_seq_rm_c(ctx, 0, 0) {
        0 => {
            sys::llama_kv_cache_clear_c(ctx);
            Ok(true)
        }
        -1 => Ok(false),
        _ => Err(native_failure("clearing the context's memory")),
    }
}

/// Byte size of a state snapshot of `seq_id`. Zero on failure, including
/// a `seq_id` outside `[0, n_seq_max)` (checked by the shim).
///
/// # Safety
///
/// `ctx` must be a live, non-null context pointer.
pub(crate) unsafe fn state_seq_get_size(ctx: *mut c_void, seq_id: i32) -> usize {
    sys::llama_state_seq_get_size_c(ctx, seq_id)
}

/// Write a state snapshot of `seq_id` into `dst`, returning the bytes
/// written. Zero on failure, including `dst` being too small.
///
/// # Safety
///
/// `ctx` must be a live, non-null context pointer.
pub(crate) unsafe fn state_seq_get_data(ctx: *mut c_void, dst: &mut [u8], seq_id: i32) -> usize {
    sys::llama_state_seq_get_data_c(ctx, dst.as_mut_ptr(), dst.len(), seq_id)
}

/// Load a snapshot from [`state_seq_get_data`] into `seq_id`, returning
/// the bytes read. Zero on failure, which leaves `seq_id` empty.
///
/// # Safety
///
/// `ctx` must be a live, non-null context pointer, and `src` an unmodified
/// [`state_seq_get_data`] snapshot from a context on the same model.
/// llama.cpp does not validate the bytes: anything else can hit one of its
/// asserts and abort the process.
pub(crate) unsafe fn state_seq_set_data(ctx: *mut c_void, src: &[u8], seq_id: i32) -> usize {
    sys::llama_state_seq_set_data_c(ctx, src.as_ptr(), src.len(), seq_id)
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
    #[cfg(test)]
    record_native_call();
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
    #[cfg(test)]
    record_native_call();
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

/// # Safety: `model` must be a live, non-null model pointer.
pub(crate) unsafe fn vocab_get_add_bos(model: *const c_void) -> bool {
    sys::llama_vocab_get_add_bos_c(model)
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

/// Whether the model has an encoder; `None` when the guarded query failed.
///
/// # Safety: `model` must be a live model pointer or null (`None`).
pub(crate) unsafe fn model_has_encoder(model: *const c_void) -> Option<bool> {
    #[cfg(test)]
    record_native_call();
    match sys::llama_model_has_encoder_c(model) {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

/// Number of scalar metadata keys llama.cpp loaded with the model, or
/// [`FAILED`].
///
/// # Safety: `model` must be a live model pointer or null ([`FAILED`]).
pub(crate) unsafe fn model_meta_count(model: *const c_void) -> i32 {
    #[cfg(test)]
    record_native_call();
    sys::llama_model_meta_count_c(model)
}

/// Writes the value of `key` into `buf` (NUL-terminated, truncated to fit)
/// and returns its whole length, [`ABSENT`] when the model has no such
/// scalar key, or [`FAILED`].
///
/// # Safety
///
/// `model` must be a live model pointer or null, and `key` outlive the call.
/// `buf` must be writable for `buf_len` bytes, or null with `buf_len == 0`
/// to probe the length.
pub(crate) unsafe fn model_meta_val(
    model: *const c_void,
    key: &CString,
    buf: *mut c_char,
    buf_len: usize,
) -> i32 {
    #[cfg(test)]
    record_native_call();
    sys::llama_model_meta_val_str_c(model, key.as_ptr(), buf, buf_len)
}

/// The values of `keys` in two native calls whatever their number (one to
/// size the buffer, one to fill it), each a hash lookup per key: `None` for
/// a key the model does not have.
///
/// # Safety: `model` must be a live model pointer or null (an error).
pub(crate) unsafe fn model_meta_lookup(
    model: *const c_void,
    keys: &[CString],
) -> Result<Vec<Option<Vec<u8>>>, LlamaError> {
    let pointers: Vec<*const c_char> = keys.iter().map(|key| key.as_ptr()).collect();
    let mut lens = vec![0i64; keys.len()];
    #[cfg(test)]
    record_native_call();
    let total = sys::llama_model_meta_lookup_c(
        model,
        pointers.as_ptr(),
        pointers.len(),
        std::ptr::null_mut(),
        0,
        lens.as_mut_ptr(),
    );
    let total = usize::try_from(total).map_err(|_| native_failure("reading model metadata"))?;
    let mut buf = Vec::new();
    buf.try_reserve_exact(total + 1)
        .map_err(|_| native_failure("reading model metadata"))?;
    buf.resize(total + 1, 0u8);
    let mut copied = vec![0i64; keys.len()];
    #[cfg(test)]
    record_native_call();
    let written = sys::llama_model_meta_lookup_c(
        model,
        pointers.as_ptr(),
        pointers.len(),
        buf.as_mut_ptr().cast::<c_char>(),
        buf.len(),
        copied.as_mut_ptr(),
    );
    if usize::try_from(written).ok() != Some(total) || copied != lens {
        return Err(LlamaError::Internal(
            "model metadata changed between two reads".to_string(),
        ));
    }
    let mut values = Vec::with_capacity(keys.len());
    let mut offset = 0;
    for len in lens {
        if len == i64::from(ABSENT) {
            values.push(None);
            continue;
        }
        let len = usize::try_from(len).map_err(|_| native_failure("reading model metadata"))?;
        values.push(Some(buf[offset..offset + len].to_vec()));
        offset += len;
    }
    Ok(values)
}

/// Reads a string from one of the `llama_model_meta_*` accessors, which
/// return the string's whole length like `snprintf`: a probe with no buffer
/// sizes it, a second call copies it. `None` when the accessor reports no
/// such entry ([`ABSENT`]); a failure ([`FAILED`]) is an error, never an
/// absence.
///
/// `write(buf, len)` performs one accessor call.
pub(crate) fn read_meta_string(
    mut write: impl FnMut(*mut c_char, usize) -> i32,
) -> Result<Option<Vec<u8>>, LlamaError> {
    let probe = write(std::ptr::null_mut(), 0);
    if probe == ABSENT {
        return Ok(None);
    }
    let len = usize::try_from(probe).map_err(|_| native_failure("reading model metadata"))?;
    let mut buf = vec![0u8; len + 1];
    let written = write(buf.as_mut_ptr().cast::<c_char>(), buf.len());
    if written == FAILED {
        return Err(native_failure("reading model metadata"));
    }
    if usize::try_from(written).ok() != Some(len) || buf[len] != 0 {
        return Err(LlamaError::Internal(format!(
            "a metadata string changed length between reads ({len}, then {written})"
        )));
    }
    buf.truncate(len);
    Ok(Some(buf))
}

/// The bytes of a C string the guarded shims point `out` at, given the
/// length the shim returned: `None` for [`ABSENT`], an error for anything
/// else that is not a length.
///
/// # Safety
///
/// For a length `len >= 0`, `out` must point at `len` readable bytes owned
/// by the model, valid for the call.
unsafe fn borrowed_bytes(
    len: i32,
    out: *const c_char,
    what: &str,
) -> Result<Option<Vec<u8>>, LlamaError> {
    if len == ABSENT {
        return Ok(None);
    }
    let len = usize::try_from(len).map_err(|_| native_failure(what))?;
    if out.is_null() {
        return Err(native_failure(what));
    }
    Ok(Some(
        std::slice::from_raw_parts(out.cast::<u8>(), len).to_vec(),
    ))
}

/// The chat template named `name`, copied; `None` when the model has none.
///
/// # Safety: `model` must be a live model pointer or null (an error).
pub(crate) unsafe fn model_chat_template_named(
    model: *const c_void,
    name: &CString,
) -> Result<Option<Vec<u8>>, LlamaError> {
    #[cfg(test)]
    record_native_call();
    let mut out = std::ptr::null();
    let len = sys::llama_model_chat_template_named_c(model, name.as_ptr(), &mut out);
    borrowed_bytes(len, out, "reading a chat template")
}

/// The attribute bits of `token`; `None` when it is outside the vocabulary
/// (the shim checks the range).
///
/// # Safety: `model` must be a live model pointer or null (an error).
pub(crate) unsafe fn vocab_get_attr(
    model: *const c_void,
    token: i32,
) -> Result<Option<u32>, LlamaError> {
    #[cfg(test)]
    record_native_call();
    match sys::llama_vocab_get_attr_c(model, token) {
        ABSENT => Ok(None),
        bits => u32::try_from(bits)
            .map(Some)
            .map_err(|_| native_failure("reading a token's attributes")),
    }
}

/// The stored text of `token`, copied; `None` when it is outside the
/// vocabulary.
///
/// # Safety: `model` must be a live model pointer or null (an error).
pub(crate) unsafe fn vocab_get_text(
    model: *const c_void,
    token: i32,
) -> Result<Option<Vec<u8>>, LlamaError> {
    #[cfg(test)]
    record_native_call();
    let mut out = std::ptr::null();
    let len = sys::llama_vocab_get_text_c(model, token, &mut out);
    borrowed_bytes(len, out, "reading a token's text")
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
/// `ctx` and `model` must each be live or null, and a live `ctx` must have
/// been created from `model`. The native wrapper returns an error for a null
/// handle before using either. `input_tokens` valid for `n_input` elements.
/// `output_tokens` writable for `max_tokens` elements. `stop_seqs` and
/// `stop_lens` either both null with
/// `n_stop_seqs == 0`, or both valid. `grammar` / `grammar_root` are either
/// null (unconstrained generation) or NUL-terminated C strings that outlive
/// the call.
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
    grammar: *const c_char,
    grammar_root: *const c_char,
    stop_seqs: *const i32,
    stop_lens: *const c_int,
    n_stop_seqs: c_int,
) -> c_int {
    #[cfg(test)]
    record_native_call();
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
        grammar,
        grammar_root,
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
    grammar: *const c_char,
    grammar_root: *const c_char,
    stop_seqs: *const i32,
    stop_lens: *const c_int,
    n_stop_seqs: c_int,
    callback: Option<sys::TokenCallback>,
    user_data: *mut c_void,
    n_past_in: usize,
) -> c_int {
    #[cfg(test)]
    record_native_call();
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
        grammar,
        grammar_root,
        stop_seqs,
        stop_lens,
        n_stop_seqs,
        callback,
        user_data,
        n_past_in.min(c_int::MAX as usize) as c_int,
    )
}

/// Continue autoregressive generation from logits already present in
/// `ctx`.
///
/// # Safety
///
/// `ctx` and `model` must each be live or null, and a live `ctx` must have
/// been created from `model`. The native wrapper returns an error for a null
/// handle before using either. `output_tokens` writable for `max_tokens`
/// elements. Stop arrays and callback/user_data follow the same contract as
/// [`generate_streaming`].
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "vision")]
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
    grammar: *const c_char,
    grammar_root: *const c_char,
    stop_seqs: *const i32,
    stop_lens: *const c_int,
    n_stop_seqs: c_int,
    callback: Option<sys::TokenCallback>,
    user_data: *mut c_void,
    n_past: usize,
) -> c_int {
    #[cfg(test)]
    record_native_call();
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
        grammar,
        grammar_root,
        stop_seqs,
        stop_lens,
        n_stop_seqs,
        callback,
        user_data,
        n_past.min(c_int::MAX as usize) as c_int,
    )
}

/// Decode `batch` on `ctx`. Returns llama.cpp's status: `0` on success, and
/// [`DECODE_EXCEPTION`] when llama.cpp raised an exception (see
/// [`decode_failure`]).
///
/// # Safety
///
/// `ctx` must be a live context. Every token id in `batch` must be inside the
/// context's vocabulary and every position inside its window, and the batch
/// must not be longer than the context's native batch size (its
/// `decode_limit`): llama.cpp aborts the process on a longer batch.
pub(crate) unsafe fn decode(ctx: *mut c_void, batch: &mut TokenBatch) -> c_int {
    #[cfg(test)]
    record_native_call();
    // The view points into `batch`, which stays borrowed for the call;
    // llama.cpp reads the arrays during the call and keeps no pointer to them.
    let view = batch.view();
    sys::llama_decode_c(ctx, &view)
}

/// What a failed [`decode`] status means, for an error message.
pub(crate) fn decode_failure(code: c_int) -> &'static str {
    match code {
        DECODE_EXCEPTION => "llama.cpp raised an exception (such as running out of memory)",
        1 => "llama.cpp found no memory slot for the batch",
        2 => "llama.cpp aborted the evaluation",
        -1 => "llama.cpp refused the batch",
        _ => "llama.cpp failed to evaluate the batch",
    }
}

/// The logits of the last decode's output rows, owned by `ctx`.
///
/// # Safety
///
/// `ctx` must be a live context. The pointer, when not null, holds one row of
/// `n_vocab` floats per token the last decode flagged for output, and stays
/// valid until the next decode or the context's destruction.
pub(crate) unsafe fn logits(ctx: *mut c_void) -> *const f32 {
    sys::llama_get_logits_c(ctx)
}

/// A batch of tokens from sequence `0`, in memory Rust owns.
///
/// [`TokenBatch::set`] fills it and [`decode`] hands llama.cpp a view of it.
/// It does not use `llama_batch_init`, which writes through its allocations
/// without checking them, so sizes and allocation stay on the Rust side.
pub(crate) struct TokenBatch {
    capacity: usize,
    tokens: Vec<sys::llama_token>,
    positions: Vec<sys::llama_pos>,
    seq_counts: Vec<i32>,
    seq_ids: Vec<sys::llama_seq_id>,
    seq_id_ptrs: Vec<*mut sys::llama_seq_id>,
    outputs: Vec<i8>,
}

impl TokenBatch {
    /// An empty batch with room for `capacity` tokens.
    ///
    /// # Panics
    ///
    /// When `capacity` is zero or does not fit a C `int`: a caller sizes the
    /// batch from the context's decode limit, which is positive and fits.
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0 && c_int::try_from(capacity).is_ok(),
            "a batch holds between one token and c_int::MAX, not {capacity}"
        );
        Self {
            capacity,
            tokens: Vec::with_capacity(capacity),
            positions: Vec::with_capacity(capacity),
            seq_counts: Vec::with_capacity(capacity),
            seq_ids: Vec::with_capacity(capacity),
            seq_id_ptrs: Vec::with_capacity(capacity),
            outputs: Vec::with_capacity(capacity),
        }
    }

    /// The most tokens [`TokenBatch::set`] accepts.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Fill the batch with `tokens` at positions `first_pos..`, in sequence
    /// `0`, flagging only the last token for output when `output_last`.
    ///
    /// # Panics
    ///
    /// When `tokens` is longer than the batch's capacity, or a position does
    /// not fit llama.cpp's `i32`: both are bugs in the caller.
    pub(crate) fn set(&mut self, tokens: &[i32], first_pos: usize, output_last: bool) {
        assert!(
            tokens.len() <= self.capacity,
            "{} tokens do not fit a batch of {}",
            tokens.len(),
            self.capacity
        );
        let n = tokens.len();
        self.tokens.clear();
        self.tokens.extend_from_slice(tokens);
        self.positions.clear();
        self.positions.extend(
            (first_pos..first_pos + n)
                .map(|pos| sys::llama_pos::try_from(pos).expect("token positions fit an i32")),
        );
        self.seq_counts.clear();
        self.seq_counts.resize(n, 1);
        self.seq_ids.clear();
        self.seq_ids.resize(n, 0);
        self.outputs.clear();
        self.outputs
            .extend((0..n).map(|i| i8::from(output_last && i + 1 == n)));
    }

    /// The llama.cpp view of the batch: its length and pointers into these
    /// arrays, valid until the batch is next changed or dropped.
    pub(crate) fn view(&mut self) -> sys::llama_batch {
        self.seq_id_ptrs.clear();
        self.seq_id_ptrs.extend(
            self.seq_ids
                .iter_mut()
                .map(|id| id as *mut sys::llama_seq_id),
        );
        sys::llama_batch {
            n_tokens: c_int::try_from(self.tokens.len()).expect("the capacity fits a C int"),
            token: self.tokens.as_mut_ptr(),
            embd: std::ptr::null_mut(),
            pos: self.positions.as_mut_ptr(),
            n_seq_id: self.seq_counts.as_mut_ptr(),
            seq_id: self.seq_id_ptrs.as_mut_ptr(),
            logits: self.outputs.as_mut_ptr(),
        }
    }
}

/// # Safety: callable from any thread at any time.
pub(crate) unsafe fn log_set_verbosity(level: i32) {
    sys::llama_log_set_verbosity_c(level as c_int);
}

/// # Safety: callable from any thread at any time.
pub(crate) unsafe fn log_get_verbosity() -> i32 {
    sys::llama_log_get_verbosity_c() as i32
}

/// Load an mtmd projector context.
///
/// # Safety
///
/// `path` must be a live C string. `model` must be a live llama model
/// pointer that stays live until the returned handle is freed with
/// [`mtmd_free`]: the projector keeps reading it.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_init_from_file(
    path: &CString,
    model: *const c_void,
    use_gpu: bool,
    warmup: bool,
    n_threads: usize,
    flash_attn: bool,
    max_image_tokens: Option<usize>,
) -> *mut MtmdContextRaw {
    sys::mtmd_init_from_file_c(
        path.as_ptr(),
        model,
        use_gpu,
        warmup,
        n_threads.min(c_int::MAX as usize) as c_int,
        flash_attn,
        // 0 keeps the projector's own limit.
        max_image_tokens.map_or(0, |tokens| tokens.min(c_int::MAX as usize) as c_int),
    )
}

/// # Safety: `ctx` must be a live mtmd context returned by
/// [`mtmd_init_from_file`]. Null is tolerated.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_free(ctx: *mut MtmdContextRaw) {
    if !ctx.is_null() {
        sys::mtmd_free_c(ctx);
    }
}

/// # Safety: `buf` must be valid for `len` bytes. `ctx` may be null for
/// image-only decode paths.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_init_from_buf(
    ctx: *mut MtmdContextRaw,
    buf: *const u8,
    len: usize,
) -> *mut MtmdBitmapRaw {
    sys::mtmd_bitmap_init_from_buf_c(ctx, buf, len)
}

/// # Safety: `data` must point to exactly `nx * ny * 3` tightly-packed RGB
/// bytes (RGBRGB... order, no row-stride padding, no alpha). The caller is
/// responsible for upholding that invariant before crossing the boundary.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_init_rgb(nx: u32, ny: u32, data: *const u8) -> *mut MtmdBitmapRaw {
    sys::mtmd_bitmap_init_rgb_c(nx, ny, data)
}

/// # Safety: `bitmap` must be a live mtmd bitmap. Null is tolerated.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_free(bitmap: *mut MtmdBitmapRaw) {
    if !bitmap.is_null() {
        sys::mtmd_bitmap_free_c(bitmap);
    }
}

/// # Safety: `bitmap` must be a live mtmd bitmap.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_get_nx(bitmap: *const MtmdBitmapRaw) -> u32 {
    sys::mtmd_bitmap_get_nx_c(bitmap)
}

/// # Safety: `bitmap` must be a live mtmd bitmap.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_get_ny(bitmap: *const MtmdBitmapRaw) -> u32 {
    sys::mtmd_bitmap_get_ny_c(bitmap)
}

/// # Safety: `bitmap` must be a live mtmd bitmap.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_get_n_bytes(bitmap: *const MtmdBitmapRaw) -> usize {
    sys::mtmd_bitmap_get_n_bytes_c(bitmap)
}

/// # Safety: `bitmap` must be a live mtmd bitmap. Returned string is owned
/// by mtmd and must not be freed.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_get_id(bitmap: *const MtmdBitmapRaw) -> *const c_char {
    sys::mtmd_bitmap_get_id_c(bitmap)
}

/// # Safety: `bitmap` must be live and `id` must be a live C string.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_bitmap_set_id(bitmap: *mut MtmdBitmapRaw, id: &CString) {
    sys::mtmd_bitmap_set_id_c(bitmap, id.as_ptr());
}

/// # Safety: returns an owned mtmd chunks allocation or null.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunks_init() -> *mut MtmdInputChunksRaw {
    sys::mtmd_input_chunks_init_c()
}

/// # Safety: `chunks` must be a live mtmd chunks handle. Null is tolerated.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunks_free(chunks: *mut MtmdInputChunksRaw) {
    if !chunks.is_null() {
        sys::mtmd_input_chunks_free_c(chunks);
    }
}

/// # Safety: `chunks` must be a live mtmd chunks handle.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunks_size(chunks: *const MtmdInputChunksRaw) -> usize {
    sys::mtmd_input_chunks_size_c(chunks)
}

/// # Safety: `chunks` must be live and `idx < size`.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunks_get(
    chunks: *const MtmdInputChunksRaw,
    idx: usize,
) -> *const MtmdInputChunkRaw {
    sys::mtmd_input_chunks_get_c(chunks, idx)
}

/// # Safety: `chunk` must be live.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunk_get_type(chunk: *const MtmdInputChunkRaw) -> c_int {
    sys::mtmd_input_chunk_get_type_c(chunk)
}

/// # Safety: `chunk` must be a live text chunk. `n_tokens_output` must be
/// writable.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunk_get_tokens_text(
    chunk: *const MtmdInputChunkRaw,
    n_tokens_output: *mut usize,
) -> *const i32 {
    sys::mtmd_input_chunk_get_tokens_text_c(chunk, n_tokens_output)
}

/// # Safety: `chunk` must be a live image chunk.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunk_get_tokens_image(
    chunk: *const MtmdInputChunkRaw,
) -> *const MtmdImageTokensRaw {
    sys::mtmd_input_chunk_get_tokens_image_c(chunk)
}

/// # Safety: `chunk` must be live.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunk_get_n_tokens(chunk: *const MtmdInputChunkRaw) -> usize {
    sys::mtmd_input_chunk_get_n_tokens_c(chunk)
}

/// # Safety: `chunk` must be live.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_input_chunk_get_n_pos(chunk: *const MtmdInputChunkRaw) -> i32 {
    sys::mtmd_input_chunk_get_n_pos_c(chunk)
}

/// # Safety: `image_tokens` must be live.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_image_tokens_get_n_tokens(
    image_tokens: *const MtmdImageTokensRaw,
) -> usize {
    sys::mtmd_image_tokens_get_n_tokens_c(image_tokens)
}

/// # Safety: `image_tokens` must be live.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_image_tokens_get_n_pos(image_tokens: *const MtmdImageTokensRaw) -> i32 {
    sys::mtmd_image_tokens_get_n_pos_c(image_tokens)
}

/// # Safety: `image_tokens` must be live and `i` must be in bounds.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_image_tokens_get_decoder_pos(
    image_tokens: *const MtmdImageTokensRaw,
    pos_0: i32,
    i: usize,
) -> MtmdDecoderPosRaw {
    sys::mtmd_image_tokens_get_decoder_pos_c(image_tokens, pos_0, i)
}

/// # Safety: `chunks` must be live.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_helper_get_n_tokens(chunks: *const MtmdInputChunksRaw) -> usize {
    sys::mtmd_helper_get_n_tokens_c(chunks)
}

/// # Safety: `chunks` must be live.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_helper_get_n_pos(chunks: *const MtmdInputChunksRaw) -> i32 {
    sys::mtmd_helper_get_n_pos_c(chunks)
}

/// # Safety: all handles/pointers must be live for the call duration.
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_tokenize(
    ctx: *mut MtmdContextRaw,
    output: *mut MtmdInputChunksRaw,
    text: &CString,
    add_special: bool,
    parse_special: bool,
    bitmaps: *mut *const MtmdBitmapRaw,
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

/// # Safety: each handle/pointer must be live for the call duration or null,
/// and live `ctx` and `lctx` must have been created from the same model. The
/// native wrapper returns an error for a null handle before using any.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "vision")]
pub(crate) unsafe fn mtmd_helper_eval_chunks(
    ctx: *mut MtmdContextRaw,
    lctx: *mut c_void,
    chunks: *const MtmdInputChunksRaw,
    n_past: i32,
    seq_id: i32,
    n_batch: usize,
    logits_last: bool,
    new_n_past: *mut i32,
) -> c_int {
    #[cfg(test)]
    record_native_call();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// One call of an accessor that behaves like `llama_model_meta_*`:
    /// `snprintf` of `value` into the buffer, -1 for a missing entry.
    fn accessor_call(value: Option<&[u8]>, buf: *mut c_char, len: usize) -> i32 {
        let Some(value) = value else { return -1 };
        if len > 0 {
            let n = value.len().min(len - 1);
            // SAFETY: the reader passes a buffer writable for `len` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(value.as_ptr(), buf.cast::<u8>(), n);
                *buf.add(n) = 0;
            }
        }
        i32::try_from(value.len()).unwrap()
    }

    #[test]
    fn a_metadata_string_is_sized_then_copied_whole() {
        let read = |value: Option<&[u8]>| {
            read_meta_string(|buf, len| accessor_call(value, buf, len)).unwrap()
        };
        assert_eq!(read(None), None);
        assert_eq!(read(Some(b"1.337398")), Some(b"1.337398".to_vec()));
        assert_eq!(read(Some(b"")), Some(Vec::new()));
    }

    #[test]
    fn a_metadata_string_that_changes_between_reads_is_an_error() {
        let changing = |second: Option<&'static [u8]>| {
            let mut calls = 0;
            read_meta_string(move |buf, len| {
                calls += 1;
                let value = if calls == 1 {
                    Some(&b"abc"[..])
                } else {
                    second
                };
                accessor_call(value, buf, len)
            })
        };
        assert!(matches!(
            changing(Some(b"abcd")),
            Err(LlamaError::Internal(_))
        ));
        assert!(matches!(
            changing(Some(b"ab")),
            Err(LlamaError::Internal(_))
        ));
        assert!(matches!(changing(None), Err(LlamaError::Internal(_))));
        assert_eq!(changing(Some(b"abc")).unwrap(), Some(b"abc".to_vec()));
    }
}
