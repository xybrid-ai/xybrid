//! Autoregressive generation paths: streaming + non-streaming, with
//! KV-cache prefix-reuse and stop-sequence support.
//!
//! The [`StreamingContext`] / [`streaming_trampoline`] pair carries the
//! `Option<extern "C" fn(...)>` type-erasure verbatim from the
//! pre-refactor home inside `xybrid-core::runtime_adapter::llama_cpp`.
//! The `CloudFallbackAbort` round-trip path through this trampoline is
//! the single most behavior-sensitive surface in the refactor; the
//! regression test at `crates/xybrid-llama/tests/cloud_fallback_abort.rs`
//! watches it.

use std::error::Error as StdError;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::time::SystemTime;

use llama_cpp_sys::bindings::TokenCallback;

use crate::context::LlamaContext;
use crate::error::{LlamaError, LlamaResult};
use crate::ffi::{self, SamplingArgs};
use crate::model::LlamaModel;

/// Closure type alias for per-token callbacks. The signature deliberately
/// boxes the error so any `Send + Sync` error (notably
/// `xybrid-core::abort::CloudFallbackAbort`) can survive the trampoline
/// round-trip.
pub type StreamingCallback<'a> =
    &'a mut dyn FnMut(i32, &str) -> Result<(), Box<dyn StdError + Send + Sync>>;

/// Minimal chat-message view consumed by [`format_chat`].
///
/// Trait-based so callers can use their own `ChatMessage` type
/// (`xybrid-core::runtime_adapter::types::ChatMessage`) without forcing
/// `xybrid-llama` to take a dep on `xybrid-core`. The two getters return
/// borrowed `&str` so callers pay zero allocations.
pub trait ChatMessageView {
    fn role(&self) -> &str;
    fn content(&self) -> &str;
}

/// Heap-side state passed through the C callback to the Rust closure.
///
/// Generic over `F` so the monomorphised trampoline knows the closure
/// shape statically. The `error` slot captures any `Err(_)` the closure
/// returned so the safe-wrapper caller can recover it after the C side
/// returns.
///
/// Exposed as `pub` (rather than `pub(crate)`) only so the
/// `__test_hooks` module in `crate::lib` can re-export it for integration
/// tests; `#[doc(hidden)]` keeps it out of rustdoc and IDE autocomplete.
/// Treat as crate-private — the shape is unstable.
#[doc(hidden)]
pub struct StreamingContext<'a, F>
where
    F: FnMut(i32, &str) -> Result<(), Box<dyn StdError + Send + Sync>>,
{
    pub callback: &'a mut F,
    pub error: Option<Box<dyn StdError + Send + Sync>>,
}

/// C-compatible trampoline that bridges llama.cpp's token callback into
/// the Rust closure stored in [`StreamingContext`].
///
/// Returns 0 to keep generating, non-zero to stop.
///
/// # Safety
///
/// The C side must invoke this with `user_data` being a live, exclusive
/// pointer to a `StreamingContext<F>` whose lifetime brackets every
/// invocation (`StreamingContext` lives on the safe-wrapper's stack across
/// the generation call). The matching `extern "C" fn` ABI is what makes
/// the `Option<TokenCallback>` parameter on
/// `llama_generate_streaming_c` accept this function pointer.
///
/// Exposed `pub` for `__test_hooks` re-export; `#[doc(hidden)]` keeps
/// it out of public surface in rustdoc. After Phase 5's bindgen
/// migration the trampoline must match the bindgen-emitted
/// `llama_token_callback_c` typedef, which carries `unsafe extern "C"
/// fn(...)` semantics.
#[doc(hidden)]
pub unsafe extern "C" fn streaming_trampoline<F>(
    token_id: i32,
    token_text: *const c_char,
    user_data: *mut c_void,
) -> c_int
where
    F: FnMut(i32, &str) -> Result<(), Box<dyn StdError + Send + Sync>>,
{
    // SAFETY: caller upholds the trampoline's `# Safety` block above.
    // The whole function is `unsafe extern "C" fn`, so the body is in
    // an implicit unsafe scope — no `unsafe { ... }` blocks needed
    // around the raw-pointer derefs.
    let ctx = &mut *(user_data as *mut StreamingContext<F>);

    let text = if token_text.is_null() {
        ""
    } else {
        CStr::from_ptr(token_text).to_str().unwrap_or("")
    };

    match (ctx.callback)(token_id, text) {
        Ok(()) => 0,
        Err(e) => {
            ctx.error = Some(e);
            1
        }
    }
}

fn build_stop_token_arrays(
    model: &LlamaModel,
    stop_sequences: &[String],
) -> LlamaResult<(Vec<i32>, Vec<c_int>)> {
    let mut tokens: Vec<i32> = Vec::new();
    let mut lens: Vec<c_int> = Vec::new();
    for seq in stop_sequences {
        // Tokenize WITH special-token parsing — stop sequences like
        // `<|im_end|>` are typically special tokens.
        let toks = model.tokenize_special(seq, false)?;
        if !toks.is_empty() {
            lens.push(toks.len() as c_int);
            tokens.extend(toks);
        }
    }
    Ok((tokens, lens))
}

fn time_seed() -> u32 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(42)
}

fn decode_hard_error(code: i32, n_past_in: usize) -> LlamaError {
    let detail = match code {
        -1 => "invalid arguments (null context/model/input or non-positive sizes)",
        -2 => "sampler chain creation failed",
        -3 => {
            // The wrapper unconditionally logs the actual llama_decode
            // return code + n_past_in / chunk position to stderr (see
            // `llama_generate_streaming_c` in llama_wrapper.cpp); the
            // diagnostic is not gated on `XYBRID_LLAMACPP_VERBOSITY`,
            // which only controls llama.cpp's own log callback path.
            // When n_past_in > 0 the prefix-reuse path was in play; that
            // is the path that triggers KV-cache state mismatches on
            // recurrent / hybrid models. Gate via
            // `LlamaModel::has_recurrent_state`.
            "llama_decode failed on prefill (KV-cache state mismatch likely; see stderr for the wrapper-level diagnostic line emitted by `llama_generate_streaming_c`)"
        }
        -4 => "input + prefix exceeds context window (n_past_in + n_input >= n_ctx)",
        _ => "unknown",
    };
    LlamaError::DecodeFailed {
        code,
        n_past_in,
        detail: detail.to_string(),
    }
}

/// Autoregressive generation without streaming. Returns the generated
/// token IDs.
///
/// `stop_sequences` are tokenised with special-token parsing enabled — a
/// sequence that tokenises to zero tokens is silently dropped, matching
/// the pre-refactor wrapper behavior. The count passed to the C side is
/// the *filtered* length, not the original `stop_sequences.len()`.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_stops(
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
) -> LlamaResult<Vec<i32>> {
    if input_tokens.is_empty() {
        return Err(LlamaError::InvalidInput("empty input tokens".to_string()));
    }

    let (stop_tokens, stop_lens) = build_stop_token_arrays(model, stop_sequences)?;
    let mut output_tokens = vec![0i32; max_tokens];

    let (stop_seqs_ptr, stop_lens_ptr, n_stop_seqs) = if stop_lens.is_empty() {
        (ptr::null(), ptr::null(), 0)
    } else {
        (
            stop_tokens.as_ptr(),
            stop_lens.as_ptr(),
            stop_lens.len() as c_int,
        )
    };

    let sampling = SamplingArgs {
        temperature,
        top_p,
        min_p,
        top_k,
        repeat_penalty,
    };

    // SAFETY: all pointers checked / sourced from owned buffers; sizes
    // honest; ctx + model live for the call.
    let result = unsafe {
        ffi::generate(
            ctx.as_ptr(),
            model.as_ptr(),
            input_tokens.as_ptr(),
            input_tokens.len(),
            output_tokens.as_mut_ptr(),
            max_tokens,
            sampling,
            time_seed(),
            stop_seqs_ptr,
            stop_lens_ptr,
            n_stop_seqs,
        )
    };

    if result < 0 {
        return Err(decode_hard_error(result, 0));
    }

    output_tokens.truncate(result as usize);
    Ok(output_tokens)
}

/// Streaming generation. Calls `on_token` for each generated token; an
/// `Err(_)` from the closure aborts generation and surfaces as
/// [`LlamaError::StreamingCallbackAborted`], preserving the boxed error.
///
/// `n_past_in` is the KV-cache prefix position the caller has prepared
/// via [`LlamaContext::kv_cache_seq_rm`] (`0` = fresh prefill). The
/// returned bool indicates whether the closure stopped generation early.
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming<F>(
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
    n_past_in: usize,
) -> LlamaResult<(Vec<i32>, bool)>
where
    F: FnMut(i32, &str) -> Result<(), Box<dyn StdError + Send + Sync>>,
{
    if input_tokens.is_empty() {
        return Err(LlamaError::InvalidInput("empty input tokens".to_string()));
    }

    let (stop_tokens, stop_lens) = build_stop_token_arrays(model, stop_sequences)?;
    let mut output_tokens = vec![0i32; max_tokens];

    let (stop_seqs_ptr, stop_lens_ptr, n_stop_seqs) = if stop_lens.is_empty() {
        (ptr::null(), ptr::null(), 0)
    } else {
        (
            stop_tokens.as_ptr(),
            stop_lens.as_ptr(),
            stop_lens.len() as c_int,
        )
    };

    let mut streaming_ctx = StreamingContext {
        callback: &mut on_token,
        error: None,
    };

    let sampling = SamplingArgs {
        temperature,
        top_p,
        min_p,
        top_k,
        repeat_penalty,
    };

    let callback: Option<TokenCallback> = Some(streaming_trampoline::<F>);

    // SAFETY: all pointers checked / sourced from owned buffers; the
    // user_data pointer is a stack-pinned `&mut StreamingContext<F>`
    // that lives for the duration of the C call.
    let result = unsafe {
        ffi::generate_streaming(
            ctx.as_ptr(),
            model.as_ptr(),
            input_tokens.as_ptr(),
            input_tokens.len(),
            output_tokens.as_mut_ptr(),
            max_tokens,
            sampling,
            time_seed(),
            stop_seqs_ptr,
            stop_lens_ptr,
            n_stop_seqs,
            callback,
            &mut streaming_ctx as *mut StreamingContext<F> as *mut c_void,
            n_past_in,
        )
    };

    // Hard error codes first — these are never callback-stop.
    if (-4..=-1).contains(&result) {
        return Err(decode_hard_error(result, n_past_in));
    }

    // Callback error wins over the silent "stopped by callback" path.
    if let Some(err) = streaming_ctx.error.take() {
        return Err(LlamaError::StreamingCallbackAborted(err));
    }

    let (n_generated, stopped_by_callback) = if result < 0 {
        ((-result) as usize, true)
    } else {
        (result as usize, false)
    };

    output_tokens.truncate(n_generated);
    Ok((output_tokens, stopped_by_callback))
}

/// Render `messages` through `model`'s built-in chat template, with a
/// fallback to a minimal ChatML format if the C-side template invocation
/// fails. Returns the formatted prompt string.
///
/// Caller supplies `messages` as anything implementing
/// [`ChatMessageView`], so `xybrid-llama` does not need to depend on
/// `xybrid-core` for its `ChatMessage` type.
pub fn format_chat<M: ChatMessageView>(model: &LlamaModel, messages: &[M]) -> LlamaResult<String> {
    if messages.is_empty() {
        return Err(LlamaError::InvalidInput("empty messages".to_string()));
    }

    let roles: Vec<CString> = messages
        .iter()
        .map(|m| ffi::cstring(m.role(), "chat role"))
        .collect::<Result<Vec<_>, _>>()?;
    let contents: Vec<CString> = messages
        .iter()
        .map(|m| ffi::cstring(m.content(), "chat content"))
        .collect::<Result<Vec<_>, _>>()?;

    let role_ptrs: Vec<*const c_char> = roles.iter().map(|s| s.as_ptr()).collect();
    let content_ptrs: Vec<*const c_char> = contents.iter().map(|s| s.as_ptr()).collect();

    let mut buf = vec![0u8; 4096];

    // SAFETY: model.as_ptr is live; role_ptrs / content_ptrs are valid
    // for the call duration (the underlying CStrings live in
    // `roles` / `contents`); buf is writable for buf.len() bytes.
    let result = unsafe {
        ffi::format_chat_with_model(
            model.as_ptr(),
            role_ptrs.as_ptr(),
            content_ptrs.as_ptr(),
            messages.len(),
            buf.as_mut_ptr() as *mut c_char,
            buf.len(),
        )
    };

    if result < 0 {
        tracing::warn!(
            target: "xybrid_llama",
            code = result,
            "model chat template failed; falling back to ChatML format"
        );
        return Ok(format_chat_chatml(messages));
    }

    let len = if result as usize >= buf.len() {
        buf.resize((result as usize) + 1, 0);
        // SAFETY: buf resized above.
        let retry_result = unsafe {
            ffi::format_chat_with_model(
                model.as_ptr(),
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
            )
        };
        if retry_result < 0 {
            return Ok(format_chat_chatml(messages));
        }
        retry_result as usize
    } else {
        result as usize
    };

    match std::str::from_utf8(&buf[..len]) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => Ok(format_chat_chatml(messages)),
    }
}

/// Minimal ChatML fallback for models without an embedded template.
fn format_chat_chatml<M: ChatMessageView>(messages: &[M]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role() {
            "system" => prompt.push_str(&format!(
                "<|im_start|>system\n{}<|im_end|>\n",
                msg.content()
            )),
            "user" => prompt.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n",
                msg.content()
            )),
            "assistant" => prompt.push_str(&format!(
                "<|im_start|>assistant\n{}<|im_end|>\n",
                msg.content()
            )),
            _ => prompt.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n",
                msg.content()
            )),
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}
