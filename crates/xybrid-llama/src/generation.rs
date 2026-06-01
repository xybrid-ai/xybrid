//! Autoregressive generation paths: streaming + non-streaming, with
//! KV-cache prefix-reuse and stop-sequence support.
//!
//! The [`StreamingContext`] / [`streaming_trampoline`] pair carries the
//! `Option<extern "C" fn(...)>` type-erasure verbatim from the
//! pre-refactor home inside `xybrid-core::runtime_adapter::llama_cpp`.
//! The `CloudFallbackAbort` round-trip path through this trampoline is
//! the single most behavior-sensitive surface in the refactor; the
//! inline regression test below watches it without exposing public hooks.

use std::error::Error as StdError;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::time::SystemTime;

use llama_cpp_sys::bindings::TokenCallback;

use crate::context::LlamaContext;
use crate::error::{LlamaError, LlamaResult};
use crate::ffi;
use crate::model::LlamaModel;

/// Closure type alias for per-token callbacks. The signature deliberately
/// boxes the error so any `Send + Sync` error (notably
/// `xybrid-core::abort::CloudFallbackAbort`) can survive the trampoline
/// round-trip.
pub type StreamingCallback<'a> =
    &'a mut dyn FnMut(i32, &str) -> Result<(), Box<dyn StdError + Send + Sync>>;

/// Heap-side state passed through the C callback to the Rust closure.
///
/// Generic over `F` so the monomorphised trampoline knows the closure
/// shape statically. The `error` slot captures any `Err(_)` the closure
/// returned so the safe-wrapper caller can recover it after the C side
/// returns.
struct StreamingContext<'a, F>
where
    F: FnMut(i32, &str) -> Result<(), Box<dyn StdError + Send + Sync>>,
{
    callback: &'a mut F,
    error: Option<Box<dyn StdError + Send + Sync>>,
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
/// After Phase 5's bindgen migration the trampoline must match the
/// bindgen-emitted `llama_token_callback_c` typedef, which carries
/// `unsafe extern "C" fn(...)` semantics.
unsafe extern "C" fn streaming_trampoline<F>(
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

/// Shared prelude for the generation entry points: rejects empty input,
/// tokenises the stop sequences, and allocates the output buffer. Returns
/// the owned `(stop_tokens, stop_lens, output_tokens)` triple; the caller
/// keeps them alive for the FFI call and derives raw pointers via
/// [`stop_array_ptrs`].
fn prepare_generation(
    model: &LlamaModel,
    input_tokens: &[i32],
    max_tokens: usize,
    stop_sequences: &[String],
) -> LlamaResult<(Vec<i32>, Vec<c_int>, Vec<i32>)> {
    if input_tokens.is_empty() {
        return Err(LlamaError::InvalidInput("empty input tokens".to_string()));
    }
    let (stop_tokens, stop_lens) = build_stop_token_arrays(model, stop_sequences)?;
    let output_tokens = vec![0i32; max_tokens];
    Ok((stop_tokens, stop_lens, output_tokens))
}

/// Raw `(seqs, lens, count)` the C generate functions expect for the
/// stop-sequence arrays. An empty stop list passes `null / null / 0` (the
/// wrapper reads that as "no stop sequences"); a populated list passes the
/// owned buffers' pointers and the *filtered* count.
fn stop_array_ptrs(stop_tokens: &[i32], stop_lens: &[c_int]) -> (*const i32, *const c_int, c_int) {
    if stop_lens.is_empty() {
        (ptr::null(), ptr::null(), 0)
    } else {
        (
            stop_tokens.as_ptr(),
            stop_lens.as_ptr(),
            stop_lens.len() as c_int,
        )
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
    let (stop_tokens, stop_lens, mut output_tokens) =
        prepare_generation(model, input_tokens, max_tokens, stop_sequences)?;
    let (stop_seqs_ptr, stop_lens_ptr, n_stop_seqs) = stop_array_ptrs(&stop_tokens, &stop_lens);

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
            temperature,
            top_p,
            min_p,
            top_k,
            repeat_penalty,
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
    let (stop_tokens, stop_lens, mut output_tokens) =
        prepare_generation(model, input_tokens, max_tokens, stop_sequences)?;
    let (stop_seqs_ptr, stop_lens_ptr, n_stop_seqs) = stop_array_ptrs(&stop_tokens, &stop_lens);

    let mut streaming_ctx = StreamingContext {
        callback: &mut on_token,
        error: None,
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
            temperature,
            top_p,
            min_p,
            top_k,
            repeat_penalty,
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

/// Render role/content slices through `model`'s built-in chat template.
///
/// Returns `Ok(None)` only when the model has no usable embedded template —
/// either the GGUF metadata key is absent or its value is empty/whitespace
/// (which the native wrapper rejects anyway) — so the caller can apply its own
/// model-family fallback policy. If a non-empty template exists but native
/// rendering fails, this returns [`LlamaError::ChatTemplateFailed`] instead of
/// silently switching prompt families.
pub fn format_chat(
    model: &LlamaModel,
    roles: &[&str],
    contents: &[&str],
) -> LlamaResult<Option<String>> {
    if roles.is_empty() {
        return Err(LlamaError::InvalidInput("empty messages".to_string()));
    }
    if roles.len() != contents.len() {
        return Err(LlamaError::InvalidInput(format!(
            "chat roles/content length mismatch: roles={}, contents={}",
            roles.len(),
            contents.len()
        )));
    }
    if model
        .chat_template()
        .filter(|t| !t.trim().is_empty())
        .is_none()
    {
        return Ok(None);
    }

    let roles: Vec<CString> = roles
        .iter()
        .map(|role| ffi::cstring(role, "chat role"))
        .collect::<Result<Vec<_>, _>>()?;
    let contents: Vec<CString> = contents
        .iter()
        .map(|content| ffi::cstring(content, "chat content"))
        .collect::<Result<Vec<_>, _>>()?;

    let role_ptrs: Vec<*const c_char> = roles.iter().map(|s| s.as_ptr()).collect();
    let content_ptrs: Vec<*const c_char> = contents.iter().map(|s| s.as_ptr()).collect();

    let input_bytes: usize = roles.iter().map(|s| s.as_bytes().len()).sum::<usize>()
        + contents.iter().map(|s| s.as_bytes().len()).sum::<usize>();
    let initial_cap = (input_bytes * 3).max(4096);
    let mut buf = vec![0u8; initial_cap];

    // SAFETY: model.as_ptr is live; role_ptrs / content_ptrs are valid
    // for the call duration (the underlying CStrings live in
    // `roles` / `contents`); buf is writable for buf.len() bytes.
    let result = unsafe {
        ffi::format_chat_with_model(
            model.as_ptr(),
            role_ptrs.as_ptr(),
            content_ptrs.as_ptr(),
            roles.len(),
            buf.as_mut_ptr() as *mut c_char,
            buf.len(),
        )
    };

    if result < 0 {
        tracing::warn!(
            target: "xybrid_llama",
            code = result,
            "model chat template failed"
        );
        return Err(chat_template_render_failed(result, "initial render"));
    }

    let len = if result as usize >= buf.len() {
        buf.resize((result as usize) + 1, 0);
        // SAFETY: buf resized above.
        let retry_result = unsafe {
            ffi::format_chat_with_model(
                model.as_ptr(),
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                roles.len(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
            )
        };
        if retry_result < 0 {
            return Err(chat_template_render_failed(retry_result, "retry render"));
        }
        if retry_result as usize >= buf.len() {
            return Err(LlamaError::ChatTemplateFailed {
                detail: format!(
                    "retry render still required {} bytes after resizing buffer to {} bytes",
                    retry_result,
                    buf.len()
                ),
            });
        }
        retry_result as usize
    } else {
        result as usize
    };

    Ok(Some(prompt_from_template_bytes(&buf, len)?))
}

fn chat_template_render_failed(code: c_int, phase: &'static str) -> LlamaError {
    LlamaError::ChatTemplateFailed {
        detail: format!("native formatter {phase} returned error code {code}"),
    }
}

fn prompt_from_template_bytes(buf: &[u8], len: usize) -> LlamaResult<String> {
    if len > buf.len() {
        return Err(LlamaError::ChatTemplateFailed {
            detail: format!(
                "native formatter reported {} bytes but buffer only has {} bytes",
                len,
                buf.len()
            ),
        });
    }

    std::str::from_utf8(&buf[..len])
        .map(str::to_owned)
        .map_err(|err| LlamaError::ChatTemplateFailed {
            detail: format!("native formatter produced invalid UTF-8: {err}"),
        })
}

#[cfg(test)]
mod tests {
    use super::{
        chat_template_render_failed, prompt_from_template_bytes, streaming_trampoline, LlamaError,
        StreamingContext,
    };
    use std::error::Error;
    use std::ffi::CString;
    use std::fmt;
    use std::os::raw::c_void;
    use std::time::{Duration, Instant};

    #[derive(Debug)]
    struct MarkerError(&'static str);

    impl fmt::Display for MarkerError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl Error for MarkerError {}

    type Callback = fn(i32, &str) -> Result<(), Box<dyn Error + Send + Sync>>;

    fn abort_callback(_token_id: i32, _text: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        Err(Box::new(MarkerError("marker preserved through trampoline")))
    }

    #[test]
    fn stop_array_ptrs_null_when_empty_and_filtered_count_otherwise() {
        // Empty stop list -> null/null/0; the wrapper reads that as "no
        // stop sequences". Shared by both generate paths after the prelude
        // extraction, so this guards the null sentinel for both.
        let (seqs, lens, count) = super::stop_array_ptrs(&[], &[]);
        assert!(seqs.is_null());
        assert!(lens.is_null());
        assert_eq!(count, 0);

        // Populated -> pointers into the owned buffers + the *filtered*
        // count (number of stop sequences that tokenised to >0 tokens),
        // not the flattened token count.
        let stop_tokens = vec![1i32, 2, 3];
        let stop_lens = vec![2 as std::os::raw::c_int, 1];
        let (seqs, lens, count) = super::stop_array_ptrs(&stop_tokens, &stop_lens);
        assert_eq!(seqs, stop_tokens.as_ptr());
        assert_eq!(lens, stop_lens.as_ptr());
        assert_eq!(count, 2);
    }

    #[test]
    fn streaming_trampoline_preserves_boxed_error_marker() {
        let mut callback: Callback = abort_callback;
        let mut ctx = StreamingContext {
            callback: &mut callback,
            error: None,
        };
        let token_text = CString::new("token").unwrap();

        let started = Instant::now();
        // SAFETY: ctx is a valid `&mut StreamingContext<Callback>`, the
        // token_text CString lives for the duration of the call, and the
        // callback pointer in ctx is live.
        let stop = unsafe {
            streaming_trampoline::<Callback>(
                42,
                token_text.as_ptr(),
                &mut ctx as *mut StreamingContext<Callback> as *mut c_void,
            )
        };
        let elapsed = started.elapsed();

        assert_eq!(stop, 1, "callback errors must stop the C stream");
        assert!(
            elapsed <= Duration::from_millis(50),
            "llama.cpp trampoline abort exceeded M-series cancellation budget: {:?}",
            elapsed
        );
        let err = ctx.error.take().expect("callback error must be stored");
        let downcast: &MarkerError = err
            .downcast_ref::<MarkerError>()
            .expect("typed marker must survive the trampoline boundary");
        assert_eq!(downcast.0, "marker preserved through trampoline");
    }

    #[test]
    fn chat_template_render_error_is_not_fallback() {
        let err = chat_template_render_failed(-1, "initial render");

        match err {
            LlamaError::ChatTemplateFailed { detail } => {
                assert!(detail.contains("initial render"));
                assert!(detail.contains("-1"));
            }
            other => panic!("expected ChatTemplateFailed, got {other:?}"),
        }
    }

    #[test]
    fn invalid_chat_template_utf8_is_not_fallback() {
        let err = prompt_from_template_bytes(&[0xff, 0xfe], 2)
            .expect_err("invalid rendered prompt must be a template error");

        match err {
            LlamaError::ChatTemplateFailed { detail } => {
                assert!(detail.contains("invalid UTF-8"));
            }
            other => panic!("expected ChatTemplateFailed, got {other:?}"),
        }
    }
}
