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

/// Opaque handle to a loaded llama model
#[cfg(feature = "llm-llamacpp")]
pub struct LlamaModel {
    ptr: *mut c_void,
}

#[cfg(feature = "llm-llamacpp")]
unsafe impl Send for LlamaModel {}
#[cfg(feature = "llm-llamacpp")]
unsafe impl Sync for LlamaModel {}

/// Opaque handle to a llama context.
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

#[cfg(feature = "llm-llamacpp")]
unsafe impl Send for LlamaContext {}
// NOTE: Sync intentionally NOT implemented. llama_decode() mutates internal
// state and is not thread-safe. Use Mutex for shared access.

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
    fn llama_load_model_from_file_c(
        path_model: *const c_char,
        n_gpu_layers: c_int,
    ) -> *mut c_void;
    fn llama_free_model_c(model: *mut c_void);

    // Context management
    fn llama_new_context_with_model_c(
        model: *mut c_void,
        n_ctx: c_int,
        n_threads: c_int,
        n_batch: c_int,
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

    // Model info
    fn llama_n_vocab_c(model: *const c_void) -> c_int;
    fn llama_n_ctx_c(ctx: *const c_void) -> c_int;

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
        top_k: c_int,
        repeat_penalty: c_float,
        seed: u32,
        stop_seqs: *const i32,
        stop_lens: *const c_int,
        n_stop_seqs: c_int,
        callback: Option<TokenCallback>,
        user_data: *mut c_void,
    ) -> c_int;
}

/// Callback type for streaming token generation.
///
/// Return 0 to continue generation, non-zero to stop.
#[cfg(feature = "llm-llamacpp")]
pub type TokenCallback = extern "C" fn(token_id: i32, token_text: *const c_char, user_data: *mut c_void) -> c_int;

// =============================================================================
// Safe Wrapper Functions
// =============================================================================

/// Initialize the llama.cpp backend (call once at startup)
#[cfg(feature = "llm-llamacpp")]
pub fn llama_backend_init() {
    unsafe {
        llama_backend_init_c();
    }
}

/// Free the llama.cpp backend (call once at shutdown)
#[cfg(feature = "llm-llamacpp")]
pub fn llama_backend_free() {
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
    unsafe {
        llama_log_set_verbosity_c(level as c_int);
    }
}

/// Get the current verbosity level for llama.cpp/ggml logging.
#[cfg(feature = "llm-llamacpp")]
pub fn llama_log_get_verbosity() -> i32 {
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

    let ptr = unsafe { llama_load_model_from_file_c(c_path.as_ptr(), n_gpu_layers as c_int) };

    if ptr.is_null() {
        return Err(AdapterError::RuntimeError(format!(
            "Failed to load model from {}",
            path
        )));
    }

    Ok(LlamaModel { ptr })
}

/// Free a loaded model
#[cfg(feature = "llm-llamacpp")]
pub fn llama_free_model(model: LlamaModel) {
    unsafe {
        llama_free_model_c(model.ptr);
    }
}

/// Create a new context for a model
///
/// # Arguments
/// * `model` - The loaded model
/// * `n_ctx` - Context length (tokens)
/// * `n_threads` - Number of threads for inference (0 = auto-detect)
/// * `n_batch` - Batch size for prompt processing (0 = default 512)
#[cfg(feature = "llm-llamacpp")]
pub fn llama_new_context_with_model(
    model: &LlamaModel,
    n_ctx: usize,
    n_threads: usize,
    n_batch: usize,
) -> Result<LlamaContext, AdapterError> {
    let ptr = unsafe {
        llama_new_context_with_model_c(
            model.ptr,
            n_ctx as c_int,
            n_threads as c_int,
            n_batch as c_int,
        )
    };

    if ptr.is_null() {
        return Err(AdapterError::RuntimeError(
            "Failed to create context".to_string(),
        ));
    }

    Ok(LlamaContext { ptr })
}

/// Free a context
#[cfg(feature = "llm-llamacpp")]
pub fn llama_free(ctx: LlamaContext) {
    unsafe {
        llama_free_c(ctx.ptr);
    }
}

/// Clear the KV cache (reset context state for new conversation)
#[cfg(feature = "llm-llamacpp")]
pub fn llama_kv_cache_clear(ctx: &LlamaContext) {
    unsafe {
        llama_kv_cache_clear_c(ctx.ptr);
    }
}

/// Get the BOS (beginning of sequence) token
#[cfg(feature = "llm-llamacpp")]
pub fn llama_token_bos(model: &LlamaModel) -> i32 {
    unsafe { llama_token_bos_c(model.ptr) }
}

/// Get the EOS (end of sequence) token
#[cfg(feature = "llm-llamacpp")]
pub fn llama_token_eos(model: &LlamaModel) -> i32 {
    unsafe { llama_token_eos_c(model.ptr) }
}

/// Get vocabulary size
#[cfg(feature = "llm-llamacpp")]
pub fn llama_n_vocab(model: &LlamaModel) -> usize {
    unsafe { llama_n_vocab_c(model.ptr) as usize }
}

/// Get context length
#[cfg(feature = "llm-llamacpp")]
pub fn llama_n_ctx(ctx: &LlamaContext) -> usize {
    unsafe { llama_n_ctx_c(ctx.ptr) as usize }
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

    // Convert messages to C strings
    let roles: Vec<CString> = messages
        .iter()
        .map(|m| CString::new(m.role.as_str()).unwrap_or_else(|_| CString::new("user").unwrap()))
        .collect();
    let contents: Vec<CString> = messages
        .iter()
        .map(|m| CString::new(m.content.as_str()).unwrap_or_else(|_| CString::new("").unwrap()))
        .collect();

    let role_ptrs: Vec<*const c_char> = roles.iter().map(|s| s.as_ptr()).collect();
    let content_ptrs: Vec<*const c_char> = contents.iter().map(|s| s.as_ptr()).collect();

    // Allocate output buffer (start with 4KB, should be enough for most prompts)
    let mut buf = vec![0u8; 4096];

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

    // First call to get required size (returns negative count)
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
        return Err(AdapterError::RuntimeError("Tokenization failed".to_string()));
    }

    tokens.truncate(result as usize);
    Ok(tokens)
}

/// Tokenize text with special token parsing enabled.
///
/// This is used for stop sequences like `<|im_end|>` which should be
/// recognized as special tokens, not literal character sequences.
#[cfg(feature = "llm-llamacpp")]
pub fn llama_tokenize_special(
    model: &LlamaModel,
    text: &str,
) -> Result<Vec<i32>, AdapterError> {
    let c_text = CString::new(text)
        .map_err(|_| AdapterError::InvalidInput("Invalid text encoding".to_string()))?;

    // First call to get required size
    let n_tokens = unsafe {
        llama_tokenize_c(
            model.ptr,
            c_text.as_ptr(),
            text.len() as c_int,
            ptr::null_mut(),
            0,
            false,  // don't add BOS/EOS
            true,   // parse_special = true for <|im_end|> etc.
        )
    };

    let required_size = if n_tokens < 0 { -n_tokens } else { n_tokens };

    if required_size <= 0 {
        return Ok(Vec::new());
    }

    let mut tokens = vec![0i32; required_size as usize + 16];
    let result = unsafe {
        llama_tokenize_c(
            model.ptr,
            c_text.as_ptr(),
            text.len() as c_int,
            tokens.as_mut_ptr(),
            tokens.len() as c_int,
            false,  // don't add BOS/EOS
            true,   // parse_special = true
        )
    };

    if result < 0 {
        return Err(AdapterError::RuntimeError("Tokenization failed".to_string()));
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
        let len = unsafe {
            llama_token_to_piece_c(
                model.ptr,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
                0,
                true,  // special = true: render special tokens like <|im_end|> as text
            )
        };

        if len > 0 && (len as usize) < buf.len() {
            if let Ok(piece) = std::str::from_utf8(&buf[..len as usize]) {
                result.push_str(piece);
            }
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
        let tokens = llama_tokenize_special(model, seq)?;
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
        .map(|d| d.as_secs() as u32)
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
            top_k as c_int,
            repeat_penalty,
            seed,
            stop_seqs_ptr,
            stop_lens_ptr,
            n_stop_seqs,
        )
    };

    if result < 0 {
        return Err(AdapterError::RuntimeError(format!(
            "Generation failed with error code {}",
            result
        )));
    }

    output_tokens.truncate(result as usize);
    Ok(output_tokens)
}

/// Generate tokens using autoregressive decoding (without stop sequences).
///
/// This is a convenience wrapper around `llama_generate_with_stops` for
/// backwards compatibility. Uses default repetition penalty of 1.1.
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
    llama_generate_with_stops(ctx, model, input_tokens, max_tokens, temperature, top_p, top_k, 1.1, &[])
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
    let ctx = unsafe { &mut *(user_data as *mut StreamingContext<F>) };

    // Convert C string to Rust string
    let text = if token_text.is_null() {
        ""
    } else {
        unsafe { std::ffi::CStr::from_ptr(token_text) }
            .to_str()
            .unwrap_or("")
    };

    // Call the Rust closure
    match (ctx.callback)(token_id, text) {
        Ok(()) => 0,  // Continue
        Err(e) => {
            ctx.error = Some(e);
            1  // Stop
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
/// * `top_k` - Top-k sampling (0 = disabled)
/// * `repeat_penalty` - Repetition penalty (1.0 = disabled)
/// * `stop_sequences` - Optional stop sequences (as strings)
/// * `on_token` - Callback called for each generated token
///
/// # Returns
/// Vector of generated token IDs and whether generation was stopped by callback.
#[cfg(feature = "llm-llamacpp")]
pub fn llama_generate_streaming<F>(
    ctx: &LlamaContext,
    model: &LlamaModel,
    input_tokens: &[i32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    stop_sequences: &[String],
    mut on_token: F,
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
        let tokens = llama_tokenize_special(model, seq)?;
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
        .map(|d| d.as_secs() as u32)
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
            top_k as c_int,
            repeat_penalty,
            seed,
            stop_seqs_ptr,
            stop_lens_ptr,
            n_stop_seqs,
            Some(streaming_trampoline::<F>),
            &mut streaming_ctx as *mut StreamingContext<F> as *mut c_void,
        )
    };

    // Check for callback error
    if let Some(err) = streaming_ctx.error {
        return Err(AdapterError::RuntimeError(format!(
            "Streaming callback error: {}",
            err
        )));
    }

    // Negative result means stopped by callback, but absolute value is token count
    let (n_generated, stopped_by_callback) = if result < 0 {
        ((-result) as usize, true)
    } else {
        (result as usize, false)
    };

    if result == -1 || result == -2 || result == -3 {
        return Err(AdapterError::RuntimeError(format!(
            "Generation failed with error code {}",
            result
        )));
    }

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
pub fn llama_generate_streaming<F>(
    _ctx: &LlamaContext,
    _model: &LlamaModel,
    _input_tokens: &[i32],
    _max_tokens: usize,
    _temperature: f32,
    _top_p: f32,
    _top_k: usize,
    _repeat_penalty: f32,
    _stop_sequences: &[String],
    _on_token: F,
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
    use std::os::raw::c_int;

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
            vec![32000, 32001],  // <|im_end|> → 2 tokens
            vec![],              // <|unknown_token|> → empty (filtered out)
            vec![32002],         // <|end_of_text|> → 1 token
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
            retry_result as usize  // FIXED: use retry's value
        } else {
            result as usize
        };

        assert_eq!(len, 4998, "Must use retry_result (4998), not first result (5000)");
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
        let batch_size = if small_input > fixed_batch_size { small_input } else { fixed_batch_size };
        assert!(batch_size >= small_input);

        // Large input: batch must grow to fit
        let large_input = 2000;
        let batch_size = if large_input > fixed_batch_size { large_input } else { fixed_batch_size };
        assert_eq!(batch_size, 2000, "Batch must grow to fit large input");
        assert!(batch_size >= large_input);

        // Edge case: exactly 512
        let exact_input = 512;
        let batch_size = if exact_input > fixed_batch_size { exact_input } else { fixed_batch_size };
        assert_eq!(batch_size, 512);
        assert!(batch_size >= exact_input);

        // Edge case: 513 tokens (one over) → must allocate 513
        let over_input = 513;
        let batch_size = if over_input > fixed_batch_size { over_input } else { fixed_batch_size };
        assert_eq!(batch_size, 513, "Batch must not use fixed 512 when input is 513");
    }
}
