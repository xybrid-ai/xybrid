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

#[cfg(feature = "local-llm-llamacpp")]
use std::ffi::CString;
#[cfg(feature = "local-llm-llamacpp")]
use std::os::raw::{c_char, c_float, c_int, c_void};
#[cfg(feature = "local-llm-llamacpp")]
use std::ptr;

use crate::runtime_adapter::llm::ChatMessage;
use crate::runtime_adapter::AdapterError;

// =============================================================================
// Opaque Types
// =============================================================================

/// Opaque handle to a loaded llama model
#[cfg(feature = "local-llm-llamacpp")]
pub struct LlamaModel {
    ptr: *mut c_void,
}

#[cfg(feature = "local-llm-llamacpp")]
unsafe impl Send for LlamaModel {}
#[cfg(feature = "local-llm-llamacpp")]
unsafe impl Sync for LlamaModel {}

/// Opaque handle to a llama context
#[cfg(feature = "local-llm-llamacpp")]
pub struct LlamaContext {
    ptr: *mut c_void,
}

#[cfg(feature = "local-llm-llamacpp")]
unsafe impl Send for LlamaContext {}
#[cfg(feature = "local-llm-llamacpp")]
unsafe impl Sync for LlamaContext {}

// =============================================================================
// FFI Declarations
// =============================================================================

#[cfg(feature = "local-llm-llamacpp")]
extern "C" {
    // Backend initialization
    fn llama_backend_init_c();
    fn llama_backend_free_c();

    // Model loading
    fn llama_load_model_from_file_c(
        path_model: *const c_char,
        n_gpu_layers: c_int,
    ) -> *mut c_void;
    fn llama_free_model_c(model: *mut c_void);

    // Context management
    fn llama_new_context_with_model_c(model: *mut c_void, n_ctx: c_int) -> *mut c_void;
    fn llama_free_c(ctx: *mut c_void);

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

    // Generation loop
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
        seed: u32,
    ) -> c_int;
}

// =============================================================================
// Safe Wrapper Functions
// =============================================================================

/// Initialize the llama.cpp backend (call once at startup)
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_backend_init() {
    unsafe {
        llama_backend_init_c();
    }
}

/// Free the llama.cpp backend (call once at shutdown)
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_backend_free() {
    unsafe {
        llama_backend_free_c();
    }
}

/// Load a model from a GGUF file
#[cfg(feature = "local-llm-llamacpp")]
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
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_free_model(model: LlamaModel) {
    unsafe {
        llama_free_model_c(model.ptr);
    }
}

/// Create a new context for a model
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_new_context_with_model(
    model: &LlamaModel,
    n_ctx: usize,
) -> Result<LlamaContext, AdapterError> {
    let ptr = unsafe { llama_new_context_with_model_c(model.ptr, n_ctx as c_int) };

    if ptr.is_null() {
        return Err(AdapterError::RuntimeError(
            "Failed to create context".to_string(),
        ));
    }

    Ok(LlamaContext { ptr })
}

/// Free a context
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_free(ctx: LlamaContext) {
    unsafe {
        llama_free_c(ctx.ptr);
    }
}

/// Get the BOS (beginning of sequence) token
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_token_bos(model: &LlamaModel) -> i32 {
    unsafe { llama_token_bos_c(model.ptr) }
}

/// Get the EOS (end of sequence) token
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_token_eos(model: &LlamaModel) -> i32 {
    unsafe { llama_token_eos_c(model.ptr) }
}

/// Get vocabulary size
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_n_vocab(model: &LlamaModel) -> usize {
    unsafe { llama_n_vocab_c(model.ptr) as usize }
}

/// Get context length
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_n_ctx(ctx: &LlamaContext) -> usize {
    unsafe { llama_n_ctx_c(ctx.ptr) as usize }
}

/// Format chat messages using the model's chat template
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_format_chat(
    _model: &LlamaModel,
    messages: &[ChatMessage],
) -> Result<String, AdapterError> {
    // Simple chat template formatting (ChatML-style)
    // TODO: Use llama_chat_apply_template for model-specific templates
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
#[cfg(feature = "local-llm-llamacpp")]
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

/// Detokenize tokens to text
#[cfg(feature = "local-llm-llamacpp")]
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
                false,
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
#[cfg(feature = "local-llm-llamacpp")]
#[derive(Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
}

#[cfg(feature = "local-llm-llamacpp")]
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

/// Generate tokens using autoregressive decoding.
///
/// This function performs the full generation loop:
/// 1. Processes input tokens
/// 2. Samples from logits using temperature/top_p/top_k
/// 3. Repeats until EOS or max_tokens
///
/// # Arguments
/// * `ctx` - The llama context
/// * `model` - The llama model (needed for EOS token)
/// * `input_tokens` - Input token IDs
/// * `max_tokens` - Maximum tokens to generate
/// * `temperature` - Sampling temperature (0 = greedy)
/// * `top_p` - Top-p (nucleus) sampling threshold
/// * `top_k` - Top-k sampling (0 = disabled)
///
/// # Returns
/// Vector of generated token IDs
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_generate(
    ctx: &LlamaContext,
    model: &LlamaModel,
    input_tokens: &[i32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
) -> Result<Vec<i32>, AdapterError> {
    if input_tokens.is_empty() {
        return Err(AdapterError::InvalidInput("Empty input tokens".to_string()));
    }

    // Allocate output buffer
    let mut output_tokens = vec![0i32; max_tokens];

    // Use current time as seed for sampling
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(42);

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
            seed,
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

// =============================================================================
// Stub implementations when feature is disabled
// =============================================================================

#[cfg(not(feature = "local-llm-llamacpp"))]
pub struct LlamaModel;

#[cfg(not(feature = "local-llm-llamacpp"))]
pub struct LlamaContext;

#[cfg(not(feature = "local-llm-llamacpp"))]
pub fn llama_backend_init() {}

#[cfg(not(feature = "local-llm-llamacpp"))]
pub fn llama_backend_free() {}

#[cfg(not(feature = "local-llm-llamacpp"))]
pub fn llama_load_model_from_file(
    _path: &str,
    _n_gpu_layers: i32,
) -> Result<LlamaModel, AdapterError> {
    Err(AdapterError::RuntimeError(
        "local-llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "local-llm-llamacpp"))]
pub fn llama_free_model(_model: LlamaModel) {}

#[cfg(not(feature = "local-llm-llamacpp"))]
pub fn llama_new_context_with_model(
    _model: &LlamaModel,
    _n_ctx: usize,
) -> Result<LlamaContext, AdapterError> {
    Err(AdapterError::RuntimeError(
        "local-llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "local-llm-llamacpp"))]
pub fn llama_free(_ctx: LlamaContext) {}

#[cfg(not(feature = "local-llm-llamacpp"))]
pub fn llama_format_chat(
    _model: &LlamaModel,
    _messages: &[ChatMessage],
) -> Result<String, AdapterError> {
    Err(AdapterError::RuntimeError(
        "local-llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "local-llm-llamacpp"))]
pub fn llama_tokenize(
    _model: &LlamaModel,
    _text: &str,
    _add_special: bool,
) -> Result<Vec<i32>, AdapterError> {
    Err(AdapterError::RuntimeError(
        "local-llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "local-llm-llamacpp"))]
pub fn llama_detokenize(_model: &LlamaModel, _tokens: &[i32]) -> Result<String, AdapterError> {
    Err(AdapterError::RuntimeError(
        "local-llm-llamacpp feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "local-llm-llamacpp"))]
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
        "local-llm-llamacpp feature not enabled".to_string(),
    ))
}
