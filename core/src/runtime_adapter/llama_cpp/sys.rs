//! FFI bindings for llama.cpp
//!
//! This module provides safe Rust wrappers around the llama.cpp C API.
//!
//! # Build Requirements
//!
//! llama.cpp is built from source via the `cc` or `cmake` crate in build.rs.
//! The build system handles:
//! - Android NDK cross-compilation
//! - Metal support on Apple platforms
//! - CUDA support on Linux/Windows (optional)
//!
//! # Safety
//!
//! All FFI functions are wrapped in safe Rust APIs that handle:
//! - Null pointer checks
//! - Lifetime management via Drop
//! - Error conversion to AdapterError

use crate::runtime_adapter::llm::ChatMessage;
use crate::runtime_adapter::AdapterError;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};

// =============================================================================
// Opaque Types
// =============================================================================

/// Opaque handle to a loaded llama model
#[cfg(feature = "local-llm-llamacpp")]
pub struct LlamaModel {
    ptr: *mut llama_model,
}

#[cfg(feature = "local-llm-llamacpp")]
unsafe impl Send for LlamaModel {}
#[cfg(feature = "local-llm-llamacpp")]
unsafe impl Sync for LlamaModel {}

/// Opaque handle to a llama context
#[cfg(feature = "local-llm-llamacpp")]
pub struct LlamaContext {
    ptr: *mut llama_context,
}

#[cfg(feature = "local-llm-llamacpp")]
unsafe impl Send for LlamaContext {}
#[cfg(feature = "local-llm-llamacpp")]
unsafe impl Sync for LlamaContext {}

// =============================================================================
// FFI Declarations
// =============================================================================

#[cfg(feature = "local-llm-llamacpp")]
#[repr(C)]
struct llama_model {
    _private: [u8; 0],
}

#[cfg(feature = "local-llm-llamacpp")]
#[repr(C)]
struct llama_context {
    _private: [u8; 0],
}

#[cfg(feature = "local-llm-llamacpp")]
#[repr(C)]
struct llama_model_params {
    n_gpu_layers: c_int,
    // ... other fields omitted for brevity, will be added
}

#[cfg(feature = "local-llm-llamacpp")]
#[repr(C)]
struct llama_context_params {
    n_ctx: c_int,
    // ... other fields omitted for brevity, will be added
}

#[cfg(feature = "local-llm-llamacpp")]
extern "C" {
    // Backend initialization
    fn llama_backend_init_c();
    fn llama_backend_free_c();

    // Model loading
    fn llama_load_model_from_file_c(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    fn llama_free_model_c(model: *mut llama_model);

    // Context management
    fn llama_new_context_with_model_c(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    fn llama_free_c(ctx: *mut llama_context);

    // Tokenization
    fn llama_tokenize_c(
        model: *const llama_model,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut i32,
        n_tokens_max: c_int,
        add_special: bool,
        parse_special: bool,
    ) -> c_int;

    fn llama_token_to_piece_c(
        model: *const llama_model,
        token: i32,
        buf: *mut c_char,
        length: c_int,
        lstrip: c_int,
        special: bool,
    ) -> c_int;

    // Generation
    fn llama_decode_c(ctx: *mut llama_context, batch: *const c_void) -> c_int;
    fn llama_get_logits_c(ctx: *mut llama_context) -> *mut c_float;

    // Chat template
    fn llama_chat_apply_template_c(
        model: *const llama_model,
        tmpl: *const c_char,
        chat: *const c_void,
        n_msg: usize,
        add_ass: bool,
        buf: *mut c_char,
        length: c_int,
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

    let params = llama_model_params {
        n_gpu_layers: n_gpu_layers as c_int,
    };

    let ptr = unsafe { llama_load_model_from_file_c(c_path.as_ptr(), params) };

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
    let params = llama_context_params {
        n_ctx: n_ctx as c_int,
    };

    let ptr = unsafe { llama_new_context_with_model_c(model.ptr, params) };

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

/// Format chat messages using the model's chat template
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_format_chat(
    _model: &LlamaModel,
    messages: &[ChatMessage],
) -> Result<String, AdapterError> {
    // TODO: Implement proper chat template formatting via llama_chat_apply_template
    // For now, simple concatenation
    let mut prompt = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<|system|>\n{}<|end|>\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("<|user|>\n{}<|end|>\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<|assistant|>\n{}<|end|>\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("<|user|>\n{}<|end|>\n", msg.content));
            }
        }
    }

    // Add assistant prefix for generation
    prompt.push_str("<|assistant|>\n");

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

    // First call to get required size
    let n_tokens = unsafe {
        llama_tokenize_c(
            model.ptr,
            c_text.as_ptr(),
            text.len() as c_int,
            std::ptr::null_mut(),
            0,
            add_special,
            false,
        )
    };

    if n_tokens < 0 {
        return Err(AdapterError::RuntimeError("Tokenization failed".to_string()));
    }

    // Allocate and tokenize
    let mut tokens = vec![0i32; n_tokens as usize];
    let result = unsafe {
        llama_tokenize_c(
            model.ptr,
            c_text.as_ptr(),
            text.len() as c_int,
            tokens.as_mut_ptr(),
            n_tokens,
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
    let mut buf = [0u8; 256];

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

        if len > 0 {
            let piece = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
            if let Ok(s) = piece.to_str() {
                result.push_str(s);
            }
        }
    }

    Ok(result)
}

/// Generate tokens
#[cfg(feature = "local-llm-llamacpp")]
pub fn llama_generate(
    _ctx: &LlamaContext,
    _input_tokens: &[i32],
    _max_tokens: usize,
    _temperature: f32,
    _top_p: f32,
    _top_k: usize,
) -> Result<Vec<i32>, AdapterError> {
    // TODO: Implement actual generation loop
    // This is a placeholder - real implementation needs:
    // 1. Create batch with input tokens
    // 2. llama_decode to process input
    // 3. Sample from logits
    // 4. Add new token to batch
    // 5. Repeat until EOS or max_tokens

    Err(AdapterError::RuntimeError(
        "llama_generate not yet implemented - FFI bindings incomplete".to_string(),
    ))
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
