//! # xybrid-ffi
//!
//! C ABI FFI bindings for xybrid-sdk.
//!
//! This crate provides a C-compatible interface to the xybrid SDK,
//! enabling integration with Unity, C/C++, and other languages that
//! can consume C libraries.
//!
//! ## Usage
//!
//! Build the library:
//! ```sh
//! cargo build -p xybrid-ffi --release
//! ```
//!
//! The output will be:
//! - macOS: `libxybrid_ffi.dylib` (dynamic) and `libxybrid_ffi.a` (static)
//! - Linux: `libxybrid_ffi.so` (dynamic) and `libxybrid_ffi.a` (static)
//! - Windows: `xybrid_ffi.dll` (dynamic) and `xybrid_ffi.lib` (static)
//!
//! Include the generated C header (`include/xybrid.h`) in your C/C++ project.

#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;

// Import SDK types
use xybrid_sdk::ir::{Envelope, EnvelopeKind, MessageRole};
use xybrid_sdk::{InferenceResult, ConversationContext, ModelLoader, PartialToken, XybridModel};

// ============================================================================
// Opaque Handle Types (US-009)
// ============================================================================
//
// These opaque handles allow C consumers to hold references to Rust objects
// without knowing their internal structure. Each handle wraps a raw pointer
// to a boxed Rust type.
//
// Safety: Handles must be:
// - Created via the corresponding xybrid_*_create functions
// - Freed via the corresponding xybrid_*_free functions
// - Not used after being freed
// - Not shared across threads without synchronization

/// Opaque handle to a model loader.
///
/// This handle is created by `xybrid_model_loader_from_registry` or
/// `xybrid_model_loader_from_bundle` and must be freed with
/// `xybrid_model_loader_free`.
#[repr(C)]
pub struct XybridModelLoaderHandle(*mut c_void);

/// Opaque handle to a loaded model.
///
/// This handle is created by `xybrid_model_loader_load` and must be
/// freed with `xybrid_model_free`.
#[repr(C)]
pub struct XybridModelHandle(*mut c_void);

/// Opaque handle to an envelope (input data).
///
/// This handle is created by `xybrid_envelope_audio` or `xybrid_envelope_text`
/// and must be freed with `xybrid_envelope_free`.
#[repr(C)]
pub struct XybridEnvelopeHandle(*mut c_void);

/// Opaque handle to an inference result.
///
/// This handle is created by `xybrid_model_run` and must be freed with
/// `xybrid_result_free`.
#[repr(C)]
pub struct XybridResultHandle(*mut c_void);

/// Opaque handle to a conversation context.
///
/// This handle is created by `xybrid_context_new` and must be freed with
/// `xybrid_context_free`.
#[repr(C)]
pub struct XybridContextHandle(*mut c_void);

// ============================================================================
// Internal Boxed Types
// ============================================================================
//
// These type aliases define the actual Rust types that the opaque handles
// point to. They are boxed (heap-allocated) so we can convert them to/from
// raw pointers for FFI.

/// Internal state for a model loader.
pub(crate) struct LoaderState {
    /// The SDK ModelLoader instance.
    pub loader: ModelLoader,
    /// The model ID for reference.
    pub model_id: String,
}

/// Internal state for a loaded model.
pub(crate) struct ModelState {
    /// The SDK XybridModel instance (Arc for thread-safety).
    pub model: Arc<XybridModel>,
    /// The model ID for reference.
    pub model_id: String,
}

/// Internal envelope data.
pub(crate) enum EnvelopeData {
    /// Audio data with sample rate and channel count.
    Audio {
        bytes: Vec<u8>,
        sample_rate: u32,
        channels: u32,
    },
    /// Text data with optional voice, speed, and message role.
    Text {
        text: String,
        voice_id: Option<String>,
        speed: Option<f64>,
        /// Message role for conversation context (None for non-context usage).
        role: Option<MessageRole>,
    },
}

/// Internal inference result.
pub(crate) struct ResultData {
    /// Whether inference succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
    /// Type of output produced.
    pub output_type: String,
    /// Text output (for ASR/LLM).
    pub text: Option<String>,
    /// Embedding output.
    pub embedding: Option<Vec<f32>>,
    /// Audio bytes (for TTS).
    pub audio_bytes: Option<Vec<u8>>,
    /// Inference latency in milliseconds.
    pub latency_ms: u32,
}

/// Internal conversation context.
pub(crate) struct ContextData {
    /// The conversation context instance.
    pub context: ConversationContext,
}

/// Type alias for a boxed loader.
pub(crate) type BoxedLoader = Box<LoaderState>;

/// Type alias for a boxed model.
pub(crate) type BoxedModel = Box<ModelState>;

/// Type alias for a boxed envelope.
pub(crate) type BoxedEnvelope = Box<EnvelopeData>;

/// Type alias for a boxed result.
pub(crate) type BoxedResult = Box<ResultData>;

/// Type alias for a boxed context.
pub(crate) type BoxedContext = Box<ContextData>;

// ============================================================================
// Callback Types
// ============================================================================

/// Callback function type for streaming inference.
///
/// This callback is invoked for each token generated during streaming inference.
/// All string parameters are null-terminated UTF-8 and valid only for the duration
/// of the callback invocation. The caller must copy any data they want to retain.
///
/// # Parameters
///
/// - `token`: The generated token text
/// - `token_id`: The raw token ID (-1 if not available)
/// - `index`: Zero-based index of this token in the generation sequence
/// - `cumulative_text`: All generated text so far (concatenation of all tokens)
/// - `finish_reason`: Reason for stopping, or null if generation is still in progress
/// - `user_data`: The opaque pointer passed to `xybrid_model_run_streaming`
pub type XybridStreamCallback = Option<
    unsafe extern "C" fn(
        token: *const c_char,
        token_id: i64,
        index: u32,
        cumulative_text: *const c_char,
        finish_reason: *const c_char,
        user_data: *mut c_void,
    ),
>;

// ============================================================================
// Internal Helpers
// ============================================================================

/// Send-safe wrapper for streaming callback context.
///
/// # Safety
/// The caller must ensure that `user_data` is valid for the duration of
/// the streaming call and that no data races occur. Function pointers are
/// inherently thread-safe (just addresses).
struct StreamCallbackCtx {
    callback: unsafe extern "C" fn(*const c_char, i64, u32, *const c_char, *const c_char, *mut c_void),
    user_data: *mut c_void,
}
unsafe impl Send for StreamCallbackCtx {}
unsafe impl Sync for StreamCallbackCtx {}

impl StreamCallbackCtx {
    unsafe fn invoke(&self, token: &PartialToken) {
        let c_token = CString::new(token.token.as_str()).unwrap_or_default();
        let c_cumulative = CString::new(token.cumulative_text.as_str()).unwrap_or_default();
        let c_finish = token.finish_reason.as_ref().map(|r| CString::new(r.as_str()).unwrap_or_default());

        (self.callback)(
            c_token.as_ptr(),
            token.token_id.unwrap_or(-1),
            token.index as u32,
            c_cumulative.as_ptr(),
            c_finish.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            self.user_data,
        );
    }
}

/// Convert EnvelopeData to SDK Envelope.
fn envelope_data_to_sdk(data: &EnvelopeData) -> Envelope {
    match data {
        EnvelopeData::Audio {
            bytes,
            sample_rate,
            channels,
        } => {
            let mut metadata = HashMap::new();
            metadata.insert("sample_rate".to_string(), sample_rate.to_string());
            metadata.insert("channels".to_string(), channels.to_string());
            Envelope {
                kind: EnvelopeKind::Audio(bytes.clone()),
                metadata,
            }
        }
        EnvelopeData::Text {
            text,
            voice_id,
            speed,
            role,
        } => {
            let mut metadata = HashMap::new();
            if let Some(v) = voice_id {
                metadata.insert("voice_id".to_string(), v.clone());
            }
            if let Some(s) = speed {
                metadata.insert("speed".to_string(), s.to_string());
            }
            let mut envelope = Envelope {
                kind: EnvelopeKind::Text(text.clone()),
                metadata,
            };
            if let Some(r) = role {
                envelope = envelope.with_role(*r);
            }
            envelope
        }
    }
}

/// Convert SDK InferenceResult to FFI ResultData.
fn inference_result_to_data(result: &xybrid_sdk::InferenceResult) -> ResultData {
    ResultData {
        success: true,
        error: None,
        output_type: match result.text() {
            Some(_) => "text".to_string(),
            None => match result.audio_bytes() {
                Some(_) => "audio".to_string(),
                None => match result.embedding() {
                    Some(_) => "embedding".to_string(),
                    None => "unknown".to_string(),
                },
            },
        },
        text: result.text().map(|s| s.to_string()),
        embedding: result.embedding().map(|e| e.to_vec()),
        audio_bytes: result.audio_bytes().map(|b| b.to_vec()),
        latency_ms: result.latency_ms(),
    }
}

// ============================================================================
// Handle Conversion Utilities
// ============================================================================
//
// These functions convert between opaque handles and boxed types.
// They are used internally by the C ABI functions.

impl XybridModelLoaderHandle {
    /// Create a handle from a boxed loader (takes ownership).
    pub(crate) fn from_boxed(loader: BoxedLoader) -> *mut Self {
        let ptr = Box::into_raw(loader) as *mut c_void;
        Box::into_raw(Box::new(XybridModelLoaderHandle(ptr)))
    }

    /// Convert handle back to boxed loader (takes ownership of handle).
    ///
    /// # Safety
    /// The handle must be valid and not already freed.
    pub(crate) unsafe fn into_boxed(handle: *mut Self) -> Option<BoxedLoader> {
        if handle.is_null() {
            return None;
        }
        let wrapper = Box::from_raw(handle);
        if wrapper.0.is_null() {
            return None;
        }
        Some(Box::from_raw(wrapper.0 as *mut LoaderState))
    }

    /// Borrow the loader state from a handle.
    ///
    /// # Safety
    /// The handle must be valid and not already freed.
    pub(crate) unsafe fn as_ref<'a>(handle: *mut Self) -> Option<&'a LoaderState> {
        if handle.is_null() {
            return None;
        }
        let wrapper = &*handle;
        if wrapper.0.is_null() {
            return None;
        }
        Some(&*(wrapper.0 as *const LoaderState))
    }
}

impl XybridModelHandle {
    /// Create a handle from a boxed model (takes ownership).
    pub(crate) fn from_boxed(model: BoxedModel) -> *mut Self {
        let ptr = Box::into_raw(model) as *mut c_void;
        Box::into_raw(Box::new(XybridModelHandle(ptr)))
    }

    /// Convert handle back to boxed model (takes ownership of handle).
    ///
    /// # Safety
    /// The handle must be valid and not already freed.
    pub(crate) unsafe fn into_boxed(handle: *mut Self) -> Option<BoxedModel> {
        if handle.is_null() {
            return None;
        }
        let wrapper = Box::from_raw(handle);
        if wrapper.0.is_null() {
            return None;
        }
        Some(Box::from_raw(wrapper.0 as *mut ModelState))
    }

    /// Borrow the model state from a handle.
    ///
    /// # Safety
    /// The handle must be valid and not already freed.
    pub(crate) unsafe fn as_ref<'a>(handle: *mut Self) -> Option<&'a ModelState> {
        if handle.is_null() {
            return None;
        }
        let wrapper = &*handle;
        if wrapper.0.is_null() {
            return None;
        }
        Some(&*(wrapper.0 as *const ModelState))
    }
}

impl XybridEnvelopeHandle {
    /// Create a handle from a boxed envelope (takes ownership).
    pub(crate) fn from_boxed(envelope: BoxedEnvelope) -> *mut Self {
        let ptr = Box::into_raw(envelope) as *mut c_void;
        Box::into_raw(Box::new(XybridEnvelopeHandle(ptr)))
    }

    /// Convert handle back to boxed envelope (takes ownership of handle).
    ///
    /// # Safety
    /// The handle must be valid and not already freed.
    pub(crate) unsafe fn into_boxed(handle: *mut Self) -> Option<BoxedEnvelope> {
        if handle.is_null() {
            return None;
        }
        let wrapper = Box::from_raw(handle);
        if wrapper.0.is_null() {
            return None;
        }
        Some(Box::from_raw(wrapper.0 as *mut EnvelopeData))
    }

    /// Borrow the envelope data from a handle.
    ///
    /// # Safety
    /// The handle must be valid and not already freed.
    pub(crate) unsafe fn as_ref<'a>(handle: *mut Self) -> Option<&'a EnvelopeData> {
        if handle.is_null() {
            return None;
        }
        let wrapper = &*handle;
        if wrapper.0.is_null() {
            return None;
        }
        Some(&*(wrapper.0 as *const EnvelopeData))
    }
}

impl XybridResultHandle {
    /// Create a handle from a boxed result (takes ownership).
    pub(crate) fn from_boxed(result: BoxedResult) -> *mut Self {
        let ptr = Box::into_raw(result) as *mut c_void;
        Box::into_raw(Box::new(XybridResultHandle(ptr)))
    }

    /// Convert handle back to boxed result (takes ownership of handle).
    ///
    /// # Safety
    /// The handle must be valid and not already freed.
    pub(crate) unsafe fn into_boxed(handle: *mut Self) -> Option<BoxedResult> {
        if handle.is_null() {
            return None;
        }
        let wrapper = Box::from_raw(handle);
        if wrapper.0.is_null() {
            return None;
        }
        Some(Box::from_raw(wrapper.0 as *mut ResultData))
    }

    /// Borrow the result data from a handle.
    ///
    /// # Safety
    /// The handle must be valid and not already freed.
    pub(crate) unsafe fn as_ref<'a>(handle: *mut Self) -> Option<&'a ResultData> {
        if handle.is_null() {
            return None;
        }
        let wrapper = &*handle;
        if wrapper.0.is_null() {
            return None;
        }
        Some(&*(wrapper.0 as *const ResultData))
    }
}

impl XybridContextHandle {
    /// Create a handle from a boxed context (takes ownership).
    pub(crate) fn from_boxed(context: BoxedContext) -> *mut Self {
        let ptr = Box::into_raw(context) as *mut c_void;
        Box::into_raw(Box::new(XybridContextHandle(ptr)))
    }

    /// Convert handle back to boxed context (takes ownership of handle).
    ///
    /// # Safety
    /// The handle must be valid and not already freed.
    pub(crate) unsafe fn into_boxed(handle: *mut Self) -> Option<BoxedContext> {
        if handle.is_null() {
            return None;
        }
        let wrapper = Box::from_raw(handle);
        if wrapper.0.is_null() {
            return None;
        }
        Some(Box::from_raw(wrapper.0 as *mut ContextData))
    }

    /// Borrow the context data from a handle.
    ///
    /// # Safety
    /// The handle must be valid and not already freed.
    pub(crate) unsafe fn as_ref<'a>(handle: *mut Self) -> Option<&'a ContextData> {
        if handle.is_null() {
            return None;
        }
        let wrapper = &*handle;
        if wrapper.0.is_null() {
            return None;
        }
        Some(&*(wrapper.0 as *const ContextData))
    }

    /// Mutably borrow the context data from a handle.
    ///
    /// # Safety
    /// The handle must be valid and not already freed.
    pub(crate) unsafe fn as_mut<'a>(handle: *mut Self) -> Option<&'a mut ContextData> {
        if handle.is_null() {
            return None;
        }
        let wrapper = &*handle;
        if wrapper.0.is_null() {
            return None;
        }
        Some(&mut *(wrapper.0 as *mut ContextData))
    }
}

/// Library version string.
///
/// Returns the version of the xybrid-ffi library.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ============================================================================
// Thread-Local Error Storage (US-010)
// ============================================================================
//
// Thread-local storage for the last error message. This allows C consumers
// to retrieve error details after a function returns an error status.

thread_local! {
    /// Thread-local storage for the last error message.
    ///
    /// This is set by C ABI functions when an error occurs and can be
    /// retrieved by calling `xybrid_last_error()`.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Set the last error message.
///
/// This is called internally by C ABI functions when an error occurs.
fn set_last_error(message: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(message).ok();
    });
}

/// Clear the last error message.
#[allow(dead_code)]
fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

// ============================================================================
// C ABI Utility Functions (US-010)
// ============================================================================
//
// These functions provide basic library initialization and utilities
// for C consumers.

/// Initialize the xybrid library.
///
/// This function should be called once before using any other xybrid functions.
/// Currently this is a no-op but may perform initialization in the future.
///
/// # Returns
///
/// - `0` on success
/// - Non-zero on failure (check `xybrid_last_error()` for details)
///
/// # Example (C)
///
/// ```c
/// if (xybrid_init() != 0) {
///     const char* error = xybrid_last_error();
///     fprintf(stderr, "Failed to initialize: %s\n", error);
///     return 1;
/// }
/// ```
#[no_mangle]
pub extern "C" fn xybrid_init() -> i32 {
    // Clear any previous error
    clear_last_error();

    // Future: Initialize logging, runtime, etc.
    // For now, just return success.
    0
}

/// Get the library version string.
///
/// Returns a pointer to a null-terminated string containing the library version.
/// The returned pointer is valid for the lifetime of the library and should NOT
/// be freed by the caller.
///
/// # Returns
///
/// A pointer to a static null-terminated version string, or null on error.
///
/// # Example (C)
///
/// ```c
/// const char* version = xybrid_version();
/// printf("xybrid version: %s\n", version);
/// ```
#[no_mangle]
pub extern "C" fn xybrid_version() -> *const c_char {
    // Use a static CString to ensure the pointer remains valid.
    // This is safe because VERSION is a compile-time constant.
    static VERSION_CSTRING: std::sync::OnceLock<CString> = std::sync::OnceLock::new();

    VERSION_CSTRING
        .get_or_init(|| CString::new(VERSION).expect("VERSION contains no null bytes"))
        .as_ptr()
}

/// Get the last error message.
///
/// Returns a pointer to a null-terminated string containing the last error
/// message, or null if no error has occurred. The returned pointer is valid
/// until the next xybrid function call on the same thread.
///
/// # Returns
///
/// A pointer to the last error message, or null if no error.
///
/// # Example (C)
///
/// ```c
/// XybridModelHandle* model = xybrid_model_loader_load(loader);
/// if (model == NULL) {
///     const char* error = xybrid_last_error();
///     fprintf(stderr, "Failed to load: %s\n", error ? error : "unknown error");
/// }
/// ```
#[no_mangle]
pub extern "C" fn xybrid_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        match e.borrow().as_ref() {
            Some(cstr) => cstr.as_ptr(),
            None => std::ptr::null(),
        }
    })
}

/// Free a string allocated by the library.
///
/// This function should be called to free strings returned by xybrid functions
/// that specify the caller must free the result. Do NOT use this to free
/// strings returned by `xybrid_version()` or `xybrid_last_error()`.
///
/// # Safety
///
/// The pointer must be a valid pointer to a string allocated by xybrid,
/// or null. Passing an invalid pointer causes undefined behavior.
///
/// # Example (C)
///
/// ```c
/// char* model_id = xybrid_model_id(model);
/// printf("Model: %s\n", model_id);
/// xybrid_free_string(model_id);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_free_string(s: *mut c_char) {
    if !s.is_null() {
        // Reconstruct the CString and let it drop to free the memory
        let _ = CString::from_raw(s);
    }
}

// ============================================================================
// C ABI Model Loader Functions (US-011)
// ============================================================================
//
// These functions allow C consumers to create model loaders and load models.
// Loaders can be created from a registry model ID or a local bundle path.

/// Create a model loader from a registry model ID.
///
/// This creates a loader that will fetch the model from the xybrid registry
/// when `xybrid_model_loader_load` is called.
///
/// # Parameters
///
/// - `model_id`: A null-terminated string containing the model ID (e.g., "kokoro-82m").
///
/// # Returns
///
/// A handle to the model loader, or null on failure.
/// On failure, call `xybrid_last_error()` to get the error message.
///
/// # Example (C)
///
/// ```c
/// XybridModelLoaderHandle* loader = xybrid_model_loader_from_registry("kokoro-82m");
/// if (loader == NULL) {
///     fprintf(stderr, "Failed: %s\n", xybrid_last_error());
///     return 1;
/// }
/// // Use loader...
/// xybrid_model_loader_free(loader);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_model_loader_from_registry(
    model_id: *const c_char,
) -> *mut XybridModelLoaderHandle {
    clear_last_error();

    // Validate input
    if model_id.is_null() {
        set_last_error("model_id is null");
        return std::ptr::null_mut();
    }

    // Convert C string to Rust string
    let model_id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_last_error("model_id is not valid UTF-8");
            return std::ptr::null_mut();
        }
    };

    if model_id_str.is_empty() {
        set_last_error("model_id is empty");
        return std::ptr::null_mut();
    }

    // Create SDK ModelLoader
    let sdk_loader = ModelLoader::from_registry(&model_id_str);

    // Create loader state
    let loader = Box::new(LoaderState {
        loader: sdk_loader,
        model_id: model_id_str,
    });

    XybridModelLoaderHandle::from_boxed(loader)
}

/// Create a model loader from a local bundle path.
///
/// This creates a loader that will load the model from the specified local path
/// when `xybrid_model_loader_load` is called.
///
/// # Parameters
///
/// - `path`: A null-terminated string containing the path to the model bundle.
///
/// # Returns
///
/// A handle to the model loader, or null on failure.
/// On failure, call `xybrid_last_error()` to get the error message.
///
/// # Example (C)
///
/// ```c
/// XybridModelLoaderHandle* loader = xybrid_model_loader_from_bundle("/path/to/model");
/// if (loader == NULL) {
///     fprintf(stderr, "Failed: %s\n", xybrid_last_error());
///     return 1;
/// }
/// // Use loader...
/// xybrid_model_loader_free(loader);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_model_loader_from_bundle(
    path: *const c_char,
) -> *mut XybridModelLoaderHandle {
    clear_last_error();

    // Validate input
    if path.is_null() {
        set_last_error("path is null");
        return std::ptr::null_mut();
    }

    // Convert C string to Rust string
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_last_error("path is not valid UTF-8");
            return std::ptr::null_mut();
        }
    };

    if path_str.is_empty() {
        set_last_error("path is empty");
        return std::ptr::null_mut();
    }

    // Extract model ID from path (use the last path component)
    let model_id = std::path::Path::new(&path_str)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(&path_str)
        .to_string();

    // Create SDK ModelLoader from bundle
    let sdk_loader = match ModelLoader::from_bundle(&path_str) {
        Ok(loader) => loader,
        Err(e) => {
            set_last_error(&format!("Failed to create loader from bundle: {}", e));
            return std::ptr::null_mut();
        }
    };

    // Create loader state
    let loader = Box::new(LoaderState {
        loader: sdk_loader,
        model_id,
    });

    XybridModelLoaderHandle::from_boxed(loader)
}

/// Load a model using the loader.
///
/// This function loads the model from the registry or local bundle,
/// depending on how the loader was created.
///
/// # Parameters
///
/// - `handle`: A handle to the model loader created by `xybrid_model_loader_from_registry`
///   or `xybrid_model_loader_from_bundle`.
///
/// # Returns
///
/// A handle to the loaded model, or null on failure.
/// On failure, call `xybrid_last_error()` to get the error message.
///
/// # Example (C)
///
/// ```c
/// XybridModelHandle* model = xybrid_model_loader_load(loader);
/// if (model == NULL) {
///     fprintf(stderr, "Failed: %s\n", xybrid_last_error());
///     xybrid_model_loader_free(loader);
///     return 1;
/// }
/// // Use model...
/// xybrid_model_free(model);
/// xybrid_model_loader_free(loader);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_model_loader_load(
    handle: *mut XybridModelLoaderHandle,
) -> *mut XybridModelHandle {
    clear_last_error();

    // Validate handle
    if handle.is_null() {
        set_last_error("loader handle is null");
        return std::ptr::null_mut();
    }

    // Borrow the loader state
    let loader_state = match XybridModelLoaderHandle::as_ref(handle) {
        Some(state) => state,
        None => {
            set_last_error("invalid loader handle");
            return std::ptr::null_mut();
        }
    };

    // Load the model using the SDK
    let xybrid_model = match loader_state.loader.load() {
        Ok(model) => model,
        Err(e) => {
            set_last_error(&format!("Failed to load model: {}", e));
            return std::ptr::null_mut();
        }
    };

    let model_id = loader_state.model_id.clone();

    // Create model state
    let model = Box::new(ModelState {
        model: Arc::new(xybrid_model),
        model_id,
    });

    XybridModelHandle::from_boxed(model)
}

/// Free a model loader handle.
///
/// This function frees the memory associated with a model loader handle.
/// After calling this function, the handle is no longer valid.
///
/// # Parameters
///
/// - `handle`: A handle to the model loader to free. May be null (no-op).
///
/// # Example (C)
///
/// ```c
/// xybrid_model_loader_free(loader);
/// loader = NULL; // Good practice: null out after freeing
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_model_loader_free(handle: *mut XybridModelLoaderHandle) {
    if !handle.is_null() {
        // Take ownership and let it drop to free memory
        let _ = XybridModelLoaderHandle::into_boxed(handle);
    }
}

// ============================================================================
// C ABI Envelope Functions (US-012)
// ============================================================================
//
// These functions allow C consumers to create envelopes (input data) for
// inference. Envelopes can contain audio data or text data.

/// Create an envelope containing audio data.
///
/// This function creates an envelope containing raw audio bytes with the
/// specified sample rate and channel count.
///
/// # Parameters
///
/// - `bytes`: Pointer to the raw audio bytes. May be null if `len` is 0.
/// - `len`: Length of the audio bytes array.
/// - `sample_rate`: Sample rate in Hz (e.g., 16000 for 16kHz).
/// - `channels`: Number of audio channels (e.g., 1 for mono, 2 for stereo).
///
/// # Returns
///
/// A handle to the envelope, or null on failure.
/// On failure, call `xybrid_last_error()` to get the error message.
///
/// # Example (C)
///
/// ```c
/// uint8_t audio_data[] = { /* PCM audio bytes */ };
/// XybridEnvelopeHandle* envelope = xybrid_envelope_audio(
///     audio_data, sizeof(audio_data), 16000, 1);
/// if (envelope == NULL) {
///     fprintf(stderr, "Failed: %s\n", xybrid_last_error());
///     return 1;
/// }
/// // Use envelope...
/// xybrid_envelope_free(envelope);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_envelope_audio(
    bytes: *const u8,
    len: usize,
    sample_rate: u32,
    channels: u32,
) -> *mut XybridEnvelopeHandle {
    clear_last_error();

    // Handle the case where len is 0 (empty audio is valid)
    let audio_bytes = if len == 0 {
        Vec::new()
    } else if bytes.is_null() {
        set_last_error("bytes is null but len is non-zero");
        return std::ptr::null_mut();
    } else {
        // Copy the audio bytes into a Rust Vec
        std::slice::from_raw_parts(bytes, len).to_vec()
    };

    // Validate sample rate and channels
    if sample_rate == 0 {
        set_last_error("sample_rate must be non-zero");
        return std::ptr::null_mut();
    }

    if channels == 0 {
        set_last_error("channels must be non-zero");
        return std::ptr::null_mut();
    }

    // Create envelope
    let envelope = Box::new(EnvelopeData::Audio {
        bytes: audio_bytes,
        sample_rate,
        channels,
    });

    XybridEnvelopeHandle::from_boxed(envelope)
}

/// Create an envelope containing text data.
///
/// This function creates an envelope containing text for TTS or LLM inference.
///
/// # Parameters
///
/// - `text`: A null-terminated string containing the text. Must not be null.
///
/// # Returns
///
/// A handle to the envelope, or null on failure.
/// On failure, call `xybrid_last_error()` to get the error message.
///
/// # Example (C)
///
/// ```c
/// XybridEnvelopeHandle* envelope = xybrid_envelope_text("Hello, world!");
/// if (envelope == NULL) {
///     fprintf(stderr, "Failed: %s\n", xybrid_last_error());
///     return 1;
/// }
/// // Use envelope...
/// xybrid_envelope_free(envelope);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_envelope_text(text: *const c_char) -> *mut XybridEnvelopeHandle {
    clear_last_error();

    // Validate input
    if text.is_null() {
        set_last_error("text is null");
        return std::ptr::null_mut();
    }

    // Convert C string to Rust string
    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_last_error("text is not valid UTF-8");
            return std::ptr::null_mut();
        }
    };

    // Note: Empty text is allowed (for edge cases)

    // Create envelope with no voice_id, speed, or role
    let envelope = Box::new(EnvelopeData::Text {
        text: text_str,
        voice_id: None,
        speed: None,
        role: None,
    });

    XybridEnvelopeHandle::from_boxed(envelope)
}

/// Free an envelope handle.
///
/// This function frees the memory associated with an envelope handle.
/// After calling this function, the handle is no longer valid.
///
/// # Parameters
///
/// - `handle`: A handle to the envelope to free. May be null (no-op).
///
/// # Example (C)
///
/// ```c
/// xybrid_envelope_free(envelope);
/// envelope = NULL; // Good practice: null out after freeing
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_envelope_free(handle: *mut XybridEnvelopeHandle) {
    if !handle.is_null() {
        // Take ownership and let it drop to free memory
        let _ = XybridEnvelopeHandle::into_boxed(handle);
    }
}

// ============================================================================
// C ABI Conversation Context Functions
// ============================================================================
//
// These functions allow C consumers to manage conversation context for
// multi-turn LLM interactions.

/// Message role constants for conversation context.
///
/// Use these values with `xybrid_envelope_text_with_role`:
/// - `XYBRID_ROLE_SYSTEM` (0): System prompt
/// - `XYBRID_ROLE_USER` (1): User message
/// - `XYBRID_ROLE_ASSISTANT` (2): Assistant response
pub const XYBRID_ROLE_SYSTEM: i32 = 0;
pub const XYBRID_ROLE_USER: i32 = 1;
pub const XYBRID_ROLE_ASSISTANT: i32 = 2;

/// Create a new conversation context with a generated UUID.
///
/// # Returns
///
/// A handle to the conversation context, or null on failure.
///
/// # Example (C)
///
/// ```c
/// XybridContextHandle* ctx = xybrid_context_new();
/// if (ctx == NULL) {
///     fprintf(stderr, "Failed: %s\n", xybrid_last_error());
///     return 1;
/// }
/// // Use context...
/// xybrid_context_free(ctx);
/// ```
#[no_mangle]
pub extern "C" fn xybrid_context_new() -> *mut XybridContextHandle {
    clear_last_error();

    let context = Box::new(ContextData {
        context: ConversationContext::new(),
    });

    XybridContextHandle::from_boxed(context)
}

/// Create a new conversation context with a specific ID.
///
/// # Parameters
///
/// - `id`: A null-terminated string containing the context ID.
///
/// # Returns
///
/// A handle to the conversation context, or null on failure.
///
/// # Example (C)
///
/// ```c
/// XybridContextHandle* ctx = xybrid_context_with_id("session-123");
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_context_with_id(id: *const c_char) -> *mut XybridContextHandle {
    clear_last_error();

    if id.is_null() {
        set_last_error("id is null");
        return std::ptr::null_mut();
    }

    let id_str = match CStr::from_ptr(id).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_last_error("id is not valid UTF-8");
            return std::ptr::null_mut();
        }
    };

    let context = Box::new(ContextData {
        context: ConversationContext::with_id(id_str),
    });

    XybridContextHandle::from_boxed(context)
}

/// Set the system prompt for a conversation context.
///
/// The system prompt defines the assistant's behavior and persists
/// across `xybrid_context_clear()` calls.
///
/// # Parameters
///
/// - `handle`: A handle to the conversation context.
/// - `text`: A null-terminated string containing the system prompt.
///
/// # Returns
///
/// - `0` on success
/// - Non-zero on failure (check `xybrid_last_error()`)
///
/// # Example (C)
///
/// ```c
/// xybrid_context_set_system(ctx, "You are a helpful assistant.");
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_context_set_system(
    handle: *mut XybridContextHandle,
    text: *const c_char,
) -> i32 {
    clear_last_error();

    if handle.is_null() {
        set_last_error("context handle is null");
        return -1;
    }

    if text.is_null() {
        set_last_error("text is null");
        return -1;
    }

    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_last_error("text is not valid UTF-8");
            return -1;
        }
    };

    let ctx_data = match XybridContextHandle::as_mut(handle) {
        Some(data) => data,
        None => {
            set_last_error("invalid context handle");
            return -1;
        }
    };

    // Create system envelope
    let system_envelope = Envelope::new(EnvelopeKind::Text(text_str)).with_role(MessageRole::System);

    // Rebuild context with system (preserving ID and max_history_len)
    let id = ctx_data.context.id().to_string();
    let max_len = ctx_data.context.max_history_len();
    let history: Vec<_> = ctx_data.context.history().to_vec();

    let mut new_ctx = ConversationContext::with_id(id)
        .with_max_history_len(max_len)
        .with_system(system_envelope);

    for envelope in history {
        new_ctx.push(envelope);
    }

    ctx_data.context = new_ctx;
    0
}

/// Set the maximum history length for a conversation context.
///
/// When the history exceeds this limit, oldest messages are dropped (FIFO).
/// Default is 50 messages.
///
/// # Parameters
///
/// - `handle`: A handle to the conversation context.
/// - `max_len`: Maximum number of history entries.
///
/// # Returns
///
/// - `0` on success
/// - Non-zero on failure
#[no_mangle]
pub unsafe extern "C" fn xybrid_context_set_max_history_len(
    handle: *mut XybridContextHandle,
    max_len: u32,
) -> i32 {
    clear_last_error();

    if handle.is_null() {
        set_last_error("context handle is null");
        return -1;
    }

    let ctx_data = match XybridContextHandle::as_mut(handle) {
        Some(data) => data,
        None => {
            set_last_error("invalid context handle");
            return -1;
        }
    };

    // Rebuild context with new max_history_len
    let id = ctx_data.context.id().to_string();
    let system = ctx_data.context.system_envelope().cloned();
    let history: Vec<_> = ctx_data.context.history().to_vec();

    let mut new_ctx = ConversationContext::with_id(id).with_max_history_len(max_len as usize);

    if let Some(sys) = system {
        new_ctx = new_ctx.with_system(sys);
    }

    for envelope in history {
        new_ctx.push(envelope);
    }

    ctx_data.context = new_ctx;
    0
}

/// Push an envelope to the conversation history.
///
/// The envelope should have a role set (use `xybrid_envelope_text_with_role`).
///
/// # Parameters
///
/// - `handle`: A handle to the conversation context.
/// - `envelope`: A handle to the envelope to push.
///
/// # Returns
///
/// - `0` on success
/// - Non-zero on failure
///
/// # Example (C)
///
/// ```c
/// XybridEnvelopeHandle* msg = xybrid_envelope_text_with_role("Hello!", XYBRID_ROLE_USER);
/// xybrid_context_push(ctx, msg);
/// xybrid_envelope_free(msg);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_context_push(
    handle: *mut XybridContextHandle,
    envelope: *mut XybridEnvelopeHandle,
) -> i32 {
    clear_last_error();

    if handle.is_null() {
        set_last_error("context handle is null");
        return -1;
    }

    if envelope.is_null() {
        set_last_error("envelope handle is null");
        return -1;
    }

    let ctx_data = match XybridContextHandle::as_mut(handle) {
        Some(data) => data,
        None => {
            set_last_error("invalid context handle");
            return -1;
        }
    };

    let envelope_data = match XybridEnvelopeHandle::as_ref(envelope) {
        Some(data) => data,
        None => {
            set_last_error("invalid envelope handle");
            return -1;
        }
    };

    // Convert to SDK envelope and push
    let sdk_envelope = match envelope_data {
        EnvelopeData::Text {
            text,
            voice_id,
            speed,
            role,
        } => {
            let mut metadata = HashMap::new();
            if let Some(v) = voice_id {
                metadata.insert("voice_id".to_string(), v.clone());
            }
            if let Some(s) = speed {
                metadata.insert("speed".to_string(), s.to_string());
            }
            // Use the role from envelope, default to User
            let msg_role = role.unwrap_or(MessageRole::User);

            Envelope::with_metadata(EnvelopeKind::Text(text.clone()), metadata).with_role(msg_role)
        }
        EnvelopeData::Audio { .. } => {
            set_last_error("audio envelopes cannot be pushed to context");
            return -1;
        }
    };

    ctx_data.context.push(sdk_envelope);
    0
}

/// Clear the conversation history but preserve the system prompt and ID.
///
/// # Parameters
///
/// - `handle`: A handle to the conversation context.
///
/// # Returns
///
/// - `0` on success
/// - Non-zero on failure
#[no_mangle]
pub unsafe extern "C" fn xybrid_context_clear(handle: *mut XybridContextHandle) -> i32 {
    clear_last_error();

    if handle.is_null() {
        set_last_error("context handle is null");
        return -1;
    }

    let ctx_data = match XybridContextHandle::as_mut(handle) {
        Some(data) => data,
        None => {
            set_last_error("invalid context handle");
            return -1;
        }
    };

    ctx_data.context.clear();
    0
}

/// Get the conversation context ID.
///
/// Returns a pointer to a null-terminated string containing the context ID.
/// The returned pointer is valid until the context handle is freed.
/// Do NOT free the returned string.
///
/// # Parameters
///
/// - `handle`: A handle to the conversation context.
///
/// # Returns
///
/// A pointer to the context ID string, or null on failure.
#[no_mangle]
pub unsafe extern "C" fn xybrid_context_id(handle: *mut XybridContextHandle) -> *const c_char {
    if handle.is_null() {
        return std::ptr::null();
    }

    match XybridContextHandle::as_ref(handle) {
        Some(data) => {
            thread_local! {
                static CONTEXT_ID: RefCell<Option<CString>> = const { RefCell::new(None) };
            }
            CONTEXT_ID.with(|e| {
                *e.borrow_mut() = CString::new(data.context.id()).ok();
                match e.borrow().as_ref() {
                    Some(cstr) => cstr.as_ptr(),
                    None => std::ptr::null(),
                }
            })
        }
        None => std::ptr::null(),
    }
}

/// Get the current history length (excluding system prompt).
///
/// # Parameters
///
/// - `handle`: A handle to the conversation context.
///
/// # Returns
///
/// The number of messages in the history, or 0 if the handle is null/invalid.
#[no_mangle]
pub unsafe extern "C" fn xybrid_context_history_len(handle: *mut XybridContextHandle) -> u32 {
    if handle.is_null() {
        return 0;
    }

    match XybridContextHandle::as_ref(handle) {
        Some(data) => data.context.history().len() as u32,
        None => 0,
    }
}

/// Check if a system prompt is set.
///
/// # Parameters
///
/// - `handle`: A handle to the conversation context.
///
/// # Returns
///
/// - `1` if a system prompt is set
/// - `0` if not, or if the handle is null/invalid
#[no_mangle]
pub unsafe extern "C" fn xybrid_context_has_system(handle: *mut XybridContextHandle) -> i32 {
    if handle.is_null() {
        return 0;
    }

    match XybridContextHandle::as_ref(handle) {
        Some(data) => {
            if data.context.system_envelope().is_some() {
                1
            } else {
                0
            }
        }
        None => 0,
    }
}

/// Free a conversation context handle.
///
/// This function frees the memory associated with a context handle.
/// After calling this function, the handle is no longer valid.
///
/// # Parameters
///
/// - `handle`: A handle to the context to free. May be null (no-op).
#[no_mangle]
pub unsafe extern "C" fn xybrid_context_free(handle: *mut XybridContextHandle) {
    if !handle.is_null() {
        let _ = XybridContextHandle::into_boxed(handle);
    }
}

/// Create an envelope containing text data with a message role.
///
/// This is used for building conversation context with proper role tagging.
///
/// # Parameters
///
/// - `text`: A null-terminated string containing the text.
/// - `role`: The message role (0=System, 1=User, 2=Assistant).
///
/// # Returns
///
/// A handle to the envelope, or null on failure.
///
/// # Example (C)
///
/// ```c
/// XybridEnvelopeHandle* user_msg = xybrid_envelope_text_with_role("Hello!", XYBRID_ROLE_USER);
/// xybrid_context_push(ctx, user_msg);
/// xybrid_envelope_free(user_msg);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_envelope_text_with_role(
    text: *const c_char,
    role: i32,
) -> *mut XybridEnvelopeHandle {
    clear_last_error();

    if text.is_null() {
        set_last_error("text is null");
        return std::ptr::null_mut();
    }

    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_last_error("text is not valid UTF-8");
            return std::ptr::null_mut();
        }
    };

    let msg_role = match role {
        XYBRID_ROLE_SYSTEM => MessageRole::System,
        XYBRID_ROLE_USER => MessageRole::User,
        XYBRID_ROLE_ASSISTANT => MessageRole::Assistant,
        _ => {
            set_last_error("invalid role value (use 0=System, 1=User, 2=Assistant)");
            return std::ptr::null_mut();
        }
    };

    let envelope = Box::new(EnvelopeData::Text {
        text: text_str,
        voice_id: None,
        speed: None,
        role: Some(msg_role),
    });

    XybridEnvelopeHandle::from_boxed(envelope)
}

// ============================================================================
// C ABI Inference Functions (US-013)
// ============================================================================
//
// These functions allow C consumers to run inference on loaded models.

/// Run inference on a model with the given input envelope.
///
/// This function executes inference using the loaded model and returns
/// a result handle containing the output or error information.
///
/// # Parameters
///
/// - `model`: A handle to the loaded model (from `xybrid_model_loader_load`).
/// - `envelope`: A handle to the input envelope (from `xybrid_envelope_audio` or `xybrid_envelope_text`).
///
/// # Returns
///
/// A handle to the result, or null on failure.
/// On failure, call `xybrid_last_error()` to get the error message.
/// The envelope is NOT consumed - it can be reused for multiple inferences.
///
/// # Example (C)
///
/// ```c
/// XybridResultHandle* result = xybrid_model_run(model, envelope);
/// if (result == NULL) {
///     fprintf(stderr, "Inference failed: %s\n", xybrid_last_error());
///     return 1;
/// }
///
/// if (xybrid_result_success(result)) {
///     const char* text = xybrid_result_text(result);
///     printf("Result: %s\n", text);
/// } else {
///     const char* error = xybrid_result_error(result);
///     printf("Error: %s\n", error);
/// }
///
/// xybrid_result_free(result);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_model_run(
    model: *mut XybridModelHandle,
    envelope: *mut XybridEnvelopeHandle,
) -> *mut XybridResultHandle {
    clear_last_error();

    // Validate model handle
    if model.is_null() {
        set_last_error("model handle is null");
        return std::ptr::null_mut();
    }

    // Validate envelope handle
    if envelope.is_null() {
        set_last_error("envelope handle is null");
        return std::ptr::null_mut();
    }

    // Borrow the model state
    let model_state = match XybridModelHandle::as_ref(model) {
        Some(state) => state,
        None => {
            set_last_error("invalid model handle");
            return std::ptr::null_mut();
        }
    };

    // Borrow the envelope data
    let envelope_data = match XybridEnvelopeHandle::as_ref(envelope) {
        Some(data) => data,
        None => {
            set_last_error("invalid envelope handle");
            return std::ptr::null_mut();
        }
    };

    // Convert EnvelopeData to SDK Envelope
    let sdk_envelope = envelope_data_to_sdk(envelope_data);

    // Run inference using the SDK
    let inference_result = match model_state.model.run(&sdk_envelope) {
        Ok(result) => result,
        Err(e) => {
            // Return error result
            let result = ResultData {
                success: false,
                error: Some(format!("Inference failed: {}", e)),
                output_type: "".to_string(),
                text: None,
                embedding: None,
                audio_bytes: None,
                latency_ms: 0,
            };
            return XybridResultHandle::from_boxed(Box::new(result));
        }
    };

    // Convert InferenceResult to ResultData
    XybridResultHandle::from_boxed(Box::new(inference_result_to_data(&inference_result)))
}

/// Run inference on a model with conversation context.
///
/// This function executes inference using the loaded model with conversation
/// history. The context provides previous messages which are formatted into
/// the prompt using the model's chat template.
///
/// # Parameters
///
/// - `model`: A handle to the loaded model.
/// - `envelope`: A handle to the input envelope (current user message).
/// - `context`: A handle to the conversation context.
///
/// # Returns
///
/// A handle to the result, or null on failure.
/// The envelope and context are NOT consumed - they can be reused.
///
/// # Example (C)
///
/// ```c
/// XybridContextHandle* ctx = xybrid_context_new();
/// xybrid_context_set_system(ctx, "You are a helpful assistant.");
///
/// XybridEnvelopeHandle* user_msg = xybrid_envelope_text_with_role("Hello!", XYBRID_ROLE_USER);
/// xybrid_context_push(ctx, user_msg);
///
/// XybridResultHandle* result = xybrid_model_run_with_context(model, user_msg, ctx);
/// if (xybrid_result_success(result)) {
///     const char* response = xybrid_result_text(result);
///     printf("Assistant: %s\n", response);
///
///     // Add assistant response to context
///     XybridEnvelopeHandle* asst_msg = xybrid_envelope_text_with_role(response, XYBRID_ROLE_ASSISTANT);
///     xybrid_context_push(ctx, asst_msg);
///     xybrid_envelope_free(asst_msg);
/// }
///
/// xybrid_result_free(result);
/// xybrid_envelope_free(user_msg);
/// xybrid_context_free(ctx);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_model_run_with_context(
    model: *mut XybridModelHandle,
    envelope: *mut XybridEnvelopeHandle,
    context: *mut XybridContextHandle,
) -> *mut XybridResultHandle {
    clear_last_error();

    // Validate handles
    if model.is_null() {
        set_last_error("model handle is null");
        return std::ptr::null_mut();
    }

    if envelope.is_null() {
        set_last_error("envelope handle is null");
        return std::ptr::null_mut();
    }

    if context.is_null() {
        set_last_error("context handle is null");
        return std::ptr::null_mut();
    }

    // Borrow the model state
    let model_state = match XybridModelHandle::as_ref(model) {
        Some(state) => state,
        None => {
            set_last_error("invalid model handle");
            return std::ptr::null_mut();
        }
    };

    // Borrow the envelope data
    let envelope_data = match XybridEnvelopeHandle::as_ref(envelope) {
        Some(data) => data,
        None => {
            set_last_error("invalid envelope handle");
            return std::ptr::null_mut();
        }
    };

    // Borrow the context data
    let ctx_data = match XybridContextHandle::as_ref(context) {
        Some(data) => data,
        None => {
            set_last_error("invalid context handle");
            return std::ptr::null_mut();
        }
    };

    // Convert EnvelopeData to SDK Envelope
    let sdk_envelope = envelope_data_to_sdk(envelope_data);

    // Run inference with context using the SDK
    let inference_result = match model_state.model.run_with_context(&sdk_envelope, &ctx_data.context) {
        Ok(result) => result,
        Err(e) => {
            // Return error result
            let result = ResultData {
                success: false,
                error: Some(format!("Inference failed: {}", e)),
                output_type: "".to_string(),
                text: None,
                embedding: None,
                audio_bytes: None,
                latency_ms: 0,
            };
            return XybridResultHandle::from_boxed(Box::new(result));
        }
    };

    // Convert InferenceResult to ResultData
    XybridResultHandle::from_boxed(Box::new(inference_result_to_data(&inference_result)))
}

/// Get the model ID of a loaded model.
///
/// Returns a pointer to a null-terminated string containing the model ID.
/// The caller is responsible for freeing the returned string using
/// `xybrid_free_string()`.
///
/// # Parameters
///
/// - `model`: A handle to the loaded model.
///
/// # Returns
///
/// A pointer to a null-terminated string containing the model ID,
/// or null on failure. The caller must free this string with `xybrid_free_string()`.
///
/// # Example (C)
///
/// ```c
/// char* model_id = xybrid_model_id(model);
/// if (model_id != NULL) {
///     printf("Model ID: %s\n", model_id);
///     xybrid_free_string(model_id);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_model_id(model: *mut XybridModelHandle) -> *mut c_char {
    clear_last_error();

    // Validate handle
    if model.is_null() {
        set_last_error("model handle is null");
        return std::ptr::null_mut();
    }

    // Borrow the model state
    let model_state = match XybridModelHandle::as_ref(model) {
        Some(state) => state,
        None => {
            set_last_error("invalid model handle");
            return std::ptr::null_mut();
        }
    };

    // Create a CString from the model ID and return it
    // The caller is responsible for freeing this with xybrid_free_string()
    match CString::new(model_state.model_id.clone()) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => {
            set_last_error("model_id contains null bytes");
            std::ptr::null_mut()
        }
    }
}

/// Check if a model supports token-by-token streaming.
///
/// Returns 1 if the model supports true token-by-token streaming (LLM models
/// with GGUF format when LLM features are enabled), 0 otherwise.
///
/// Note: `xybrid_model_run_streaming()` works for all models, but only LLM
/// models get true token-by-token streaming; others emit a single result.
///
/// # Parameters
///
/// - `model`: A handle to the loaded model.
///
/// # Returns
///
/// - `1` if the model supports token streaming
/// - `0` if it does not, or if the handle is null/invalid
///
/// # Example (C)
///
/// ```c
/// if (xybrid_model_supports_token_streaming(model)) {
///     // Use streaming inference
/// } else {
///     // Use batch inference
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_model_supports_token_streaming(
    model: *mut XybridModelHandle,
) -> i32 {
    // Don't clear last error - this is a read-only accessor
    if model.is_null() {
        return 0;
    }

    match XybridModelHandle::as_ref(model) {
        Some(state) => {
            if state.model.supports_token_streaming() {
                1
            } else {
                0
            }
        }
        None => 0,
    }
}

/// Run streaming inference on a model with the given input envelope.
///
/// This function blocks until inference is complete. For each token generated,
/// the callback is invoked with the token data. After all tokens are emitted,
/// the function returns a result handle with the final output.
///
/// For non-LLM models, a single callback invocation occurs with the complete result.
///
/// # Parameters
///
/// - `model`: A handle to the loaded model.
/// - `envelope`: A handle to the input envelope.
/// - `callback`: Function pointer invoked for each generated token.
/// - `user_data`: Opaque pointer passed through to every callback invocation.
///
/// # Returns
///
/// A handle to the final result, or null on failure.
/// On failure, call `xybrid_last_error()` to get the error message.
///
/// # Thread Safety
///
/// The callback is invoked from the calling thread. The caller must ensure
/// that `user_data` is valid for the duration of the call.
///
/// # Example (C)
///
/// ```c
/// void on_token(const char* token, int64_t token_id, uint32_t index,
///               const char* cumulative, const char* finish, void* ctx) {
///     printf("%s", token);
///     fflush(stdout);
/// }
///
/// XybridResultHandle* result = xybrid_model_run_streaming(
///     model, envelope, on_token, NULL);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_model_run_streaming(
    model: *mut XybridModelHandle,
    envelope: *mut XybridEnvelopeHandle,
    callback: XybridStreamCallback,
    user_data: *mut c_void,
) -> *mut XybridResultHandle {
    clear_last_error();

    // Validate handles
    if model.is_null() {
        set_last_error("model handle is null");
        return std::ptr::null_mut();
    }
    if envelope.is_null() {
        set_last_error("envelope handle is null");
        return std::ptr::null_mut();
    }
    let callback_fn = match callback {
        Some(f) => f,
        None => {
            set_last_error("callback is null");
            return std::ptr::null_mut();
        }
    };

    let model_state = match XybridModelHandle::as_ref(model) {
        Some(state) => state,
        None => {
            set_last_error("invalid model handle");
            return std::ptr::null_mut();
        }
    };
    let envelope_data = match XybridEnvelopeHandle::as_ref(envelope) {
        Some(data) => data,
        None => {
            set_last_error("invalid envelope handle");
            return std::ptr::null_mut();
        }
    };

    let sdk_envelope = envelope_data_to_sdk(envelope_data);

    // Wrap callback + user_data in a Send-safe context
    let ctx = StreamCallbackCtx { callback: callback_fn, user_data };

    // Build the on_token closure that bridges to the C callback
    let on_token = move |token: PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        unsafe { ctx.invoke(&token) };
        Ok(())
    };

    // Call the SDK streaming method
    match model_state.model.run_streaming(&sdk_envelope, on_token) {
        Ok(result) => {
            XybridResultHandle::from_boxed(Box::new(inference_result_to_data(&result)))
        }
        Err(e) => {
            let result = ResultData {
                success: false,
                error: Some(format!("Streaming inference failed: {}", e)),
                output_type: "".to_string(),
                text: None,
                embedding: None,
                audio_bytes: None,
                latency_ms: 0,
            };
            XybridResultHandle::from_boxed(Box::new(result))
        }
    }
}

/// Run streaming inference on a model with conversation context.
///
/// Same as `xybrid_model_run_streaming` but includes conversation history
/// for multi-turn LLM interactions.
///
/// # Parameters
///
/// - `model`: A handle to the loaded model.
/// - `envelope`: A handle to the input envelope (current user message).
/// - `context`: A handle to the conversation context.
/// - `callback`: Function pointer invoked for each generated token.
/// - `user_data`: Opaque pointer passed through to every callback invocation.
///
/// # Returns
///
/// A handle to the final result, or null on failure.
/// The envelope and context are NOT consumed - they can be reused.
///
/// # Example (C)
///
/// ```c
/// XybridContextHandle* ctx = xybrid_context_new();
/// xybrid_context_set_system(ctx, "You are a helpful assistant.");
///
/// XybridEnvelopeHandle* msg = xybrid_envelope_text_with_role("Hello!", XYBRID_ROLE_USER);
/// xybrid_context_push(ctx, msg);
///
/// XybridResultHandle* result = xybrid_model_run_streaming_with_context(
///     model, msg, ctx, on_token, NULL);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_model_run_streaming_with_context(
    model: *mut XybridModelHandle,
    envelope: *mut XybridEnvelopeHandle,
    context: *mut XybridContextHandle,
    callback: XybridStreamCallback,
    user_data: *mut c_void,
) -> *mut XybridResultHandle {
    clear_last_error();

    // Validate all handles
    if model.is_null() {
        set_last_error("model handle is null");
        return std::ptr::null_mut();
    }
    if envelope.is_null() {
        set_last_error("envelope handle is null");
        return std::ptr::null_mut();
    }
    if context.is_null() {
        set_last_error("context handle is null");
        return std::ptr::null_mut();
    }
    let callback_fn = match callback {
        Some(f) => f,
        None => {
            set_last_error("callback is null");
            return std::ptr::null_mut();
        }
    };

    let model_state = match XybridModelHandle::as_ref(model) {
        Some(state) => state,
        None => {
            set_last_error("invalid model handle");
            return std::ptr::null_mut();
        }
    };
    let envelope_data = match XybridEnvelopeHandle::as_ref(envelope) {
        Some(data) => data,
        None => {
            set_last_error("invalid envelope handle");
            return std::ptr::null_mut();
        }
    };
    let ctx_data = match XybridContextHandle::as_ref(context) {
        Some(data) => data,
        None => {
            set_last_error("invalid context handle");
            return std::ptr::null_mut();
        }
    };

    let sdk_envelope = envelope_data_to_sdk(envelope_data);

    // Wrap callback + user_data in a Send-safe context
    let cb_ctx = StreamCallbackCtx { callback: callback_fn, user_data };

    // Build the on_token closure that bridges to the C callback
    let on_token = move |token: PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        unsafe { cb_ctx.invoke(&token) };
        Ok(())
    };

    // Call the SDK streaming method with context
    match model_state.model.run_streaming_with_context(&sdk_envelope, &ctx_data.context, on_token) {
        Ok(result) => {
            XybridResultHandle::from_boxed(Box::new(inference_result_to_data(&result)))
        }
        Err(e) => {
            let result = ResultData {
                success: false,
                error: Some(format!("Streaming inference with context failed: {}", e)),
                output_type: "".to_string(),
                text: None,
                embedding: None,
                audio_bytes: None,
                latency_ms: 0,
            };
            XybridResultHandle::from_boxed(Box::new(result))
        }
    }
}

/// Free a model handle.
///
/// This function frees the memory associated with a model handle.
/// After calling this function, the handle is no longer valid.
///
/// # Parameters
///
/// - `handle`: A handle to the model to free. May be null (no-op).
///
/// # Example (C)
///
/// ```c
/// xybrid_model_free(model);
/// model = NULL; // Good practice: null out after freeing
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_model_free(handle: *mut XybridModelHandle) {
    if !handle.is_null() {
        // Take ownership and let it drop to free memory
        let _ = XybridModelHandle::into_boxed(handle);
    }
}

// ============================================================================
// C ABI Result Accessor Functions (US-014)
// ============================================================================
//
// These functions allow C consumers to extract data from inference results.
// Results are created by xybrid_model_run and must be freed with xybrid_result_free.

/// Check if the inference was successful.
///
/// Returns 1 if the inference succeeded, 0 if it failed.
/// If the handle is null or invalid, returns 0.
///
/// # Parameters
///
/// - `result`: A handle to the inference result.
///
/// # Returns
///
/// - `1` if success is true
/// - `0` if success is false, or if the handle is null/invalid
///
/// # Example (C)
///
/// ```c
/// if (xybrid_result_success(result)) {
///     const char* text = xybrid_result_text(result);
///     printf("Result: %s\n", text);
/// } else {
///     const char* error = xybrid_result_error(result);
///     printf("Error: %s\n", error ? error : "unknown");
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_result_success(result: *mut XybridResultHandle) -> i32 {
    // Don't clear last error - this is a read-only accessor
    if result.is_null() {
        return 0;
    }

    match XybridResultHandle::as_ref(result) {
        Some(data) => {
            if data.success {
                1
            } else {
                0
            }
        }
        None => 0,
    }
}

/// Get the error message from a failed inference.
///
/// Returns a pointer to a null-terminated string containing the error message,
/// or null if there was no error. The returned pointer is valid for the
/// lifetime of the result handle - do NOT free it with xybrid_free_string().
///
/// # Parameters
///
/// - `result`: A handle to the inference result.
///
/// # Returns
///
/// A pointer to the error message string, or null if no error.
/// The pointer is valid until the result handle is freed.
///
/// # Example (C)
///
/// ```c
/// if (!xybrid_result_success(result)) {
///     const char* error = xybrid_result_error(result);
///     fprintf(stderr, "Inference failed: %s\n", error ? error : "unknown error");
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_result_error(result: *mut XybridResultHandle) -> *const c_char {
    // Don't clear last error - this is a read-only accessor
    if result.is_null() {
        return std::ptr::null();
    }

    match XybridResultHandle::as_ref(result) {
        Some(data) => {
            match &data.error {
                Some(error_str) => {
                    // Store the CString in thread-local storage so the pointer remains valid
                    // until the next call to this function on the same thread.
                    // This is a trade-off between simplicity and thread-safety.
                    thread_local! {
                        static RESULT_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
                    }
                    RESULT_ERROR.with(|e| {
                        *e.borrow_mut() = CString::new(error_str.as_str()).ok();
                        match e.borrow().as_ref() {
                            Some(cstr) => cstr.as_ptr(),
                            None => std::ptr::null(),
                        }
                    })
                }
                None => std::ptr::null(),
            }
        }
        None => std::ptr::null(),
    }
}

/// Get the text output from an inference result.
///
/// Returns a pointer to a null-terminated string containing the text output,
/// or null if the result does not contain text. The returned pointer is valid
/// for the lifetime of the result handle - do NOT free it with xybrid_free_string().
///
/// # Parameters
///
/// - `result`: A handle to the inference result.
///
/// # Returns
///
/// A pointer to the text output string, or null if no text output.
/// The pointer is valid until the result handle is freed.
///
/// # Example (C)
///
/// ```c
/// XybridResultHandle* result = xybrid_model_run(model, envelope);
/// if (xybrid_result_success(result)) {
///     const char* text = xybrid_result_text(result);
///     if (text != NULL) {
///         printf("Transcription: %s\n", text);
///     }
/// }
/// xybrid_result_free(result);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_result_text(result: *mut XybridResultHandle) -> *const c_char {
    // Don't clear last error - this is a read-only accessor
    if result.is_null() {
        return std::ptr::null();
    }

    match XybridResultHandle::as_ref(result) {
        Some(data) => {
            match &data.text {
                Some(text_str) => {
                    // Store the CString in thread-local storage so the pointer remains valid
                    // until the next call to this function on the same thread.
                    thread_local! {
                        static RESULT_TEXT: RefCell<Option<CString>> = const { RefCell::new(None) };
                    }
                    RESULT_TEXT.with(|e| {
                        *e.borrow_mut() = CString::new(text_str.as_str()).ok();
                        match e.borrow().as_ref() {
                            Some(cstr) => cstr.as_ptr(),
                            None => std::ptr::null(),
                        }
                    })
                }
                None => std::ptr::null(),
            }
        }
        None => std::ptr::null(),
    }
}

/// Get the latency in milliseconds from an inference result.
///
/// Returns the inference latency in milliseconds.
/// If the handle is null or invalid, returns 0.
///
/// # Parameters
///
/// - `result`: A handle to the inference result.
///
/// # Returns
///
/// The inference latency in milliseconds, or 0 if the handle is null/invalid.
///
/// # Example (C)
///
/// ```c
/// uint32_t latency = xybrid_result_latency_ms(result);
/// printf("Inference took %u ms\n", latency);
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_result_latency_ms(result: *mut XybridResultHandle) -> u32 {
    // Don't clear last error - this is a read-only accessor
    if result.is_null() {
        return 0;
    }

    match XybridResultHandle::as_ref(result) {
        Some(data) => data.latency_ms,
        None => 0,
    }
}

/// Free an inference result handle.
///
/// This function frees the memory associated with an inference result handle.
/// After calling this function, the handle is no longer valid.
///
/// # Parameters
///
/// - `handle`: A handle to the result to free. May be null (no-op).
///
/// # Example (C)
///
/// ```c
/// XybridResultHandle* result = xybrid_model_run(model, envelope);
/// // ... use result ...
/// xybrid_result_free(result);
/// result = NULL; // Good practice: null out after freeing
/// ```
#[no_mangle]
pub unsafe extern "C" fn xybrid_result_free(handle: *mut XybridResultHandle) {
    if !handle.is_null() {
        // Take ownership and let it drop to free memory
        let _ = XybridResultHandle::into_boxed(handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }

    #[test]
    fn test_sdk_dependency() {
        // Verify SDK is accessible
        let _ = xybrid_sdk::current_platform();
    }

    // Note: LoaderState and ModelState hold SDK objects (ModelLoader, XybridModel)
    // which require actual model paths or registry access to construct.
    // Handle roundtrip is tested implicitly through the integration tests below.

    #[test]
    fn test_envelope_handle_audio() {
        let envelope = Box::new(EnvelopeData::Audio {
            bytes: vec![1, 2, 3, 4],
            sample_rate: 16000,
            channels: 1,
        });

        let handle = XybridEnvelopeHandle::from_boxed(envelope);
        assert!(!handle.is_null());

        unsafe {
            let data = XybridEnvelopeHandle::as_ref(handle).expect("should have data");
            match data {
                EnvelopeData::Audio {
                    bytes,
                    sample_rate,
                    channels,
                } => {
                    assert_eq!(bytes, &vec![1, 2, 3, 4]);
                    assert_eq!(*sample_rate, 16000);
                    assert_eq!(*channels, 1);
                }
                _ => panic!("Expected Audio variant"),
            }

            // Clean up
            let _ = XybridEnvelopeHandle::into_boxed(handle);
        }
    }

    #[test]
    fn test_envelope_handle_text() {
        let envelope = Box::new(EnvelopeData::Text {
            text: "Hello, world!".to_string(),
            voice_id: Some("voice-1".to_string()),
            speed: Some(1.0),
            role: None,
        });

        let handle = XybridEnvelopeHandle::from_boxed(envelope);
        assert!(!handle.is_null());

        unsafe {
            let data = XybridEnvelopeHandle::as_ref(handle).expect("should have data");
            match data {
                EnvelopeData::Text {
                    text,
                    voice_id,
                    speed,
                    role,
                } => {
                    assert_eq!(text, "Hello, world!");
                    assert_eq!(voice_id.as_deref(), Some("voice-1"));
                    assert_eq!(*speed, Some(1.0));
                    assert!(role.is_none());
                }
                _ => panic!("Expected Text variant"),
            }

            // Clean up
            let _ = XybridEnvelopeHandle::into_boxed(handle);
        }
    }

    #[test]
    fn test_result_handle_success() {
        let result = Box::new(ResultData {
            success: true,
            error: None,
            output_type: "text".to_string(),
            text: Some("Transcribed text".to_string()),
            embedding: None,
            audio_bytes: None,
            latency_ms: 100,
        });

        let handle = XybridResultHandle::from_boxed(result);
        assert!(!handle.is_null());

        unsafe {
            let data = XybridResultHandle::as_ref(handle).expect("should have data");
            assert!(data.success);
            assert!(data.error.is_none());
            assert_eq!(data.output_type, "text");
            assert_eq!(data.text.as_deref(), Some("Transcribed text"));
            assert_eq!(data.latency_ms, 100);

            // Clean up
            let _ = XybridResultHandle::into_boxed(handle);
        }
    }

    #[test]
    fn test_result_handle_error() {
        let result = Box::new(ResultData {
            success: false,
            error: Some("Model not found".to_string()),
            output_type: "".to_string(),
            text: None,
            embedding: None,
            audio_bytes: None,
            latency_ms: 0,
        });

        let handle = XybridResultHandle::from_boxed(result);
        assert!(!handle.is_null());

        unsafe {
            let data = XybridResultHandle::as_ref(handle).expect("should have data");
            assert!(!data.success);
            assert_eq!(data.error.as_deref(), Some("Model not found"));

            // Clean up
            let _ = XybridResultHandle::into_boxed(handle);
        }
    }

    #[test]
    fn test_null_handle_safety() {
        use std::ptr;

        unsafe {
            // All as_ref methods should return None for null handles
            assert!(XybridModelLoaderHandle::as_ref(ptr::null_mut()).is_none());
            assert!(XybridModelHandle::as_ref(ptr::null_mut()).is_none());
            assert!(XybridEnvelopeHandle::as_ref(ptr::null_mut()).is_none());
            assert!(XybridResultHandle::as_ref(ptr::null_mut()).is_none());

            // All into_boxed methods should return None for null handles
            assert!(XybridModelLoaderHandle::into_boxed(ptr::null_mut()).is_none());
            assert!(XybridModelHandle::into_boxed(ptr::null_mut()).is_none());
            assert!(XybridEnvelopeHandle::into_boxed(ptr::null_mut()).is_none());
            assert!(XybridResultHandle::into_boxed(ptr::null_mut()).is_none());
        }
    }

    // ========================================================================
    // Tests for C ABI Utility Functions (US-010)
    // ========================================================================

    #[test]
    fn test_xybrid_init() {
        // Init should return 0 (success)
        let result = xybrid_init();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_xybrid_version() {
        let version_ptr = xybrid_version();
        assert!(!version_ptr.is_null());

        // Convert to Rust string and verify
        let version_str = unsafe { CStr::from_ptr(version_ptr) }
            .to_str()
            .expect("version should be valid UTF-8");

        assert_eq!(version_str, VERSION);
        assert_eq!(version_str, "0.1.0");
    }

    #[test]
    fn test_xybrid_last_error_empty() {
        // Clear any previous error first
        clear_last_error();

        // Last error should be null when no error has occurred
        let error_ptr = xybrid_last_error();
        assert!(error_ptr.is_null());
    }

    #[test]
    fn test_xybrid_last_error_set() {
        // Set an error
        set_last_error("Test error message");

        // Last error should return the message
        let error_ptr = xybrid_last_error();
        assert!(!error_ptr.is_null());

        let error_str = unsafe { CStr::from_ptr(error_ptr) }
            .to_str()
            .expect("error should be valid UTF-8");

        assert_eq!(error_str, "Test error message");

        // Clean up
        clear_last_error();
    }

    #[test]
    fn test_xybrid_free_string_null() {
        // Freeing null should not panic
        unsafe {
            xybrid_free_string(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_xybrid_free_string_valid() {
        // Create a CString and convert to raw pointer
        let s = CString::new("test string").expect("valid string");
        let ptr = s.into_raw();

        // Free should not panic
        unsafe {
            xybrid_free_string(ptr);
        }
        // Note: We can't verify the memory is freed, but no panic means success
    }

    #[test]
    fn test_error_persistence() {
        // Verify error persists until cleared
        set_last_error("First error");
        assert!(!xybrid_last_error().is_null());

        // Set another error
        set_last_error("Second error");
        let error_ptr = xybrid_last_error();
        let error_str = unsafe { CStr::from_ptr(error_ptr) }
            .to_str()
            .expect("valid UTF-8");
        assert_eq!(error_str, "Second error");

        // Clear
        clear_last_error();
        assert!(xybrid_last_error().is_null());
    }

    // ========================================================================
    // Tests for C ABI Model Loader Functions (US-011)
    // ========================================================================

    #[test]
    fn test_model_loader_from_registry() {
        let model_id = CString::new("kokoro-82m").unwrap();

        unsafe {
            let handle = xybrid_model_loader_from_registry(model_id.as_ptr());
            assert!(!handle.is_null());

            // Verify state
            let state = XybridModelLoaderHandle::as_ref(handle).unwrap();
            assert_eq!(state.model_id, "kokoro-82m");

            // Clean up
            xybrid_model_loader_free(handle);
        }
    }

    #[test]
    fn test_model_loader_from_registry_null() {
        unsafe {
            let handle = xybrid_model_loader_from_registry(std::ptr::null());
            assert!(handle.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "model_id is null");
        }
    }

    #[test]
    fn test_model_loader_from_registry_empty() {
        let model_id = CString::new("").unwrap();

        unsafe {
            let handle = xybrid_model_loader_from_registry(model_id.as_ptr());
            assert!(handle.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "model_id is empty");
        }
    }

    #[test]
    #[ignore] // Requires a real model bundle path
    fn test_model_loader_from_bundle() {
        // Note: This test is ignored because it requires a valid model bundle path.
        // ModelLoader::from_bundle validates that the path exists.
        let path = CString::new("/path/to/my-model").unwrap();

        unsafe {
            let handle = xybrid_model_loader_from_bundle(path.as_ptr());
            // This fails for non-existent paths
            if !handle.is_null() {
                let state = XybridModelLoaderHandle::as_ref(handle).unwrap();
                assert_eq!(state.model_id, "my-model");
                xybrid_model_loader_free(handle);
            }
        }
    }

    #[test]
    fn test_model_loader_from_bundle_null() {
        unsafe {
            let handle = xybrid_model_loader_from_bundle(std::ptr::null());
            assert!(handle.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "path is null");
        }
    }

    #[test]
    fn test_model_loader_from_bundle_empty() {
        let path = CString::new("").unwrap();

        unsafe {
            let handle = xybrid_model_loader_from_bundle(path.as_ptr());
            assert!(handle.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "path is empty");
        }
    }

    #[test]
    #[ignore] // Requires real model from registry
    fn test_model_loader_load_from_registry() {
        // Note: This test requires a real model to be available in the registry.
        // Run with: cargo test -p xybrid-ffi -- --ignored
        let model_id = CString::new("kokoro-82m").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            assert!(!loader.is_null());

            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                let error = xybrid_last_error();
                if !error.is_null() {
                    eprintln!(
                        "Model load failed: {}",
                        CStr::from_ptr(error).to_str().unwrap()
                    );
                }
            }
            assert!(!model.is_null(), "Model should load from registry");

            // Verify model state
            let state = XybridModelHandle::as_ref(model).unwrap();
            assert_eq!(state.model_id, "kokoro-82m");

            // Clean up
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    #[ignore] // Requires real model bundle path
    fn test_model_loader_load_from_bundle() {
        // Note: This test requires a real model bundle path.
        // Adjust the path to a real model bundle to run this test.
        let path = CString::new("/path/to/bundle").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_bundle(path.as_ptr());
            if loader.is_null() {
                // Expected for non-existent paths
                return;
            }

            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            // Verify model state
            let state = XybridModelHandle::as_ref(model).unwrap();
            assert!(!state.model_id.is_empty());

            // Clean up
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    fn test_model_loader_load_null() {
        unsafe {
            let model = xybrid_model_loader_load(std::ptr::null_mut());
            assert!(model.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "loader handle is null");
        }
    }

    #[test]
    fn test_model_loader_free_null() {
        // Freeing null should not panic
        unsafe {
            xybrid_model_loader_free(std::ptr::null_mut());
        }
    }

    // ========================================================================
    // Tests for C ABI Envelope Functions (US-012)
    // ========================================================================

    #[test]
    fn test_envelope_audio_basic() {
        let audio_bytes: [u8; 4] = [1, 2, 3, 4];

        unsafe {
            let handle = xybrid_envelope_audio(audio_bytes.as_ptr(), audio_bytes.len(), 16000, 1);
            assert!(!handle.is_null());

            // Verify envelope data
            let data = XybridEnvelopeHandle::as_ref(handle).unwrap();
            match data {
                EnvelopeData::Audio {
                    bytes,
                    sample_rate,
                    channels,
                } => {
                    assert_eq!(bytes, &vec![1, 2, 3, 4]);
                    assert_eq!(*sample_rate, 16000);
                    assert_eq!(*channels, 1);
                }
                _ => panic!("Expected Audio variant"),
            }

            // Clean up
            xybrid_envelope_free(handle);
        }
    }

    #[test]
    fn test_envelope_audio_stereo() {
        let audio_bytes: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

        unsafe {
            let handle = xybrid_envelope_audio(audio_bytes.as_ptr(), audio_bytes.len(), 44100, 2);
            assert!(!handle.is_null());

            // Verify envelope data
            let data = XybridEnvelopeHandle::as_ref(handle).unwrap();
            match data {
                EnvelopeData::Audio {
                    bytes,
                    sample_rate,
                    channels,
                } => {
                    assert_eq!(bytes.len(), 8);
                    assert_eq!(*sample_rate, 44100);
                    assert_eq!(*channels, 2);
                }
                _ => panic!("Expected Audio variant"),
            }

            // Clean up
            xybrid_envelope_free(handle);
        }
    }

    #[test]
    fn test_envelope_audio_empty() {
        // Empty audio is valid (len=0)
        unsafe {
            let handle = xybrid_envelope_audio(std::ptr::null(), 0, 16000, 1);
            assert!(!handle.is_null());

            // Verify empty audio
            let data = XybridEnvelopeHandle::as_ref(handle).unwrap();
            match data {
                EnvelopeData::Audio { bytes, .. } => {
                    assert!(bytes.is_empty());
                }
                _ => panic!("Expected Audio variant"),
            }

            // Clean up
            xybrid_envelope_free(handle);
        }
    }

    #[test]
    fn test_envelope_audio_null_with_length() {
        // Null bytes with non-zero length is an error
        unsafe {
            let handle = xybrid_envelope_audio(std::ptr::null(), 10, 16000, 1);
            assert!(handle.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "bytes is null but len is non-zero");
        }
    }

    #[test]
    fn test_envelope_audio_zero_sample_rate() {
        let audio_bytes: [u8; 4] = [1, 2, 3, 4];

        unsafe {
            let handle = xybrid_envelope_audio(audio_bytes.as_ptr(), audio_bytes.len(), 0, 1);
            assert!(handle.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "sample_rate must be non-zero");
        }
    }

    #[test]
    fn test_envelope_audio_zero_channels() {
        let audio_bytes: [u8; 4] = [1, 2, 3, 4];

        unsafe {
            let handle = xybrid_envelope_audio(audio_bytes.as_ptr(), audio_bytes.len(), 16000, 0);
            assert!(handle.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "channels must be non-zero");
        }
    }

    #[test]
    fn test_envelope_text_basic() {
        let text = CString::new("Hello, world!").unwrap();

        unsafe {
            let handle = xybrid_envelope_text(text.as_ptr());
            assert!(!handle.is_null());

            // Verify envelope data
            let data = XybridEnvelopeHandle::as_ref(handle).unwrap();
            match data {
                EnvelopeData::Text {
                    text,
                    voice_id,
                    speed,
                    role,
                } => {
                    assert_eq!(text, "Hello, world!");
                    assert!(voice_id.is_none());
                    assert!(speed.is_none());
                    assert!(role.is_none());
                }
                _ => panic!("Expected Text variant"),
            }

            // Clean up
            xybrid_envelope_free(handle);
        }
    }

    #[test]
    fn test_envelope_text_empty() {
        // Empty text is allowed
        let text = CString::new("").unwrap();

        unsafe {
            let handle = xybrid_envelope_text(text.as_ptr());
            assert!(!handle.is_null());

            // Verify envelope data
            let data = XybridEnvelopeHandle::as_ref(handle).unwrap();
            match data {
                EnvelopeData::Text { text, .. } => {
                    assert!(text.is_empty());
                }
                _ => panic!("Expected Text variant"),
            }

            // Clean up
            xybrid_envelope_free(handle);
        }
    }

    #[test]
    fn test_envelope_text_null() {
        unsafe {
            let handle = xybrid_envelope_text(std::ptr::null());
            assert!(handle.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "text is null");
        }
    }

    #[test]
    fn test_envelope_text_unicode() {
        // Test with unicode characters
        let text = CString::new("こんにちは世界 🌍").unwrap();

        unsafe {
            let handle = xybrid_envelope_text(text.as_ptr());
            assert!(!handle.is_null());

            // Verify envelope data
            let data = XybridEnvelopeHandle::as_ref(handle).unwrap();
            match data {
                EnvelopeData::Text { text, .. } => {
                    assert_eq!(text, "こんにちは世界 🌍");
                }
                _ => panic!("Expected Text variant"),
            }

            // Clean up
            xybrid_envelope_free(handle);
        }
    }

    #[test]
    fn test_envelope_free_null() {
        // Freeing null should not panic
        unsafe {
            xybrid_envelope_free(std::ptr::null_mut());
        }
    }

    // ========================================================================
    // Tests for C ABI Inference Functions (US-013)
    // ========================================================================

    #[test]
    #[ignore] // Requires real ASR model from registry
    fn test_model_run_with_audio() {
        // Note: This test requires a real ASR model (e.g., whisper-tiny)
        let model_id = CString::new("whisper-tiny").unwrap();
        let audio_bytes: [u8; 4] = [1, 2, 3, 4]; // Would need real audio data

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            assert!(!loader.is_null());

            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            let envelope =
                xybrid_envelope_audio(audio_bytes.as_ptr(), audio_bytes.len(), 16000, 1);
            assert!(!envelope.is_null());

            let result = xybrid_model_run(model, envelope);
            assert!(!result.is_null());

            // Check result structure
            let result_data = XybridResultHandle::as_ref(result).unwrap();
            // Result may or may not succeed depending on audio data validity

            xybrid_result_free(result);
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    #[ignore] // Requires real TTS model from registry
    fn test_model_run_with_text() {
        // Note: This test requires a real TTS model (e.g., kokoro-82m)
        let model_id = CString::new("kokoro-82m").unwrap();
        let text = CString::new("Hello, world!").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            assert!(!loader.is_null());

            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            let envelope = xybrid_envelope_text(text.as_ptr());
            assert!(!envelope.is_null());

            let result = xybrid_model_run(model, envelope);
            assert!(!result.is_null());

            let result_data = XybridResultHandle::as_ref(result).unwrap();
            if result_data.success {
                assert_eq!(result_data.output_type, "audio");
                assert!(result_data.audio_bytes.is_some());
            }

            xybrid_result_free(result);
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    fn test_model_run_null_model() {
        let text = CString::new("Hello").unwrap();

        unsafe {
            // Create envelope
            let envelope = xybrid_envelope_text(text.as_ptr());
            assert!(!envelope.is_null());

            // Run with null model
            let result = xybrid_model_run(std::ptr::null_mut(), envelope);
            assert!(result.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "model handle is null");

            // Clean up
            xybrid_envelope_free(envelope);
        }
    }

    #[test]
    #[ignore] // Requires real model from registry
    fn test_model_run_null_envelope() {
        let model_id = CString::new("kokoro-82m").unwrap();

        unsafe {
            // Create loader and load model
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            // Run with null envelope
            let result = xybrid_model_run(model, std::ptr::null_mut());
            assert!(result.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "envelope handle is null");

            // Clean up
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    #[ignore] // Requires real model from registry
    fn test_model_run_envelope_reuse() {
        let model_id = CString::new("kokoro-82m").unwrap();
        let text = CString::new("Test text").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            let envelope = xybrid_envelope_text(text.as_ptr());
            assert!(!envelope.is_null());

            // Run inference twice with the same envelope
            let result1 = xybrid_model_run(model, envelope);
            assert!(!result1.is_null());

            let result2 = xybrid_model_run(model, envelope);
            assert!(!result2.is_null());

            xybrid_result_free(result1);
            xybrid_result_free(result2);
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    #[ignore] // Requires real model from registry
    fn test_model_id_basic() {
        let model_name = CString::new("kokoro-82m").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_name.as_ptr());
            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            let id_ptr = xybrid_model_id(model);
            assert!(!id_ptr.is_null());

            let id_str = CStr::from_ptr(id_ptr).to_str().unwrap();
            assert_eq!(id_str, "kokoro-82m");

            xybrid_free_string(id_ptr);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    fn test_model_id_null_handle() {
        unsafe {
            let id_ptr = xybrid_model_id(std::ptr::null_mut());
            assert!(id_ptr.is_null());

            // Verify error was set
            let error = xybrid_last_error();
            assert!(!error.is_null());
            let error_str = CStr::from_ptr(error).to_str().unwrap();
            assert_eq!(error_str, "model handle is null");
        }
    }

    #[test]
    fn test_model_supports_token_streaming_null_handle() {
        unsafe {
            // Null handle should return 0
            assert_eq!(xybrid_model_supports_token_streaming(std::ptr::null_mut()), 0);
        }
    }

    #[test]
    #[ignore] // Requires real model from registry
    fn test_model_supports_token_streaming_tts() {
        // TTS model should NOT support token streaming
        let model_id = CString::new("kokoro-82m").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            // TTS models don't support token streaming
            let supports = xybrid_model_supports_token_streaming(model);
            assert_eq!(supports, 0);

            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    #[ignore] // Requires real model from registry
    fn test_model_free_basic() {
        let model_id = CString::new("kokoro-82m").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            // Free model (should not panic)
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    fn test_model_free_null() {
        // Freeing null should not panic
        unsafe {
            xybrid_model_free(std::ptr::null_mut());
        }
    }

    #[test]
    #[ignore] // Requires real model from registry
    fn test_model_run_latency_recorded() {
        let model_id = CString::new("kokoro-82m").unwrap();
        let text = CString::new("Hello").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            let envelope = xybrid_envelope_text(text.as_ptr());
            let result = xybrid_model_run(model, envelope);
            assert!(!result.is_null());

            let result_data = XybridResultHandle::as_ref(result).unwrap();
            // Latency should be recorded
            assert!(result_data.latency_ms < 60000); // Less than 60 seconds

            xybrid_result_free(result);
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    // ========================================================================
    // Tests for C ABI Result Accessor Functions (US-014)
    // ========================================================================

    #[test]
    #[ignore] // Requires real model from registry
    fn test_result_success_true() {
        let model_id = CString::new("kokoro-82m").unwrap();
        let text = CString::new("Hello").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            let envelope = xybrid_envelope_text(text.as_ptr());
            let result = xybrid_model_run(model, envelope);

            // Check if success
            let success = xybrid_result_success(result);
            // Success should be 0 or 1
            assert!(success == 0 || success == 1);

            xybrid_result_free(result);
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    fn test_result_success_null_handle() {
        unsafe {
            // Null handle should return 0
            assert_eq!(xybrid_result_success(std::ptr::null_mut()), 0);
        }
    }

    #[test]
    #[ignore] // Requires real model from registry
    fn test_result_error_no_error() {
        let model_id = CString::new("kokoro-82m").unwrap();
        let text = CString::new("Hello").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            let envelope = xybrid_envelope_text(text.as_ptr());
            let result = xybrid_model_run(model, envelope);

            // If successful, error should be null
            if xybrid_result_success(result) == 1 {
                let error = xybrid_result_error(result);
                assert!(error.is_null());
            }

            xybrid_result_free(result);
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    fn test_result_error_null_handle() {
        unsafe {
            // Null handle should return null
            let error = xybrid_result_error(std::ptr::null_mut());
            assert!(error.is_null());
        }
    }

    #[test]
    #[ignore] // Requires real ASR model from registry
    fn test_result_text_with_audio_input() {
        // Note: This test requires a real ASR model (e.g., whisper-tiny)
        let model_id = CString::new("whisper-tiny").unwrap();
        let audio_bytes: [u8; 4] = [1, 2, 3, 4]; // Would need real audio data

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            let envelope = xybrid_envelope_audio(audio_bytes.as_ptr(), audio_bytes.len(), 16000, 1);
            let result = xybrid_model_run(model, envelope);

            // Check result structure
            let text_ptr = xybrid_result_text(result);
            // May or may not have text depending on model and input

            xybrid_result_free(result);
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    fn test_result_text_null_handle() {
        unsafe {
            // Null handle should return null
            let text = xybrid_result_text(std::ptr::null_mut());
            assert!(text.is_null());
        }
    }

    #[test]
    #[ignore] // Requires real model from registry
    fn test_result_latency_ms_basic() {
        let model_id = CString::new("kokoro-82m").unwrap();
        let text = CString::new("Hello").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            let envelope = xybrid_envelope_text(text.as_ptr());
            let result = xybrid_model_run(model, envelope);

            let latency = xybrid_result_latency_ms(result);
            assert!(latency < 60000); // Less than 60 seconds

            xybrid_result_free(result);
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    fn test_result_latency_ms_null_handle() {
        unsafe {
            // Null handle should return 0
            let latency = xybrid_result_latency_ms(std::ptr::null_mut());
            assert_eq!(latency, 0);
        }
    }

    #[test]
    fn test_result_free_null() {
        unsafe {
            // Freeing null should not panic
            xybrid_result_free(std::ptr::null_mut());
        }
    }

    #[test]
    #[ignore] // Requires real model from registry
    fn test_result_free_basic() {
        let model_id = CString::new("kokoro-82m").unwrap();
        let text = CString::new("Hello").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            if model.is_null() {
                xybrid_model_loader_free(loader);
                return;
            }

            let envelope = xybrid_envelope_text(text.as_ptr());
            let result = xybrid_model_run(model, envelope);

            // Free result (should not panic)
            xybrid_result_free(result);

            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }
}
