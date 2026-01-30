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
use std::ffi::{c_char, c_void, CStr, CString};

// Re-export SDK for internal use
#[allow(unused_imports)]
use xybrid_sdk as sdk;

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

// ============================================================================
// Internal Boxed Types
// ============================================================================
//
// These type aliases define the actual Rust types that the opaque handles
// point to. They are boxed (heap-allocated) so we can convert them to/from
// raw pointers for FFI.

/// Internal state for a model loader.
pub(crate) struct LoaderState {
    /// The model ID (from registry) or path (from bundle).
    pub model_id: String,
    /// Whether this loader was created from a bundle path.
    pub from_bundle: bool,
    /// The bundle path if from_bundle is true.
    pub bundle_path: Option<String>,
}

/// Internal state for a loaded model.
pub(crate) struct ModelState {
    /// The model ID.
    pub model_id: String,
    /// The path to the loaded model bundle.
    pub bundle_path: String,
    // Future: Add TemplateExecutor, ModelMetadata, etc.
}

/// Internal envelope data.
pub(crate) enum EnvelopeData {
    /// Audio data with sample rate and channel count.
    Audio {
        bytes: Vec<u8>,
        sample_rate: u32,
        channels: u32,
    },
    /// Text data with optional voice and speed.
    Text {
        text: String,
        voice_id: Option<String>,
        speed: Option<f64>,
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

/// Type alias for a boxed loader.
pub(crate) type BoxedLoader = Box<LoaderState>;

/// Type alias for a boxed model.
pub(crate) type BoxedModel = Box<ModelState>;

/// Type alias for a boxed envelope.
pub(crate) type BoxedEnvelope = Box<EnvelopeData>;

/// Type alias for a boxed result.
pub(crate) type BoxedResult = Box<ResultData>;

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

    // Create loader state
    let loader = Box::new(LoaderState {
        model_id: model_id_str,
        from_bundle: false,
        bundle_path: None,
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

    // Create loader state
    let loader = Box::new(LoaderState {
        model_id,
        from_bundle: true,
        bundle_path: Some(path_str),
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

    // Determine bundle path
    let bundle_path = if loader_state.from_bundle {
        match &loader_state.bundle_path {
            Some(path) => path.clone(),
            None => {
                set_last_error("bundle loader has no path");
                return std::ptr::null_mut();
            }
        }
    } else {
        // For registry models, we would use SDK's RegistryClient to fetch.
        // For now, return a placeholder path (actual fetching will be wired later).
        // In a real implementation, this would call:
        //   let client = sdk::RegistryClient::default_client()?;
        //   let bundle = client.fetch(&loader_state.model_id, None, |_| {})?;
        //   bundle.to_string_lossy().to_string()
        format!("~/.xybrid/cache/{}", loader_state.model_id)
    };

    // Create model state
    let model = Box::new(ModelState {
        model_id: loader_state.model_id.clone(),
        bundle_path,
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

    // Create envelope with no voice_id or speed (those would require additional parameters)
    let envelope = Box::new(EnvelopeData::Text {
        text: text_str,
        voice_id: None,
        speed: None,
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

    // Record start time for latency measurement
    let start = std::time::Instant::now();

    // Determine output type and perform inference
    // Note: In a real implementation, this would use TemplateExecutor from xybrid-core.
    // For now, we return a placeholder result that indicates success with mock data.
    let result = match envelope_data {
        EnvelopeData::Audio { .. } => {
            // ASR: Audio input → Text output
            // In a real implementation:
            //   1. Load model_metadata.json from model_state.bundle_path
            //   2. Create TemplateExecutor
            //   3. Execute with audio envelope
            //   4. Extract text from result
            ResultData {
                success: true,
                error: None,
                output_type: "text".to_string(),
                text: Some(format!("Transcribed from model: {}", model_state.model_id)),
                embedding: None,
                audio_bytes: None,
                latency_ms: start.elapsed().as_millis() as u32,
            }
        }
        EnvelopeData::Text { text, .. } => {
            // TTS: Text input → Audio output
            // In a real implementation:
            //   1. Load model_metadata.json from model_state.bundle_path
            //   2. Create TemplateExecutor
            //   3. Execute with text envelope
            //   4. Extract audio from result
            ResultData {
                success: true,
                error: None,
                output_type: "audio".to_string(),
                text: None,
                embedding: None,
                // Placeholder: return some audio bytes to indicate success
                audio_bytes: Some(format!("Audio for: {}", text).into_bytes()),
                latency_ms: start.elapsed().as_millis() as u32,
            }
        }
    };

    XybridResultHandle::from_boxed(Box::new(result))
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
        let _ = sdk::current_platform();
    }

    #[test]
    fn test_loader_handle_roundtrip() {
        let loader = Box::new(LoaderState {
            model_id: "test-model".to_string(),
            from_bundle: false,
            bundle_path: None,
        });

        let handle = XybridModelLoaderHandle::from_boxed(loader);
        assert!(!handle.is_null());

        unsafe {
            let state = XybridModelLoaderHandle::as_ref(handle).expect("should have state");
            assert_eq!(state.model_id, "test-model");
            assert!(!state.from_bundle);

            // Clean up
            let _ = XybridModelLoaderHandle::into_boxed(handle);
        }
    }

    #[test]
    fn test_model_handle_roundtrip() {
        let model = Box::new(ModelState {
            model_id: "test-model".to_string(),
            bundle_path: "/path/to/bundle".to_string(),
        });

        let handle = XybridModelHandle::from_boxed(model);
        assert!(!handle.is_null());

        unsafe {
            let state = XybridModelHandle::as_ref(handle).expect("should have state");
            assert_eq!(state.model_id, "test-model");
            assert_eq!(state.bundle_path, "/path/to/bundle");

            // Clean up
            let _ = XybridModelHandle::into_boxed(handle);
        }
    }

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
                } => {
                    assert_eq!(text, "Hello, world!");
                    assert_eq!(voice_id.as_deref(), Some("voice-1"));
                    assert_eq!(*speed, Some(1.0));
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
            assert!(!state.from_bundle);
            assert!(state.bundle_path.is_none());

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
    fn test_model_loader_from_bundle() {
        let path = CString::new("/path/to/my-model").unwrap();

        unsafe {
            let handle = xybrid_model_loader_from_bundle(path.as_ptr());
            assert!(!handle.is_null());

            // Verify state
            let state = XybridModelLoaderHandle::as_ref(handle).unwrap();
            assert_eq!(state.model_id, "my-model"); // Extracted from path
            assert!(state.from_bundle);
            assert_eq!(state.bundle_path.as_deref(), Some("/path/to/my-model"));

            // Clean up
            xybrid_model_loader_free(handle);
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
    fn test_model_loader_load_from_registry() {
        let model_id = CString::new("test-model").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            assert!(!loader.is_null());

            let model = xybrid_model_loader_load(loader);
            assert!(!model.is_null());

            // Verify model state
            let state = XybridModelHandle::as_ref(model).unwrap();
            assert_eq!(state.model_id, "test-model");
            assert!(state.bundle_path.contains("test-model"));

            // Clean up
            let _ = XybridModelHandle::into_boxed(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    fn test_model_loader_load_from_bundle() {
        let path = CString::new("/path/to/bundle").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_bundle(path.as_ptr());
            assert!(!loader.is_null());

            let model = xybrid_model_loader_load(loader);
            assert!(!model.is_null());

            // Verify model state
            let state = XybridModelHandle::as_ref(model).unwrap();
            assert_eq!(state.model_id, "bundle");
            assert_eq!(state.bundle_path, "/path/to/bundle");

            // Clean up
            let _ = XybridModelHandle::into_boxed(model);
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
                } => {
                    assert_eq!(text, "Hello, world!");
                    assert!(voice_id.is_none());
                    assert!(speed.is_none());
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
    fn test_model_run_with_audio() {
        let model_id = CString::new("test-asr-model").unwrap();
        let audio_bytes: [u8; 4] = [1, 2, 3, 4];

        unsafe {
            // Create loader and load model
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            assert!(!loader.is_null());

            let model = xybrid_model_loader_load(loader);
            assert!(!model.is_null());

            // Create audio envelope
            let envelope =
                xybrid_envelope_audio(audio_bytes.as_ptr(), audio_bytes.len(), 16000, 1);
            assert!(!envelope.is_null());

            // Run inference
            let result = xybrid_model_run(model, envelope);
            assert!(!result.is_null());

            // Verify result
            let result_data = XybridResultHandle::as_ref(result).unwrap();
            assert!(result_data.success);
            assert!(result_data.error.is_none());
            assert_eq!(result_data.output_type, "text");
            assert!(result_data.text.is_some());
            assert!(result_data.text.as_ref().unwrap().contains("test-asr-model"));

            // Clean up
            let _ = XybridResultHandle::into_boxed(result);
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    fn test_model_run_with_text() {
        let model_id = CString::new("test-tts-model").unwrap();
        let text = CString::new("Hello, world!").unwrap();

        unsafe {
            // Create loader and load model
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            assert!(!loader.is_null());

            let model = xybrid_model_loader_load(loader);
            assert!(!model.is_null());

            // Create text envelope
            let envelope = xybrid_envelope_text(text.as_ptr());
            assert!(!envelope.is_null());

            // Run inference
            let result = xybrid_model_run(model, envelope);
            assert!(!result.is_null());

            // Verify result
            let result_data = XybridResultHandle::as_ref(result).unwrap();
            assert!(result_data.success);
            assert!(result_data.error.is_none());
            assert_eq!(result_data.output_type, "audio");
            assert!(result_data.audio_bytes.is_some());
            // The placeholder returns text content as bytes
            let audio_bytes = result_data.audio_bytes.as_ref().unwrap();
            let audio_str = String::from_utf8_lossy(audio_bytes);
            assert!(audio_str.contains("Hello, world!"));

            // Clean up
            let _ = XybridResultHandle::into_boxed(result);
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
    fn test_model_run_null_envelope() {
        let model_id = CString::new("test-model").unwrap();

        unsafe {
            // Create loader and load model
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            assert!(!model.is_null());

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
    fn test_model_run_envelope_reuse() {
        let model_id = CString::new("test-model").unwrap();
        let text = CString::new("Test text").unwrap();

        unsafe {
            // Create loader and load model
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            assert!(!model.is_null());

            // Create envelope
            let envelope = xybrid_envelope_text(text.as_ptr());
            assert!(!envelope.is_null());

            // Run inference twice with the same envelope
            let result1 = xybrid_model_run(model, envelope);
            assert!(!result1.is_null());

            let result2 = xybrid_model_run(model, envelope);
            assert!(!result2.is_null());

            // Both should succeed
            let result_data1 = XybridResultHandle::as_ref(result1).unwrap();
            let result_data2 = XybridResultHandle::as_ref(result2).unwrap();
            assert!(result_data1.success);
            assert!(result_data2.success);

            // Clean up
            let _ = XybridResultHandle::into_boxed(result1);
            let _ = XybridResultHandle::into_boxed(result2);
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    #[test]
    fn test_model_id_basic() {
        let model_name = CString::new("kokoro-82m").unwrap();

        unsafe {
            // Create loader and load model
            let loader = xybrid_model_loader_from_registry(model_name.as_ptr());
            let model = xybrid_model_loader_load(loader);
            assert!(!model.is_null());

            // Get model ID
            let id_ptr = xybrid_model_id(model);
            assert!(!id_ptr.is_null());

            // Verify model ID
            let id_str = CStr::from_ptr(id_ptr).to_str().unwrap();
            assert_eq!(id_str, "kokoro-82m");

            // Clean up (caller must free the string)
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
    fn test_model_free_basic() {
        let model_id = CString::new("test-model").unwrap();

        unsafe {
            // Create loader and load model
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            assert!(!model.is_null());

            // Free model (should not panic)
            xybrid_model_free(model);

            // Clean up loader
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
    fn test_model_run_latency_recorded() {
        let model_id = CString::new("test-model").unwrap();
        let text = CString::new("Hello").unwrap();

        unsafe {
            // Create loader and load model
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            let envelope = xybrid_envelope_text(text.as_ptr());

            // Run inference
            let result = xybrid_model_run(model, envelope);
            assert!(!result.is_null());

            // Verify latency was recorded (should be > 0 or at least not negative)
            let result_data = XybridResultHandle::as_ref(result).unwrap();
            // Latency should be a reasonable value (placeholder returns almost immediately)
            assert!(result_data.latency_ms < 10000); // Less than 10 seconds

            // Clean up
            let _ = XybridResultHandle::into_boxed(result);
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }

    // ========================================================================
    // Tests for C ABI Result Accessor Functions (US-014)
    // ========================================================================

    #[test]
    fn test_result_success_true() {
        let model_id = CString::new("test-model").unwrap();
        let text = CString::new("Hello").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            let envelope = xybrid_envelope_text(text.as_ptr());
            let result = xybrid_model_run(model, envelope);

            // Verify success returns 1
            assert_eq!(xybrid_result_success(result), 1);

            // Clean up
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
    fn test_result_error_no_error() {
        let model_id = CString::new("test-model").unwrap();
        let text = CString::new("Hello").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            let envelope = xybrid_envelope_text(text.as_ptr());
            let result = xybrid_model_run(model, envelope);

            // Successful result should have no error
            let error = xybrid_result_error(result);
            assert!(error.is_null());

            // Clean up
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
    fn test_result_text_with_audio_input() {
        let model_id = CString::new("test-asr-model").unwrap();
        let audio_bytes: [u8; 4] = [1, 2, 3, 4];

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            let envelope = xybrid_envelope_audio(audio_bytes.as_ptr(), audio_bytes.len(), 16000, 1);
            let result = xybrid_model_run(model, envelope);

            // ASR result should have text
            let text_ptr = xybrid_result_text(result);
            assert!(!text_ptr.is_null());

            let text_str = CStr::from_ptr(text_ptr).to_str().unwrap();
            assert!(text_str.contains("test-asr-model"));

            // Clean up
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
    fn test_result_latency_ms_basic() {
        let model_id = CString::new("test-model").unwrap();
        let text = CString::new("Hello").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            let envelope = xybrid_envelope_text(text.as_ptr());
            let result = xybrid_model_run(model, envelope);

            // Latency should be recorded and reasonable
            let latency = xybrid_result_latency_ms(result);
            assert!(latency < 10000); // Less than 10 seconds

            // Clean up
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
    fn test_result_free_basic() {
        let model_id = CString::new("test-model").unwrap();
        let text = CString::new("Hello").unwrap();

        unsafe {
            let loader = xybrid_model_loader_from_registry(model_id.as_ptr());
            let model = xybrid_model_loader_load(loader);
            let envelope = xybrid_envelope_text(text.as_ptr());
            let result = xybrid_model_run(model, envelope);

            // Free result (should not panic)
            xybrid_result_free(result);

            // Clean up remaining handles
            xybrid_envelope_free(envelope);
            xybrid_model_free(model);
            xybrid_model_loader_free(loader);
        }
    }
}
