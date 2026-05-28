//! Safe ownership wrappers for llama.cpp's experimental mtmd multimodal API.
//!
//! The raw C ABI remains in `llama-cpp-sys`. This module owns mtmd lifetimes
//! and error mapping so downstream crates can stay on a safe surface.

use std::ffi::{c_void, CStr};
use std::ptr;

use crate::context::LlamaContext;
use crate::error::{LlamaError, LlamaResult};
use crate::ffi;
use crate::model::LlamaModel;

const MTMD_INPUT_CHUNK_TYPE_TEXT: i32 = 0;
const MTMD_INPUT_CHUNK_TYPE_IMAGE: i32 = 1;
const MTMD_INPUT_CHUNK_TYPE_AUDIO: i32 = 2;

/// Opaque handle to an mtmd multimodal projector context.
///
/// The projector references the loaded llama text model, so it must be
/// dropped before the corresponding [`LlamaModel`].
pub struct MtmdContext {
    ptr: *mut c_void,
}

impl MtmdContext {
    /// Load an mtmd projector (`mmproj`) for a loaded llama text model.
    pub fn load(
        path: &str,
        model: &LlamaModel,
        use_gpu: bool,
        warmup: bool,
        n_threads: usize,
        flash_attn: bool,
    ) -> LlamaResult<Self> {
        let c_path = ffi::cstring(path, "mtmd projector path")?;
        // SAFETY: c_path outlives the call; model.as_ptr is live.
        let ptr = unsafe {
            ffi::mtmd_init_from_file(
                &c_path,
                model.as_ptr(),
                use_gpu,
                warmup,
                n_threads,
                flash_attn,
            )
        };
        if ptr.is_null() {
            return Err(LlamaError::Internal(format!(
                "failed to initialize mtmd context from {path}"
            )));
        }
        Ok(Self { ptr })
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: pointer came from mtmd_init_from_file and is owned here.
            unsafe { ffi::mtmd_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

unsafe impl Send for MtmdContext {}

/// Opaque handle to an mtmd decoded bitmap.
pub struct MtmdBitmap {
    ptr: *mut c_void,
}

impl MtmdBitmap {
    /// Decode encoded image bytes into an mtmd bitmap using the projector
    /// context for model-specific decoding policy.
    pub fn from_encoded_bytes(ctx: &MtmdContext, bytes: &[u8]) -> LlamaResult<Self> {
        Self::from_encoded_bytes_with_context(ctx.as_ptr(), bytes)
    }

    /// Decode encoded image bytes without an mtmd context.
    ///
    /// This is valid for image bytes; audio inputs need a context so mtmd can
    /// read the model's expected sample rate.
    pub fn from_encoded_image_bytes(bytes: &[u8]) -> LlamaResult<Self> {
        Self::from_encoded_bytes_with_context(ptr::null_mut(), bytes)
    }

    fn from_encoded_bytes_with_context(ctx: *mut c_void, bytes: &[u8]) -> LlamaResult<Self> {
        if bytes.is_empty() {
            return Err(LlamaError::InvalidInput(
                "encoded image bytes must not be empty".to_string(),
            ));
        }
        // SAFETY: bytes points to bytes.len() readable bytes for the call.
        let ptr = unsafe { ffi::mtmd_bitmap_init_from_buf(ctx, bytes.as_ptr(), bytes.len()) };
        if ptr.is_null() {
            return Err(LlamaError::InvalidInput(
                "mtmd failed to decode encoded image bytes".to_string(),
            ));
        }
        Ok(Self { ptr })
    }

    pub fn width(&self) -> u32 {
        // SAFETY: ptr is a live bitmap handle.
        unsafe { ffi::mtmd_bitmap_width(self.ptr) }
    }

    pub fn height(&self) -> u32 {
        // SAFETY: ptr is a live bitmap handle.
        unsafe { ffi::mtmd_bitmap_height(self.ptr) }
    }

    pub fn n_bytes(&self) -> usize {
        // SAFETY: ptr is a live bitmap handle.
        unsafe { ffi::mtmd_bitmap_n_bytes(self.ptr) }
    }

    pub fn id(&self) -> Option<String> {
        // SAFETY: ptr is a live bitmap handle; C string is copied.
        let ptr = unsafe { ffi::mtmd_bitmap_id(self.ptr) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: mtmd returns a non-null NUL-terminated C string.
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .ok()
            .map(ToOwned::to_owned)
    }

    pub fn set_id(&mut self, id: &str) -> LlamaResult<()> {
        let c_id = ffi::cstring(id, "image id")?;
        // SAFETY: ptr is a live bitmap handle; c_id lives for the call.
        unsafe { ffi::mtmd_bitmap_set_id(self.ptr, &c_id) };
        Ok(())
    }

    #[inline]
    fn as_ptr(&self) -> *const c_void {
        self.ptr
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: pointer came from mtmd_bitmap_init_from_buf and is owned here.
            unsafe { ffi::mtmd_bitmap_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

unsafe impl Send for MtmdBitmap {}

/// Ordered summary of tokenized mtmd chunks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MtmdChunksSummary {
    pub total_chunks: usize,
    pub text_chunks: usize,
    pub image_chunks: usize,
    pub audio_chunks: usize,
    pub text_tokens: usize,
    pub image_tokens: usize,
    pub image_n_pos: usize,
    pub helper_total_tokens: usize,
    pub helper_total_n_pos: i32,
}

/// Owned mtmd input chunks produced by `mtmd_tokenize`.
pub struct MtmdInputChunks {
    ptr: *mut c_void,
}

impl MtmdInputChunks {
    pub fn empty() -> LlamaResult<Self> {
        // SAFETY: returns an owned pointer or null.
        let ptr = unsafe { ffi::mtmd_input_chunks_init() };
        if ptr.is_null() {
            return Err(LlamaError::Internal(
                "mtmd failed to allocate input chunks".to_string(),
            ));
        }
        Ok(Self { ptr })
    }

    pub fn tokenize(
        ctx: &MtmdContext,
        text: &str,
        add_special: bool,
        parse_special: bool,
        bitmaps: &[MtmdBitmap],
    ) -> LlamaResult<Self> {
        let chunks = Self::empty()?;
        let c_text = ffi::cstring(text, "mtmd prompt text")?;
        let mut bitmap_ptrs = bitmaps.iter().map(MtmdBitmap::as_ptr).collect::<Vec<_>>();

        // SAFETY: all pointers are live for the call; chunks owns output.
        let result = unsafe {
            ffi::mtmd_tokenize(
                ctx.as_ptr(),
                chunks.ptr,
                &c_text,
                add_special,
                parse_special,
                bitmap_ptrs.as_mut_ptr(),
                bitmap_ptrs.len(),
            )
        };
        if result != 0 {
            let detail = match result {
                -1 => "invalid arguments",
                1 => "number of bitmaps does not match media markers",
                2 => "image preprocessing failed",
                _ => "unknown",
            };
            return Err(LlamaError::Internal(format!(
                "mtmd_tokenize failed with error code {result} ({detail})"
            )));
        }

        Ok(chunks)
    }

    pub fn summary(&self) -> LlamaResult<MtmdChunksSummary> {
        // SAFETY: ptr is a live chunk list.
        let total_chunks = unsafe { ffi::mtmd_input_chunks_size(self.ptr) };
        let mut summary = MtmdChunksSummary {
            total_chunks,
            // SAFETY: ptr is a live chunk list.
            helper_total_tokens: unsafe { ffi::mtmd_helper_n_tokens(self.ptr) },
            // SAFETY: ptr is a live chunk list.
            helper_total_n_pos: unsafe { ffi::mtmd_helper_n_pos(self.ptr) },
            ..MtmdChunksSummary::default()
        };

        for idx in 0..total_chunks {
            // SAFETY: idx is in range of the reported chunk count.
            let chunk = unsafe { ffi::mtmd_input_chunks_get(self.ptr, idx) };
            if chunk.is_null() {
                return Err(LlamaError::Internal(format!(
                    "mtmd returned null chunk at index {idx}"
                )));
            }

            // SAFETY: chunk is non-null and live.
            match unsafe { ffi::mtmd_input_chunk_type(chunk) } {
                MTMD_INPUT_CHUNK_TYPE_TEXT => {
                    let mut n_tokens = 0usize;
                    // SAFETY: n_tokens is writable; chunk is a live text chunk.
                    let _ = unsafe { ffi::mtmd_input_chunk_tokens_text(chunk, &mut n_tokens) };
                    summary.text_chunks += 1;
                    summary.text_tokens += n_tokens;
                }
                MTMD_INPUT_CHUNK_TYPE_IMAGE => {
                    summary.image_chunks += 1;
                    // SAFETY: chunk is a live image chunk.
                    let image_tokens = unsafe { ffi::mtmd_input_chunk_tokens_image(chunk) };
                    if !image_tokens.is_null() {
                        // SAFETY: image_tokens is a live image-token handle.
                        let image_tokens_count =
                            unsafe { ffi::mtmd_image_tokens_n_tokens(image_tokens) };
                        let image_n_pos = unsafe { ffi::mtmd_image_tokens_n_pos(image_tokens) };
                        summary.image_tokens += image_tokens_count;
                        summary.image_n_pos += image_n_pos.max(0) as usize;
                    }
                }
                MTMD_INPUT_CHUNK_TYPE_AUDIO => {
                    summary.audio_chunks += 1;
                }
                other => {
                    return Err(LlamaError::Internal(format!(
                        "mtmd chunk at index {idx} has unknown type {other}"
                    )));
                }
            }
        }

        Ok(summary)
    }

    pub fn validate_for_generation(&self) -> LlamaResult<MtmdChunksSummary> {
        let summary = self.summary()?;

        if summary.total_chunks == 0 {
            return Err(LlamaError::Internal(
                "mtmd produced no prompt chunks for generation".to_string(),
            ));
        }
        if summary.helper_total_tokens == 0 {
            return Err(LlamaError::Internal(
                "mtmd prompt chunks contain no tokens for generation".to_string(),
            ));
        }
        if summary.image_chunks > 0 && (summary.image_tokens == 0 || summary.image_n_pos == 0) {
            return Err(LlamaError::Internal(
                "mtmd image chunks contain no image token positions for generation".to_string(),
            ));
        }

        Ok(summary)
    }

    #[inline]
    fn as_ptr(&self) -> *const c_void {
        self.ptr
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: pointer came from mtmd_input_chunks_init and is owned here.
            unsafe { ffi::mtmd_input_chunks_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

unsafe impl Send for MtmdInputChunks {}

/// Prefill mtmd text/image chunks into the llama context.
pub fn helper_eval_chunks(
    ctx: &MtmdContext,
    lctx: &LlamaContext,
    chunks: &MtmdInputChunks,
    n_past: i32,
    seq_id: i32,
    n_batch: usize,
    logits_last: bool,
) -> LlamaResult<i32> {
    let mut new_n_past = n_past;
    // SAFETY: caller serializes access to lctx; all handles are live.
    let result = unsafe {
        ffi::mtmd_helper_eval_chunks(
            ctx.as_ptr(),
            lctx.as_ptr(),
            chunks.as_ptr(),
            n_past,
            seq_id,
            n_batch,
            logits_last,
            &mut new_n_past,
        )
    };
    if result != 0 {
        return Err(LlamaError::Internal(format!(
            "mtmd_helper_eval_chunks failed with error code {result}"
        )));
    }
    Ok(new_n_past)
}
