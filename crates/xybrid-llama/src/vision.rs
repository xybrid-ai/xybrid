//! Safe mtmd wrappers for llama.cpp vision-language models.
//!
//! This module owns the mtmd FFI boundary after the llama.cpp crate split.
//! `xybrid-core` should only orchestrate prompts/images and call these safe
//! handles; it must not reach into `llama-cpp-sys` directly.

use std::ffi::CStr;
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
/// The context owns projector-side state that references the loaded llama
/// model, so callers must drop it before dropping the text model.
pub struct MtmdContext {
    ptr: *mut ffi::MtmdContextRaw,
}

impl MtmdContext {
    /// Test-only null handle for wrapper-state tests that never cross FFI.
    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn test_stub() -> Self {
        Self {
            ptr: ptr::null_mut(),
        }
    }

    /// Load an mtmd projector for `model` from `path`.
    pub fn from_file(
        path: &str,
        model: &LlamaModel,
        use_gpu: bool,
        warmup: bool,
        n_threads: usize,
        flash_attn: bool,
    ) -> LlamaResult<Self> {
        let c_path = ffi::cstring(path, "mtmd projector path")?;
        // SAFETY: c_path lives for the call; model.as_ptr() is a live
        // llama_model handle. Null return is mapped below.
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
    pub(crate) fn as_ptr(&self) -> *mut ffi::MtmdContextRaw {
        self.ptr
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from mtmd_init_from_file and Drop runs once.
            unsafe { ffi::mtmd_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

unsafe impl Send for MtmdContext {}

/// Opaque handle to an mtmd decoded bitmap.
pub struct MtmdBitmap {
    ptr: *mut ffi::MtmdBitmapRaw,
}

impl MtmdBitmap {
    /// Decode encoded image bytes into an mtmd bitmap.
    pub fn from_encoded_bytes(ctx: &MtmdContext, bytes: &[u8]) -> LlamaResult<Self> {
        Self::from_encoded_bytes_with_context(ctx.as_ptr(), bytes)
    }

    /// Decode encoded image bytes without an mtmd context.
    ///
    /// Valid only for image bytes; audio inputs need a context so mtmd can
    /// read the model's expected sample rate.
    pub fn from_encoded_image_bytes(bytes: &[u8]) -> LlamaResult<Self> {
        Self::from_encoded_bytes_with_context(ptr::null_mut(), bytes)
    }

    fn from_encoded_bytes_with_context(
        ctx: *mut ffi::MtmdContextRaw,
        bytes: &[u8],
    ) -> LlamaResult<Self> {
        if bytes.is_empty() {
            return Err(LlamaError::InvalidInput(
                "encoded image bytes must not be empty".to_string(),
            ));
        }

        // SAFETY: bytes pointer is valid for bytes.len(); ctx is either a
        // live mtmd context or null for image-only decode.
        let ptr = unsafe { ffi::mtmd_bitmap_init_from_buf(ctx, bytes.as_ptr(), bytes.len()) };
        if ptr.is_null() {
            return Err(LlamaError::InvalidInput(
                "mtmd failed to decode encoded image bytes".to_string(),
            ));
        }

        Ok(Self { ptr })
    }

    /// Build an mtmd bitmap directly from tightly-packed RGB pixels.
    ///
    /// `rgb` must contain exactly `width * height * 3` bytes in RGBRGB...
    /// order (no row-stride padding, no alpha). This is the raw-frame path
    /// that lets pre-decoded camera frames skip a per-frame JPEG round-trip;
    /// clip (inside mtmd) still does its own resize and normalization, so
    /// callers pass full-resolution pixels.
    ///
    /// The mtmd packed-RGB constructor does not consult the projector
    /// context, but `_ctx` is accepted to mirror [`Self::from_encoded_bytes`]
    /// and keep call sites uniform.
    pub fn from_raw_rgb(
        _ctx: &MtmdContext,
        width: u32,
        height: u32,
        rgb: &[u8],
    ) -> LlamaResult<Self> {
        Self::from_raw_rgb_inner(width, height, rgb)
    }

    fn from_raw_rgb_inner(width: u32, height: u32, rgb: &[u8]) -> LlamaResult<Self> {
        if width == 0 || height == 0 {
            return Err(LlamaError::InvalidInput(
                "raw RGB bitmap dimensions must be non-zero".to_string(),
            ));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| LlamaError::InvalidInput("raw RGB bitmap size overflow".to_string()))?;
        if rgb.len() != expected {
            return Err(LlamaError::InvalidInput(format!(
                "raw RGB bitmap requires exactly {expected} packed bytes for {width}x{height}, got {}",
                rgb.len()
            )));
        }

        // SAFETY: rgb is exactly width * height * 3 tightly-packed RGB bytes,
        // matching the contract of the packed-RGB ctor; the pointer is valid
        // for that length for the duration of the call.
        let ptr = unsafe { ffi::mtmd_bitmap_init_rgb(width, height, rgb.as_ptr()) };
        if ptr.is_null() {
            return Err(LlamaError::InvalidInput(
                "mtmd failed to build a bitmap from packed RGB pixels".to_string(),
            ));
        }

        Ok(Self { ptr })
    }

    pub fn width(&self) -> u32 {
        // SAFETY: self.ptr is live.
        unsafe { ffi::mtmd_bitmap_get_nx(self.ptr) }
    }

    pub fn height(&self) -> u32 {
        // SAFETY: self.ptr is live.
        unsafe { ffi::mtmd_bitmap_get_ny(self.ptr) }
    }

    pub fn n_bytes(&self) -> usize {
        // SAFETY: self.ptr is live.
        unsafe { ffi::mtmd_bitmap_get_n_bytes(self.ptr) }
    }

    pub fn id(&self) -> Option<String> {
        // SAFETY: self.ptr is live; C string is owned by mtmd.
        let ptr = unsafe { ffi::mtmd_bitmap_get_id(self.ptr) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: mtmd returned a null-terminated string pointer valid for
        // the bitmap lifetime.
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .ok()
            .map(ToOwned::to_owned)
    }

    pub fn set_id(&mut self, id: &str) -> LlamaResult<()> {
        let c_id = ffi::cstring(id, "image id")?;
        // SAFETY: self.ptr is live; c_id lives for the call.
        unsafe { ffi::mtmd_bitmap_set_id(self.ptr, &c_id) };
        Ok(())
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const ffi::MtmdBitmapRaw {
        self.ptr
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from mtmd_bitmap_init_from_buf and Drop runs once.
            unsafe { ffi::mtmd_bitmap_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

unsafe impl Send for MtmdBitmap {}

/// Summary of the mtmd chunk list produced for a multimodal prompt.
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

/// Owned mtmd input chunks.
pub struct MtmdInputChunks {
    ptr: *mut ffi::MtmdInputChunksRaw,
}

impl MtmdInputChunks {
    pub fn empty() -> LlamaResult<Self> {
        // SAFETY: allocates an owned chunk container.
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

        // SAFETY: ctx/chunks are live; c_text and bitmap_ptrs live for
        // the call; n_bitmaps matches bitmap_ptrs.len().
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
        // SAFETY: self.ptr is live.
        let total_chunks = unsafe { ffi::mtmd_input_chunks_size(self.ptr) };
        let mut summary = MtmdChunksSummary {
            total_chunks,
            // SAFETY: self.ptr is live.
            helper_total_tokens: unsafe { ffi::mtmd_helper_get_n_tokens(self.ptr) },
            // SAFETY: self.ptr is live.
            helper_total_n_pos: unsafe { ffi::mtmd_helper_get_n_pos(self.ptr) },
            ..MtmdChunksSummary::default()
        };

        for idx in 0..total_chunks {
            // SAFETY: self.ptr is live and idx < total_chunks.
            let chunk = unsafe { ffi::mtmd_input_chunks_get(self.ptr, idx) };
            if chunk.is_null() {
                return Err(LlamaError::Internal(format!(
                    "mtmd returned null chunk at index {idx}"
                )));
            }

            // SAFETY: chunk is non-null and live for self's lifetime.
            match unsafe { ffi::mtmd_input_chunk_get_type(chunk) } {
                MTMD_INPUT_CHUNK_TYPE_TEXT => {
                    let mut n_tokens = 0usize;
                    // SAFETY: chunk is a text chunk; n_tokens pointer is writable.
                    let tokens =
                        unsafe { ffi::mtmd_input_chunk_get_tokens_text(chunk, &mut n_tokens) };
                    if tokens.is_null() || n_tokens == 0 {
                        return Err(LlamaError::Internal(format!(
                            "mtmd text chunk at index {idx} has no tokens"
                        )));
                    }
                    summary.text_chunks += 1;
                    summary.text_tokens += n_tokens;
                }
                MTMD_INPUT_CHUNK_TYPE_IMAGE => {
                    // SAFETY: chunk is an image chunk.
                    let image_tokens = unsafe { ffi::mtmd_input_chunk_get_tokens_image(chunk) };
                    if image_tokens.is_null() {
                        return Err(LlamaError::Internal(format!(
                            "mtmd image chunk at index {idx} has no image tokens"
                        )));
                    }

                    // SAFETY: chunk/image_tokens are live.
                    let chunk_tokens = unsafe { ffi::mtmd_input_chunk_get_n_tokens(chunk) };
                    let image_tokens_count =
                        unsafe { ffi::mtmd_image_tokens_get_n_tokens(image_tokens) };
                    if chunk_tokens == 0 || image_tokens_count == 0 {
                        return Err(LlamaError::Internal(format!(
                            "mtmd image chunk at index {idx} has zero tokens"
                        )));
                    }

                    // SAFETY: chunk/image_tokens are live.
                    let chunk_n_pos = unsafe { ffi::mtmd_input_chunk_get_n_pos(chunk) };
                    let image_n_pos = unsafe { ffi::mtmd_image_tokens_get_n_pos(image_tokens) };
                    if chunk_n_pos <= 0 || image_n_pos <= 0 {
                        return Err(LlamaError::Internal(format!(
                            "mtmd image chunk at index {idx} has no decoder positions"
                        )));
                    }

                    // SAFETY: image_tokens_count > 0, so last index is valid.
                    let last_pos = unsafe {
                        ffi::mtmd_image_tokens_get_decoder_pos(
                            image_tokens,
                            0,
                            image_tokens_count - 1,
                        )
                    };
                    if last_pos.x == 0 && last_pos.y == 0 && image_tokens_count > 1 {
                        return Err(LlamaError::Internal(format!(
                            "mtmd image chunk at index {idx} lacks spatial decoder metadata"
                        )));
                    }

                    summary.image_chunks += 1;
                    summary.image_tokens += image_tokens_count;
                    summary.image_n_pos += image_n_pos as usize;
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

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const ffi::MtmdInputChunksRaw {
        self.ptr
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from mtmd_input_chunks_init and Drop runs once.
            unsafe { ffi::mtmd_input_chunks_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

unsafe impl Send for MtmdInputChunks {}

/// Evaluate mtmd text/image chunks into `lctx`, returning the new `n_past`.
///
/// Upstream documents `mtmd_helper_eval_chunks()` as not thread-safe, so
/// callers must hold the serialized llama context lock around this call.
#[allow(clippy::too_many_arguments)]
pub fn mtmd_helper_eval_chunks(
    ctx: &MtmdContext,
    lctx: &LlamaContext,
    chunks: &MtmdInputChunks,
    n_past: i32,
    seq_id: i32,
    n_batch: usize,
    logits_last: bool,
) -> LlamaResult<i32> {
    let mut new_n_past = n_past;
    // SAFETY: handles are live; new_n_past is writable for the call.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Length validation runs before any FFI call, so a wrong-sized buffer is
    /// rejected with a null (test-stub) context without crossing the boundary.
    #[test]
    fn from_raw_rgb_rejects_wrong_length_buffer() {
        let ctx = MtmdContext::test_stub();
        // 4x4 RGB needs 48 bytes; supply 47 to force the length guard.
        let rgb = vec![0_u8; 47];
        match MtmdBitmap::from_raw_rgb(&ctx, 4, 4, &rgb) {
            Err(LlamaError::InvalidInput(message)) => {
                assert!(
                    message.contains("48") && message.contains("47"),
                    "error should report expected/actual byte counts, got: {message}"
                );
            }
            Err(other) => panic!("expected InvalidInput, got {other:?}"),
            Ok(_) => panic!("undersized packed RGB buffer must be rejected"),
        }
    }

    /// Zero dimensions are rejected before the size computation and FFI call.
    #[test]
    fn from_raw_rgb_rejects_zero_dimensions() {
        let ctx = MtmdContext::test_stub();
        assert!(matches!(
            MtmdBitmap::from_raw_rgb(&ctx, 0, 4, &[]),
            Err(LlamaError::InvalidInput(_))
        ));
        assert!(matches!(
            MtmdBitmap::from_raw_rgb(&ctx, 4, 0, &[]),
            Err(LlamaError::InvalidInput(_))
        ));
    }

    /// Build a real mtmd bitmap from a small synthetic packed-RGB buffer and
    /// confirm the dimensions/byte-count accessors agree. This exercises the
    /// `mtmd_bitmap_init_rgb_c` shim end to end, so it only runs when the
    /// llama.cpp runtime is linked (`vision` feature). The packed-RGB ctor is
    /// context-free, so a null test-stub context is sufficient — no mmproj.
    #[cfg(feature = "vision")]
    #[test]
    fn from_raw_rgb_constructs_bitmap_with_expected_geometry() {
        let ctx = MtmdContext::test_stub();
        let width = 4_u32;
        let height = 4_u32;
        let rgb = (0..(width * height * 3))
            .map(|i| (i % 256) as u8)
            .collect::<Vec<u8>>();

        let bitmap = MtmdBitmap::from_raw_rgb(&ctx, width, height, &rgb)
            .expect("packed RGB bitmap constructs from a valid buffer");

        assert_eq!(bitmap.width(), width);
        assert_eq!(bitmap.height(), height);
        assert_eq!(bitmap.n_bytes(), (width * height * 3) as usize);
    }
}
