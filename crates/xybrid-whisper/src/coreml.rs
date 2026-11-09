//! Pairing a loaded GGML model with its Core ML encoder (iOS).
//!
//! [`LoadScope`] declares the loaded context's dimensions (never re-read from
//! the file) and the encoder path ([`encoder_path`]) around
//! `whisper_init_state`, so xybrid's bridge never copies output that doesn't
//! fit. Builds without the bridge get no-op stand-ins.

/// The encoder path whisper.cpp derives from `model` (drop the extension and
/// any `-qX_Y` tag, append `-encoder.mlmodelc`). Ported byte for byte, quirks
/// included, because the bridge only serves an init for exactly this path.
#[cfg(any(test, all(feature = "coreml", target_os = "ios")))]
pub(crate) fn encoder_path(model: &str) -> String {
    let mut base = model.rfind('.').map_or(model, |dot| &model[..dot]);
    if let Some(dash) = base.rfind('-') {
        let tag = &base.as_bytes()[dash..];
        if tag.len() == 5 && tag[1] == b'q' && tag[3] == b'_' {
            base = &base[..dash];
        }
    }
    format!("{base}-encoder.mlmodelc")
}

#[cfg(all(feature = "coreml", target_os = "ios"))]
mod bridge {
    use super::encoder_path;
    use std::ffi::CString;
    use std::marker::PhantomData;
    use std::path::Path;
    use std::ptr::NonNull;
    use xybrid_whisper_sys::bindings as whisper;
    use xybrid_whisper_sys::coreml as sys;

    /// Brackets one `whisper_init_state` so the bridge knows what the encoder
    /// must match. Begin right before the init and `finish` right after, on
    /// the same thread; dropping it unfinished still closes the declaration.
    pub(crate) struct LoadScope {
        encoder: String,
        /// The context's encoder frames per window, as declared.
        n_audio_ctx: i32,
        /// Whether a declaration is open and must be closed.
        open: bool,
        _same_thread: PhantomData<*const ()>,
    }

    impl LoadScope {
        /// Declares `ctx`'s dimensions if `model` has an encoder beside it.
        pub(crate) fn begin(ctx: NonNull<whisper::whisper_context>, model: &str) -> Self {
            let encoder = encoder_path(model);
            // SAFETY: `ctx` is live; the accessor only reads loaded hyper-parameters.
            let n_audio_ctx = unsafe { whisper::whisper_model_n_audio_ctx(ctx.as_ptr()) };
            let declared = Path::new(&encoder).exists()
                && match CString::new(encoder.as_str()) {
                    Ok(c_encoder) => {
                        // SAFETY: as above; the bridge copies the path before returning.
                        unsafe {
                            sys::xybrid_whisper_coreml_begin_load(
                                c_encoder.as_ptr(),
                                whisper::whisper_model_n_mels(ctx.as_ptr()),
                                n_audio_ctx,
                                whisper::whisper_model_n_audio_state(ctx.as_ptr()),
                            );
                        }
                        true
                    }
                    Err(_) => false,
                };
            Self {
                encoder,
                n_audio_ctx,
                open: declared,
                _same_thread: PhantomData,
            }
        }

        /// The full-window `audio_ctx` if Core ML runs the encoder, else `None`.
        pub(crate) fn finish(mut self) -> Option<u32> {
            if !self.open {
                return None;
            }
            self.open = false;
            // SAFETY: closes the declaration this scope opened on this thread.
            let status = unsafe { sys::xybrid_whisper_coreml_end_load() };
            match status {
                sys::XYBRID_WHISPER_COREML_ACTIVE => {
                    tracing::debug!(encoder = %self.encoder, "whisper encoder runs on Core ML");
                    u32::try_from(self.n_audio_ctx).ok()
                }
                sys::XYBRID_WHISPER_COREML_MISMATCH => {
                    tracing::warn!(
                        encoder = %self.encoder,
                        "Core ML encoder does not match its model; using the ggml encoder"
                    );
                    None
                }
                sys::XYBRID_WHISPER_COREML_UNSUPPORTED_OS => {
                    tracing::debug!(
                        encoder = %self.encoder,
                        "Core ML encoder needs iOS 16; using the ggml encoder"
                    );
                    None
                }
                sys::XYBRID_WHISPER_COREML_LOAD_FAILED => {
                    tracing::warn!(
                        encoder = %self.encoder,
                        "Core ML could not load the encoder; using the ggml encoder"
                    );
                    None
                }
                _ => None,
            }
        }
    }

    impl Drop for LoadScope {
        fn drop(&mut self) {
            if self.open {
                // SAFETY: as in `finish`; keeps the bracket balanced if abandoned.
                unsafe { sys::xybrid_whisper_coreml_end_load() };
            }
        }
    }

    /// Failed Core ML encoder runs on this thread; compare before and after a call.
    pub(crate) fn failure_count() -> u64 {
        // SAFETY: no arguments; reads this thread's counter.
        unsafe { sys::xybrid_whisper_coreml_failure_count() }
    }
}

#[cfg(all(feature = "coreml", target_os = "ios"))]
pub(crate) use bridge::{failure_count, LoadScope};

/// Builds without the Core ML encoder: nothing to declare, nothing can fail.
#[cfg(all(feature = "bindings", not(all(feature = "coreml", target_os = "ios"))))]
pub(crate) struct LoadScope;

#[cfg(all(feature = "bindings", not(all(feature = "coreml", target_os = "ios"))))]
impl LoadScope {
    pub(crate) fn begin(
        _ctx: std::ptr::NonNull<xybrid_whisper_sys::bindings::whisper_context>,
        _model: &str,
    ) -> Self {
        Self
    }

    pub(crate) fn finish(self) -> Option<u32> {
        None
    }
}

#[cfg(all(feature = "bindings", not(all(feature = "coreml", target_os = "ios"))))]
pub(crate) fn failure_count() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::encoder_path;

    #[test]
    fn a_quantized_model_names_its_unquantized_encoder() {
        assert_eq!(
            encoder_path("/models/ggml-base.en-q5_1.bin"),
            "/models/ggml-base.en-encoder.mlmodelc"
        );
        assert_eq!(
            encoder_path("ggml-large-v3-turbo-q5_0.bin"),
            "ggml-large-v3-turbo-encoder.mlmodelc"
        );
    }

    #[test]
    fn an_unquantized_model_keeps_its_name() {
        assert_eq!(encoder_path("ggml-tiny.bin"), "ggml-tiny-encoder.mlmodelc");
        // `-turbo` is not a quantization tag.
        assert_eq!(
            encoder_path("ggml-large-v3-turbo.bin"),
            "ggml-large-v3-turbo-encoder.mlmodelc"
        );
    }

    #[test]
    fn the_extension_cut_matches_whisper_cpp_quirks() {
        // No extension: the last dot in the path wins, as upstream does.
        assert_eq!(encoder_path("/a.b/model"), "/a-encoder.mlmodelc");
        assert_eq!(encoder_path("model"), "model-encoder.mlmodelc");
    }
}
