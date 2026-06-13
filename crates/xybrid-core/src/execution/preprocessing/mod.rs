//! Preprocessing step implementations.
//!
//! This module contains all preprocessing operations that transform input data
//! before model execution:
//!
//! | Module | Operations |
//! |--------|-----------|
//! | [`audio`] | `AudioDecode`, `MelSpectrogram` |
//! | [`image`] | `ImageDecode`, `Resize`, `CenterCrop` |
//! | [`text`] | `Tokenize`, `Phonemize` |
//! | [`tensor`] | `Normalize`, `Reshape` |

pub mod audio;
pub mod backends;
pub mod image;
pub mod tensor;
pub mod text;

use super::path::resolve_file_path;
use super::types::{ExecutorResult, PreprocessedData};
use crate::execution::template::PreprocessingStep;
use crate::ir::Envelope;

/// Apply a preprocessing step to data.
///
/// This is the main dispatcher that routes to the appropriate step implementation.
pub fn apply_preprocessing_step(
    step: &PreprocessingStep,
    data: PreprocessedData,
    input_envelope: &Envelope,
    base_path: &str,
) -> ExecutorResult<PreprocessedData> {
    match step {
        PreprocessingStep::MelSpectrogram {
            preset,
            n_mels,
            sample_rate,
            fft_size,
            hop_length,
            mel_scale,
            max_frames,
        } => audio::mel_spectrogram_step(
            data,
            preset.as_deref(),
            *n_mels,
            *sample_rate,
            *fft_size,
            *hop_length,
            *mel_scale,
            *max_frames,
        ),

        PreprocessingStep::AudioDecode {
            sample_rate,
            channels,
        } => audio::decode_audio_step(data, input_envelope, *sample_rate, *channels),

        #[cfg(feature = "vision")]
        PreprocessingStep::ImageDecode { channels, layout } => {
            image::image_decode_step(data, *channels, *layout)
        }

        #[cfg(feature = "vision")]
        PreprocessingStep::ImageIngress { channels, layout } => {
            image::image_ingress_step(data, *channels, *layout)
        }

        PreprocessingStep::Tokenize {
            vocab_file,
            tokenizer_type,
            max_length,
        } => {
            let vocab_path = resolve_file_path(base_path, vocab_file);
            text::tokenize_step(data, &vocab_path, tokenizer_type, *max_length)
        }

        PreprocessingStep::PhonemeRaw { .. } => {
            Err(crate::runtime_adapter::AdapterError::InvalidInput(
                "PhonemeRaw is handled by CodecTtsStrategy, not the generic preprocessing dispatcher".into(),
            ))
        }

        PreprocessingStep::Phonemize {
            tokens_file,
            backend,
            dict_file,
            language,
            add_padding,
            normalize_text,
            silence_tokens,
        } => {
            let tokens_path = resolve_file_path(base_path, tokens_file);
            let dict_path = dict_file.as_ref().map(|p| resolve_file_path(base_path, p));
            text::phonemize_step(
                data,
                &tokens_path,
                backend,
                dict_path.as_deref(),
                language.as_deref(),
                *add_padding,
                *normalize_text,
                silence_tokens.unwrap_or(0),
            )
        }

        PreprocessingStep::Normalize { mean, std } => tensor::normalize_step(data, mean, std),

        PreprocessingStep::Reshape { shape } => tensor::reshape_step(data, shape),

        #[cfg(feature = "vision")]
        PreprocessingStep::ImageResize {
            width,
            height,
            mode,
            interpolation,
            fill,
            layout,
        } => image::image_resize_step(
            data,
            *width,
            *height,
            *mode,
            interpolation,
            fill,
            *layout,
        ),

        #[cfg(feature = "vision")]
        PreprocessingStep::ImageNormalize { preset, layout } => {
            image::image_normalize_step(data, preset, *layout)
        }

        PreprocessingStep::CenterCrop { width, height } => {
            image::center_crop_step(data, *width, *height)
        }

        PreprocessingStep::Resize {
            width,
            height,
            interpolation,
        } => image::resize_step(data, *width, *height, interpolation),
    }
}
