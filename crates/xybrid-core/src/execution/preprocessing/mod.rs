//! Preprocessing step implementations.
//!
//! This module contains all preprocessing operations that transform input data
//! before model execution:
//!
//! | Module | Operations |
//! |--------|-----------|
//! | [`audio`] | `AudioDecode`, `MelSpectrogram` |
//! | [`image`] | `Resize`, `CenterCrop` |
//! | [`text`] | `Tokenize`, `Phonemize` |
//! | [`tensor`] | `Normalize`, `Reshape` |

pub mod audio;
pub mod backends;
pub mod image;
pub mod tensor;
pub mod text;

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

        PreprocessingStep::Tokenize {
            vocab_file,
            tokenizer_type,
            max_length,
        } => {
            let vocab_path = resolve_file_path(base_path, vocab_file);
            text::tokenize_step(data, &vocab_path, tokenizer_type, *max_length)
        }

        PreprocessingStep::Phonemize {
            tokens_file,
            backend,
            dict_file,
            language,
            add_padding,
            normalize_text,
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
            )
        }

        PreprocessingStep::Normalize { mean, std } => tensor::normalize_step(data, mean, std),

        PreprocessingStep::Reshape { shape } => tensor::reshape_step(data, shape),

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

/// Resolve a file path relative to base_path.
fn resolve_file_path(base_path: &str, file: &str) -> String {
    if base_path.is_empty() {
        file.to_string()
    } else {
        std::path::Path::new(base_path)
            .join(file)
            .to_string_lossy()
            .to_string()
    }
}
