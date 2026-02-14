//! Postprocessing step implementations.
//!
//! This module contains all postprocessing operations that transform model outputs
//! into final results:
//!
//! | Module | Operations |
//! |--------|-----------|
//! | [`decode`] | `CTCDecode`, `BPEDecode`, `WhisperDecode` |
//! | [`tensor_ops`] | `Argmax`, `Softmax`, `TopK`, `Threshold`, `MeanPool` |
//! | [`audio`] | `TTSAudioEncode` |

pub mod audio;
pub mod decode;
pub mod tensor_ops;

use super::types::{ExecutorResult, RawOutputs};
use crate::execution::template::PostprocessingStep;

/// Apply a postprocessing step to data.
///
/// This is the main dispatcher that routes to the appropriate step implementation.
pub fn apply_postprocessing_step(
    step: &PostprocessingStep,
    data: RawOutputs,
    base_path: &str,
) -> ExecutorResult<RawOutputs> {
    match step {
        PostprocessingStep::CTCDecode {
            vocab_file,
            blank_index,
        } => {
            let vocab_path = resolve_file_path(base_path, vocab_file);
            decode::ctc_decode_step(data, &vocab_path, *blank_index)
        }

        PostprocessingStep::BPEDecode { vocab_file } => {
            let vocab_path = resolve_file_path(base_path, vocab_file);
            decode::bpe_decode_step(data, &vocab_path)
        }

        PostprocessingStep::WhisperDecode { tokenizer_file } => {
            let tokenizer_path = resolve_file_path(base_path, tokenizer_file);
            decode::whisper_decode_step(data, &tokenizer_path)
        }

        PostprocessingStep::Argmax { dim } => tensor_ops::argmax_step(data, *dim),

        PostprocessingStep::Softmax { dim } => tensor_ops::softmax_step(data, *dim),

        PostprocessingStep::TopK { k, dim } => tensor_ops::topk_step(data, *k, *dim),

        PostprocessingStep::Threshold {
            threshold,
            return_indices,
        } => tensor_ops::threshold_step(data, *threshold, *return_indices),

        PostprocessingStep::MeanPool { dim } => tensor_ops::meanpool_step(data, *dim),

        PostprocessingStep::TemperatureSample {
            temperature: _,
            top_k: _,
            top_p: _,
        } => {
            // TODO: Implement temperature sampling
            Ok(data)
        }

        PostprocessingStep::Denormalize { mean: _, std: _ } => {
            // TODO: Implement denormalization
            Ok(data)
        }

        PostprocessingStep::TTSAudioEncode {
            sample_rate,
            apply_postprocessing,
            trim_trailing_silence,
        } => audio::tts_audio_encode_step(
            data,
            *sample_rate,
            *apply_postprocessing,
            *trim_trailing_silence,
        ),
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
