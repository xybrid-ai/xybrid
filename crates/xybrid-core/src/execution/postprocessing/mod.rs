//! Postprocessing step implementations.
//!
//! This module contains all postprocessing operations that transform model outputs
//! into final results:
//!
//! | Module | Operations |
//! |--------|-----------|
//! | [`decode`] | `CTCDecode`, `BPEDecode`, `WhisperDecode` |
//! | [`tensor_ops`] | `Argmax`, `Softmax`, `TopK`, `TemperatureSample`, `Threshold`, `MeanPool`, `Denormalize` |
//! | [`audio`] | `TTSAudioEncode` |

pub mod audio;
pub mod codec;
pub mod decode;
pub mod tensor_ops;

use super::path::resolve_file_path;
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
            temperature,
            top_k,
            top_p,
        } => tensor_ops::temperature_sample_step(data, *temperature, *top_k, *top_p),

        PostprocessingStep::Denormalize { mean, std } => {
            tensor_ops::denormalize_step(data, mean, std)
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

        PostprocessingStep::CodecDecode { .. } => Err(
            crate::runtime_adapter::AdapterError::InvalidInput(
                "CodecDecode is handled by CodecTtsStrategy, not the generic postprocessing dispatcher".into(),
            ),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};
    use std::collections::HashMap;

    #[test]
    fn temperature_sample_dispatches_to_class_id() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 2.0, 0.5]).expect("valid shape");
        let data = RawOutputs::TensorMap(HashMap::from([("logits".to_string(), tensor)]));
        let step = PostprocessingStep::TemperatureSample {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        };

        let result = apply_postprocessing_step(&step, data, "").expect("sampling should succeed");

        assert!(matches!(result, RawOutputs::ClassId(1)));
    }
}
