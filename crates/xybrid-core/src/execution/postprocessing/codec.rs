//! Codec-based TTS postprocessing.
//!
//! Extracts speech token IDs from LLM output text and decodes them
//! to audio via an ONNX neural codec decoder (e.g., NeuCodec).

use std::path::Path;

use log::debug;
use ndarray::Array3;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;

use crate::execution::types::ExecutorResult;
use crate::runtime_adapter::AdapterError;

/// Extract integer speech token IDs from LLM output text using a regex pattern.
///
/// The default pattern `<\|speech_(\d+)\|>` matches tokens like `<|speech_42|>`.
pub fn extract_speech_tokens(text: &str, token_pattern: &str) -> ExecutorResult<Vec<i32>> {
    let re = regex::Regex::new(token_pattern).map_err(|e| {
        AdapterError::InvalidInput(format!(
            "Invalid speech token pattern '{}': {}",
            token_pattern, e
        ))
    })?;

    let tokens: Vec<i32> = re
        .captures_iter(text)
        .filter_map(|cap| cap.get(1)?.as_str().parse().ok())
        .collect();

    if tokens.is_empty() {
        return Err(AdapterError::InvalidInput(
            "No speech tokens found in LLM output".to_string(),
        ));
    }

    debug!(
        target: "xybrid_core",
        "Extracted {} speech tokens from LLM output",
        tokens.len()
    );

    Ok(tokens)
}

/// Create an ONNX codec decoder session from a model file.
pub fn create_codec_session(decoder_model_path: &Path) -> ExecutorResult<Session> {
    Session::builder()
        .map_err(|e| AdapterError::RuntimeError(format!("ONNX session builder failed: {}", e)))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| AdapterError::RuntimeError(format!("Set optimization level failed: {}", e)))?
        .commit_from_file(decoder_model_path)
        .map_err(|e| {
            AdapterError::RuntimeError(format!(
                "Failed to load codec decoder {:?}: {}",
                decoder_model_path, e
            ))
        })
}

/// Decode raw speech token IDs to f32 audio samples using an existing decoder session.
pub fn decode_tokens_to_samples(
    session: &mut Session,
    tokens: &[i32],
    sample_rate: u32,
    apply_postprocessing: bool,
) -> ExecutorResult<Vec<f32>> {
    let n = tokens.len();
    let tensor = Array3::from_shape_vec((1, 1, n), tokens.to_vec())
        .map_err(|e| AdapterError::RuntimeError(format!("Failed to create input tensor: {}", e)))?;
    let input_value = Value::from_array(tensor)
        .map_err(|e| AdapterError::RuntimeError(format!("Failed to create ONNX value: {}", e)))?;

    let outputs = session
        .run(ort::inputs!["codes" => input_value])
        .map_err(|e| {
            AdapterError::InferenceFailed(format!("Codec decoder inference failed: {}", e))
        })?;

    let output_value = outputs
        .iter()
        .next()
        .ok_or_else(|| {
            AdapterError::InferenceFailed("Codec decoder produced no outputs".to_string())
        })?
        .1;
    let (_, waveform_view) = output_value.try_extract_tensor::<f32>().map_err(|e| {
        AdapterError::InferenceFailed(format!("Failed to extract waveform tensor: {}", e))
    })?;
    let waveform: Vec<f32> = waveform_view.to_vec();

    debug!(
        target: "xybrid_core",
        "Codec decoder produced {} audio samples at {}Hz",
        waveform.len(),
        sample_rate
    );

    if apply_postprocessing {
        Ok(crate::phonemizer::postprocess_tts_audio(
            &waveform,
            sample_rate,
        ))
    } else {
        Ok(waveform)
    }
}

/// Decode speech tokens to PCM16 WAV audio using an ONNX codec decoder.
///
/// Feeds integer tokens as a `[1, 1, N]` i32 tensor to the decoder,
/// receives a `[1, 1, T]` f32 waveform, and converts to WAV bytes.
pub fn codec_decode_step(
    text: &str,
    decoder_model_path: &Path,
    token_pattern: &str,
    sample_rate: u32,
    apply_postprocessing: bool,
) -> ExecutorResult<Vec<u8>> {
    let tokens = extract_speech_tokens(text, token_pattern)?;
    let mut session = create_codec_session(decoder_model_path)?;
    let samples =
        decode_tokens_to_samples(&mut session, &tokens, sample_rate, apply_postprocessing)?;
    Ok(crate::audio::samples_to_wav(&samples, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_speech_tokens_default_pattern() {
        let text = "<|speech_1|><|speech_2|><|speech_3|>";
        let pattern = r"<\|speech_(\d+)\|>";
        let tokens = extract_speech_tokens(text, pattern).unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_extract_speech_tokens_with_surrounding_text() {
        let text = "some preamble <|speech_100|> middle <|speech_200|> end";
        let pattern = r"<\|speech_(\d+)\|>";
        let tokens = extract_speech_tokens(text, pattern).unwrap();
        assert_eq!(tokens, vec![100, 200]);
    }

    #[test]
    fn test_extract_speech_tokens_empty_match() {
        let text = "no speech tokens here";
        let pattern = r"<\|speech_(\d+)\|>";
        let result = extract_speech_tokens(text, pattern);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_speech_tokens_invalid_regex() {
        let text = "test";
        let pattern = r"[invalid";
        let result = extract_speech_tokens(text, pattern);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_speech_tokens_no_capture_group() {
        let text = "<|speech_42|>";
        let pattern = r"<\|speech_\d+\|>";
        let result = extract_speech_tokens(text, pattern);
        assert!(result.is_err());
    }
}
