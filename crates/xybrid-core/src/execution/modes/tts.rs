//! TTS execution mode.
//!
//! This module handles execution of TTS models that require phoneme IDs,
//! voice embeddings, and speed parameters.
//!
//! Input mapping is based on ONNX metadata (dtype + shape), not input names,
//! so any TTS model with the standard input signature works without code changes.

use crate::runtime_adapter::onnx::ONNXSession;
use crate::runtime_adapter::AdapterError;
use ndarray::{Array1, Array2, ArrayD};
use ort::tensor::TensorElementType;
use ort::value::Value;
use std::collections::HashMap;

use super::super::types::ExecutorResult;

/// Execute TTS inference with phoneme IDs, voice embedding, and speed.
///
/// Inputs are mapped by dtype and shape pattern, not by name:
/// - int64 input with shape [1, N] (dynamic) → token/phoneme IDs
/// - f32 input with shape [1, 256] → voice/style embedding
/// - f32 input with shape [1] → speed multiplier
///
/// This makes the function model-agnostic: KittenTTS (input_ids, style, speed)
/// and Kokoro (tokens, style, speed) both work without name-specific code.
pub fn execute_tts_inference(
    session: &ONNXSession,
    phoneme_ids: &[i64],
    voice_embedding: Vec<f32>,
    speed: f32,
    piper: PiperInferenceConfig,
) -> ExecutorResult<HashMap<String, ArrayD<f32>>> {
    let input_names = session.input_names();
    let input_shapes = session.input_shapes();
    let input_dtypes = session.input_dtypes();

    let batch_size = 1;
    let seq_len = phoneme_ids.len();
    let embedding_len = voice_embedding.len();

    let mut value_inputs: HashMap<String, Value> = HashMap::new();

    for (i, input_name) in input_names.iter().enumerate() {
        let dtype = input_dtypes.get(i).and_then(|d| *d);
        let shape = input_shapes.get(i).map(|s| s.as_slice()).unwrap_or(&[]);

        match classify_tts_input(input_name, dtype, shape) {
            TtsInputKind::Tokens => {
                let arr =
                    Array2::<i64>::from_shape_vec((batch_size, seq_len), phoneme_ids.to_vec())
                        .map_err(|e| {
                            AdapterError::InvalidInput(format!(
                                "Failed to create token array for '{}': {}",
                                input_name, e
                            ))
                        })?;
                let val: Value = Value::from_array(arr)
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create token value for '{}': {}",
                            input_name, e
                        ))
                    })?
                    .into();
                value_inputs.insert(input_name.clone(), val);
            }
            TtsInputKind::VoiceEmbedding => {
                let arr =
                    Array2::<f32>::from_shape_vec((1, embedding_len), voice_embedding.clone())
                        .map_err(|e| {
                            AdapterError::InvalidInput(format!(
                                "Failed to create voice embedding array for '{}': {}",
                                input_name, e
                            ))
                        })?;
                let val: Value = Value::from_array(arr)
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create voice embedding value for '{}': {}",
                            input_name, e
                        ))
                    })?
                    .into();
                value_inputs.insert(input_name.clone(), val);
            }
            TtsInputKind::Speed => {
                let arr = Array1::<f32>::from_vec(vec![speed]);
                let val: Value = Value::from_array(arr)
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create speed value for '{}': {}",
                            input_name, e
                        ))
                    })?
                    .into();
                value_inputs.insert(input_name.clone(), val);
            }
            TtsInputKind::InputLengths => {
                let val: Value = Value::from_array(Array1::from_vec(vec![seq_len as i64]))
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create input length value: {e}"
                        ))
                    })?
                    .into();
                value_inputs.insert(input_name.clone(), val);
            }
            TtsInputKind::Scales => {
                let scales = vec![piper.noise_scale, piper.length_scale / speed, piper.noise_w];
                let val: Value = Value::from_array(Array1::from_vec(scales))
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create Piper scales value: {e}"
                        ))
                    })?
                    .into();
                value_inputs.insert(input_name.clone(), val);
            }
            TtsInputKind::SpeakerId => {
                let val: Value = Value::from_array(Array1::from_vec(vec![piper.speaker_id]))
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create Piper speaker value: {e}"
                        ))
                    })?
                    .into();
                value_inputs.insert(input_name.clone(), val);
            }
            TtsInputKind::Unknown => {
                // Not classified — will trigger the mismatch error below
            }
        }
    }

    // Verify we mapped all inputs
    if value_inputs.len() != input_names.len() {
        let found: Vec<String> = input_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let dtype = input_dtypes
                    .get(i)
                    .and_then(|d| *d)
                    .map_or("unknown".to_string(), |d| format!("{:?}", d));
                let shape = input_shapes
                    .get(i)
                    .map(|s| format!("{:?}", s))
                    .unwrap_or_default();
                format!("'{}' (dtype={}, shape={})", name, dtype, shape)
            })
            .collect();

        return Err(AdapterError::InvalidInput(format!(
            "TTS model has unexpected inputs. Expected patterns: \
             int64 [1, N] (tokens), Piper input_lengths/scales/sid, \
             f32 [1, 256] (voice embedding), or f32 [1] (speed). \
             Found: [{}]",
            found.join(", ")
        )));
    }

    session.run_with_values(value_inputs)
}

/// Classification of a TTS model input based on dtype and shape.
enum TtsInputKind {
    /// int64 [1, N] — phoneme/token IDs
    Tokens,
    /// f32 [1, 256] — voice/style embedding
    VoiceEmbedding,
    /// f32 [1] — speed multiplier
    Speed,
    InputLengths,
    Scales,
    SpeakerId,
    /// Unrecognized input pattern
    Unknown,
}

/// Classify a TTS input by its element type and shape dimensions.
///
/// Rules:
/// - int64 with 2D shape [1, N] (N fixed or dynamic) → Tokens
/// - f32 with 2D shape [1, 256] → VoiceEmbedding
/// - f32 with 1D shape [1] (or dynamic [N]) → Speed
fn classify_tts_input(name: &str, dtype: Option<TensorElementType>, shape: &[i64]) -> TtsInputKind {
    match (name, dtype) {
        ("input", Some(TensorElementType::Int64)) => return TtsInputKind::Tokens,
        ("input_lengths", Some(TensorElementType::Int64)) => return TtsInputKind::InputLengths,
        ("scales", Some(TensorElementType::Float32)) => return TtsInputKind::Scales,
        ("sid", Some(TensorElementType::Int64)) => return TtsInputKind::SpeakerId,
        _ => {}
    }
    match dtype {
        Some(TensorElementType::Int64) if shape.len() == 2 && (shape[0] == 1 || shape[0] == -1) => {
            return TtsInputKind::Tokens;
        }
        Some(TensorElementType::Float32) => {
            if shape.len() == 2 && (shape[0] == 1 || shape[0] == -1) {
                // f32 [1, 256] → voice embedding
                // Distinguish from tokens: embedding dim is typically 256 (fixed)
                if shape[1] > 1 {
                    return TtsInputKind::VoiceEmbedding;
                }
            }
            if shape.len() == 1 {
                // f32 [1] or f32 [-1] → speed
                return TtsInputKind::Speed;
            }
        }
        _ => {}
    }
    TtsInputKind::Unknown
}

#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub struct PiperInferenceConfig {
    #[serde(default = "default_noise_scale")]
    pub noise_scale: f32,
    #[serde(default = "default_length_scale")]
    pub length_scale: f32,
    #[serde(default = "default_noise_w")]
    pub noise_w: f32,
    #[serde(skip)]
    pub speaker_id: i64,
}

impl Default for PiperInferenceConfig {
    fn default() -> Self {
        Self {
            noise_scale: default_noise_scale(),
            length_scale: default_length_scale(),
            noise_w: default_noise_w(),
            speaker_id: 0,
        }
    }
}

impl PiperInferenceConfig {
    pub fn from_model_path(model_path: &std::path::Path) -> Self {
        #[derive(serde::Deserialize)]
        struct VoiceConfig {
            #[serde(default)]
            inference: PiperInferenceConfig,
        }
        let path = model_path.with_extension("onnx.json");
        std::fs::read_to_string(path)
            .ok()
            .and_then(|value| serde_json::from_str::<VoiceConfig>(&value).ok())
            .map(|config| config.inference)
            .unwrap_or_default()
    }
}

fn default_noise_scale() -> f32 {
    0.667
}
fn default_length_scale() -> f32 {
    1.0
}
fn default_noise_w() -> f32 {
    0.8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_piper_inputs_by_name() {
        assert!(matches!(
            classify_tts_input("input", Some(TensorElementType::Int64), &[1, -1]),
            TtsInputKind::Tokens
        ));
        assert!(matches!(
            classify_tts_input("input_lengths", Some(TensorElementType::Int64), &[1]),
            TtsInputKind::InputLengths
        ));
        assert!(matches!(
            classify_tts_input("scales", Some(TensorElementType::Float32), &[3]),
            TtsInputKind::Scales
        ));
        assert!(matches!(
            classify_tts_input("sid", Some(TensorElementType::Int64), &[1]),
            TtsInputKind::SpeakerId
        ));
    }

    #[test]
    fn preserves_shape_fallback_for_embedding_tts() {
        assert!(matches!(
            classify_tts_input("style", Some(TensorElementType::Float32), &[1, 256]),
            TtsInputKind::VoiceEmbedding
        ));
        assert!(matches!(
            classify_tts_input("speed", Some(TensorElementType::Float32), &[1]),
            TtsInputKind::Speed
        ));
    }
}
