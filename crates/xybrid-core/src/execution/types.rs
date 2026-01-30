//! Data types for the template executor.
//!
//! This module defines the intermediate data types used during execution:
//! - [`PreprocessedData`]: Output from preprocessing, input to model execution
//! - [`RawOutputs`]: Output from model execution, input to postprocessing

use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::AdapterError;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

/// Result type for executor operations.
pub type ExecutorResult<T> = Result<T, AdapterError>;

/// Preprocessed data intermediate representation.
///
/// This enum represents the output of preprocessing steps and serves as
/// input to the model execution phase.
#[derive(Debug, Clone)]
pub enum PreprocessedData {
    /// Raw audio bytes (WAV format, not yet decoded)
    AudioBytes(Vec<u8>),

    /// Decoded PCM audio samples (f32, normalized to [-1.0, 1.0])
    AudioSamples(Vec<f32>),

    /// Text string
    Text(String),

    /// Tensor data (mel spectrogram, embeddings, etc.)
    Tensor(ArrayD<f32>),

    /// Token IDs for BERT-style models
    TokenIds {
        ids: Vec<usize>,
        attention_mask: Vec<usize>,
        token_type_ids: Vec<usize>,
        vocab_file: String,
        original_text: String,
    },

    /// Phoneme IDs for TTS models
    PhonemeIds {
        ids: Vec<i64>,
        phonemes: String,
        original_text: String,
    },
}

impl PreprocessedData {
    /// Create preprocessed data from an envelope.
    pub fn from_envelope(envelope: &Envelope) -> ExecutorResult<Self> {
        match &envelope.kind {
            EnvelopeKind::Audio(bytes) => Ok(PreprocessedData::AudioBytes(bytes.clone())),
            EnvelopeKind::Text(text) => Ok(PreprocessedData::Text(text.clone())),
            EnvelopeKind::Embedding(floats) => {
                let tensor = ArrayD::from_shape_vec(IxDyn(&[floats.len()]), floats.clone())
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!("Failed to create tensor: {:?}", e))
                    })?;
                Ok(PreprocessedData::Tensor(tensor))
            }
        }
    }

    /// Convert to tensor if possible.
    pub fn to_tensor(&self) -> ExecutorResult<ArrayD<f32>> {
        match self {
            PreprocessedData::Tensor(t) => Ok(t.clone()),
            PreprocessedData::AudioSamples(samples) => {
                let batch_size = 1;
                let num_samples = samples.len();
                ArrayD::from_shape_vec(IxDyn(&[batch_size, num_samples]), samples.clone()).map_err(
                    |e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create audio tensor: {:?}",
                            e
                        ))
                    },
                )
            }
            _ => Err(AdapterError::InvalidInput(
                "Cannot convert to tensor".to_string(),
            )),
        }
    }

    /// Check if this is phoneme data (for TTS).
    pub fn is_phoneme_ids(&self) -> bool {
        matches!(self, PreprocessedData::PhonemeIds { .. })
    }

    /// Get phoneme IDs if this is phoneme data.
    pub fn as_phoneme_ids(&self) -> Option<&Vec<i64>> {
        match self {
            PreprocessedData::PhonemeIds { ids, .. } => Some(ids),
            _ => None,
        }
    }

    /// Check if this is token data (for BERT-style models).
    pub fn is_token_ids(&self) -> bool {
        matches!(self, PreprocessedData::TokenIds { .. })
    }

    /// Get token IDs if this is token data.
    pub fn as_token_ids(&self) -> Option<(&Vec<usize>, &Vec<usize>, &Vec<usize>)> {
        match self {
            PreprocessedData::TokenIds {
                ids,
                attention_mask,
                token_type_ids,
                ..
            } => Some((ids, attention_mask, token_type_ids)),
            _ => None,
        }
    }

    /// Get text content if available.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            PreprocessedData::Text(text) => Some(text),
            PreprocessedData::PhonemeIds { original_text, .. } => Some(original_text),
            PreprocessedData::TokenIds { original_text, .. } => Some(original_text),
            _ => None,
        }
    }
    /// Convert to envelope.
    pub fn to_envelope(&self) -> ExecutorResult<Envelope> {
        match self {
            PreprocessedData::AudioBytes(bytes) => {
                Ok(Envelope::new(EnvelopeKind::Audio(bytes.clone())))
            }
            PreprocessedData::AudioSamples(samples) => {
                // If samples are f32, we might want to encode to WAV or just pass raw bytes?
                // Envelope::Audio expects "bytes", usually WAV/encoded.
                // But Envelope implies "transportable".
                // If we want to pass samples, we can convert to WAV using audio utils.
                // Assuming crate::audio::samples_to_wav exists and is accessible.
                // Or implementing a simpler conversion to bytes.
                // For now, let's use a placeholder or assume Envelope accepts raw f32 bytes?
                // EnvelopeKind::Audio is Vec<u8>.
                let bytes = crate::audio::samples_to_wav(samples, 16000); // hardcoded rate?
                Ok(Envelope::new(EnvelopeKind::Audio(bytes)))
            }
            PreprocessedData::Text(text) => Ok(Envelope::new(EnvelopeKind::Text(text.clone()))),
            PreprocessedData::Tensor(tensor) => {
                let data = tensor.as_slice().ok_or_else(|| {
                    AdapterError::InvalidInput("Tensor not contiguous".to_string())
                })?;
                // Store shape in metadata so it can be reconstructed
                let shape_str = tensor
                    .shape()
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("tensor_shape".to_string(), shape_str);
                Ok(Envelope::with_metadata(
                    EnvelopeKind::Embedding(data.to_vec()),
                    metadata,
                ))
            }
            PreprocessedData::PhonemeIds { original_text, .. } => {
                // Fallback to original text for now
                Ok(Envelope::new(EnvelopeKind::Text(original_text.clone())))
            }
            PreprocessedData::TokenIds { original_text, .. } => {
                Ok(Envelope::new(EnvelopeKind::Text(original_text.clone())))
            }
        }
    }
}

/// Raw outputs from model execution.
///
/// This enum represents the output from model execution before postprocessing.
#[derive(Debug, Clone)]
pub enum RawOutputs {
    /// Map of named tensors (typical ONNX output)
    TensorMap(HashMap<String, ArrayD<f32>>),

    /// Token IDs (for sequence generation)
    TokenIds(Vec<usize>),

    /// Text string
    Text(String),

    /// Class ID (for classification)
    ClassId(usize),

    /// Raw audio bytes (PCM or WAV)
    AudioBytes(Vec<u8>),
}

impl RawOutputs {
    /// Convert to an Envelope.
    pub fn to_envelope(&self) -> ExecutorResult<Envelope> {
        match self {
            RawOutputs::Text(text) => Ok(Envelope::new(EnvelopeKind::Text(text.clone()))),
            RawOutputs::ClassId(id) => {
                Ok(Envelope::new(EnvelopeKind::Text(format!("Class: {}", id))))
            }
            RawOutputs::TensorMap(map) => {
                let tensor = map
                    .values()
                    .next()
                    .ok_or_else(|| AdapterError::InvalidInput("No outputs".to_string()))?;

                let data = tensor.as_slice().ok_or_else(|| {
                    AdapterError::InvalidInput("Tensor not contiguous".to_string())
                })?;

                Ok(Envelope::new(EnvelopeKind::Embedding(data.to_vec())))
            }
            RawOutputs::TokenIds(ids) => {
                Ok(Envelope::new(EnvelopeKind::Text(format!("{:?}", ids))))
            }
            RawOutputs::AudioBytes(bytes) => Ok(Envelope::new(EnvelopeKind::Audio(bytes.clone()))),
        }
    }

    /// Create from a tensor map.
    pub fn from_tensor_map(map: HashMap<String, ArrayD<f32>>) -> Self {
        RawOutputs::TensorMap(map)
    }

    /// Create from text.
    pub fn from_text(text: String) -> Self {
        RawOutputs::Text(text)
    }
    /// Create from Envelope.
    pub fn from_envelope(envelope: &Envelope) -> ExecutorResult<Self> {
        match &envelope.kind {
            EnvelopeKind::Text(text) => Ok(RawOutputs::Text(text.clone())),
            EnvelopeKind::Audio(bytes) => Ok(RawOutputs::AudioBytes(bytes.clone())),
            EnvelopeKind::Embedding(floats) => {
                // Check if shape metadata is available
                let shape: Vec<usize> = envelope
                    .metadata
                    .get("tensor_shape")
                    .and_then(|s| {
                        let parts: Result<Vec<usize>, _> =
                            s.split(',').map(|p| p.parse()).collect();
                        parts.ok()
                    })
                    .unwrap_or_else(|| vec![floats.len()]); // Default to 1D if no shape

                let tensor =
                    ArrayD::from_shape_vec(IxDyn(&shape), floats.clone()).map_err(|e| {
                        AdapterError::InvalidInput(format!("Failed to create tensor: {:?}", e))
                    })?;
                // Map it to "output" key by default? Or TensorMap
                let mut map = HashMap::new();
                map.insert("output".to_string(), tensor);
                Ok(RawOutputs::TensorMap(map))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessed_data_from_envelope_text() {
        let envelope = Envelope::new(EnvelopeKind::Text("hello".to_string()));
        let data = PreprocessedData::from_envelope(&envelope).unwrap();
        assert!(matches!(data, PreprocessedData::Text(_)));
    }

    #[test]
    fn test_preprocessed_data_from_envelope_audio() {
        let envelope = Envelope::new(EnvelopeKind::Audio(vec![1, 2, 3]));
        let data = PreprocessedData::from_envelope(&envelope).unwrap();
        assert!(matches!(data, PreprocessedData::AudioBytes(_)));
    }

    #[test]
    fn test_raw_outputs_to_envelope_text() {
        let outputs = RawOutputs::Text("result".to_string());
        let envelope = outputs.to_envelope().unwrap();
        assert!(matches!(envelope.kind, EnvelopeKind::Text(_)));
    }

    #[test]
    fn test_raw_outputs_to_envelope_audio() {
        let outputs = RawOutputs::AudioBytes(vec![1, 2, 3, 4]);
        let envelope = outputs.to_envelope().unwrap();
        assert!(matches!(envelope.kind, EnvelopeKind::Audio(_)));
    }
}
