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
                ArrayD::from_shape_vec(IxDyn(&[batch_size, num_samples]), samples.clone())
                    .map_err(|e| {
                        AdapterError::InvalidInput(format!(
                            "Failed to create audio tensor: {:?}",
                            e
                        ))
                    })
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

    /// Get text content if available.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            PreprocessedData::Text(text) => Some(text),
            PreprocessedData::PhonemeIds { original_text, .. } => Some(original_text),
            PreprocessedData::TokenIds { original_text, .. } => Some(original_text),
            _ => None,
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

                let data = tensor
                    .as_slice()
                    .ok_or_else(|| AdapterError::InvalidInput("Tensor not contiguous".to_string()))?;

                Ok(Envelope::new(EnvelopeKind::Embedding(data.to_vec())))
            }
            RawOutputs::TokenIds(ids) => {
                Ok(Envelope::new(EnvelopeKind::Text(format!("{:?}", ids))))
            }
            RawOutputs::AudioBytes(bytes) => {
                Ok(Envelope::new(EnvelopeKind::Audio(bytes.clone())))
            }
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
