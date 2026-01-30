//! Tensor conversion utilities for converting between Envelope types and ONNX tensors.
//!
//! This module provides functions to convert:
//! - `Envelope` (Audio/Text/Embedding) → ONNX tensors (`ndarray::Array`)
//! - ONNX tensors → `Envelope`

use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{AdapterError, AdapterResult};
use ndarray::{Array, ArrayD, IxDyn};

/// Converts an `Envelope` to ONNX tensor inputs.
///
/// # Arguments
///
/// * `envelope` - The input envelope containing Audio, Text, or Embedding data
/// * `input_shapes` - Expected input shapes for the model (one per input)
/// * `input_names` - Names of the model inputs
///
/// # Returns
///
/// A HashMap mapping input names to `ndarray::ArrayD<f32>` tensors ready for inference
///
/// # Errors
///
/// Returns an error if:
/// - The envelope type is not supported
/// - Tensor shape conversion fails
/// - Audio/text conversion fails
pub fn envelope_to_tensors(
    envelope: &Envelope,
    input_shapes: &[Vec<i64>],
    input_names: &[String],
) -> AdapterResult<std::collections::HashMap<String, ArrayD<f32>>> {
    if input_shapes.is_empty() || input_names.is_empty() {
        return Err(AdapterError::InvalidInput(
            "No input shapes or names provided".to_string(),
        ));
    }

    let input_name = &input_names[0]; // Use first input name

    // Check if envelope has shape metadata (from preprocessed tensor)
    let shape_from_metadata: Option<Vec<i64>> =
        envelope.metadata.get("tensor_shape").and_then(|s| {
            let parts: Result<Vec<i64>, _> = s.split(',').map(|p| p.parse::<i64>()).collect();
            parts.ok()
        });

    // Use shape from metadata if available, otherwise use model's declared shape
    let target_shape = match &shape_from_metadata {
        Some(shape) => shape.as_slice(),
        None => &input_shapes[0],
    };

    let tensor = match &envelope.kind {
        EnvelopeKind::Audio(audio_data) => audio_to_tensor(audio_data, target_shape)?,
        EnvelopeKind::Text(text) => text_to_tensor(text, target_shape)?,
        EnvelopeKind::Embedding(embedding) => embedding_to_tensor(embedding, target_shape)?,
    };

    let mut result = std::collections::HashMap::new();
    result.insert(input_name.clone(), tensor);
    Ok(result)
}

/// Converts ONNX tensor outputs to an `Envelope`.
///
/// # Arguments
///
/// * `outputs` - HashMap of output names to tensors from ONNX inference
/// * `output_names` - Names of the model outputs (for ordering/selection)
///
/// # Returns
///
/// An `Envelope` containing the inference result
///
/// # Errors
///
/// Returns an error if:
/// - No outputs provided
/// - Tensor extraction fails
/// - Output type detection fails
pub fn tensors_to_envelope(
    outputs: &std::collections::HashMap<String, ArrayD<f32>>,
    output_names: &[String],
) -> AdapterResult<Envelope> {
    if outputs.is_empty() {
        return Err(AdapterError::InvalidInput(
            "No output tensors provided".to_string(),
        ));
    }

    // Use first output for now
    let output_name = output_names.get(0).map(|s| s.as_str()).unwrap_or("output");
    let output = outputs
        .get(output_name)
        .ok_or_else(|| AdapterError::InvalidInput(format!("Output '{}' not found", output_name)))?;

    // Convert tensor to embedding, preserving shape in metadata
    let data = output
        .as_slice()
        .ok_or_else(|| AdapterError::InvalidInput("Output tensor not contiguous".to_string()))?;

    // Store shape in metadata so postprocessing can reconstruct the tensor
    let shape_str = output
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

/// Converts audio bytes to an ONNX tensor.
///
/// # Arguments
///
/// * `audio_data` - Raw audio bytes (WAV file or PCM bytes)
/// * `target_shape` - Expected tensor shape (e.g., `[1, 16000]` for wav2vec2)
///
/// # Returns
///
/// An `ndarray::ArrayD<f32>` tensor containing audio samples as `f32`
///
/// # Supported Formats
///
/// - WAV files: Auto-decoded with resampling to 16kHz mono
/// - Raw 16-bit PCM: Converted directly to f32
///
/// # Wav2Vec2 Format
///
/// Wav2Vec2 expects:
/// - 16kHz sample rate
/// - Mono (1 channel)
/// - Normalized to [-1.0, 1.0]
/// - Shape: `[batch, samples]` (2D tensor with batch dimension)
fn audio_to_tensor(audio_data: &[u8], target_shape: &[i64]) -> AdapterResult<ArrayD<f32>> {
    // Try to decode as WAV file first (most common case)
    let samples = decode_audio_to_samples(audio_data)?;

    if samples.is_empty() {
        return Err(AdapterError::InvalidInput(
            "Audio data is empty".to_string(),
        ));
    }

    // Determine output shape based on target_shape
    // Models like wav2vec2 expect [batch, samples] (2D)
    // Models like Whisper may expect [batch, channels, samples] (3D)
    let final_shape: Vec<usize> = if target_shape.len() == 2 {
        // 2D shape [batch, samples] - most common for wav2vec2
        let batch = if target_shape[0] == -1 {
            1
        } else {
            target_shape[0] as usize
        };
        let num_samples = if target_shape[1] == -1 {
            samples.len()
        } else {
            target_shape[1] as usize
        };
        vec![batch, num_samples]
    } else if target_shape.len() == 3 {
        // 3D shape [batch, channels, samples] - for Whisper-style models
        let batch = if target_shape[0] == -1 {
            1
        } else {
            target_shape[0] as usize
        };
        let channels = if target_shape[1] == -1 {
            1
        } else {
            target_shape[1] as usize
        };
        let num_samples = if target_shape[2] == -1 {
            samples.len()
        } else {
            target_shape[2] as usize
        };
        vec![batch, channels, num_samples]
    } else if target_shape.len() == 1 {
        // 1D shape - add batch dimension to make 2D for model compatibility
        let num_samples = if target_shape[0] == -1 {
            samples.len()
        } else {
            target_shape[0] as usize
        };
        vec![1, num_samples]
    } else {
        return Err(AdapterError::InvalidInput(format!(
            "Unsupported target shape dimensions: {:?}",
            target_shape
        )));
    };

    // Calculate expected size from shape
    let expected_size: usize = final_shape.iter().product();

    // Handle sample count mismatch (pad or truncate)
    let final_samples = if samples.len() < expected_size {
        // Pad with zeros (silence)
        let mut padded = samples;
        padded.resize(expected_size, 0.0);
        padded
    } else if samples.len() > expected_size {
        // Truncate to expected size
        samples[..expected_size].to_vec()
    } else {
        samples
    };

    Array::from_shape_vec(IxDyn(&final_shape), final_samples)
        .map_err(|e| AdapterError::RuntimeError(format!("Failed to create audio tensor: {}", e)))
}

/// Decodes audio bytes (WAV or raw PCM) to f32 samples.
///
/// Returns samples normalized to [-1.0, 1.0] at 16kHz mono.
fn decode_audio_to_samples(audio_data: &[u8]) -> AdapterResult<Vec<f32>> {
    use std::io::Cursor;

    // Try WAV decoding first
    let cursor = Cursor::new(audio_data);
    match hound::WavReader::new(cursor) {
        Ok(mut reader) => {
            let spec = reader.spec();
            let source_sample_rate = spec.sample_rate;
            let source_channels = spec.channels as usize;

            // Read samples as f32
            let samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => {
                    reader.samples::<f32>().filter_map(|s| s.ok()).collect()
                }
                hound::SampleFormat::Int => {
                    // Convert int samples to f32 normalized to [-1.0, 1.0]
                    let bits = spec.bits_per_sample;
                    let max_value = (1 << (bits - 1)) as f32;
                    reader
                        .samples::<i32>()
                        .filter_map(|s| s.ok())
                        .map(|s| s as f32 / max_value)
                        .collect()
                }
            };

            // Convert to mono if needed
            let mono_samples = if source_channels > 1 {
                // Average channels for mono conversion
                samples
                    .chunks(source_channels)
                    .map(|chunk| chunk.iter().sum::<f32>() / source_channels as f32)
                    .collect()
            } else {
                samples
            };

            // Resample to 16kHz if needed
            const TARGET_SAMPLE_RATE: u32 = 16000;
            let resampled = if source_sample_rate != TARGET_SAMPLE_RATE {
                let ratio = TARGET_SAMPLE_RATE as f32 / source_sample_rate as f32;
                let target_len = (mono_samples.len() as f32 * ratio) as usize;

                (0..target_len)
                    .map(|i| {
                        let source_idx = (i as f32 / ratio) as usize;
                        mono_samples.get(source_idx).copied().unwrap_or(0.0)
                    })
                    .collect()
            } else {
                mono_samples
            };

            Ok(resampled)
        }
        Err(_) => {
            // Not a WAV file, try raw PCM (16-bit little-endian)
            if audio_data.len() % 2 != 0 {
                return Err(AdapterError::InvalidInput(format!(
                    "Audio data length ({}) must be even for 16-bit PCM",
                    audio_data.len()
                )));
            }

            let samples: Vec<f32> = audio_data
                .chunks_exact(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    sample as f32 / 32768.0
                })
                .collect();

            Ok(samples)
        }
    }
}

/// Converts text to an ONNX tensor (token IDs).
///
/// # Arguments
///
/// * `text` - Input text string
/// * `target_shape` - Expected tensor shape (e.g., `[1, 512]` for sequence length 512)
///
/// # Returns
///
/// An `ndarray::ArrayD<f32>` tensor containing token IDs as `f32` (converted from i64)
///
/// # Note
///
/// This is a simple tokenization implementation for MVP.
/// Real implementations should use proper tokenizers (BPE, SentencePiece, etc.)
fn text_to_tensor(text: &str, target_shape: &[i64]) -> AdapterResult<ArrayD<f32>> {
    // Simple tokenization: split by whitespace and map to IDs
    // This is a placeholder - real tokenizers should be integrated later
    let tokens: Vec<i64> = text
        .split_whitespace()
        .enumerate()
        .map(|(i, _)| i as i64)
        .collect();

    let actual_size = tokens.len();

    // Check if shape contains dynamic dimensions (-1)
    let has_dynamic = target_shape.iter().any(|&d| d < 0);

    if has_dynamic {
        // For dynamic shapes, use the actual token count
        let shape: Vec<usize> = if target_shape == &[-1] {
            vec![actual_size]
        } else if target_shape.len() == 2 {
            let batch = if target_shape[0] > 0 {
                target_shape[0] as usize
            } else {
                1
            };
            vec![batch, actual_size]
        } else {
            vec![actual_size]
        };

        let tokens_f32: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        return Array::from_shape_vec(IxDyn(&shape), tokens_f32).map_err(|e| {
            AdapterError::RuntimeError(format!("Failed to create text tensor: {}", e))
        });
    }

    // Calculate expected size from static shape
    let expected_size: i64 = target_shape.iter().product();

    // Pad or truncate to match shape
    let final_tokens = if (actual_size as i64) < expected_size {
        // Pad with zeros (or special token ID)
        let mut padded = tokens;
        padded.resize(expected_size as usize, 0);
        padded
    } else {
        // Truncate
        tokens[..expected_size as usize].to_vec()
    };

    // Create tensor (convert i64 to f32)
    let shape: Vec<usize> = target_shape.iter().map(|&s| s as usize).collect();
    let tokens_f32: Vec<f32> = final_tokens.iter().map(|&t| t as f32).collect();

    Array::from_shape_vec(IxDyn(&shape), tokens_f32)
        .map_err(|e| AdapterError::RuntimeError(format!("Failed to create text tensor: {}", e)))
}

/// Converts an embedding vector to an ONNX tensor.
///
/// # Arguments
///
/// * `embedding` - Embedding vector (f32 values)
/// * `target_shape` - Expected tensor shape
///
/// # Returns
///
/// An `ndarray::ArrayD<f32>` tensor containing embedding values as `f32`
fn embedding_to_tensor(embedding: &[f32], target_shape: &[i64]) -> AdapterResult<ArrayD<f32>> {
    let actual_size = embedding.len();

    // Check if shape contains dynamic dimensions (-1)
    let has_dynamic = target_shape.iter().any(|&d| d < 0);

    if has_dynamic {
        // For dynamic shapes, infer the shape from data
        // If shape is just [-1], treat the embedding as-is (1D tensor)
        // Common patterns:
        // - [-1] → [actual_size] (1D)
        // - [1, -1] → [1, actual_size] (batch of 1)
        // - [-1, -1] → [1, actual_size] (assume batch=1)

        let shape: Vec<usize> = if target_shape == &[-1] {
            // Single dynamic dimension: create 1D tensor
            vec![actual_size]
        } else if target_shape.len() == 2 {
            // 2D with dynamics: [batch, features] or similar
            let batch = if target_shape[0] > 0 {
                target_shape[0] as usize
            } else {
                1
            };
            let features = if target_shape[1] > 0 {
                target_shape[1] as usize
            } else {
                actual_size / batch
            };
            vec![batch, features]
        } else {
            // Fallback: treat as 1D
            vec![actual_size]
        };

        return Array::from_shape_vec(IxDyn(&shape), embedding.to_vec()).map_err(|e| {
            AdapterError::RuntimeError(format!("Failed to create embedding tensor: {}", e))
        });
    }

    // Calculate expected size from static shape
    let expected_size: i64 = target_shape.iter().product();

    // Validate shape matches data size
    if actual_size as i64 != expected_size {
        return Err(AdapterError::InvalidInput(format!(
            "Embedding size mismatch: expected {}, got {}",
            expected_size, actual_size
        )));
    }

    // Create tensor with specified shape
    let shape: Vec<usize> = target_shape.iter().map(|&s| s as usize).collect();

    Array::from_shape_vec(IxDyn(&shape), embedding.to_vec()).map_err(|e| {
        AdapterError::RuntimeError(format!("Failed to create embedding tensor: {}", e))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_to_tensor() {
        // Create sample PCM audio (16-bit, 2 bytes per sample)
        let audio_data = vec![0u8; 32000]; // 16k samples * 2 bytes = 32k bytes
        let target_shape = vec![1, 1, 16000]; // [batch, channels, samples]

        let result = audio_to_tensor(&audio_data, &target_shape);
        assert!(result.is_ok());
    }

    #[test]
    fn test_text_to_tensor() {
        let text = "hello world test";
        let target_shape = vec![1, 512]; // [batch, sequence_length]

        let result = text_to_tensor(text, &target_shape);
        assert!(result.is_ok());
    }

    #[test]
    fn test_embedding_to_tensor() {
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let target_shape = vec![1, 4]; // [batch, embedding_dim]

        let result = embedding_to_tensor(&embedding, &target_shape);
        assert!(result.is_ok());
    }

    #[test]
    fn test_envelope_to_tensors_audio() {
        let envelope = Envelope::new(EnvelopeKind::Audio(vec![0u8; 32000]));
        let input_shapes = vec![vec![1, 1, 16000]];
        let input_names = vec!["audio_input".to_string()];

        let result = envelope_to_tensors(&envelope, &input_shapes, &input_names);
        assert!(result.is_ok());
        let tensors = result.unwrap();
        assert!(tensors.contains_key("audio_input"));
    }

    #[test]
    fn test_envelope_to_tensors_text() {
        let envelope = Envelope::new(EnvelopeKind::Text("hello world".to_string()));
        let input_shapes = vec![vec![1, 512]];
        let input_names = vec!["text_input".to_string()];

        let result = envelope_to_tensors(&envelope, &input_shapes, &input_names);
        assert!(result.is_ok());
        let tensors = result.unwrap();
        assert!(tensors.contains_key("text_input"));
    }
}
