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

    let target_shape = &input_shapes[0]; // Use first input shape for now
    let input_name = &input_names[0]; // Use first input name

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
    let output = outputs.get(output_name).ok_or_else(|| {
        AdapterError::InvalidInput(format!("Output '{}' not found", output_name))
    })?;

    // Try to detect output type and convert
    // For MVP, we'll try to convert to Text (most common for ASR/TTS)
    tensor_to_text_envelope(output, output_name)
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
        return Err(AdapterError::InvalidInput("Audio data is empty".to_string()));
    }

    // Determine output shape based on target_shape
    // Models like wav2vec2 expect [batch, samples] (2D)
    // Models like Whisper may expect [batch, channels, samples] (3D)
    let final_shape: Vec<usize> = if target_shape.len() == 2 {
        // 2D shape [batch, samples] - most common for wav2vec2
        let batch = if target_shape[0] == -1 { 1 } else { target_shape[0] as usize };
        let num_samples = if target_shape[1] == -1 { samples.len() } else { target_shape[1] as usize };
        vec![batch, num_samples]
    } else if target_shape.len() == 3 {
        // 3D shape [batch, channels, samples] - for Whisper-style models
        let batch = if target_shape[0] == -1 { 1 } else { target_shape[0] as usize };
        let channels = if target_shape[1] == -1 { 1 } else { target_shape[1] as usize };
        let num_samples = if target_shape[2] == -1 { samples.len() } else { target_shape[2] as usize };
        vec![batch, channels, num_samples]
    } else if target_shape.len() == 1 {
        // 1D shape - add batch dimension to make 2D for model compatibility
        let num_samples = if target_shape[0] == -1 { samples.len() } else { target_shape[0] as usize };
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
                    reader.samples::<f32>()
                        .filter_map(|s| s.ok())
                        .collect()
                }
                hound::SampleFormat::Int => {
                    // Convert int samples to f32 normalized to [-1.0, 1.0]
                    let bits = spec.bits_per_sample;
                    let max_value = (1 << (bits - 1)) as f32;
                    reader.samples::<i32>()
                        .filter_map(|s| s.ok())
                        .map(|s| s as f32 / max_value)
                        .collect()
                }
            };

            // Convert to mono if needed
            let mono_samples = if source_channels > 1 {
                // Average channels for mono conversion
                samples.chunks(source_channels)
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
                return Err(AdapterError::InvalidInput(
                    format!("Audio data length ({}) must be even for 16-bit PCM", audio_data.len())
                ));
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

    // Calculate expected size from shape
    let expected_size: i64 = target_shape.iter().product();
    let actual_size = tokens.len() as i64;

    // Pad or truncate to match shape
    let final_tokens = if actual_size < expected_size {
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
    // Calculate expected size from shape
    let expected_size: i64 = target_shape.iter().product();
    let actual_size = embedding.len() as i64;

    // Validate or reshape
    if actual_size != expected_size {
        return Err(AdapterError::InvalidInput(format!(
            "Embedding size mismatch: expected {}, got {}",
            expected_size, actual_size
        )));
    }

    // Create tensor
    let shape: Vec<usize> = target_shape.iter().map(|&s| s as usize).collect();
    
    Array::from_shape_vec(IxDyn(&shape), embedding.to_vec())
        .map_err(|e| AdapterError::RuntimeError(format!("Failed to create embedding tensor: {}", e)))
}

/// Converts an ONNX tensor output to a Text `Envelope`.
///
/// # Arguments
///
/// * `output` - Output tensor from ONNX inference (`ndarray::ArrayD<f32>`)
/// * `output_name` - Name of the output (for metadata)
///
/// # Returns
///
/// An `Envelope` with `EnvelopeKind::Text`
///
/// # Note
///
/// This is a simplified conversion. Real implementations should:
/// - Detect output type (logits, probabilities, token IDs)
/// - Apply proper decoding (argmax, sampling, tokenizer decode)
fn tensor_to_text_envelope(output: &ArrayD<f32>, output_name: &str) -> AdapterResult<Envelope> {
    // DEBUG: Comprehensive logging to understand Whisper output format
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    eprintln!("🔍 DEBUG: Whisper Output Analysis");
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Get tensor shape and data
    let shape = output.shape();
    let data = output.as_slice().unwrap_or(&[]);
    
    eprintln!("📊 Output Name: {}", output_name);
    eprintln!("📐 Output Shape: {:?}", shape);
    eprintln!("📏 Output Dimensions: {}D", shape.len());
    eprintln!("🔢 Total Elements: {}", data.len());
    
    if data.is_empty() {
        eprintln!("⚠️  WARNING: Output tensor is empty!");
        let text = format!("onnx-output-{}", output_name);
        let mut envelope = Envelope::new(EnvelopeKind::Text(text));
        envelope.metadata.insert("output_name".to_string(), output_name.to_string());
        envelope.metadata.insert("output_shape".to_string(), format!("{:?}", shape));
        return Ok(envelope);
    }
    
    // Calculate statistics
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();
    
    eprintln!("📈 Statistics:");
    eprintln!("   Min: {:.6}", min_val);
    eprintln!("   Max: {:.6}", max_val);
    eprintln!("   Mean: {:.6}", mean);
    eprintln!("   Std Dev: {:.6}", std_dev);
    
    // Analyze value ranges to guess output type
    eprintln!("🔬 Output Type Analysis:");
    if min_val >= 0.0 && max_val <= 1.0 {
        eprintln!("   → Looks like PROBABILITIES (values in [0, 1])");
    } else if min_val >= 0.0 && max_val > 1.0 && max_val < 1000.0 {
        eprintln!("   → Looks like LOGITS (positive values, moderate range)");
    } else if min_val < 0.0 {
        eprintln!("   → Looks like LOGITS (contains negative values)");
    } else if data.iter().all(|&x| x == x.floor() && x >= 0.0 && x < 100000.0) {
        eprintln!("   → Looks like TOKEN IDs (integer-like values)");
    } else {
        eprintln!("   → Unknown format (need investigation)");
    }
    
    // Show sample values
    eprintln!("📝 Sample Values (first 30):");
    let sample_size = data.len().min(30);
    for (i, val) in data.iter().take(sample_size).enumerate() {
        if i % 10 == 0 && i > 0 {
            eprintln!();
        }
        eprint!("  [{:3}]: {:8.4}", i, val);
    }
    eprintln!();
    
    // Analyze shape to understand structure
    eprintln!("🏗️  Shape Analysis:");
    match shape.len() {
        1 => {
            eprintln!("   1D tensor: [{}]", shape[0]);
            eprintln!("   → Could be: token IDs (1D sequence) or logits (flattened)");
        }
        2 => {
            eprintln!("   2D tensor: [{} x {}]", shape[0], shape[1]);
            if shape[1] > 1000 {
                eprintln!("   → Likely: [batch, vocab_size] logits");
            } else if shape[1] < 100 {
                eprintln!("   → Likely: [batch, seq_len] token IDs");
            } else {
                eprintln!("   → Could be: [batch, seq_len] or [batch, vocab_size]");
            }
        }
        3 => {
            eprintln!("   3D tensor: [{} x {} x {}]", shape[0], shape[1], shape[2]);
            eprintln!("   → Likely: [batch, seq_len, vocab_size] logits");
        }
        _ => {
            eprintln!("   {}D tensor: {:?}", shape.len(), shape);
            eprintln!("   → Complex shape, need to investigate");
        }
    }
    
    // Try to find patterns
    eprintln!("🔍 Pattern Detection:");
    let unique_values = {
        let mut unique = std::collections::HashSet::new();
        for val in data.iter().take(1000) {
            unique.insert((val * 1000.0).round() as i32); // Round to 3 decimals
        }
        unique.len()
    };
    eprintln!("   Unique values (first 1000, rounded): {}", unique_values);
    
    if unique_values < 50 {
        eprintln!("   → Low diversity: likely token IDs or discrete values");
    } else if unique_values > 500 {
        eprintln!("   → High diversity: likely logits or probabilities");
    }
    
    // Find argmax for each dimension if 2D+
    if shape.len() >= 2 {
        eprintln!("🎯 Argmax Analysis:");
        if shape.len() == 2 {
            // [batch, vocab] or [batch, seq_len]
            let second_dim = shape[1];
            eprintln!("   Checking first batch element (batch_size: {}, second_dim: {}):", shape[0], second_dim);
            
            let start_idx = 0;
            let end_idx = second_dim.min(100); // Check first 100 elements
            let slice = &data[start_idx..end_idx];
            
            if let Some((max_idx, max_val)) = slice.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                eprintln!("   Max value at index {}: {:.6}", max_idx, max_val);
            }
        }
    }
    
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // For now, return placeholder with all debug info in metadata
    let text = if data.is_empty() {
        format!("onnx-output-{}", output_name)
    } else {
        // Find max value index (argmax) for placeholder
        let max_idx = data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        format!("[DEBUG] onnx-output-{}-idx{} (shape: {:?}, min: {:.2}, max: {:.2})", 
                output_name, max_idx, shape, min_val, max_val)
    };
    
    let mut envelope = Envelope::new(EnvelopeKind::Text(text));
    envelope.metadata.insert("output_name".to_string(), output_name.to_string());
    envelope.metadata.insert("output_shape".to_string(), format!("{:?}", shape));
    envelope.metadata.insert("output_size".to_string(), data.len().to_string());
    envelope.metadata.insert("output_min".to_string(), format!("{:.6}", min_val));
    envelope.metadata.insert("output_max".to_string(), format!("{:.6}", max_val));
    envelope.metadata.insert("output_mean".to_string(), format!("{:.6}", mean));
    envelope.metadata.insert("output_stddev".to_string(), format!("{:.6}", std_dev));
    
    Ok(envelope)
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

