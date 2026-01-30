//! Test fixtures for common inputs and outputs.
//!
//! Provides pre-built test data that can be used across tests without
//! requiring external files or real model outputs.

use crate::ir::{Envelope, EnvelopeKind};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

/// Generate sample audio data at 16kHz (silence).
///
/// # Arguments
/// * `duration_secs` - Duration in seconds
///
/// # Returns
/// Vector of f32 samples (silence = 0.0)
pub fn sample_audio_16khz(duration_secs: f32) -> Vec<f32> {
    let num_samples = (16000.0 * duration_secs) as usize;
    vec![0.0f32; num_samples]
}

/// Generate sample audio data at 24kHz (silence).
pub fn sample_audio_24khz(duration_secs: f32) -> Vec<f32> {
    let num_samples = (24000.0 * duration_secs) as usize;
    vec![0.0f32; num_samples]
}

/// Generate a simple sine wave for audio testing.
///
/// # Arguments
/// * `sample_rate` - Sample rate in Hz
/// * `frequency` - Frequency of sine wave in Hz
/// * `duration_secs` - Duration in seconds
pub fn sine_wave(sample_rate: u32, frequency: f32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin()
        })
        .collect()
}

/// Create a text envelope for testing.
pub fn text_envelope(text: &str) -> Envelope {
    Envelope {
        kind: EnvelopeKind::Text(text.to_string()),
        metadata: HashMap::new(),
    }
}

/// Create an audio envelope from raw bytes.
pub fn audio_envelope(bytes: Vec<u8>) -> Envelope {
    Envelope {
        kind: EnvelopeKind::Audio(bytes),
        metadata: HashMap::new(),
    }
}

/// Create an embedding envelope.
pub fn embedding_envelope(values: Vec<f32>) -> Envelope {
    Envelope {
        kind: EnvelopeKind::Embedding(values),
        metadata: HashMap::new(),
    }
}

/// Create a sample mel spectrogram tensor.
///
/// Returns a tensor of shape [1, n_mels, time_steps] filled with zeros.
pub fn sample_mel_tensor(n_mels: usize, time_steps: usize) -> ArrayD<f32> {
    ArrayD::zeros(IxDyn(&[1, n_mels, time_steps]))
}

/// Create a sample logits tensor for classification.
///
/// Returns a tensor of shape [1, num_classes] with uniform distribution.
pub fn sample_logits(num_classes: usize) -> ArrayD<f32> {
    let value = 1.0 / num_classes as f32;
    ArrayD::from_elem(IxDyn(&[1, num_classes]), value)
}

/// Create sample token IDs for testing.
pub fn sample_token_ids(length: usize) -> Vec<i64> {
    (0..length as i64).collect()
}

/// Sample model metadata JSON for TTS testing.
pub fn sample_tts_metadata_json() -> &'static str {
    r#"{
        "model_id": "test-tts",
        "version": "1.0",
        "description": "Test TTS model",
        "execution_template": {
            "type": "SimpleMode",
            "model_file": "model.onnx"
        },
        "preprocessing": [
            {
                "type": "Phonemize",
                "backend": "CmuDictionary",
                "tokens_file": "tokens.txt"
            }
        ],
        "postprocessing": [
            {
                "type": "TTSAudioEncode",
                "sample_rate": 24000,
                "apply_postprocessing": true
            }
        ],
        "files": ["model.onnx", "tokens.txt"],
        "metadata": {
            "task": "text-to-speech",
            "sample_rate": 24000
        }
    }"#
}

/// Sample model metadata JSON for ASR testing.
pub fn sample_asr_metadata_json() -> &'static str {
    r#"{
        "model_id": "test-asr",
        "version": "1.0",
        "description": "Test ASR model",
        "execution_template": {
            "type": "SimpleMode",
            "model_file": "model.onnx"
        },
        "preprocessing": [
            {
                "type": "AudioDecode",
                "sample_rate": 16000,
                "channels": 1
            }
        ],
        "postprocessing": [
            {
                "type": "CTCDecode",
                "vocab_file": "vocab.json",
                "blank_index": 0
            }
        ],
        "files": ["model.onnx", "vocab.json"],
        "metadata": {
            "task": "speech-recognition",
            "sample_rate": 16000
        }
    }"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_audio_16khz() {
        let audio = sample_audio_16khz(1.0);
        assert_eq!(audio.len(), 16000);
        assert!(audio.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_sine_wave() {
        let wave = sine_wave(16000, 440.0, 0.1);
        assert_eq!(wave.len(), 1600);
        // Sine wave should have values between -1 and 1
        assert!(wave.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_text_envelope() {
        let envelope = text_envelope("Hello");
        match envelope.kind {
            EnvelopeKind::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("Expected Text envelope"),
        }
    }

    #[test]
    fn test_sample_mel_tensor() {
        let mel = sample_mel_tensor(80, 100);
        assert_eq!(mel.shape(), &[1, 80, 100]);
    }

    #[test]
    fn test_sample_logits() {
        let logits = sample_logits(10);
        assert_eq!(logits.shape(), &[1, 10]);
    }
}
