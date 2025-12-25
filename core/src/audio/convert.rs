//! Audio conversion utilities.

use thiserror::Error;

/// Resampling method for audio conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMethod {
    /// Nearest neighbor (fast, low quality)
    NearestNeighbor,
    /// Linear interpolation (fast, medium quality)
    Linear,
}

impl Default for ResampleMethod {
    fn default() -> Self {
        ResampleMethod::Linear
    }
}

/// Resamples audio samples from one sample rate to another.
///
/// # Arguments
///
/// * `samples` - Input audio samples (f32, normalized to [-1.0, 1.0])
/// * `from_rate` - Source sample rate in Hz
/// * `to_rate` - Target sample rate in Hz
/// * `method` - Resampling method to use
///
/// # Returns
///
/// Resampled audio samples or an error.
///
/// # Example
///
/// ```rust
/// use xybrid_core::audio::{resample_audio, ResampleMethod};
///
/// // Resample from 44.1kHz to 16kHz
/// let samples_44k = vec![0.0f32; 44100]; // 1 second
/// let samples_16k = resample_audio(&samples_44k, 44100, 16000, ResampleMethod::Linear).unwrap();
/// assert_eq!(samples_16k.len(), 16000);
/// ```
pub fn resample_audio(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
    method: ResampleMethod,
) -> Result<Vec<f32>, ConvertError> {
    if from_rate == 0 || to_rate == 0 {
        return Err(ConvertError::InvalidParameter(
            "Sample rate must be greater than 0".to_string(),
        ));
    }

    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;

    if output_len == 0 {
        return Ok(Vec::new());
    }

    let mut output = Vec::with_capacity(output_len);

    match method {
        ResampleMethod::NearestNeighbor => {
            for i in 0..output_len {
                let src_idx = (i as f64 / ratio).floor() as usize;
                let src_idx = src_idx.min(samples.len() - 1);
                output.push(samples[src_idx]);
            }
        }
        ResampleMethod::Linear => {
            for i in 0..output_len {
                let src_pos = i as f64 / ratio;
                let src_idx = src_pos.floor() as usize;
                let frac = (src_pos - src_idx as f64) as f32;

                if src_idx + 1 < samples.len() {
                    // Linear interpolation between two samples
                    let sample = samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac;
                    output.push(sample);
                } else if src_idx < samples.len() {
                    output.push(samples[src_idx]);
                }
            }
        }
    }

    Ok(output)
}

/// Converts 16-bit signed PCM bytes to f32 samples.
///
/// # Arguments
///
/// * `pcm_bytes` - Raw PCM16 bytes (little-endian)
///
/// # Returns
///
/// Audio samples normalized to [-1.0, 1.0].
///
/// # Example
///
/// ```rust
/// use xybrid_core::audio::normalize_pcm16_to_f32;
///
/// let pcm_bytes: Vec<u8> = vec![0, 0, 0xFF, 0x7F]; // silence, max positive
/// let samples = normalize_pcm16_to_f32(&pcm_bytes);
/// assert_eq!(samples.len(), 2);
/// assert!((samples[0] - 0.0).abs() < 0.001); // ~0
/// assert!((samples[1] - 1.0).abs() < 0.001); // ~1
/// ```
pub fn normalize_pcm16_to_f32(pcm_bytes: &[u8]) -> Vec<f32> {
    let num_samples = pcm_bytes.len() / 2;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let idx = i * 2;
        if idx + 1 < pcm_bytes.len() {
            let sample_i16 = i16::from_le_bytes([pcm_bytes[idx], pcm_bytes[idx + 1]]);
            let sample_f32 = sample_i16 as f32 / 32768.0;
            samples.push(sample_f32);
        }
    }

    samples
}

/// Converts f32 samples to 16-bit signed PCM bytes.
///
/// # Arguments
///
/// * `samples` - Audio samples (expected range [-1.0, 1.0])
///
/// # Returns
///
/// Raw PCM16 bytes (little-endian).
///
/// # Example
///
/// ```rust
/// use xybrid_core::audio::f32_to_pcm16;
///
/// let samples = vec![0.0f32, 1.0, -1.0];
/// let pcm_bytes = f32_to_pcm16(&samples);
/// assert_eq!(pcm_bytes.len(), 6); // 3 samples * 2 bytes
/// ```
pub fn f32_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut pcm_bytes = Vec::with_capacity(samples.len() * 2);

    for &sample in samples {
        // Clamp to [-1.0, 1.0] range
        let clamped = sample.clamp(-1.0, 1.0);
        // Convert to i16
        let sample_i16 = (clamped * 32767.0) as i16;
        pcm_bytes.extend_from_slice(&sample_i16.to_le_bytes());
    }

    pcm_bytes
}

/// Converts stereo audio to mono by averaging channels.
///
/// # Arguments
///
/// * `samples` - Interleaved stereo samples (L, R, L, R, ...)
///
/// # Returns
///
/// Mono samples.
pub fn stereo_to_mono(samples: &[f32]) -> Vec<f32> {
    let num_frames = samples.len() / 2;
    let mut mono = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let left = samples[i * 2];
        let right = samples.get(i * 2 + 1).copied().unwrap_or(left);
        mono.push((left + right) / 2.0);
    }

    mono
}

/// Converts multi-channel audio to mono by averaging all channels.
///
/// # Arguments
///
/// * `samples` - Interleaved multi-channel samples
/// * `channels` - Number of channels
///
/// # Returns
///
/// Mono samples.
pub fn multichannel_to_mono(samples: &[f32], channels: u32) -> Vec<f32> {
    if channels == 0 {
        return Vec::new();
    }
    if channels == 1 {
        return samples.to_vec();
    }
    if channels == 2 {
        return stereo_to_mono(samples);
    }

    let channels = channels as usize;
    let num_frames = samples.len() / channels;
    let mut mono = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let mut sum = 0.0f32;
        for ch in 0..channels {
            sum += samples.get(i * channels + ch).copied().unwrap_or(0.0);
        }
        mono.push(sum / channels as f32);
    }

    mono
}

/// Converts f32 samples to a complete WAV file.
///
/// Creates a valid WAV file with RIFF header, fmt chunk, and data chunk.
/// Output is 16-bit PCM, mono.
///
/// # Arguments
///
/// * `samples` - Audio samples (expected range [-1.0, 1.0])
/// * `sample_rate` - Sample rate in Hz (e.g., 16000 for ASR)
///
/// # Returns
///
/// Complete WAV file as bytes.
///
/// # Example
///
/// ```rust
/// use xybrid_core::audio::samples_to_wav;
///
/// let samples = vec![0.0f32; 16000]; // 1 second of silence at 16kHz
/// let wav_bytes = samples_to_wav(&samples, 16000);
///
/// // Verify WAV header
/// assert_eq!(&wav_bytes[0..4], b"RIFF");
/// assert_eq!(&wav_bytes[8..12], b"WAVE");
/// ```
pub fn samples_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let pcm_bytes = f32_to_pcm16(samples);
    let data_size = pcm_bytes.len() as u32;
    let file_size = 36 + data_size; // 36 = header size before data chunk

    let mut buffer = Vec::with_capacity(44 + pcm_bytes.len());

    // RIFF header
    buffer.extend_from_slice(b"RIFF");
    buffer.extend_from_slice(&file_size.to_le_bytes());
    buffer.extend_from_slice(b"WAVE");

    // fmt chunk
    buffer.extend_from_slice(b"fmt ");
    buffer.extend_from_slice(&16u32.to_le_bytes()); // chunk size (16 for PCM)
    buffer.extend_from_slice(&1u16.to_le_bytes()); // audio format (1 = PCM)
    buffer.extend_from_slice(&1u16.to_le_bytes()); // num channels (mono)
    buffer.extend_from_slice(&sample_rate.to_le_bytes()); // sample rate
    buffer.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate (sample_rate * channels * bytes_per_sample)
    buffer.extend_from_slice(&2u16.to_le_bytes()); // block align (channels * bytes_per_sample)
    buffer.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    buffer.extend_from_slice(b"data");
    buffer.extend_from_slice(&data_size.to_le_bytes());
    buffer.extend_from_slice(&pcm_bytes);

    buffer
}

/// Error type for audio conversion operations.
#[derive(Error, Debug)]
pub enum ConvertError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Conversion error: {0}")]
    ConversionFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_same_rate() {
        let samples = vec![0.0f32, 0.5, 1.0, 0.5, 0.0];
        let result = resample_audio(&samples, 16000, 16000, ResampleMethod::Linear).unwrap();
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resample_downsample() {
        let samples: Vec<f32> = (0..44100).map(|i| (i as f32 / 44100.0)).collect();
        let result = resample_audio(&samples, 44100, 16000, ResampleMethod::Linear).unwrap();
        // Should be approximately 16000 samples for 1 second
        assert!((result.len() as i32 - 16000).abs() <= 1);
    }

    #[test]
    fn test_resample_upsample() {
        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 / 16000.0)).collect();
        let result = resample_audio(&samples, 16000, 44100, ResampleMethod::Linear).unwrap();
        // Should be approximately 44100 samples for 1 second
        assert!((result.len() as i32 - 44100).abs() <= 1);
    }

    #[test]
    fn test_resample_invalid_rate() {
        let samples = vec![0.0f32; 100];
        let result = resample_audio(&samples, 0, 16000, ResampleMethod::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_pcm16_to_f32() {
        // Test silence (0)
        let silence = vec![0u8, 0];
        let result = normalize_pcm16_to_f32(&silence);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.0).abs() < 0.0001);

        // Test max positive (32767 = 0x7FFF)
        let max_positive = vec![0xFF, 0x7F];
        let result = normalize_pcm16_to_f32(&max_positive);
        assert!((result[0] - 1.0).abs() < 0.001);

        // Test max negative (-32768 = 0x8000)
        let max_negative = vec![0x00, 0x80];
        let result = normalize_pcm16_to_f32(&max_negative);
        assert!((result[0] - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_f32_to_pcm16() {
        let samples = vec![0.0f32, 1.0, -1.0];
        let pcm = f32_to_pcm16(&samples);

        assert_eq!(pcm.len(), 6);

        // Silence
        let silence = i16::from_le_bytes([pcm[0], pcm[1]]);
        assert_eq!(silence, 0);

        // Max positive
        let max_pos = i16::from_le_bytes([pcm[2], pcm[3]]);
        assert_eq!(max_pos, 32767);

        // Max negative
        let max_neg = i16::from_le_bytes([pcm[4], pcm[5]]);
        assert_eq!(max_neg, -32767);
    }

    #[test]
    fn test_stereo_to_mono() {
        let stereo = vec![0.5f32, 0.3, 1.0, 0.0, -0.5, 0.5];
        let mono = stereo_to_mono(&stereo);

        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.4).abs() < 0.001); // (0.5 + 0.3) / 2
        assert!((mono[1] - 0.5).abs() < 0.001); // (1.0 + 0.0) / 2
        assert!((mono[2] - 0.0).abs() < 0.001); // (-0.5 + 0.5) / 2
    }

    #[test]
    fn test_multichannel_to_mono() {
        // Test with 4 channels
        let multi = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mono = multichannel_to_mono(&multi, 4);

        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.25).abs() < 0.001); // (0.1 + 0.2 + 0.3 + 0.4) / 4
        assert!((mono[1] - 0.65).abs() < 0.001); // (0.5 + 0.6 + 0.7 + 0.8) / 4
    }
}

/// Prepares audio samples by converting channels and resampling if necessary.
pub fn prepare_audio_samples(
    samples: Vec<f32>,
    source_rate: u32,
    source_channels: usize,
    target_rate: u32,
    target_channels: usize,
) -> Vec<f32> {
    // Step 1: Channel Conversion
    let samples = if source_channels == target_channels {
        samples
    } else if target_channels == 1 {
        multichannel_to_mono(&samples, source_channels as u32)
    } else {
        // TODO: Support mono -> stereo or other mappings if needed
        // For now, just clone (or error? But we return Vec<f32>)
        // Assuming mono->stereo is rare for ASR input which expects mono
        samples
    };

    // Step 2: Resampling
    if source_rate == target_rate {
        samples
    } else {
        resample_audio(
            &samples,
            source_rate,
            target_rate,
            ResampleMethod::Linear,
        ).unwrap_or(samples) // Fallback to original if resampling fails (shouldn't happen)
    }
}

/// Decode WAV audio bytes to float32 samples.
pub fn decode_wav_audio(
    audio_bytes: &[u8],
    target_sample_rate: u32,
    target_channels: usize,
) -> Result<Vec<f32>, ConvertError> {
    use std::io::Cursor;
    let cursor = Cursor::new(audio_bytes);

    match hound::WavReader::new(cursor) {
        Ok(mut reader) => {
            let spec = reader.spec();
            let source_sample_rate = spec.sample_rate;
            let source_channels = spec.channels as usize;

            // Read samples as f32
            let samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => reader
                    .samples::<f32>()
                    .filter_map(|s| s.ok())
                    .collect(),
                hound::SampleFormat::Int => {
                    let bits = spec.bits_per_sample;
                    let max_value = match (1i32).checked_shl((bits - 1) as u32) {
                        Some(val) => val as f32,
                        None => {
                            return Err(ConvertError::InvalidParameter(format!(
                                "Unsupported bits_per_sample: {} (must be < 32)",
                                bits
                            )));
                        }
                    };
                    reader
                        .samples::<i32>()
                        .filter_map(|s| s.ok())
                        .map(|s| s as f32 / max_value)
                        .collect()
                }
            };

            let prepared = prepare_audio_samples(
                samples,
                source_sample_rate,
                source_channels,
                target_sample_rate,
                target_channels,
            );

            Ok(prepared)
        }
        Err(e) => Err(ConvertError::InvalidParameter(format!(
            "Failed to decode WAV audio: {}. Only WAV format is currently supported.",
            e
        ))),
    }
}
