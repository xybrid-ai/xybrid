//! AudioEnvelope - Type-safe audio container with rich metadata.

use crate::ir::{Envelope, EnvelopeKind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use super::convert::{
    multichannel_to_mono, normalize_pcm16_to_f32, resample_audio, ResampleMethod,
};
use super::format::AudioFormat;

/// Type-safe audio container with rich metadata.
///
/// `AudioEnvelope` provides a high-level interface for working with audio data,
/// including format detection, validation, and conversion. It integrates with
/// the pipeline `Envelope` system for seamless data flow.
///
/// ## Features
///
/// - Parse WAV files and extract metadata
/// - Store decoded f32 samples (normalized to [-1.0, 1.0])
/// - Track sample rate, channels, and format
/// - Convert to pipeline `Envelope` for inference
/// - Automatic resampling and mono conversion
///
/// ## Example
///
/// ```rust,no_run
/// use xybrid_core::audio::AudioEnvelope;
///
/// // Load from WAV file
/// let wav_bytes = std::fs::read("audio.wav").unwrap();
/// let audio = AudioEnvelope::from_wav(&wav_bytes).unwrap();
///
/// println!("Duration: {}ms", audio.duration_ms());
/// println!("Sample Rate: {}Hz", audio.sample_rate);
/// println!("Channels: {}", audio.channels);
///
/// // Convert to pipeline envelope
/// let envelope = audio.to_envelope();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEnvelope {
    /// Decoded audio samples (f32, normalized to [-1.0, 1.0])
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Original format of the audio data
    pub format: AudioFormat,
}

impl AudioEnvelope {
    /// Creates a new AudioEnvelope from f32 samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (normalized to [-1.0, 1.0])
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    ///
    /// # Returns
    ///
    /// A new `AudioEnvelope` instance.
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            format: AudioFormat::Float32 {
                sample_rate,
                channels,
            },
        }
    }

    /// Creates an AudioEnvelope from WAV file bytes.
    ///
    /// Parses the WAV header, extracts audio data, and converts to f32 samples.
    ///
    /// # Arguments
    ///
    /// * `wav_bytes` - Raw WAV file bytes
    ///
    /// # Returns
    ///
    /// An `AudioEnvelope` with parsed audio data, or an error.
    ///
    /// # Supported Formats
    ///
    /// - 8-bit PCM (unsigned)
    /// - 16-bit PCM (most common)
    /// - 32-bit PCM
    /// - 32-bit IEEE float
    ///
    /// # Implementation Note
    ///
    /// This uses a custom lightweight WAV parser to minimize dependencies.
    /// For production use cases requiring broader format support (24-bit PCM,
    /// extensible format, metadata chunks, etc.), consider replacing with
    /// the `hound` crate (<https://crates.io/crates/hound>).
    pub fn from_wav(wav_bytes: &[u8]) -> Result<Self, AudioEnvelopeError> {
        // Validate minimum WAV header size
        if wav_bytes.len() < 44 {
            return Err(AudioEnvelopeError::InvalidFormat(
                "WAV file too small for header".to_string(),
            ));
        }

        // Check RIFF header
        if &wav_bytes[0..4] != b"RIFF" {
            return Err(AudioEnvelopeError::InvalidFormat(
                "Missing RIFF header".to_string(),
            ));
        }

        // Check WAVE format
        if &wav_bytes[8..12] != b"WAVE" {
            return Err(AudioEnvelopeError::InvalidFormat(
                "Not a WAVE file".to_string(),
            ));
        }

        // Parse format chunk - find "fmt " subchunk
        let mut pos = 12;
        let mut sample_rate = 0u32;
        let mut channels = 0u16;
        let mut bits_per_sample = 0u16;
        let mut audio_format = 0u16;
        let mut data_start = 0usize;
        let mut data_size = 0usize;

        // WAVE_FORMAT_EXTENSIBLE constants
        const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE; // 65534
                                                    // SubFormat GUIDs (first 2 bytes identify the format)
        const KSDATAFORMAT_SUBTYPE_PCM: u16 = 0x0001;
        const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: u16 = 0x0003;

        while pos + 8 <= wav_bytes.len() {
            let chunk_id = &wav_bytes[pos..pos + 4];
            let chunk_size = u32::from_le_bytes([
                wav_bytes[pos + 4],
                wav_bytes[pos + 5],
                wav_bytes[pos + 6],
                wav_bytes[pos + 7],
            ]) as usize;

            if chunk_id == b"fmt " {
                if pos + 8 + 16 > wav_bytes.len() {
                    return Err(AudioEnvelopeError::InvalidFormat(
                        "fmt chunk too small".to_string(),
                    ));
                }

                audio_format = u16::from_le_bytes([wav_bytes[pos + 8], wav_bytes[pos + 9]]);
                channels = u16::from_le_bytes([wav_bytes[pos + 10], wav_bytes[pos + 11]]);
                sample_rate = u32::from_le_bytes([
                    wav_bytes[pos + 12],
                    wav_bytes[pos + 13],
                    wav_bytes[pos + 14],
                    wav_bytes[pos + 15],
                ]);
                bits_per_sample = u16::from_le_bytes([wav_bytes[pos + 22], wav_bytes[pos + 23]]);

                // Handle WAVE_FORMAT_EXTENSIBLE - extract the actual format from SubFormat GUID
                if audio_format == WAVE_FORMAT_EXTENSIBLE {
                    // WAVEFORMATEXTENSIBLE structure:
                    // - Base WAVEFORMATEX (16 bytes used above)
                    // - cbSize (2 bytes at offset 16) - size of extension
                    // - wValidBitsPerSample (2 bytes at offset 18)
                    // - dwChannelMask (4 bytes at offset 20)
                    // - SubFormat GUID (16 bytes at offset 24)
                    //   First 2 bytes of GUID contain the actual format code

                    let fmt_chunk_start = pos + 8;
                    let extension_offset = fmt_chunk_start + 24; // Offset to SubFormat GUID

                    if extension_offset + 2 <= wav_bytes.len() {
                        let sub_format = u16::from_le_bytes([
                            wav_bytes[extension_offset],
                            wav_bytes[extension_offset + 1],
                        ]);

                        // Map SubFormat to standard audio_format codes
                        audio_format = match sub_format {
                            KSDATAFORMAT_SUBTYPE_PCM => 1,        // PCM
                            KSDATAFORMAT_SUBTYPE_IEEE_FLOAT => 3, // IEEE Float
                            _ => sub_format,                      // Pass through unknown formats
                        };
                    }
                }
            } else if chunk_id == b"data" {
                data_start = pos + 8;
                data_size = chunk_size;
                break;
            }

            pos += 8 + chunk_size;
            // Align to even boundary
            if !chunk_size.is_multiple_of(2) {
                pos += 1;
            }
        }

        if data_start == 0 || data_size == 0 {
            return Err(AudioEnvelopeError::InvalidFormat(
                "Missing data chunk".to_string(),
            ));
        }

        if sample_rate == 0 {
            return Err(AudioEnvelopeError::InvalidFormat(
                "Missing fmt chunk".to_string(),
            ));
        }

        // Extract audio data
        let data_end = (data_start + data_size).min(wav_bytes.len());
        let audio_data = &wav_bytes[data_start..data_end];

        // Convert to f32 samples based on format
        let samples = match (audio_format, bits_per_sample) {
            (1, 16) => {
                // PCM 16-bit
                normalize_pcm16_to_f32(audio_data)
            }
            (1, 32) => {
                // PCM 32-bit
                let num_samples = audio_data.len() / 4;
                let mut samples = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    let idx = i * 4;
                    if idx + 3 < audio_data.len() {
                        let sample_i32 = i32::from_le_bytes([
                            audio_data[idx],
                            audio_data[idx + 1],
                            audio_data[idx + 2],
                            audio_data[idx + 3],
                        ]);
                        samples.push(sample_i32 as f32 / 2147483648.0);
                    }
                }
                samples
            }
            (3, 32) => {
                // IEEE float 32-bit
                let num_samples = audio_data.len() / 4;
                let mut samples = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    let idx = i * 4;
                    if idx + 3 < audio_data.len() {
                        let sample_f32 = f32::from_le_bytes([
                            audio_data[idx],
                            audio_data[idx + 1],
                            audio_data[idx + 2],
                            audio_data[idx + 3],
                        ]);
                        samples.push(sample_f32);
                    }
                }
                samples
            }
            (1, 8) => {
                // PCM 8-bit unsigned
                audio_data
                    .iter()
                    .map(|&b| (b as f32 - 128.0) / 128.0)
                    .collect()
            }
            _ => {
                return Err(AudioEnvelopeError::UnsupportedFormat(format!(
                    "Unsupported WAV format: audio_format={}, bits_per_sample={}",
                    audio_format, bits_per_sample
                )));
            }
        };

        let format = match bits_per_sample {
            16 => AudioFormat::Pcm16 {
                sample_rate,
                channels: channels as u32,
            },
            32 if audio_format == 3 => AudioFormat::Float32 {
                sample_rate,
                channels: channels as u32,
            },
            _ => AudioFormat::Wav,
        };

        Ok(Self {
            samples,
            sample_rate,
            channels: channels as u32,
            format,
        })
    }

    /// Creates an AudioEnvelope from raw PCM16 bytes.
    ///
    /// # Arguments
    ///
    /// * `pcm_bytes` - Raw 16-bit PCM bytes (little-endian)
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    ///
    /// # Returns
    ///
    /// A new `AudioEnvelope` with decoded samples.
    pub fn from_pcm16(pcm_bytes: &[u8], sample_rate: u32, channels: u32) -> Self {
        let samples = normalize_pcm16_to_f32(pcm_bytes);
        Self {
            samples,
            sample_rate,
            channels,
            format: AudioFormat::Pcm16 {
                sample_rate,
                channels,
            },
        }
    }

    /// Returns the duration of the audio in milliseconds.
    pub fn duration_ms(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        let num_frames = self.samples.len() / self.channels as usize;
        (num_frames as f64 / self.sample_rate as f64) * 1000.0
    }

    /// Returns the duration of the audio in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.duration_ms() / 1000.0
    }

    /// Returns the number of audio frames (samples per channel).
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Converts to mono by averaging channels.
    ///
    /// If already mono, returns a clone.
    pub fn to_mono(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }

        let mono_samples = multichannel_to_mono(&self.samples, self.channels);

        Self {
            samples: mono_samples,
            sample_rate: self.sample_rate,
            channels: 1,
            format: AudioFormat::Float32 {
                sample_rate: self.sample_rate,
                channels: 1,
            },
        }
    }

    /// Resamples to a target sample rate.
    ///
    /// # Arguments
    ///
    /// * `target_rate` - Target sample rate in Hz
    ///
    /// # Returns
    ///
    /// A new `AudioEnvelope` with resampled audio.
    pub fn resample(&self, target_rate: u32) -> Result<Self, AudioEnvelopeError> {
        if target_rate == self.sample_rate {
            return Ok(self.clone());
        }

        let resampled = resample_audio(
            &self.samples,
            self.sample_rate,
            target_rate,
            ResampleMethod::Linear,
        )
        .map_err(|e| AudioEnvelopeError::ConversionError(e.to_string()))?;

        Ok(Self {
            samples: resampled,
            sample_rate: target_rate,
            channels: self.channels,
            format: AudioFormat::Float32 {
                sample_rate: target_rate,
                channels: self.channels,
            },
        })
    }

    /// Prepares audio for ASR models (16kHz mono).
    ///
    /// Converts to mono and resamples to 16kHz if necessary.
    pub fn prepare_for_asr(&self) -> Result<Self, AudioEnvelopeError> {
        let mono = self.to_mono();
        mono.resample(16000)
    }

    /// Converts to a pipeline Envelope.
    ///
    /// The envelope contains the raw f32 samples as bytes, with metadata
    /// for sample rate, channels, and format.
    pub fn to_envelope(&self) -> Envelope {
        // Convert f32 samples to bytes
        let mut bytes = Vec::with_capacity(self.samples.len() * 4);
        for &sample in &self.samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        let mut metadata = HashMap::new();
        metadata.insert("sample_rate".to_string(), self.sample_rate.to_string());
        metadata.insert("channels".to_string(), self.channels.to_string());
        metadata.insert("format".to_string(), "float32".to_string());
        metadata.insert("num_samples".to_string(), self.samples.len().to_string());
        metadata.insert(
            "duration_ms".to_string(),
            format!("{:.2}", self.duration_ms()),
        );

        Envelope {
            kind: EnvelopeKind::Audio(bytes),
            metadata,
        }
    }

    /// Creates an AudioEnvelope from a pipeline Envelope.
    ///
    /// Requires the envelope to contain float32 audio data with proper metadata.
    pub fn from_envelope(envelope: &Envelope) -> Result<Self, AudioEnvelopeError> {
        let audio_bytes = match &envelope.kind {
            EnvelopeKind::Audio(bytes) => bytes,
            _ => {
                return Err(AudioEnvelopeError::InvalidFormat(
                    "Envelope is not an Audio type".to_string(),
                ))
            }
        };

        let sample_rate: u32 = envelope
            .get_metadata("sample_rate")
            .and_then(|s| s.parse().ok())
            .unwrap_or(16000);

        let channels: u32 = envelope
            .get_metadata("channels")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let format_str = envelope
            .get_metadata("format")
            .map(|s| s.as_str())
            .unwrap_or("pcm16");

        let samples = match format_str {
            "float32" => {
                let num_samples = audio_bytes.len() / 4;
                let mut samples = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    let idx = i * 4;
                    if idx + 3 < audio_bytes.len() {
                        let sample = f32::from_le_bytes([
                            audio_bytes[idx],
                            audio_bytes[idx + 1],
                            audio_bytes[idx + 2],
                            audio_bytes[idx + 3],
                        ]);
                        samples.push(sample);
                    }
                }
                samples
            }
            "pcm16" | _ => normalize_pcm16_to_f32(audio_bytes),
        };

        Ok(Self {
            samples,
            sample_rate,
            channels,
            format: AudioFormat::Float32 {
                sample_rate,
                channels,
            },
        })
    }
}

/// Error type for AudioEnvelope operations.
#[derive(Error, Debug)]
pub enum AudioEnvelopeError {
    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("Conversion error: {0}")]
    ConversionError(String),

    #[error("IO error: {0}")]
    IoError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data_size = samples.len() * 2;
        let file_size = 36 + data_size;

        let mut wav = Vec::new();

        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(file_size as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav.extend_from_slice(&1u16.to_le_bytes()); // audio format (PCM)
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * channels as u32 * 2;
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        let block_align = channels * 2;
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_size as u32).to_le_bytes());
        for &sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        wav
    }

    #[test]
    fn test_from_wav_basic() {
        let samples: Vec<i16> = vec![0, 16384, 32767, 16384, 0, -16384, -32767, -16384];
        let wav = create_test_wav(16000, 1, &samples);

        let audio = AudioEnvelope::from_wav(&wav).unwrap();

        assert_eq!(audio.sample_rate, 16000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 8);
    }

    #[test]
    fn test_from_wav_stereo() {
        let samples: Vec<i16> = vec![0, 0, 16384, 16384]; // L, R, L, R
        let wav = create_test_wav(44100, 2, &samples);

        let audio = AudioEnvelope::from_wav(&wav).unwrap();

        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.samples.len(), 4);
    }

    #[test]
    fn test_duration_ms() {
        let audio = AudioEnvelope::new(vec![0.0f32; 16000], 16000, 1);
        assert!((audio.duration_ms() - 1000.0).abs() < 1.0);

        let audio_stereo = AudioEnvelope::new(vec![0.0f32; 32000], 16000, 2);
        assert!((audio_stereo.duration_ms() - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_to_mono() {
        let stereo = AudioEnvelope::new(
            vec![0.5f32, 0.3, 1.0, 0.0, -0.5, 0.5], // L, R, L, R, L, R
            16000,
            2,
        );

        let mono = stereo.to_mono();

        assert_eq!(mono.channels, 1);
        assert_eq!(mono.samples.len(), 3);
        assert!((mono.samples[0] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_resample() {
        let audio = AudioEnvelope::new(vec![0.0f32; 44100], 44100, 1);
        let resampled = audio.resample(16000).unwrap();

        assert_eq!(resampled.sample_rate, 16000);
        // Should be approximately 16000 samples
        assert!((resampled.samples.len() as i32 - 16000).abs() <= 1);
    }

    #[test]
    fn test_prepare_for_asr() {
        let audio = AudioEnvelope::new(vec![0.0f32; 88200], 44100, 2); // 1 sec stereo @ 44.1kHz
        let asr_ready = audio.prepare_for_asr().unwrap();

        assert_eq!(asr_ready.sample_rate, 16000);
        assert_eq!(asr_ready.channels, 1);
        // Should be approximately 16000 samples for 1 second
        assert!((asr_ready.samples.len() as i32 - 16000).abs() <= 2);
    }

    #[test]
    fn test_to_envelope_roundtrip() {
        let original = AudioEnvelope::new(vec![0.1f32, 0.2, 0.3, 0.4], 16000, 1);
        let envelope = original.to_envelope();
        let restored = AudioEnvelope::from_envelope(&envelope).unwrap();

        assert_eq!(restored.sample_rate, original.sample_rate);
        assert_eq!(restored.channels, original.channels);
        assert_eq!(restored.samples.len(), original.samples.len());

        for (a, b) in restored.samples.iter().zip(original.samples.iter()) {
            assert!((a - b).abs() < 0.0001);
        }
    }

    #[test]
    fn test_from_pcm16() {
        let pcm: Vec<u8> = vec![0, 0, 0xFF, 0x7F, 0x00, 0x80]; // 0, max+, max-
        let audio = AudioEnvelope::from_pcm16(&pcm, 16000, 1);

        assert_eq!(audio.samples.len(), 3);
        assert!((audio.samples[0] - 0.0).abs() < 0.001);
        assert!((audio.samples[1] - 1.0).abs() < 0.001);
        assert!((audio.samples[2] - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_invalid_wav() {
        let invalid = b"not a wav file";
        let result = AudioEnvelope::from_wav(invalid);
        assert!(result.is_err());
    }

    /// Create a WAVE_FORMAT_EXTENSIBLE WAV file (format code 65534)
    fn create_extensible_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data_size = samples.len() * 2;
        // WAVEFORMATEXTENSIBLE has a larger fmt chunk: 40 bytes total (16 base + 2 cbSize + 22 extension)
        let fmt_chunk_size = 40u32;
        let file_size = 4 + 8 + fmt_chunk_size as usize + 8 + data_size; // WAVE + fmt chunk + data chunk

        let mut wav = Vec::new();

        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(file_size as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        // fmt chunk (WAVEFORMATEXTENSIBLE)
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&fmt_chunk_size.to_le_bytes()); // chunk size = 40

        // WAVEFORMATEX portion (16 bytes)
        wav.extend_from_slice(&0xFFFEu16.to_le_bytes()); // wFormatTag = WAVE_FORMAT_EXTENSIBLE (65534)
        wav.extend_from_slice(&channels.to_le_bytes()); // nChannels
        wav.extend_from_slice(&sample_rate.to_le_bytes()); // nSamplesPerSec
        let byte_rate = sample_rate * channels as u32 * 2;
        wav.extend_from_slice(&byte_rate.to_le_bytes()); // nAvgBytesPerSec
        let block_align = channels * 2;
        wav.extend_from_slice(&block_align.to_le_bytes()); // nBlockAlign
        wav.extend_from_slice(&16u16.to_le_bytes()); // wBitsPerSample

        // Extension (24 bytes)
        wav.extend_from_slice(&22u16.to_le_bytes()); // cbSize = 22 (size of extension)
        wav.extend_from_slice(&16u16.to_le_bytes()); // wValidBitsPerSample
        wav.extend_from_slice(&0u32.to_le_bytes()); // dwChannelMask (0 = unspecified)

        // SubFormat GUID for PCM: 00000001-0000-0010-8000-00aa00389b71
        // First 2 bytes = 0x0001 (PCM)
        wav.extend_from_slice(&[
            0x01, 0x00, // SubFormat = PCM
            0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
        ]);

        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_size as u32).to_le_bytes());
        for &sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        wav
    }

    #[test]
    fn test_from_wav_extensible_format() {
        // WAVE_FORMAT_EXTENSIBLE (65534) is commonly used by mobile audio recorders
        let samples: Vec<i16> = vec![0, 16384, 32767, 16384, 0, -16384, -32767, -16384];
        let wav = create_extensible_wav(16000, 1, &samples);

        let audio = AudioEnvelope::from_wav(&wav).unwrap();

        assert_eq!(audio.sample_rate, 16000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 8);
        // Verify samples are correctly normalized
        assert!((audio.samples[2] - 1.0).abs() < 0.001); // 32767 -> ~1.0
        assert!((audio.samples[6] - (-1.0)).abs() < 0.001); // -32767 -> ~-1.0
    }

    #[test]
    fn test_from_wav_extensible_stereo() {
        let samples: Vec<i16> = vec![0, 0, 16384, 16384]; // L, R, L, R
        let wav = create_extensible_wav(44100, 2, &samples);

        let audio = AudioEnvelope::from_wav(&wav).unwrap();

        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.samples.len(), 4);
    }
}
