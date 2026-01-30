//! Mel spectrogram configuration types.
//!
//! This module defines the configuration types used across all mel spectrogram
//! implementations, including the mel scale selection and padding options.

use serde::{Deserialize, Serialize};

/// Mel frequency scale variant.
///
/// Different models use different mel scale formulas. Using the wrong scale
/// will produce incorrect mel spectrograms and poor model performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MelScale {
    /// Slaney mel scale (used by Whisper, transformers.js, librosa with `htk=False`).
    ///
    /// Formula:
    /// - For freq < 1000 Hz: `mel = 3 × freq / 200`
    /// - For freq >= 1000 Hz: `mel = 15 + 27 × ln(freq / 1000) / ln(6.4)`
    ///
    /// This is the default for most modern speech models.
    #[default]
    Slaney,

    /// HTK mel scale (used by older implementations, librosa with `htk=True`).
    ///
    /// Formula: `mel = 2595 × log10(1 + freq / 700)`
    ///
    /// This produces different filter bank shapes than Slaney.
    Htk,
}

/// Padding mode for audio signal before STFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PaddingMode {
    /// Reflect padding (mirrors signal at boundaries).
    /// Used by Whisper for exact compatibility.
    #[default]
    Reflect,

    /// Zero padding (pads with zeros).
    /// Simpler but may introduce edge artifacts.
    Zero,

    /// No padding (signal is used as-is).
    /// May lose frames at boundaries.
    None,
}

/// Configuration for mel spectrogram computation.
///
/// This configuration is used by all mel spectrogram implementations.
/// Use presets for common model configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelConfig {
    /// Number of mel frequency bins.
    pub n_mels: usize,

    /// FFT window size in samples.
    pub n_fft: usize,

    /// Hop length between STFT frames in samples.
    pub hop_length: usize,

    /// Audio sample rate in Hz.
    pub sample_rate: u32,

    /// Mel frequency scale to use.
    #[serde(default)]
    pub mel_scale: MelScale,

    /// Minimum frequency for mel filter bank (Hz).
    #[serde(default)]
    pub f_min: f64,

    /// Maximum frequency for mel filter bank (Hz).
    /// If 0.0, defaults to sample_rate / 2 (Nyquist).
    #[serde(default)]
    pub f_max: f64,

    /// Padding mode for audio signal.
    #[serde(default)]
    pub padding: PaddingMode,

    /// Maximum number of output frames.
    /// If Some, output is padded/truncated to this length.
    /// If None, output length depends on input length.
    #[serde(default)]
    pub max_frames: Option<usize>,

    /// Whether to apply Whisper-style normalization.
    /// `(max(log_spec, max - 8) + 4) / 4`
    #[serde(default = "default_normalize")]
    pub normalize: bool,
}

fn default_normalize() -> bool {
    true
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            sample_rate: 16000,
            mel_scale: MelScale::Slaney,
            f_min: 0.0,
            f_max: 8000.0,
            padding: PaddingMode::Reflect,
            max_frames: Some(3000),
            normalize: true,
        }
    }
}

impl MelConfig {
    /// Create configuration from a preset name.
    ///
    /// # Presets
    ///
    /// - `"whisper"`: OpenAI Whisper / transformers.js compatible
    /// - `"whisper-large"`: Whisper large model (128 mels)
    /// - `"wav2vec2"`: Wav2Vec2 compatible
    ///
    /// Returns `None` if preset is not recognized.
    pub fn from_preset(preset: &str) -> Option<Self> {
        match preset.to_lowercase().as_str() {
            "whisper" | "whisper-tiny" | "whisper-base" | "whisper-small" | "whisper-medium" => {
                Some(Self::whisper())
            }
            "whisper-large" | "whisper-large-v2" | "whisper-large-v3" => {
                Some(Self::whisper_large())
            }
            _ => None,
        }
    }

    /// Whisper-compatible configuration (tiny/base/small/medium).
    ///
    /// - 80 mel bins
    /// - Slaney mel scale
    /// - Reflect padding (200 samples each side)
    /// - 30 seconds max (3000 frames @ 100fps)
    pub fn whisper() -> Self {
        Self {
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            sample_rate: 16000,
            mel_scale: MelScale::Slaney,
            f_min: 0.0,
            f_max: 8000.0,
            padding: PaddingMode::Reflect,
            max_frames: Some(3000),
            normalize: true,
        }
    }

    /// Whisper large model configuration.
    ///
    /// Same as whisper() but with 128 mel bins.
    pub fn whisper_large() -> Self {
        Self {
            n_mels: 128,
            ..Self::whisper()
        }
    }

    /// Generic mel spectrogram configuration using HTK scale.
    ///
    /// Compatible with older implementations and mel-spec library defaults.
    pub fn htk_default() -> Self {
        Self {
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            sample_rate: 16000,
            mel_scale: MelScale::Htk,
            f_min: 0.0,
            f_max: 0.0, // Will use Nyquist
            padding: PaddingMode::Zero,
            max_frames: None,
            normalize: true,
        }
    }

    /// Get effective f_max (uses Nyquist if f_max is 0).
    pub fn effective_f_max(&self) -> f64 {
        if self.f_max <= 0.0 {
            self.sample_rate as f64 / 2.0
        } else {
            self.f_max
        }
    }

    /// Calculate padding size for STFT.
    pub fn pad_size(&self) -> usize {
        match self.padding {
            PaddingMode::Reflect | PaddingMode::Zero => (self.n_fft - 1) / 2 + 1,
            PaddingMode::None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_whisper() {
        let config = MelConfig::from_preset("whisper").unwrap();
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.mel_scale, MelScale::Slaney);
        assert_eq!(config.max_frames, Some(3000));
    }

    #[test]
    fn test_preset_whisper_large() {
        let config = MelConfig::from_preset("whisper-large").unwrap();
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.mel_scale, MelScale::Slaney);
    }

    #[test]
    fn test_preset_unknown() {
        assert!(MelConfig::from_preset("unknown").is_none());
    }

    #[test]
    fn test_effective_f_max() {
        let config = MelConfig {
            sample_rate: 16000,
            f_max: 0.0,
            ..Default::default()
        };
        assert_eq!(config.effective_f_max(), 8000.0);

        let config2 = MelConfig {
            sample_rate: 16000,
            f_max: 4000.0,
            ..Default::default()
        };
        assert_eq!(config2.effective_f_max(), 4000.0);
    }

    #[test]
    fn test_serde_roundtrip() {
        let config = MelConfig::whisper();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: MelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.n_mels, config.n_mels);
        assert_eq!(parsed.mel_scale, config.mel_scale);
    }
}
