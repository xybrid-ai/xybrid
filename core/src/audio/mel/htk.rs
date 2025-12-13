//! HTK mel scale implementation.
//!
//! This implements the HTK (Hidden Markov Model Toolkit) mel scale used by:
//! - Older speech recognition systems
//! - mel-spec crate
//! - librosa (with `htk=True`)
//!
//! ## HTK Formula
//!
//! The HTK mel scale uses a logarithmic formula:
//!
//! `mel = 2595 × log10(1 + freq / 700)`
//!
//! This is a simpler formula than Slaney but produces different filter shapes.

// ============================================================================
// Mel Scale Conversion
// ============================================================================

/// Convert frequency in Hz to mel scale using HTK formula.
pub fn hz_to_mel(freq: f64) -> f64 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

/// Convert mel scale to frequency in Hz using HTK formula.
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

// ============================================================================
// Filter Bank Creation
// ============================================================================

/// Create HTK-style mel filter bank.
///
/// Uses standard triangular filters without Slaney normalization.
pub fn create_filter_bank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    f_min: f64,
    f_max: f64,
) -> Vec<Vec<f64>> {
    let n_freqs = n_fft / 2 + 1;

    // FFT bin frequencies
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| (i as f64 * sample_rate as f64) / n_fft as f64)
        .collect();

    // Mel scale center frequencies
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let n_mel_points = n_mels + 2;
    let mel_points: Vec<f64> = (0..n_mel_points)
        .map(|i| mel_min + (mel_max - mel_min) * (i as f64) / ((n_mel_points - 1) as f64))
        .collect();

    let freq_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Create triangular filters (no Slaney normalization)
    let mut filter_bank = vec![vec![0.0; n_freqs]; n_mels];

    for i in 0..n_mels {
        let f_lower = freq_points[i];
        let f_center = freq_points[i + 1];
        let f_upper = freq_points[i + 2];

        for (j, &freq) in fft_freqs.iter().enumerate() {
            if freq >= f_lower && freq <= f_center {
                filter_bank[i][j] = (freq - f_lower) / (f_center - f_lower);
            } else if freq > f_center && freq <= f_upper {
                filter_bank[i][j] = (f_upper - freq) / (f_upper - f_center);
            }
        }
    }

    filter_bank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_scale_roundtrip() {
        for freq in [100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel(freq);
            let freq_back = mel_to_hz(mel);
            assert!(
                (freq - freq_back).abs() < 0.001,
                "Failed for freq {}",
                freq
            );
        }
    }

    #[test]
    fn test_mel_scale_known_values() {
        // At 1000 Hz, HTK mel = 2595 * log10(1 + 1000/700) ≈ 999.98
        let mel_1000 = hz_to_mel(1000.0);
        assert!(
            (mel_1000 - 1000.0).abs() < 5.0,
            "Expected ~1000, got {}",
            mel_1000
        );
    }

    #[test]
    fn test_filter_bank_shape() {
        let filters = create_filter_bank(80, 400, 16000, 0.0, 8000.0);
        assert_eq!(filters.len(), 80);
        assert_eq!(filters[0].len(), 201); // n_fft/2 + 1
    }

    #[test]
    fn test_htk_vs_slaney_difference() {
        // HTK and Slaney should give different values
        let htk_mel = hz_to_mel(2000.0);
        let slaney_mel = super::super::slaney::hz_to_mel(2000.0);

        // They should be noticeably different
        assert!((htk_mel - slaney_mel).abs() > 100.0);
    }
}
