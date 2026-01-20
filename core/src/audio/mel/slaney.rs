//! Slaney mel scale implementation.
//!
//! This implements the Slaney mel scale used by:
//! - OpenAI Whisper
//! - transformers.js
//! - librosa (with `htk=False`, the default)
//!
//! ## Slaney Formula
//!
//! The Slaney mel scale uses a piecewise linear/logarithmic formula:
//!
//! - For freq < 1000 Hz: `mel = 3 × freq / 200`
//! - For freq >= 1000 Hz: `mel = 15 + 27 × ln(freq / 1000) / ln(6.4)`
//!
//! This produces mel filter banks that are linear below 1kHz and logarithmic above.

// ============================================================================
// Mel Scale Conversion
// ============================================================================

/// Convert frequency in Hz to mel scale using Slaney formula.
pub fn hz_to_mel(freq: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    const LOGSTEP: f64 = 0.06875177742094912; // log(6.4) / 27

    if freq < MIN_LOG_HZ {
        freq / F_SP
    } else {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

/// Convert mel scale to frequency in Hz using Slaney formula.
pub fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    const LOGSTEP: f64 = 0.06875177742094912;

    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOGSTEP).exp()
    }
}

// ============================================================================
// Filter Bank Creation
// ============================================================================

/// Create Slaney-normalized mel filter bank.
///
/// Each triangular filter is normalized by `2 / (upper_freq - lower_freq)`,
/// which is the Slaney normalization (area normalization).
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

    // Create triangular filters with Slaney normalization
    let mut filter_bank = vec![vec![0.0; n_freqs]; n_mels];

    for i in 0..n_mels {
        let f_lower = freq_points[i];
        let f_center = freq_points[i + 1];
        let f_upper = freq_points[i + 2];

        // Slaney normalization: area = 1
        let enorm = 2.0 / (f_upper - f_lower);

        for (j, &freq) in fft_freqs.iter().enumerate() {
            if freq >= f_lower && freq <= f_center {
                filter_bank[i][j] = enorm * (freq - f_lower) / (f_center - f_lower);
            } else if freq > f_center && freq <= f_upper {
                filter_bank[i][j] = enorm * (f_upper - freq) / (f_upper - f_center);
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
            assert!((freq - freq_back).abs() < 0.001, "Failed for freq {}", freq);
        }
    }

    #[test]
    fn test_mel_scale_known_values() {
        // At 1000 Hz, mel should be exactly 15 (the breakpoint)
        let mel_1000 = hz_to_mel(1000.0);
        assert!((mel_1000 - 15.0).abs() < 0.001);

        // Below 1000 Hz, should be linear
        let mel_500 = hz_to_mel(500.0);
        let mel_250 = hz_to_mel(250.0);
        assert!((mel_500 - 2.0 * mel_250).abs() < 0.001);
    }

    #[test]
    fn test_filter_bank_shape() {
        let filters = create_filter_bank(80, 400, 16000, 0.0, 8000.0);
        assert_eq!(filters.len(), 80);
        assert_eq!(filters[0].len(), 201); // n_fft/2 + 1
    }

    #[test]
    fn test_filter_bank_coverage() {
        let filters = create_filter_bank(80, 400, 16000, 0.0, 8000.0);

        // Each filter should have non-zero values
        for (i, filter) in filters.iter().enumerate() {
            let sum: f64 = filter.iter().sum();
            assert!(sum > 0.0, "Filter {} has zero sum", i);
        }
    }
}
