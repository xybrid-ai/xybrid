//! Runtime feature-flag introspection.
//!
//! Reports which Cargo features were enabled at compile time so callers
//! (e.g. the registry telemetry header) can advertise the active backend set.

use std::sync::OnceLock;

/// The registry serves the `ios-arm64` whisper bundles (Core ML encoder
/// included) only to clients that list this backend.
#[cfg(feature = "asr-whispercpp")]
const WHISPER_COREML: bool = xybrid_whisper::COREML_ENCODER;
#[cfg(not(feature = "asr-whispercpp"))]
const WHISPER_COREML: bool = false;

// Forward-compatible feature names: some entries (`ort-cuda`, `espeak`) are
// not yet declared in Cargo.toml. The `cfg!()` checks resolve to `false`
// for any feature that does not exist, and `#[allow(unexpected_cfgs)]`
// suppresses the corresponding lint so the list can grow without a
// parallel Cargo.toml edit.
#[allow(unexpected_cfgs)]
const ALL_FEATURES: &[(&str, bool)] = &[
    ("asr-whispercpp", cfg!(feature = "asr-whispercpp")),
    ("asr-whispercpp-coreml", WHISPER_COREML),
    ("candle-cuda", cfg!(feature = "candle-cuda")),
    ("candle-metal", cfg!(feature = "candle-metal")),
    ("espeak", cfg!(feature = "espeak")),
    ("llm-llamacpp", cfg!(feature = "llm-llamacpp")),
    ("llm-llamacpp-vision", cfg!(feature = "llm-llamacpp-vision")),
    ("llm-mistral", cfg!(feature = "llm-mistral")),
    ("ort-coreml", cfg!(feature = "ort-coreml")),
    ("ort-cuda", cfg!(feature = "ort-cuda")),
    ("ort-download", cfg!(feature = "ort-download")),
    ("ort-dynamic", cfg!(feature = "ort-dynamic")),
    ("tts-zzz", cfg!(feature = "tts-zzz")),
    // Vision (image envelopes + preprocessing pipeline) is always compiled in;
    // the heavy native VLM backend is reported via `llm-llamacpp-vision`.
    ("vision", true),
    // The registry serves `phonon-31m-en` only to clients that list this
    // backend. Every `tts-zzz` build pins a zzz engine that loads it; the
    // rc.6 engine pinned by 0.12.0 rejects it, so `tts-zzz` alone cannot gate it.
    ("zzz-phonon-31m", cfg!(feature = "tts-zzz")),
];

/// Return the sorted list of enabled runtime feature names.
///
/// The result is computed once and cached, so subsequent calls are cheap
/// and return a slice with a stable address.
pub fn enabled() -> &'static [&'static str] {
    static ENABLED: OnceLock<Vec<&'static str>> = OnceLock::new();
    ENABLED
        .get_or_init(|| filter_enabled(ALL_FEATURES))
        .as_slice()
}

fn filter_enabled(items: &[(&'static str, bool)]) -> Vec<&'static str> {
    let mut names: Vec<&'static str> = items
        .iter()
        .filter_map(|(name, on)| if *on { Some(*name) } else { None })
        .collect();
    names.sort();
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_empty_slice() {
        let result = filter_enabled(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn no_features_enabled_yields_empty_slice() {
        let items: &[(&'static str, bool)] = &[
            ("candle-metal", false),
            ("ort-coreml", false),
            ("llm-llamacpp", false),
        ];
        assert!(filter_enabled(items).is_empty());
    }

    #[test]
    fn output_is_sorted() {
        let items: &[(&'static str, bool)] = &[
            ("ort-download", true),
            ("candle-metal", true),
            ("llm-llamacpp", true),
        ];
        assert_eq!(
            filter_enabled(items),
            vec!["candle-metal", "llm-llamacpp", "ort-download"]
        );
    }

    #[test]
    fn enabled_is_deterministic_across_calls() {
        let first = enabled();
        let second = enabled();
        assert_eq!(first, second);
        assert_eq!(first.as_ptr(), second.as_ptr());
    }

    #[test]
    fn enabled_is_alphabetically_sorted() {
        let names = enabled();
        for window in names.windows(2) {
            assert!(window[0] <= window[1], "not sorted: {:?}", names);
        }
    }

    #[cfg(feature = "ort-coreml")]
    #[test]
    fn platform_macos_enables_expected_features() {
        // Under --features platform-macos, ort-coreml must be reported.
        assert!(enabled().contains(&"ort-coreml"));
    }

    #[cfg(feature = "ort-download")]
    #[test]
    fn ort_download_branch_is_exercised() {
        assert!(enabled().contains(&"ort-download"));
    }

    #[test]
    fn vision_branch_is_exercised() {
        assert!(enabled().contains(&"vision"));
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    #[test]
    fn llama_cpp_vision_branch_is_exercised() {
        assert!(enabled().contains(&"llm-llamacpp-vision"));
    }

    #[cfg(feature = "llm-llamacpp")]
    #[test]
    fn llamacpp_reports_enabled() {
        assert!(enabled().contains(&"llm-llamacpp"));
    }

    #[cfg(feature = "asr-whispercpp")]
    #[test]
    fn whisper_coreml_follows_the_encoder_bridge() {
        assert_eq!(
            enabled().contains(&"asr-whispercpp-coreml"),
            xybrid_whisper::COREML_ENCODER
        );
    }

    #[test]
    fn zzz_phonon_31m_follows_tts_zzz() {
        assert_eq!(
            enabled().contains(&"zzz-phonon-31m"),
            cfg!(feature = "tts-zzz")
        );
    }
}
