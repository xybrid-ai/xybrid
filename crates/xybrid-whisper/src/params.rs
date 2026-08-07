//! Transcription parameters — the safe, validated view of
//! `whisper_full_params`.
//!
//! Compiles on every target (no FFI imports), so callers can construct and
//! pass these even in a no-bindings build.

/// The audio sample rate whisper.cpp requires. Callers must resample before
/// handing PCM to [`crate::WhisperModel::transcribe`].
pub const SAMPLE_RATE: u32 = 16_000;

/// Number of mel frames the encoder sees for a full 30 s window. `audio_ctx`
/// is expressed in these units: 1500 covers the whole window, 500 covers 10 s.
pub const FULL_AUDIO_CTX: u32 = 1500;

/// What the model should do with the audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
    /// Transcribe in the source language.
    #[default]
    Transcribe,
    /// Translate into English.
    Translate,
}

/// Validated transcription parameters.
///
/// Defaults mirror xybrid's live streaming path: greedy sampling, no
/// temperature fallback, no timestamps — and, deliberately, **no encoder
/// truncation**.
///
/// # On `audio_ctx`
///
/// whisper's encoder always processes a 30 s window, padding short audio with
/// silence. `audio_ctx` truncates that context, which is the single largest
/// speed lever for a live 5 s window — but lowering it too far makes the
/// decoder emit repetition loops ("ask not to ask not to ask not"). Measured on
/// a Pixel 8 with `tiny.en` Q5_1: 500 is safe and 2.8x faster than full; 250
/// degenerates. Multilingual quantized models degenerate earlier than `.en`
/// ones.
///
/// So the default is [`FULL_AUDIO_CTX`] — no truncation. A model opts into
/// truncation through its own metadata after its own quality check, rather than
/// inheriting a fast-but-fragile default.
#[derive(Debug, Clone)]
pub struct TranscribeParams {
    /// Language code (e.g. `"en"`). `None` auto-detects.
    pub language: Option<String>,
    /// Transcribe or translate.
    pub task: Task,
    /// Encoder context in mel frames, capped at [`FULL_AUDIO_CTX`]. See the
    /// type-level note above before lowering this.
    pub audio_ctx: u32,
    /// Compute threads. Callers on mobile should pass the same bounded value
    /// the rest of the runtime uses rather than the core count — whisper is
    /// memory-bandwidth-bound, and oversubscription measurably slows it down.
    pub n_threads: u32,
    /// Emit per-segment timestamps. Off for live partials, on for final
    /// transcripts that need alignment.
    pub timestamps: bool,
    /// Condition each window on the previous window's text. Improves coherence
    /// on continuous speech but can propagate an early mistake through the rest
    /// of the stream.
    pub use_context: bool,
}

impl Default for TranscribeParams {
    fn default() -> Self {
        Self {
            language: None,
            task: Task::Transcribe,
            audio_ctx: FULL_AUDIO_CTX,
            n_threads: 4,
            timestamps: false,
            use_context: false,
        }
    }
}

impl TranscribeParams {
    /// Clamp `audio_ctx` into `1..=FULL_AUDIO_CTX`.
    ///
    /// `0` is whisper.cpp's own "use the default" sentinel, which means full
    /// context — so it maps to [`FULL_AUDIO_CTX`] rather than being rejected.
    /// Values above the maximum are meaningless (the encoder has no more
    /// positions) and are clamped down.
    #[must_use]
    pub fn normalized_audio_ctx(&self) -> u32 {
        if self.audio_ctx == 0 {
            FULL_AUDIO_CTX
        } else {
            self.audio_ctx.min(FULL_AUDIO_CTX)
        }
    }

    /// The encoder context that covers `window_secs` of audio, plus a safety
    /// factor.
    ///
    /// Truncating to exactly the audio's own length is what triggers the
    /// repetition loops described on this type — the decoder needs slack past
    /// the end of speech. `headroom` of 2.0 reproduces the measured-safe
    /// setting (a 5 s window maps to 500 frames, i.e. 10 s of context).
    #[must_use]
    pub fn audio_ctx_for_window(window_secs: f32, headroom: f32) -> u32 {
        // 30 s of audio maps to FULL_AUDIO_CTX frames.
        let frames = (window_secs * headroom / 30.0) * FULL_AUDIO_CTX as f32;
        (frames.ceil().max(1.0) as u32).min(FULL_AUDIO_CTX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_audio_ctx_means_full_context() {
        let p = TranscribeParams {
            audio_ctx: 0,
            ..Default::default()
        };
        assert_eq!(p.normalized_audio_ctx(), FULL_AUDIO_CTX);
    }

    #[test]
    fn audio_ctx_is_clamped_to_the_encoder_maximum() {
        let p = TranscribeParams {
            audio_ctx: 9_000,
            ..Default::default()
        };
        assert_eq!(p.normalized_audio_ctx(), FULL_AUDIO_CTX);
    }

    #[test]
    fn default_does_not_truncate_the_encoder() {
        assert_eq!(
            TranscribeParams::default().normalized_audio_ctx(),
            FULL_AUDIO_CTX
        );
    }

    #[test]
    fn five_second_window_maps_to_the_measured_safe_context() {
        // 5 s at 2x headroom is the setting measured safe on a Pixel 8.
        assert_eq!(TranscribeParams::audio_ctx_for_window(5.0, 2.0), 500);
    }

    #[test]
    fn window_context_never_exceeds_the_encoder_maximum() {
        assert_eq!(
            TranscribeParams::audio_ctx_for_window(30.0, 2.0),
            FULL_AUDIO_CTX
        );
    }

    #[test]
    fn very_short_windows_still_request_at_least_one_frame() {
        assert!(TranscribeParams::audio_ctx_for_window(0.001, 1.0) >= 1);
    }
}
