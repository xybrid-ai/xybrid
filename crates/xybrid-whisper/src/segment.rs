//! Transcript segments. No FFI imports, so this compiles on every target and
//! callers can pattern-match on results even in a no-bindings build.

/// One span of transcribed text with its position in the audio.
///
/// Times are milliseconds from the start of the PCM buffer that was passed to
/// [`crate::WhisperModel::transcribe`] — not wall-clock, and not relative to a
/// longer stream. Callers doing rolling-window streaming add their own window
/// offset. Milliseconds (rather than whisper's native centiseconds) because
/// that is xybrid's canonical cross-boundary time unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// Segment text, exactly as whisper emitted it — including the leading
    /// space it conventionally prefixes.
    pub text: String,
    /// Start offset in milliseconds.
    pub start_ms: i64,
    /// End offset in milliseconds.
    pub end_ms: i64,
}

impl Segment {
    /// Concatenate segment texts into one transcript.
    ///
    /// Segments already carry their own leading spaces, so this joins without
    /// inserting separators and trims the result.
    #[must_use]
    pub fn join(segments: &[Self]) -> String {
        let mut out = String::new();
        for s in segments {
            out.push_str(&s.text);
        }
        out.trim().to_string()
    }

    /// Duration of this segment in milliseconds, saturating at zero for the
    /// degenerate case where whisper reports `end < start`.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end_ms - self.start_ms).max(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(text: &str, start_ms: i64, end_ms: i64) -> Segment {
        Segment {
            text: text.to_string(),
            start_ms,
            end_ms,
        }
    }

    #[test]
    fn join_concatenates_without_adding_separators() {
        let segments = [
            seg(" And so my fellow", 0, 2000),
            seg(" Americans", 2000, 3000),
        ];
        assert_eq!(Segment::join(&segments), "And so my fellow Americans");
    }

    #[test]
    fn join_of_nothing_is_empty() {
        assert_eq!(Segment::join(&[]), "");
    }

    #[test]
    fn duration_is_the_span_length() {
        assert_eq!(seg(" x", 1000, 2500).duration_ms(), 1500);
    }

    #[test]
    fn inverted_span_reports_zero_rather_than_a_negative_duration() {
        assert_eq!(seg(" x", 2500, 1000).duration_ms(), 0);
    }
}
