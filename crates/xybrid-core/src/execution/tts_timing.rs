//! Per-phoneme timings for synthesized TTS audio.
//!
//! A TTS model may report, next to its waveform, how many frames each input
//! token lasts (see `modes::tts::tts_duration_output`). With the phoneme
//! behind each token id (`PreprocessedData::as_phoneme_symbols`), that says
//! when each phoneme is spoken. This module holds the sample arithmetic, free
//! of I/O, in the order the executor applies it:
//!
//! 1. [`chunk_spans`] turns one chunk's durations into sample spans on that
//!    chunk's waveform, scaling them to the waveform's real length rather than
//!    assuming a frame size.
//! 2. [`shift`] moves a chunk's spans by where the chunk starts in the
//!    crossfaded audio of a long text.
//! 3. [`apply_kept_range`] moves spans onto the samples `TTSAudioEncode` kept
//!    after trimming silence, and [`enforce_monotonic`] resolves the overlaps
//!    a crossfade leaves. [`finish`] runs both, then [`to_timings`] converts
//!    samples to seconds.
//! 4. [`metadata_json`] encodes the timings for
//!    [`Envelope::PHONEME_TIMINGS_METADATA_KEY`](crate::ir::Envelope::PHONEME_TIMINGS_METADATA_KEY).
//!
//! Padding and silence tokens have no symbol and are never reported.

use crate::ir::PhonemeTiming;
use std::ops::Range;

/// One phoneme's samples in a waveform: `start..end`, in sample indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhonemeSpan {
    /// The phoneme symbol.
    pub(crate) symbol: char,
    /// First sample of the phoneme.
    pub(crate) start: usize,
    /// One past the last sample of the phoneme.
    pub(crate) end: usize,
}

/// Sample spans of one synthesized chunk's phonemes.
///
/// `symbols` and `durations` are one entry per token id, in input order
/// (`None` for padding and silence tokens). Durations are in model frames; a
/// negative one counts as 0. Token boundaries are placed at
/// `round(raw_len * frames_so_far / total_frames)`, so the spans cover the
/// model's whole raw waveform of `raw_len` samples whatever the frame size.
/// Spans are then clipped to the first `kept_len` samples (what is left after
/// trailing samples were cut off the waveform); spans for tokens without a
/// symbol, and spans left empty, are dropped.
///
/// Returns `None` when `symbols` and `durations` differ in length or the
/// durations add up to 0: such durations do not describe this waveform.
pub(crate) fn chunk_spans(
    symbols: &[Option<char>],
    durations: &[i64],
    raw_len: usize,
    kept_len: usize,
) -> Option<Vec<PhonemeSpan>> {
    if symbols.len() != durations.len() {
        return None;
    }
    let frames: Vec<u128> = durations
        .iter()
        .map(|&d| u128::try_from(d).unwrap_or(0))
        .collect();
    let total = frames
        .iter()
        .try_fold(0u128, |sum, &f| sum.checked_add(f))?;
    if total == 0 {
        return None;
    }
    let raw = u128::try_from(raw_len).ok()?;
    let half = total / 2;

    let mut spans = Vec::new();
    let mut so_far = 0u128;
    let mut start = 0usize;
    for (symbol, &f) in symbols.iter().zip(&frames) {
        so_far += f;
        // round(raw * so_far / total), never above raw_len since so_far <= total.
        let scaled = raw.checked_mul(so_far)?.checked_add(half)? / total;
        let end = usize::try_from(scaled).ok()?;
        if let Some(symbol) = *symbol {
            let clipped_end = end.min(kept_len);
            if start < clipped_end {
                spans.push(PhonemeSpan {
                    symbol,
                    start,
                    end: clipped_end,
                });
            }
        }
        start = end;
    }
    Some(spans)
}

/// Moves `spans` later by `offset` samples: where their chunk starts in the
/// joined audio.
pub(crate) fn shift(spans: &mut [PhonemeSpan], offset: usize) {
    for span in spans {
        span.start = span.start.saturating_add(offset);
        span.end = span.end.saturating_add(offset);
    }
}

/// Makes `spans` (in start order) monotonic: each span ends no later than the
/// next one starts.
///
/// Where two spans overlap, as across a chunk crossfade, the earlier one is
/// cut short (`prev.end = min(prev.end, next.start)`); a span cut to nothing,
/// or empty already, is dropped.
pub(crate) fn enforce_monotonic(spans: &mut Vec<PhonemeSpan>) {
    let mut kept: Vec<PhonemeSpan> = Vec::with_capacity(spans.len());
    for span in spans.drain(..) {
        if span.start >= span.end {
            continue;
        }
        while let Some(prev) = kept.last_mut() {
            if prev.end <= span.start {
                break;
            }
            prev.end = span.start;
            if prev.start < prev.end {
                break;
            }
            kept.pop();
        }
        kept.push(span);
    }
    *spans = kept;
}

/// Moves `spans` onto the samples `kept` of the waveform they were measured
/// on, as silence trimming does to the audio: `kept.start` is subtracted, and
/// spans are clamped to the kept length. Spans left empty, wholly trimmed off,
/// are dropped.
pub(crate) fn apply_kept_range(spans: &mut Vec<PhonemeSpan>, kept: Range<usize>) {
    let len = kept.end.saturating_sub(kept.start);
    spans.retain_mut(|span| {
        span.start = span.start.saturating_sub(kept.start).min(len);
        span.end = span.end.saturating_sub(kept.start).min(len);
        span.start < span.end
    });
}

/// Converts sample spans to [`PhonemeTiming`]s in seconds at `sample_rate`.
/// A zero `sample_rate` yields no timings.
pub(crate) fn to_timings(spans: &[PhonemeSpan], sample_rate: u32) -> Vec<PhonemeTiming> {
    if sample_rate == 0 {
        return Vec::new();
    }
    let rate = f64::from(sample_rate);
    let seconds = |sample: usize| (sample as f64 / rate) as f32;
    spans
        .iter()
        .map(|span| PhonemeTiming {
            phoneme: span.symbol.to_string(),
            start: seconds(span.start),
            end: seconds(span.end),
        })
        .collect()
}

/// Timings for the encoded audio, from spans measured on the waveform that
/// `TTSAudioEncode` received: [`apply_kept_range`] with the samples it kept,
/// then [`enforce_monotonic`], then [`to_timings`]. The result is in order,
/// never overlaps, and lies within `[0, kept.len() / sample_rate]` seconds.
pub(crate) fn finish(
    mut spans: Vec<PhonemeSpan>,
    kept: Range<usize>,
    sample_rate: u32,
) -> Vec<PhonemeTiming> {
    apply_kept_range(&mut spans, kept);
    enforce_monotonic(&mut spans);
    to_timings(&spans, sample_rate)
}

/// The JSON value for
/// [`Envelope::PHONEME_TIMINGS_METADATA_KEY`](crate::ir::Envelope::PHONEME_TIMINGS_METADATA_KEY):
/// an array of [`PhonemeTiming`] objects. `None` when there are no timings,
/// so the key is left out rather than set to an empty array.
pub(crate) fn metadata_json(timings: &[PhonemeTiming]) -> Option<String> {
    if timings.is_empty() {
        return None;
    }
    match serde_json::to_string(timings) {
        Ok(json) => Some(json),
        Err(e) => {
            log::debug!("Could not encode phoneme timings: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(symbol: char, start: usize, end: usize) -> PhonemeSpan {
        PhonemeSpan { symbol, start, end }
    }

    /// Asserts `spans` are in order, non-empty, never overlap, and end by `len`.
    fn assert_well_formed(spans: &[PhonemeSpan], len: usize) {
        for s in spans {
            assert!(s.start < s.end, "empty span {s:?}");
            assert!(s.end <= len, "{s:?} ends past {len}");
        }
        for pair in spans.windows(2) {
            assert!(pair[0].end <= pair[1].start, "{pair:?} overlap");
        }
    }

    #[test]
    fn durations_scale_to_the_real_waveform_length() {
        let symbols = [None, Some('h'), Some('ə'), None];
        let durations = [2, 3, 5, 2]; // 12 frames

        // 600 samples per frame: the spans fall on exact frame edges.
        let spans = chunk_spans(&symbols, &durations, 7200, 7200).unwrap();
        assert_eq!(spans, vec![span('h', 1200, 3000), span('ə', 3000, 6000)]);

        // A waveform that is not a whole number of frames: edges are rounded
        // from the real length, not from a fixed frame size.
        // 7000 * 2/12 = 1166.7, * 5/12 = 2916.7, * 10/12 = 5833.3
        let spans = chunk_spans(&symbols, &durations, 7000, 7000).unwrap();
        assert_eq!(spans, vec![span('h', 1167, 2917), span('ə', 2917, 5833)]);

        // The last token ends exactly at the end of the waveform.
        let spans = chunk_spans(&[Some('a'), Some('b')], &[1, 2], 1001, 1001).unwrap();
        assert_eq!(spans, vec![span('a', 0, 334), span('b', 334, 1001)]);
    }

    #[test]
    fn pads_and_silence_tokens_are_not_reported() {
        // Pad, two silence tokens, a phoneme with no frames, two phonemes, pad.
        let symbols = [None, None, None, Some('x'), Some('a'), Some(' '), None];
        let durations = [1, 1, 1, 0, 2, 1, 2];
        let spans = chunk_spans(&symbols, &durations, 800, 800).unwrap();

        assert_eq!(spans, vec![span('a', 300, 500), span(' ', 500, 600)]);
    }

    #[test]
    fn an_audible_end_pad_is_kept_but_not_listed() {
        // KittenTTS 0.8 Micro and Mini: the end pad's frames carry the release
        // of the last sound, so the silence trim keeps them. 600 samples per
        // frame; the start pad is silent and trimmed.
        let symbols = [None, Some('a'), Some('b'), None];
        let durations = [1, 2, 2, 4];
        let spans = chunk_spans(&symbols, &durations, 5400, 5400).unwrap();
        let timings = finish(spans, 600..5400, 24_000);

        // The audio is 4800 samples (0.2 s); the last entry ends 0.1 s in, the
        // end pad's four frames before the audio does.
        assert_eq!(timings.len(), 2);
        assert_eq!((timings[0].start, timings[0].end), (0.0, 0.05));
        assert_eq!((timings[1].start, timings[1].end), (0.05, 0.1));
    }

    #[test]
    fn length_mismatch_or_zero_total_yields_no_timings() {
        assert_eq!(chunk_spans(&[Some('a'), None], &[3], 100, 100), None);
        assert_eq!(chunk_spans(&[Some('a')], &[3, 4], 100, 100), None);
        assert_eq!(chunk_spans(&[Some('a'), None], &[0, 0], 100, 100), None);
        // Negative durations count as no frames.
        assert_eq!(chunk_spans(&[Some('a'), None], &[-4, 0], 100, 100), None);
        assert_eq!(chunk_spans(&[], &[], 100, 100), None);
        // Durations that cannot be scaled without overflow are not trusted.
        assert_eq!(
            chunk_spans(&[Some('a'); 4], &[i64::MAX; 4], usize::MAX, 10),
            None
        );
    }

    #[test]
    fn trailing_truncation_clips_and_drops_spans() {
        // 1000 raw samples, the last 300 cut off before encoding.
        let symbols = [Some('a'), Some('b'), Some('c'), None];
        let durations = [4, 4, 1, 1];
        let spans = chunk_spans(&symbols, &durations, 1000, 700).unwrap();

        // `b` (400..800) is clipped at 700; `c` (800..900) is gone.
        assert_eq!(spans, vec![span('a', 0, 400), span('b', 400, 700)]);
        assert_well_formed(&spans, 700);
    }

    #[test]
    fn chunk_offsets_shift_spans_and_resolve_crossfade_overlap() {
        const CROSSFADE: usize = 480;
        // Chunk A: 2400 samples. Its last phoneme runs to its very end.
        let mut a = vec![span('h', 600, 1800), span('i', 1800, 2400)];
        // Chunk B starts CROSSFADE samples before A ends; its first phoneme
        // starts at once, inside the crossfade.
        let mut b = vec![span('j', 0, 300), span('o', 300, 1200)];
        let b_start = 2400 - CROSSFADE;
        shift(&mut b, b_start);
        assert_eq!(b[0], span('j', 1920, 2220));

        a.append(&mut b);
        enforce_monotonic(&mut a);
        assert_eq!(
            a,
            vec![
                span('h', 600, 1800),
                span('i', 1800, 1920),
                span('j', 1920, 2220),
                span('o', 2220, 3120),
            ]
        );
        assert_well_formed(&a, 2400 - CROSSFADE + 1200);
    }

    #[test]
    fn a_span_swallowed_by_the_crossfade_is_dropped() {
        // `i` lies wholly inside the overlap, after `j` starts.
        let mut spans = vec![
            span('h', 0, 1950),
            span('i', 1950, 2400),
            span('j', 1920, 2220),
        ];
        enforce_monotonic(&mut spans);
        assert_eq!(spans, vec![span('h', 0, 1920), span('j', 1920, 2220)]);

        let mut empty = vec![span('a', 5, 5), span('b', 5, 9)];
        enforce_monotonic(&mut empty);
        assert_eq!(empty, vec![span('b', 5, 9)]);
    }

    #[test]
    fn head_trim_is_subtracted_and_tail_clipped() {
        let mut spans = vec![
            span('a', 0, 80),     // trimmed off with the leading silence
            span('b', 80, 150),   // straddles the head trim
            span('c', 150, 850),  // fully kept
            span('d', 850, 1000), // straddles the tail trim
            span('e', 950, 990),  // trimmed off with the trailing silence
        ];
        apply_kept_range(&mut spans, 100..900);

        assert_eq!(
            spans,
            vec![span('b', 0, 50), span('c', 50, 750), span('d', 750, 800)]
        );
        assert_well_formed(&spans, 800);
    }

    #[test]
    fn to_timings_converts_samples_to_seconds() {
        let timings = to_timings(&[span('h', 300, 1200), span('ə', 1200, 2400)], 24_000);
        assert_eq!(
            timings,
            vec![
                PhonemeTiming {
                    phoneme: "h".to_string(),
                    start: 0.0125,
                    end: 0.05,
                },
                PhonemeTiming {
                    phoneme: "ə".to_string(),
                    start: 0.05,
                    end: 0.1,
                },
            ]
        );
        assert!(to_timings(&[span('h', 0, 10)], 0).is_empty());
    }

    #[test]
    fn finish_maps_waveform_spans_onto_the_kept_audio() {
        let spans = vec![
            span('a', 0, 2400),
            span('b', 2400, 4800),
            span('c', 4300, 9600), // overlaps `b`, as across a crossfade
        ];
        // The encoder kept samples 1200..8400 (7200 samples, 0.3 s).
        let timings = finish(spans, 1200..8400, 24_000);

        let audio_secs = 7200.0 / 24_000.0;
        assert_eq!(timings.len(), 3);
        assert_eq!(timings[0].start, 0.0);
        assert_eq!(timings[0].end, 0.05);
        assert_eq!(timings[1].end, timings[2].start);
        assert_eq!(timings[2].end, audio_secs as f32);
        for pair in timings.windows(2) {
            assert!(pair[0].end <= pair[1].start);
        }
    }

    #[test]
    fn json_round_trips_a_quote_phoneme() {
        // Kokoro's vocabulary has `"` as a phoneme token (id 11); it must
        // survive the trip through the metadata string, as must stress marks.
        let timings = to_timings(
            &[
                span('"', 0, 600),
                span('ˈ', 600, 1200),
                span('ɜ', 1200, 2400),
            ],
            24_000,
        );
        let json = metadata_json(&timings).unwrap();
        assert!(json.contains(r#""phoneme":"\"""#), "{json}");

        let parsed: Vec<PhonemeTiming> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, timings);
        assert_eq!(parsed[0].phoneme, "\"");
        assert_eq!(parsed[1].phoneme, "ˈ");
    }

    #[test]
    fn no_timings_means_no_metadata() {
        assert_eq!(metadata_json(&[]), None);
    }
}
