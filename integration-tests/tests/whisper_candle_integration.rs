//! End-to-end Whisper transcription through the Candle runtime.
//!
//! These are the regression tests for two defects that only reproduce with
//! real weights:
//!
//! - Audio longer than 240_159 samples (15.0099 s) used to hard-fail with
//!   `narrow invalid args` because the mel was encoded in one shot instead of
//!   being sliced into encoder-sized windows, and audio longer than 30 s was
//!   silently truncated. The 240_159/240_160 pair below is the boundary.
//! - `language`/`task` were baked in at model load, so every request decoded
//!   as English `transcribe` no matter what the caller asked for — and the
//!   model is cached per directory, so a fix that stores the first request's
//!   options on the model would still be wrong for the second request.
//!
//! Everything drives [`CandleRuntime`] through [`ModelRuntime`] with an
//! [`Envelope`], which is the path the executor uses in production; the
//! per-request options type itself is crate-private on purpose.
//!
//! Test clips are built in memory from the committed `jfk.wav` fixture — no
//! new English audio binaries — by looping and cutting its PCM to the exact
//! sample counts that matter, then re-encoding to WAV bytes.
//!
//! Run with:
//!   cargo test -p integration-tests --features candle --test whisper_candle_integration
//!
//! Download the model first:
//!   ./integration-tests/download.sh whisper-tiny

#![cfg(feature = "candle")]

use integration_tests::fixtures;
use xybrid_core::audio::{decode_wav_audio, samples_to_wav};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::runtime_adapter::candle::CandleRuntime;
use xybrid_core::runtime_adapter::{AdapterError, ModelRuntime};

const MODEL: &str = "whisper-tiny";

/// Everything Whisper accepts is 16 kHz mono.
const SAMPLE_RATE: u32 = 16_000;

/// English fixture: 11.0 s, 176_000 samples, 16 kHz mono.
const JFK_CLIP: &str = "jfk.wav";

/// Reference transcript for [`JFK_CLIP`] — the canonical line the same clip
/// carries in candle's and whisper.cpp's own examples.
const JFK_TRANSCRIPT: &str = "And so my fellow Americans, ask not what your country can do for \
                              you, ask what you can do for your country.";

/// French fixture: 8.0 s, 128_000 samples, 16 kHz mono. First 8 s of
/// `facebook/multilingual_librispeech` [french/test] row 2 (LibriVox, *Les
/// Mille et Une Nuits*), whose full 12.66 s reference reads "la nuit suivante
/// appela sa soeur quand il en fut temps si vous ne dormez pas ma soeur lui
/// dit-elle je vous prie en attendant le jour qui paraîtra bientôt de
/// continuer le conte du pêcheur". The trim cuts mid-sentence, so the tests
/// below compare outputs against each other rather than against that text.
const FRENCH_CLIP: &str = "mls-fr-1-8s.wav";

/// WER budget for whisper-tiny on clean speech. A threshold rather than string
/// equality because Metal and CPU decode to slightly different transcripts.
const MAX_WER: f64 = 0.15;

/// A loaded Whisper runtime, or `None` when the model is not downloaded.
///
/// `None` is only reachable with `XYBRID_REQUIRE_MODELS` unset; CI sets it, so
/// a failed download there fails the test instead of skipping it.
fn whisper_runtime() -> Option<CandleRuntime> {
    let model_dir = fixtures::model_for_test(MODEL)?;
    let mut runtime = CandleRuntime::new();
    runtime
        .load(&model_dir)
        .unwrap_or_else(|e| panic!("loading {MODEL} from {}: {e}", model_dir.display()));
    Some(runtime)
}

/// Announce the skip once, from the one place that knows why.
fn skip_notice() {
    eprintln!("Skipping: {MODEL} not downloaded. Run: ./integration-tests/download.sh {MODEL}");
}

/// Read a committed audio fixture as 16 kHz mono f32 PCM.
fn read_pcm(name: &str) -> Vec<f32> {
    let path = fixtures::input_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    decode_wav_audio(&bytes, SAMPLE_RATE, 1)
        .unwrap_or_else(|e| panic!("decoding {}: {e}", path.display()))
}

/// `pcm` looped end to end and cut to exactly `samples` samples.
///
/// Looping rather than padding with silence keeps every test clip speech all
/// the way to its final sample, so a window the decoder mishandles shows up as
/// missing words instead of being masked by a silent tail.
fn looped(pcm: &[f32], samples: usize) -> Vec<f32> {
    assert!(!pcm.is_empty(), "cannot loop empty audio");
    pcm.iter().copied().cycle().take(samples).collect()
}

/// [`JFK_TRANSCRIPT`] repeated `times` times, the reference for a looped clip.
fn transcript_repeated(times: usize) -> String {
    vec![JFK_TRANSCRIPT; times].join(" ")
}

/// Run one request the way the executor does: PCM to WAV bytes, into an audio
/// [`Envelope`] carrying the request metadata, through [`ModelRuntime`].
fn transcribe(
    runtime: &mut CandleRuntime,
    pcm: &[f32],
    metadata: &[(&str, &str)],
) -> Result<String, AdapterError> {
    let envelope = Envelope {
        kind: EnvelopeKind::Audio(samples_to_wav(pcm, SAMPLE_RATE)),
        metadata: metadata
            .iter()
            .map(|&(key, value)| (key.to_string(), value.to_string()))
            .collect(),
    };

    match runtime.execute(&envelope)?.kind {
        EnvelopeKind::Text(text) => Ok(text),
        other => panic!("expected text output, got {other:?}"),
    }
}

/// [`transcribe`], failing the test on error.
fn transcribed(runtime: &mut CandleRuntime, pcm: &[f32], metadata: &[(&str, &str)]) -> String {
    transcribe(runtime, pcm, metadata).unwrap_or_else(|e| {
        panic!(
            "transcribing {} samples ({:.2} s) with {metadata:?} failed: {e}",
            pcm.len(),
            pcm.len() as f64 / f64::from(SAMPLE_RATE),
        )
    })
}

/// Words of `text`, lowercased and stripped of punctuation.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// Word error rate: word-level Levenshtein distance over reference length.
fn wer(reference: &str, hypothesis: &str) -> f64 {
    let reference = words(reference);
    let hypothesis = words(hypothesis);
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }

    // Two rows of the edit-distance matrix: `previous` is the row for the
    // reference prefix already consumed, `current` the one being filled.
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut current = vec![0usize; hypothesis.len() + 1];
    for (i, reference_word) in reference.iter().enumerate() {
        current[0] = i + 1;
        for (j, hypothesis_word) in hypothesis.iter().enumerate() {
            let substitute = previous[j] + usize::from(reference_word != hypothesis_word);
            let delete = previous[j + 1] + 1;
            let insert = current[j] + 1;
            current[j + 1] = substitute.min(delete).min(insert);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[hypothesis.len()] as f64 / reference.len() as f64
}

/// Assert `text` is within [`MAX_WER`] of `reference`, reporting the measured
/// rate either way so a near-miss is visible in passing output too.
fn assert_wer_within_budget(label: &str, reference: &str, text: &str) {
    let measured = wer(reference, text);
    println!("{label}: WER {measured:.3}\n  {text:?}");
    assert!(
        measured <= MAX_WER,
        "{label}: WER {measured:.3} exceeds the {MAX_WER} budget\n  got:      {text:?}\n  expected: {reference:?}"
    );
}

/// Assert the transcript has words in it, naming the clip that produced it.
fn assert_transcribed_something(label: &str, text: &str) {
    println!("{label}: {text:?}");
    assert!(
        !words(text).is_empty(),
        "{label}: expected a transcript, got {text:?}"
    );
}

/// English function words, chosen to have no French homographs, so the rate
/// separates an English transcript from a French one without a language model.
const ENGLISH_FUNCTION_WORDS: &[&str] = &[
    "the", "and", "of", "to", "in", "is", "was", "were", "that", "this", "it", "he", "she", "you",
    "i", "we", "they", "his", "her", "their", "my", "for", "with", "not", "but", "have", "had",
    "said", "when", "what", "which", "who", "will", "would", "there", "from", "as", "so", "if",
    "do", "did", "does", "be", "been", "at", "by",
];

/// Share of `text`'s words that are English function words.
fn english_word_rate(text: &str) -> f64 {
    let words = words(text);
    if words.is_empty() {
        return 0.0;
    }
    let hits = words
        .iter()
        .filter(|word| ENGLISH_FUNCTION_WORDS.contains(&word.as_str()))
        .count();
    hits as f64 / words.len() as f64
}

#[test]
fn test_whisper_transcribes_five_second_slice() {
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    // A slice cuts mid-word, so there is no exact reference to score against.
    let pcm = read_pcm(JFK_CLIP);
    let text = transcribed(&mut runtime, &pcm[..5 * SAMPLE_RATE as usize], &[]);
    assert_transcribed_something("5 s slice", &text);
}

#[test]
fn test_whisper_transcribes_full_clip_within_wer_budget() {
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = read_pcm(JFK_CLIP);
    assert_eq!(pcm.len(), 176_000, "jfk.wav should be 11.0 s at 16 kHz");

    let text = transcribed(&mut runtime, &pcm, &[]);
    assert_wer_within_budget("11 s clip", JFK_TRANSCRIPT, &text);
}

/// The last sample count candle's mel arithmetic keeps inside one encoder
/// window. Anything past it used to fail; this side of the boundary always
/// worked, so it is here to prove the fix did not break the case that did.
#[test]
fn test_whisper_transcribes_exactly_240159_samples() {
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = looped(&read_pcm(JFK_CLIP), 240_159);
    let text = transcribed(&mut runtime, &pcm, &[]);
    assert_transcribed_something("240_159 samples (15.0099 s)", &text);
}

/// One sample past the boundary: this is the case that hard-failed with
/// `narrow invalid args` before the mel was windowed.
#[test]
fn test_whisper_transcribes_exactly_240160_samples() {
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = looped(&read_pcm(JFK_CLIP), 240_160);
    let text = transcribed(&mut runtime, &pcm, &[]);
    assert_transcribed_something("240_160 samples (15.01 s)", &text);
}

/// Exactly 30.00 s — the length the runtime used to clamp every longer input
/// down to, and the width of one encoder window.
#[test]
fn test_whisper_transcribes_exactly_thirty_seconds() {
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = looped(&read_pcm(JFK_CLIP), 480_000);
    let text = transcribed(&mut runtime, &pcm, &[]);
    assert_transcribed_something("480_000 samples (30.00 s)", &text);
}

/// Six copies of the clip, 66 s: three encoder windows. Scored against the
/// transcript repeated six times, so a dropped window shows up as a WER blowout
/// rather than as a plausible-looking short transcript.
#[test]
fn test_whisper_transcribes_sixty_six_seconds_without_dropping_audio() {
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let jfk = read_pcm(JFK_CLIP);
    let pcm = looped(&jfk, jfk.len() * 6);
    assert_eq!(pcm.len(), 1_056_000, "six 11 s copies at 16 kHz");

    let text = transcribed(&mut runtime, &pcm, &[]);
    assert_wer_within_budget("66 s clip", &transcript_repeated(6), &text);
}

/// Per-request `language` must not be baked into the cached model: the runtime
/// caches by directory, so the second request on one loaded runtime has to see
/// its own language, not the first request's.
#[test]
fn test_whisper_language_metadata_applies_per_request_not_per_load() {
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = read_pcm(FRENCH_CLIP);
    let english_first = transcribed(&mut runtime, &pcm, &[("language", "en")]);
    let french_first = transcribed(&mut runtime, &pcm, &[("language", "fr")]);
    let english_second = transcribed(&mut runtime, &pcm, &[("language", "en")]);
    let french_second = transcribed(&mut runtime, &pcm, &[("language", "fr")]);

    assert_transcribed_something("language=en", &english_first);
    assert_transcribed_something("language=fr", &french_first);
    assert_ne!(
        english_first.trim(),
        french_first.trim(),
        "language=fr decoded the French clip identically to language=en, so the metadata did nothing"
    );
    assert_eq!(
        english_first, english_second,
        "language=en changed after an intervening language=fr request, so options leaked into the cached model"
    );
    assert_eq!(
        french_first, french_second,
        "language=fr changed after an intervening language=en request, so options leaked into the cached model"
    );
}

/// `task=translate` on French audio has to come back in English. Measured by
/// the share of English function words rather than by an exact reference,
/// which whisper-tiny's translation quality does not support.
#[test]
fn test_whisper_translate_task_returns_english_for_french_audio() {
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = read_pcm(FRENCH_CLIP);
    let french = transcribed(
        &mut runtime,
        &pcm,
        &[("language", "fr"), ("task", "transcribe")],
    );
    let translated = transcribed(
        &mut runtime,
        &pcm,
        &[("language", "fr"), ("task", "translate")],
    );

    let french_rate = english_word_rate(&french);
    let translated_rate = english_word_rate(&translated);
    println!("transcribe (fr): rate {french_rate:.3} {french:?}");
    println!("translate  (en): rate {translated_rate:.3} {translated:?}");

    assert!(
        translated_rate > french_rate,
        "task=translate should read as English: {translated_rate:.3} English-word rate vs \
         {french_rate:.3} for the French transcript\n  translate: {translated:?}\n  transcribe: {french:?}"
    );
}

/// `prompt` has to surface as invalid input all the way through the public
/// path — accepting it and decoding without it is the failure mode this
/// guards against.
#[test]
fn test_whisper_rejects_prompt_metadata_as_invalid_input() {
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = read_pcm(FRENCH_CLIP);
    let error = transcribe(&mut runtime, &pcm, &[("prompt", "x")])
        .expect_err("'prompt' is unsupported and must be rejected");

    assert!(
        matches!(error, AdapterError::InvalidInput(_)),
        "expected InvalidInput, got {error:?}"
    );
    assert!(
        error.to_string().contains("prompt"),
        "error should name the rejected parameter: {error}"
    );
}
