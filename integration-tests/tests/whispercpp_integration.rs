//! End-to-end Whisper regression coverage through the whisper.cpp runtime.
//!
//! This carries the behavior checks that previously existed only for Candle:
//! short and boundary-length audio, multi-window transcription, per-request
//! language and task handling, translation, and non-speech-token suppression.
//! Every inference goes through [`ModelRuntime`] with an [`Envelope`], matching
//! the executor's production path.
//!
//! Run with:
//!   cargo test -p integration-tests --features asr-whispercpp \
//!     --test whispercpp_integration -- --nocapture
//!
//! Download the model first:
//!   ./integration-tests/download.sh whisper-tiny-ggml

#![cfg(feature = "asr-whispercpp")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use integration_tests::fixtures;
use xybrid_core::audio::{decode_wav_audio, samples_to_wav};
use xybrid_core::execution::{ExecutionTemplate, ModelMetadata};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::runtime_adapter::whisper_cpp::WhisperCppRuntime;
use xybrid_core::runtime_adapter::{AdapterError, ModelRuntime};

const MODEL: &str = "whisper-tiny-ggml";
const MODEL_OVERRIDE_ENV: &str = "XYBRID_WHISPER_TEST_MODEL";
const SAMPLE_RATE: u32 = 16_000;
const JFK_CLIP: &str = "jfk.wav";
const FRENCH_CLIP: &str = "mls-fr-1-8s.wav";
const JFK_TRANSCRIPT: &str = "And so my fellow Americans, ask not what your country can do for \
                              you, ask what you can do for your country.";
const MAX_WER: f64 = 0.10;

/// Serializes real-model cases so the test harness cannot create a dozen
/// ~150 MB whisper contexts and CPU-heavy inference calls at once.
static MODEL_TEST_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug)]
struct FixtureModel {
    path: PathBuf,
    language: Option<String>,
    audio_ctx: u32,
    translate: bool,
}

fn serial_model_test() -> MutexGuard<'static, ()> {
    match MODEL_TEST_LOCK.lock() {
        Ok(guard) => guard,
        // This lock carries no model state; it only limits concurrency. A
        // previous assertion poisoning it does not make the next test unsafe.
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// A loaded multilingual whisper.cpp runtime, or `None` when the model is not
/// downloaded and model fixtures are optional.
fn whisper_runtime() -> Option<WhisperCppRuntime> {
    let fixture = fixture_model()?;
    let mut runtime = WhisperCppRuntime::new().with_defaults(
        fixture.language,
        fixture.audio_ctx,
        fixture.translate,
    );
    runtime
        .load(&fixture.path)
        .unwrap_or_else(|e| panic!("loading {}: {e}", fixture.path.display()));
    Some(runtime)
}

fn fixture_model() -> Option<FixtureModel> {
    if let Some(path) = std::env::var_os(MODEL_OVERRIDE_ENV) {
        let path = PathBuf::from(path);
        assert!(
            path.is_file(),
            "{MODEL_OVERRIDE_ENV} points at a missing model: {}",
            path.display()
        );
        return Some(FixtureModel {
            path,
            language: None,
            audio_ctx: 0,
            translate: false,
        });
    }

    let model_dir = fixtures::model_for_test(MODEL)?;
    let fixture = read_bundle_config(&model_dir);
    if fixture.path.is_file() {
        return Some(fixture);
    }

    if std::env::var_os(fixtures::REQUIRE_MODELS_ENV).is_some() {
        panic!(
            "{MODEL} model payload not found at {}. Run: ./integration-tests/download.sh {MODEL}",
            fixture.path.display()
        );
    }
    None
}

fn read_bundle_config(model_dir: &Path) -> FixtureModel {
    let metadata_path = model_dir.join("model_metadata.json");
    let json = std::fs::read_to_string(&metadata_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", metadata_path.display()));
    let metadata: ModelMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("parsing {}: {e}", metadata_path.display()));

    let ExecutionTemplate::GgmlWhisper {
        model_file,
        language,
        audio_ctx,
        translate,
    } = metadata.execution_template
    else {
        panic!(
            "{MODEL} must use ExecutionTemplate::GgmlWhisper, got {:?}",
            metadata.execution_template
        );
    };

    FixtureModel {
        path: model_dir.join(model_file),
        language,
        audio_ctx,
        translate,
    }
}

fn skip_notice() {
    eprintln!("Skipping: {MODEL} not downloaded. Run: ./integration-tests/download.sh {MODEL}");
}

fn read_pcm(name: &str) -> Vec<f32> {
    let path = fixtures::input_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    decode_wav_audio(&bytes, SAMPLE_RATE, 1)
        .unwrap_or_else(|e| panic!("decoding {}: {e}", path.display()))
}

fn looped(pcm: &[f32], samples: usize) -> Vec<f32> {
    assert!(!pcm.is_empty(), "cannot loop empty audio");
    pcm.iter().copied().cycle().take(samples).collect()
}

fn transcript_repeated(times: usize) -> String {
    vec![JFK_TRANSCRIPT; times].join(" ")
}

fn transcribe(
    runtime: &mut WhisperCppRuntime,
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

fn transcribed(runtime: &mut WhisperCppRuntime, pcm: &[f32], metadata: &[(&str, &str)]) -> String {
    transcribe(runtime, pcm, metadata).unwrap_or_else(|e| {
        panic!(
            "transcribing {} samples ({:.2} s) with {metadata:?} failed: {e}",
            pcm.len(),
            pcm.len() as f64 / f64::from(SAMPLE_RATE),
        )
    })
}

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

fn wer(reference: &str, hypothesis: &str) -> f64 {
    let reference = words(reference);
    let hypothesis = words(hypothesis);
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }

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

fn assert_wer_within_budget(label: &str, reference: &str, text: &str) {
    let measured = wer(reference, text);
    println!("{label}: WER {measured:.3}\n  {text:?}");
    assert!(
        measured <= MAX_WER,
        "{label}: WER {measured:.3} exceeds the {MAX_WER} budget\n  got:      \
         {text:?}\n  expected: {reference:?}"
    );
}

fn assert_transcribed_something(label: &str, text: &str) {
    println!("{label}: {text:?}");
    assert!(
        !words(text).is_empty(),
        "{label}: expected a transcript, got {text:?}"
    );
}

fn assert_no_hallucinated_annotation(label: &str, text: &str) {
    assert!(
        !text.contains('[') && !text.contains('('),
        "{label}: transcript carries a bracketed non-speech annotation: {text:?}"
    );
}

const ENGLISH_FUNCTION_WORDS: &[&str] = &[
    "the", "and", "of", "to", "in", "is", "was", "were", "that", "this", "it", "he", "she", "you",
    "i", "we", "they", "his", "her", "their", "my", "for", "with", "not", "but", "have", "had",
    "said", "when", "what", "which", "who", "will", "would", "there", "from", "as", "so", "if",
    "do", "did", "does", "be", "been", "at", "by",
];

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
fn sub_second_audio_emits_no_annotation_token() {
    let _serial = serial_model_test();
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = read_pcm(JFK_CLIP);
    let text = transcribed(&mut runtime, &pcm[..SAMPLE_RATE as usize * 3 / 10], &[]);
    println!("0.3 s slice: {text:?}");
    assert_no_hallucinated_annotation("0.3 s slice", &text);
}

#[test]
fn forced_language_mismatch_emits_no_annotation_token() {
    let _serial = serial_model_test();
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = read_pcm(FRENCH_CLIP);
    let text = transcribed(&mut runtime, &pcm, &[("language", "en")]);
    println!("French audio forced to language=en: {text:?}");
    assert_no_hallucinated_annotation("French audio forced to language=en", &text);
}

#[test]
fn transcribes_five_second_slice() {
    let _serial = serial_model_test();
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = read_pcm(JFK_CLIP);
    let text = transcribed(&mut runtime, &pcm[..5 * SAMPLE_RATE as usize], &[]);
    assert_transcribed_something("5 s slice", &text);
}

#[test]
fn transcribes_full_clip_within_wer_budget() {
    let _serial = serial_model_test();
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = read_pcm(JFK_CLIP);
    assert_eq!(pcm.len(), 176_000, "jfk.wav should be 11.0 s at 16 kHz");
    let text = transcribed(&mut runtime, &pcm, &[]);
    assert_wer_within_budget("11 s clip", JFK_TRANSCRIPT, &text);
}

#[test]
fn transcribes_exactly_240159_samples() {
    let _serial = serial_model_test();
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = looped(&read_pcm(JFK_CLIP), 240_159);
    let text = transcribed(&mut runtime, &pcm, &[]);
    assert_transcribed_something("240_159 samples (15.0099 s)", &text);
}

#[test]
fn transcribes_exactly_240160_samples() {
    let _serial = serial_model_test();
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = looped(&read_pcm(JFK_CLIP), 240_160);
    let text = transcribed(&mut runtime, &pcm, &[]);
    assert_transcribed_something("240_160 samples (15.01 s)", &text);
    assert_no_hallucinated_annotation("240_160 samples (15.01 s)", &text);
}

#[test]
fn transcribes_exactly_thirty_seconds() {
    let _serial = serial_model_test();
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = looped(&read_pcm(JFK_CLIP), 480_000);
    let text = transcribed(&mut runtime, &pcm, &[]);
    assert_transcribed_something("480_000 samples (30.00 s)", &text);
    assert_no_hallucinated_annotation("480_000 samples (30.00 s)", &text);
}

#[test]
fn transcribes_sixty_six_seconds_without_dropping_audio() {
    let _serial = serial_model_test();
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

#[test]
fn language_metadata_applies_per_request_not_per_load() {
    let _serial = serial_model_test();
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = read_pcm(FRENCH_CLIP);
    let english_first = transcribed(&mut runtime, &pcm, &[("language", "en")]);
    let french_first = transcribed(&mut runtime, &pcm, &[("language", "fr")]);
    let english_second = transcribed(&mut runtime, &pcm, &[("language", "en")]);
    let french_second = transcribed(&mut runtime, &pcm, &[("language", "fr")]);

    assert_transcribed_something("language=fr", &french_first);
    assert_ne!(
        english_first.trim(),
        french_first.trim(),
        "language=fr decoded identically to language=en, so the metadata did nothing"
    );
    assert_eq!(
        english_first, english_second,
        "language=en changed after an intervening language=fr request"
    );
    assert_eq!(
        french_first, french_second,
        "language=fr changed after an intervening language=en request"
    );
    assert_no_hallucinated_annotation("language=en", &english_first);
}

#[test]
fn translate_task_returns_english_for_french_audio() {
    let _serial = serial_model_test();
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
        "task=translate should read as English: {translated_rate:.3} vs {french_rate:.3}\n  \
         translate: {translated:?}\n  transcribe: {french:?}"
    );
}

#[test]
fn translate_without_language_returns_english_for_french_audio() {
    let _serial = serial_model_test();
    let Some(mut runtime) = whisper_runtime() else {
        skip_notice();
        return;
    };

    let pcm = read_pcm(FRENCH_CLIP);
    let french = transcribed(&mut runtime, &pcm, &[("language", "fr")]);
    let translated = transcribed(&mut runtime, &pcm, &[("task", "translate")]);

    let french_rate = english_word_rate(&french);
    let translated_rate = english_word_rate(&translated);
    println!("transcribe (fr):          rate {french_rate:.3} {french:?}");
    println!("translate  (no language): rate {translated_rate:.3} {translated:?}");
    assert!(
        translated_rate > french_rate,
        "task=translate without a language should read as English: {translated_rate:.3} vs \
         {french_rate:.3}\n  translate: {translated:?}\n  transcribe: {french:?}"
    );
}

#[test]
fn rejects_prompt_metadata_without_exposing_it() {
    let private_prompt = "PRIVATE_PROMPT_SENTINEL_7c1f";
    let pcm = vec![0.0; SAMPLE_RATE as usize];
    let mut runtime = WhisperCppRuntime::new();
    let error = transcribe(&mut runtime, &pcm, &[("prompt", private_prompt)])
        .expect_err("prompt is unsupported and must be rejected");

    assert!(
        matches!(error, AdapterError::InvalidInput(_)),
        "expected InvalidInput, got {error:?}"
    );
    assert!(
        error.to_string().contains("prompt"),
        "error should name the rejected parameter: {error}"
    );
    assert!(
        !error.to_string().contains(private_prompt),
        "error must not expose the rejected prompt: {error}"
    );
}
