//! Per-phoneme TTS timings end to end, on two fake ONNX TTS models.
//!
//! `fixtures/tts/fake/` holds the models `tools/conformance/gen_fake_tts.py`
//! writes. Both take Kokoro's inputs and render the same waveform: each token
//! lasts a fixed number of 600-sample frames (`1 + id % 5` for a phoneme at
//! speed 1), the pad and silence tokens are silent and every phoneme is a loud
//! tone, so the raw waveform is exactly `600 * sum(duration)` samples long and
//! TTSAudioEncode trims exactly the silent pads. `with_duration` also returns
//! the frames as an INT64 `duration` output, declared before `audio`;
//! `without_duration` returns `audio` only. Neither ships a voice file.
//!
//! Each model runs through `ModelLoader` and `XybridModel::run`, the route an
//! application takes. The tests check that:
//!
//! - only `with_duration` reports timings, and `without_duration` returns the
//!   same audio bytes and no `phoneme_timings` key;
//! - spans are in order and inside the clip, each names one `tokens.txt`
//!   symbol (never a pad or silence token) and lasts its token's frames;
//! - the spans cover exactly the loud samples of the returned audio, so they
//!   survive the head and tail trim, the `trim_trailing_samples` cut and, for
//!   text over the 350-character chunk limit, the chunk crossfades;
//! - a chunked run times its first sentence exactly as a single-chunk run does.
//!
//! The tests that synthesize skip without ONNX Runtime, unless
//! `XYBRID_REQUIRE_MODELS` is set, where they fail instead.

use std::collections::HashMap;
use std::path::PathBuf;

use integration_tests::fixtures;
use xybrid_core::ir::{Envelope, EnvelopeKind, PhonemeTiming};
use xybrid_core::testing::model_fixtures::{models_required, ENV_REQUIRE_MODELS};
use xybrid_sdk::{InferenceResult, ModelLoader, XybridModel};

/// TTSAudioEncode's `sample_rate` in the fakes' metadata.
const SAMPLE_RATE: f64 = 24_000.0;
/// Samples per duration frame (`FRAME_SAMPLES` in `gen_fake_tts.py`).
const FRAME_SAMPLES: f64 = 600.0;
/// The executor's default chunk limit, which the fakes do not override.
const MAX_CHUNK_CHARS: usize = 350;
/// How far, in samples, a span edge may sit from where the audio says it is.
const SLACK_SAMPLES: f64 = 1.0;

/// Every word is in the fakes' `misaki/us_gold.json`. The quotes become a `"`
/// phoneme, which the metadata JSON has to escape.
const SENTENCE: &str = r#""Hello," she said. Welcome to the Rusty Flagon; the fire is warm!"#;
/// Misaki's phonemes for [`SENTENCE`] with the fakes' dictionary, one timing
/// per character.
const SENTENCE_PHONEMES: &str =
    r#""həlˈO," ʃi sˈɛd. wˈɛlkəm tu ði ɹˈʌsti flˈæɡən; ði fˈIəɹ ɪz wˈɔɹm!"#;
/// Copies of [`SENTENCE`] in the chunked text: 461 characters.
const REPEATS: usize = 7;

/// Whether ONNX Runtime can run here: it is linked in on Apple platforms and
/// loaded from `ORT_DYLIB_PATH` elsewhere. Without it a test that synthesizes
/// skips, unless models are required, where it fails instead.
fn onnx_runtime() -> bool {
    let available = cfg!(any(target_os = "macos", target_os = "ios"))
        || std::env::var_os("ORT_DYLIB_PATH")
            .is_some_and(|path| std::path::Path::new(&path).is_file());
    if !available {
        assert!(
            !models_required(),
            "{ENV_REQUIRE_MODELS} is set but ONNX Runtime is not loadable; set ORT_DYLIB_PATH"
        );
        eprintln!("Skipping: ONNX Runtime is not loadable (set ORT_DYLIB_PATH)");
    }
    available
}

fn fake_dir(variant: &str) -> PathBuf {
    fixtures::fixtures_dir().join("tts/fake").join(variant)
}

fn read(path: PathBuf) -> Vec<u8> {
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn load(variant: &str) -> XybridModel {
    ModelLoader::from_directory(fake_dir(variant))
        .unwrap_or_else(|e| panic!("{variant}: {e}"))
        .load()
        .unwrap_or_else(|e| panic!("load {variant}: {e}"))
}

/// Synthesizes `text`, at `speed` when given.
fn speak(model: &XybridModel, text: &str, speed: Option<&str>) -> InferenceResult {
    let mut request = Envelope::new(EnvelopeKind::Text(text.to_string()));
    if let Some(speed) = speed {
        request
            .metadata
            .insert("speed".to_string(), speed.to_string());
    }
    model
        .run(&request, None)
        .unwrap_or_else(|e| panic!("synthesize {text:?}: {e}"))
}

/// The result's 16-bit PCM samples.
fn pcm_samples(result: &InferenceResult) -> Vec<i16> {
    let bytes = result.audio_bytes().expect("TTS returns audio");
    assert_eq!(bytes.len() % 2, 0, "TTSAudioEncode returns 16-bit PCM");
    bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

/// The timings in the result's metadata, which the SDK accessor must agree with.
fn reported_timings(result: &InferenceResult) -> Vec<PhonemeTiming> {
    let raw = result
        .metadata(Envelope::PHONEME_TIMINGS_METADATA_KEY)
        .expect("a model with a duration output reports phoneme timings");
    let parsed: Vec<PhonemeTiming> = serde_json::from_str(raw)
        .unwrap_or_else(|e| panic!("phoneme timings are a JSON array of timings: {e}: {raw}"));
    assert_eq!(
        result.phoneme_timings(),
        parsed,
        "InferenceResult::phoneme_timings reads the metadata"
    );
    parsed
}

/// The fakes' `tokens.txt`: symbol to token id.
fn vocab() -> HashMap<char, i64> {
    let tokens = read(fake_dir("with_duration").join("tokens.txt"));
    let text = String::from_utf8(tokens).expect("tokens.txt is UTF-8");
    xybrid_core::phonemizer::load_tokens_map(&text)
}

/// Frames a phoneme token lasts at speed 1 (`base_frames` in `gen_fake_tts.py`).
fn frames_at_speed_1(id: i64) -> i64 {
    1 + id % 5
}

/// A time in seconds as a sample index, rounded to the nearest sample.
fn sample_at(seconds: f32) -> f64 {
    (f64::from(seconds) * SAMPLE_RATE).round()
}

fn phonemes(timings: &[PhonemeTiming]) -> String {
    timings.iter().map(|t| t.phoneme.as_str()).collect()
}

/// Checks what every timed clip promises: spans in order and inside the clip,
/// each naming one `tokens.txt` symbol. Pads and silence tokens have no symbol
/// there, so a timing for one fails here.
fn assert_well_formed(timings: &[PhonemeTiming], samples: usize, vocab: &HashMap<char, i64>) {
    assert!(!timings.is_empty(), "a spoken clip has timings");
    let clip = samples as f64 / SAMPLE_RATE;
    let mut previous_end = 0.0_f32;
    for (index, timing) in timings.iter().enumerate() {
        let mut symbols = timing.phoneme.chars();
        let symbol = symbols.next();
        assert!(
            symbols.next().is_none() && symbol.is_some_and(|s| vocab.contains_key(&s)),
            "timing {index} names one tokens.txt symbol, never a pad or silence token: {timing:?}"
        );
        assert!(
            timing.start >= previous_end,
            "timing {index} starts before the previous one ends ({previous_end} s): {timing:?}"
        );
        assert!(
            timing.end > timing.start,
            "timing {index} is empty: {timing:?}"
        );
        assert!(
            f64::from(timing.end) <= clip + 1e-6,
            "timing {index} ends after the clip ({clip} s): {timing:?}"
        );
        previous_end = timing.end;
    }
}

/// Checks that each span lasts its token's frames, at speed 1.
fn assert_frame_lengths(timings: &[PhonemeTiming], vocab: &HashMap<char, i64>) {
    for (index, timing) in timings.iter().enumerate() {
        let id = timing
            .phoneme
            .chars()
            .next()
            .and_then(|symbol| vocab.get(&symbol))
            .unwrap_or_else(|| panic!("timing {index} names a tokens.txt symbol: {timing:?}"));
        let expected = frames_at_speed_1(*id) as f64 * FRAME_SAMPLES;
        let actual = sample_at(timing.end) - sample_at(timing.start);
        assert!(
            (actual - expected).abs() <= SLACK_SAMPLES,
            "timing {index} lasts {actual} samples, its token {expected}: {timing:?}"
        );
    }
}

/// Runs of loud samples, as `[start, end)` sample ranges. The fakes' phonemes
/// are tones at least half as loud as the loudest one; pads and silence tokens
/// leave only the high-pass filter's decay, about 1% of it.
fn loud_runs(pcm: &[i16]) -> Vec<(f64, f64)> {
    let peak = pcm.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
    assert!(peak > 0, "the clip is not silent");
    let threshold = peak / 8;
    let mut runs = Vec::new();
    let mut start = None;
    for (index, sample) in pcm.iter().enumerate() {
        match (sample.unsigned_abs() > threshold, start) {
            (true, None) => start = Some(index),
            (false, Some(first)) => {
                runs.push((first as f64, index as f64));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(first) = start {
        runs.push((first as f64, pcm.len() as f64));
    }
    runs
}

/// The spans, joined into runs wherever one ends where the next starts.
fn span_runs(timings: &[PhonemeTiming]) -> Vec<(f64, f64)> {
    let mut runs: Vec<(f64, f64)> = Vec::new();
    for timing in timings {
        let (start, end) = (sample_at(timing.start), sample_at(timing.end));
        match runs.last_mut() {
            Some(run) if start - run.1 <= SLACK_SAMPLES => run.1 = end,
            _ => runs.push((start, end)),
        }
    }
    runs
}

/// Checks that the spans cover exactly the loud samples of the clip, and
/// returns how many separate runs of speech it holds: one per chunk, since
/// chunks meet in the silence of their pads.
fn assert_cover_loud_samples(timings: &[PhonemeTiming], pcm: &[i16]) -> usize {
    let loud = loud_runs(pcm);
    let spans = span_runs(timings);
    assert_eq!(
        loud.len(),
        spans.len(),
        "loud runs {loud:?} and timed runs {spans:?} differ"
    );
    for (heard, timed) in loud.iter().zip(&spans) {
        assert!(
            (heard.0 - timed.0).abs() <= SLACK_SAMPLES
                && (heard.1 - timed.1).abs() <= SLACK_SAMPLES,
            "the timings put speech at samples {timed:?}, the audio at {heard:?}"
        );
    }
    loud.len()
}

#[test]
fn the_fakes_differ_only_in_their_graph() {
    for file in ["tokens.txt", "misaki/us_gold.json"] {
        assert_eq!(
            read(fake_dir("with_duration").join(file)),
            read(fake_dir("without_duration").join(file)),
            "{file}"
        );
    }
    let metadata = |variant: &str| {
        let bytes = read(fake_dir(variant).join("model_metadata.json"));
        let mut value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("model_metadata.json is JSON");
        let object = value
            .as_object_mut()
            .expect("model_metadata.json is an object");
        object.remove("model_id");
        object.remove("description");
        value
    };
    assert_eq!(metadata("with_duration"), metadata("without_duration"));
}

#[test]
fn a_single_chunk_is_timed_phoneme_by_phoneme() {
    if !onnx_runtime() {
        return;
    }
    assert!(SENTENCE.len() <= MAX_CHUNK_CHARS);
    let vocab = vocab();
    let result = speak(&load("with_duration"), SENTENCE, None);
    let pcm = pcm_samples(&result);
    let timings = reported_timings(&result);

    assert_well_formed(&timings, pcm.len(), &vocab);
    assert_eq!(
        phonemes(&timings),
        SENTENCE_PHONEMES,
        "one timing per phoneme, in order"
    );
    assert!(timings.iter().any(|t| t.phoneme == "\""), "a quote phoneme");
    assert_frame_lengths(&timings, &vocab);
    assert_eq!(assert_cover_loud_samples(&timings, &pcm), 1);
}

#[test]
fn chunked_text_is_timed_across_the_crossfades() {
    if !onnx_runtime() {
        return;
    }
    let text = [SENTENCE; REPEATS].join(" ");
    assert!(text.len() > MAX_CHUNK_CHARS, "the text needs chunking");
    let vocab = vocab();
    let model = load("with_duration");
    let result = speak(&model, &text, None);
    let pcm = pcm_samples(&result);
    let timings = reported_timings(&result);

    assert_well_formed(&timings, pcm.len(), &vocab);
    // Each chunk is phonemized on its own and trimmed, so the space between two
    // sentences that land in different chunks is not a phoneme.
    let spoken: String = phonemes(&timings).chars().filter(|c| *c != ' ').collect();
    let expected: String = [SENTENCE_PHONEMES; REPEATS]
        .concat()
        .chars()
        .filter(|c| *c != ' ')
        .collect();
    assert_eq!(
        spoken, expected,
        "no phoneme lost or repeated at a chunk boundary"
    );
    assert_frame_lengths(&timings, &vocab);
    let chunks = assert_cover_loud_samples(&timings, &pcm);
    assert!(
        chunks >= 2,
        "text over the chunk limit is synthesized in chunks with silence between them, \
         but the audio holds {chunks} run of speech"
    );

    // The first chunk starts the way the single-chunk run does, and the head
    // trim cuts both at the same sample, so its first sentence is timed alike.
    let single = reported_timings(&speak(&model, SENTENCE, None));
    assert_eq!(
        &timings[..single.len()],
        &single[..],
        "the chunked run times its first sentence like the single-chunk run"
    );
}

#[test]
fn timings_follow_the_requested_speed() {
    if !onnx_runtime() {
        return;
    }
    let vocab = vocab();
    let model = load("with_duration");
    let normal = reported_timings(&speak(&model, SENTENCE, None));
    let result = speak(&model, SENTENCE, Some("1.25"));
    let pcm = pcm_samples(&result);
    let timings = reported_timings(&result);

    assert_well_formed(&timings, pcm.len(), &vocab);
    assert_eq!(phonemes(&timings), SENTENCE_PHONEMES);
    assert_eq!(assert_cover_loud_samples(&timings, &pcm), 1);
    let fast_end = timings.last().expect("checked by assert_well_formed").end;
    let normal_end = normal.last().expect("a spoken clip has timings").end;
    assert!(
        fast_end < normal_end,
        "speech at 1.25x ends sooner ({fast_end} s) than at 1x ({normal_end} s)"
    );
}

#[test]
fn a_model_without_a_duration_output_reports_nothing_and_sounds_the_same() {
    if !onnx_runtime() {
        return;
    }
    let timed = load("with_duration");
    let plain = load("without_duration");
    for text in [SENTENCE.to_string(), [SENTENCE; REPEATS].join(" ")] {
        let with_duration = speak(&timed, &text, None);
        let without_duration = speak(&plain, &text, None);
        assert!(
            without_duration
                .metadata(Envelope::PHONEME_TIMINGS_METADATA_KEY)
                .is_none(),
            "a model without a duration output reports no timings ({} chars)",
            text.len()
        );
        assert!(without_duration.phoneme_timings().is_empty());
        assert!(!reported_timings(&with_duration).is_empty());

        let with_audio = with_duration.audio_bytes().expect("TTS returns audio");
        let without_audio = without_duration.audio_bytes().expect("TTS returns audio");
        assert!(
            with_audio == without_audio,
            "the duration output changes no audio byte ({} chars: {} vs {} bytes)",
            text.len(),
            with_audio.len(),
            without_audio.len()
        );
    }
}
