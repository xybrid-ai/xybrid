//! End-to-end smoke test for the safe whisper.cpp wrapper.
//!
//! Skips cleanly when the GGML model fixture is absent, following the
//! `fixtures::model_if_available()` discipline the integration tests use — a
//! developer without the (gitignored, ~32 MB) model still gets a green run.
//!
//! Fetch the fixture with:
//!
//! ```text
//! ./integration-tests/download.sh whisper-ggml-tiny-en
//! ```

#![cfg(feature = "bindings")]

use std::path::{Path, PathBuf};

use xybrid_whisper::{Segment, Task, TranscribeParams, WhisperError, WhisperModel};

/// Locate the GGML whisper fixture, or `None` to skip.
///
/// `XYBRID_WHISPER_TEST_MODEL` overrides the default location so the test can
/// be pointed at a larger model when checking quality rather than plumbing.
fn model_if_available() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("XYBRID_WHISPER_TEST_MODEL") {
        let p = PathBuf::from(explicit);
        return p.exists().then_some(p);
    }
    let p =
        workspace_root().join("integration-tests/fixtures/models/whisper-ggml-tiny-en/model.bin");
    p.exists().then_some(p)
}

/// The committed 16 kHz mono fixture used by the ASR integration tests.
fn audio_if_available() -> Option<PathBuf> {
    let p = workspace_root().join("integration-tests/fixtures/input/jfk.wav");
    p.exists().then_some(p)
}

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<root>/crates/xybrid-whisper`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate lives two levels below the workspace root")
        .to_path_buf()
}

/// Decode 16-bit mono PCM WAV into `[-1.0, 1.0]` floats.
///
/// Hand-rolled rather than pulling `hound` in as a dev-dependency: the fixture
/// is a known-shape 16 kHz mono file, and this crate deliberately has a tiny
/// dependency surface.
fn read_wav_16k_mono(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("fixture is readable");
    assert!(bytes.len() > 44, "WAV shorter than its header");
    assert_eq!(&bytes[0..4], b"RIFF", "not a RIFF file");
    assert_eq!(&bytes[8..12], b"WAVE", "not a WAVE file");

    let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
    assert_eq!(channels, 1, "fixture must be mono");
    assert_eq!(sample_rate, 16_000, "fixture must be 16 kHz");
    assert_eq!(bits, 16, "fixture must be 16-bit");

    bytes[44..]
        .chunks_exact(2)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32_768.0)
        .collect()
}

#[test]
fn transcribes_the_jfk_fixture() {
    let (Some(model_path), Some(audio_path)) = (model_if_available(), audio_if_available()) else {
        eprintln!("skipping: whisper GGML fixture or jfk.wav not present");
        return;
    };

    let pcm = read_wav_16k_mono(&audio_path);
    let mut model = WhisperModel::load(&model_path).expect("fixture model loads");

    let params = TranscribeParams {
        language: Some("en".to_string()),
        task: Task::Transcribe,
        n_threads: 4,
        ..Default::default()
    };

    let segments = model
        .transcribe(&pcm, &params)
        .expect("transcription succeeds");
    let text = Segment::join(&segments).to_lowercase();

    // Assert on content words rather than an exact string: quantization and
    // upstream decoder changes legitimately shift punctuation and casing, and a
    // brittle equality check would fail on those without indicating a real
    // regression.
    assert!(
        text.contains("fellow americans"),
        "expected the JFK line, got: {text:?}"
    );
    assert!(
        text.contains("ask not"),
        "expected the JFK line, got: {text:?}"
    );

    // Segments must carry a usable span, since the streaming reconciler keys
    // off them.
    assert!(!segments.is_empty(), "at least one segment");
    assert!(
        segments.iter().any(|s| s.duration_ms() > 0),
        "at least one segment spans real time: {segments:?}"
    );
}

#[test]
fn audio_shorter_than_one_stft_window_is_rejected() {
    let Some(model_path) = model_if_available() else {
        eprintln!("skipping: whisper GGML fixture not present");
        return;
    };

    let mut model = WhisperModel::load(&model_path).expect("fixture model loads");
    let err = model
        .transcribe(&[0.0; 64], &TranscribeParams::default())
        .expect_err("a 64-sample buffer is pure padding");

    assert!(
        matches!(err, WhisperError::AudioTooShort { samples: 64, .. }),
        "expected AudioTooShort, got {err:?}"
    );
}

#[test]
fn an_english_only_model_rejects_a_foreign_language_request() {
    let Some(model_path) = model_if_available() else {
        eprintln!("skipping: whisper GGML fixture not present");
        return;
    };
    let mut model = WhisperModel::load(&model_path).expect("fixture model loads");
    if model.is_multilingual() {
        eprintln!("skipping: fixture is multilingual, so the rejection does not apply");
        return;
    }

    let params = TranscribeParams {
        language: Some("ja".to_string()),
        ..Default::default()
    };
    let err = model
        .transcribe(&[0.0; 16_000], &params)
        .expect_err("an .en model has no Japanese token");

    assert!(
        matches!(err, WhisperError::UnsupportedLanguage { .. }),
        "expected UnsupportedLanguage, got {err:?}"
    );
}

#[test]
fn a_missing_model_reports_the_path_it_tried() {
    let err = WhisperModel::load("/nonexistent/ggml-nope.bin")
        .expect_err("loading a missing file must fail");
    assert!(
        matches!(&err, WhisperError::LoadFailed(p) if p.contains("ggml-nope.bin")),
        "expected LoadFailed naming the path, got {err:?}"
    );
}
