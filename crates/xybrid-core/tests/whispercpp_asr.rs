//! End-to-end ASR through `TemplateExecutor` on the `GgmlWhisper` template.
//!
//! Proves the whole dispatch chain, not just the backend: metadata parsing →
//! template match → runtime registry lookup → whisper.cpp → text envelope.
//!
//! Skips cleanly when the (gitignored, ~32 MB) model fixture is absent, per the
//! `fixtures::model_if_available()` discipline. Fetch it with:
//!
//! ```text
//! ./integration-tests/download.sh whisper-ggml-tiny-en
//! ```

#![cfg(feature = "asr-whispercpp")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use xybrid_core::execution::{ExecutionTemplate, ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::streaming::{StreamConfig, StreamSession};

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<root>/crates/xybrid-core`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate lives two levels below the workspace root")
        .to_path_buf()
}

fn bundle_dir() -> PathBuf {
    workspace_root().join("integration-tests/fixtures/models/whisper-ggml-tiny-en")
}

/// The bundle, or `None` to skip — both the metadata and the weights must be
/// present.
fn bundle_if_available() -> Option<(PathBuf, ModelMetadata)> {
    let dir = bundle_dir();
    if !dir.join("model.bin").exists() {
        return None;
    }
    let json = std::fs::read_to_string(dir.join("model_metadata.json")).ok()?;
    let metadata: ModelMetadata = serde_json::from_str(&json).ok()?;
    Some((dir, metadata))
}

fn jfk_wav() -> Option<Vec<u8>> {
    let p = workspace_root().join("integration-tests/fixtures/input/jfk.wav");
    std::fs::read(p).ok()
}

#[test]
fn the_fixture_metadata_round_trips_as_a_ggml_whisper_template() {
    let json = std::fs::read_to_string(bundle_dir().join("model_metadata.json"))
        .expect("fixture metadata is committed, unlike the weights");
    let metadata: ModelMetadata = serde_json::from_str(&json).expect("metadata parses");

    let ExecutionTemplate::GgmlWhisper {
        model_file,
        language,
        audio_ctx,
        translate,
    } = &metadata.execution_template
    else {
        panic!(
            "expected a GgmlWhisper template, got {:?}",
            metadata.execution_template
        );
    };

    assert_eq!(model_file, "model.bin");
    assert_eq!(language.as_deref(), Some("en"));
    // 0 means no encoder truncation. The fixture asserts the *conservative*
    // default on purpose: truncation is opt-in per model after a quality check.
    assert_eq!(*audio_ctx, 0);
    assert!(!*translate);
}

#[test]
fn transcribes_audio_through_the_template_executor() {
    let Some((dir, metadata)) = bundle_if_available() else {
        eprintln!("skipping: whisper GGML fixture weights not present");
        return;
    };
    let Some(wav) = jfk_wav() else {
        eprintln!("skipping: jfk.wav fixture not present");
        return;
    };

    let mut executor = TemplateExecutor::new(&dir.display().to_string());
    let input = Envelope::new(EnvelopeKind::Audio(wav));

    let result = executor
        .execute(&metadata, &input, None)
        .expect("GgmlWhisper dispatch reaches whisper.cpp and returns text");

    let EnvelopeKind::Text(text) = &result.kind else {
        panic!("expected a Text envelope, got {:?}", result.kind_str());
    };

    // Content words rather than an exact string: quantization and upstream
    // decoder changes legitimately move punctuation and casing without being a
    // regression.
    let lower = text.to_lowercase();
    assert!(
        lower.contains("fellow americans") && lower.contains("ask not"),
        "expected the JFK line, got: {text:?}"
    );
}

#[test]
fn a_request_language_overrides_the_bundle_default() {
    let Some((dir, metadata)) = bundle_if_available() else {
        eprintln!("skipping: whisper GGML fixture weights not present");
        return;
    };
    let Some(wav) = jfk_wav() else {
        eprintln!("skipping: jfk.wav fixture not present");
        return;
    };

    // The bundle declares `language: "en"`. This asks for Japanese, which the
    // English-only fixture has no token for — so the request must be rejected
    // rather than silently falling back to the bundle's English. That is the
    // observable proof that per-request metadata takes precedence.
    let mut metadata_overridden = HashMap::new();
    metadata_overridden.insert("language".to_string(), "ja".to_string());
    let input = Envelope::with_metadata(EnvelopeKind::Audio(wav), metadata_overridden);

    let mut executor = TemplateExecutor::new(&dir.display().to_string());
    let err = executor
        .execute(&metadata, &input, None)
        .expect_err("an .en model cannot honour a Japanese request");

    let message = err.to_string();
    assert!(
        message.contains("ja") || message.to_lowercase().contains("language"),
        "error should name the rejected language, got: {message}"
    );
}

#[test]
fn live_streaming_produces_a_growing_transcript() {
    let Some((dir, _metadata)) = bundle_if_available() else {
        eprintln!("skipping: whisper GGML fixture weights not present");
        return;
    };
    let Some(wav) = jfk_wav() else {
        eprintln!("skipping: jfk.wav fixture not present");
        return;
    };

    // StreamSession reads the bundle's own metadata, so this exercises the
    // rolling-window path end to end: buffer config inference, per-window
    // dispatch to whisper.cpp, and span reconciliation across windows.
    let mut session = StreamSession::new(&dir, StreamConfig::default())
        .expect("GgmlWhisper bundle opens as a streaming session");

    let partials = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&partials);
    session.on_partial(move |partial| {
        if let Ok(mut guard) = sink.lock() {
            guard.push(partial.text);
        }
    });

    // Feed in 0.5 s slices, the same cadence a microphone callback delivers.
    let pcm = decode_wav_16k_mono(&wav);
    for slice in pcm.chunks(8_000) {
        session.feed(slice).expect("feeding audio succeeds");
    }
    let transcript = session.flush().expect("flush finalises the transcript");

    let lower = transcript.to_lowercase();
    assert!(
        lower.contains("fellow americans") && lower.contains("ask not"),
        "expected the JFK line from the streamed transcript, got: {transcript:?}"
    );

    // The reconciler must not repeat words across the 0.5 s window overlap.
    // "ask not" legitimately appears twice in the JFK line; three or more means
    // the seam dedupe failed.
    let occurrences = lower.matches("ask not").count();
    assert!(
        occurrences <= 2,
        "overlapping windows duplicated text ({occurrences} occurrences of 'ask not'): {transcript:?}"
    );

    let partials = partials.lock().expect("partial sink is not poisoned");
    assert!(
        !partials.is_empty(),
        "a live session must emit at least one partial before the final transcript"
    );
}

#[test]
fn streaming_language_override_reaches_whisper_transcribe_params() {
    let Some((dir, _metadata)) = bundle_if_available() else {
        eprintln!("skipping: whisper GGML fixture weights not present");
        return;
    };
    let Some(wav) = jfk_wav() else {
        eprintln!("skipping: jfk.wav fixture not present");
        return;
    };

    // The fixture is English-only. Asking the streaming API for Japanese can
    // only produce this error after StreamConfig reaches TranscribeParams;
    // falling back to the bundle's `en` would transcribe successfully.
    let config = StreamConfig {
        language: Some("ja".to_string()),
        ..Default::default()
    };
    let mut session =
        StreamSession::new(&dir, config).expect("GgmlWhisper bundle opens as a streaming session");
    let pcm = decode_wav_16k_mono(&wav);

    let err = pcm
        .chunks(8_000)
        .find_map(|slice| session.feed(slice).err())
        .or_else(|| session.flush().err())
        .expect("an .en model cannot honour a Japanese streaming request");

    let message = err.to_string();
    assert!(
        message.contains("ja") || message.to_lowercase().contains("language"),
        "error should name the rejected language, got: {message}"
    );
}

/// Decode the committed 16 kHz mono 16-bit fixture into float samples.
fn decode_wav_16k_mono(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() > 44, "WAV shorter than its header");
    bytes[44..]
        .chunks_exact(2)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32_768.0)
        .collect()
}
