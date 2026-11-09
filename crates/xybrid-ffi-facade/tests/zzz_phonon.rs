//! Real audio over the pull transport consumed by Bolt bindings.
#![cfg(unix)]

use std::path::PathBuf;
use std::sync::Arc;
use xybrid_ffi_facade::{
    CancellationToken, Envelope, GenerationConfig, ModelLoader, RunOptions, StreamEvent,
    XybridModel,
};

fn load() -> Option<(tempfile::TempDir, Arc<XybridModel>)> {
    if !xybrid_core::features::enabled().contains(&"tts-zzz") {
        eprintln!("skipping Phonon facade test: tts-zzz is not enabled");
        return None;
    }
    let model =
        PathBuf::from(std::env::var_os("HOME")?).join(".zzz/models/Phonon-7e71a02d-200.gguf");
    if !model.is_file() {
        assert_ne!(
            std::env::var("ZZZ_REQUIRE_FIXTURES").as_deref(),
            Ok("1"),
            "required facade Phonon fixture missing"
        );
        eprintln!("skipping Phonon facade test: model fixture unavailable");
        return None;
    }
    let directory = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(model, directory.path().join("model.gguf")).unwrap();
    let metadata = serde_json::json!({
        "model_id":"phonon-facade-test", "version":"1.0", "files":["model.gguf"],
        "execution_template":{"type":"ZzzPhonon","model_file":"model.gguf","temperature":0.8}
    });
    std::fs::write(
        directory.path().join("model_metadata.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
    let model = ModelLoader::from_directory(directory.path().to_string_lossy().into_owned())
        .unwrap()
        .load()
        .unwrap();
    Some((directory, model))
}

fn options() -> RunOptions {
    RunOptions {
        generation_config: Some(GenerationConfig {
            temperature: Some(0.3),
            ..GenerationConfig::default()
        }),
        ..RunOptions::default()
    }
}

#[test]
fn zzz_phonon_facade_delivers_owned_packets_then_one_terminal_result() {
    let Some((_directory, model)) = load() else {
        return;
    };
    let envelope = Envelope::text("Hello there.".into());
    let expected = model
        .run_with_options(envelope.clone(), options(), None)
        .unwrap()
        .audio_bytes()
        .unwrap()
        .to_vec();
    let stream = model.run_stream(envelope, options(), None).unwrap();
    let mut pcm = Vec::new();
    let mut terminals = 0;
    while let Some(event) = stream.next() {
        match event {
            StreamEvent::Audio(packet) => {
                assert_eq!(packet.first_sample, pcm.len() as u64 / 2);
                assert_eq!(packet.sample_rate, 24_000);
                assert_eq!(packet.channels, 1);
                pcm.extend(packet.pcm);
            }
            StreamEvent::Complete(result) => {
                terminals += 1;
                assert_eq!(result.envelope.metadata["tts_status"], "completed");
            }
            other => panic!("unexpected Phonon event: {other:?}"),
        }
    }
    assert_eq!(terminals, 1);
    assert_eq!(pcm, expected);
}

#[test]
fn zzz_phonon_facade_cancellation_and_disconnect_release_the_worker() {
    let Some((_directory, model)) = load() else {
        return;
    };
    let envelope = Envelope::text("Hello there. Please tell me your name and explain what brought you to the village today, before the gate closes for the night.".into());
    let cancel = CancellationToken::new();
    let stream = model
        .run_stream(envelope.clone(), options(), Some(cancel.clone()))
        .unwrap();
    assert!(matches!(stream.next(), Some(StreamEvent::Audio(_))));
    cancel.cancel();
    assert!(
        matches!(stream.next(), Some(StreamEvent::Complete(result)) if result.envelope.metadata["tts_status"] == "cancelled")
    );
    assert!(stream.next().is_none());
    let abandoned = model.run_stream(envelope, options(), None).unwrap();
    assert!(matches!(abandoned.next(), Some(StreamEvent::Audio(_))));
    drop(abandoned);
    let reused = model
        .run_with_options(Envelope::text("Hello there.".into()), options(), None)
        .unwrap();
    assert!(!reused.audio_bytes().unwrap().is_empty());
}
