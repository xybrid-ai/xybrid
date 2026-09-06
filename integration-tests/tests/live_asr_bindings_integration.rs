//! End-to-end live ASR coverage at the shared foreign-binding boundary.
//!
//! Run with:
//!   cargo test -p integration-tests --features asr-whispercpp \
//!     --test live_asr_bindings_integration -- --nocapture
//!
//! Download the model first:
//!   ./integration-tests/download.sh whisper-tiny-ggml

#![cfg(feature = "asr-whispercpp")]

use std::sync::Arc;

use integration_tests::fixtures;
use xybrid_core::audio::decode_wav_audio;
use xybrid_ffi_facade::{AsrStreamConfig, AsrStreamingSession, ModelLoader};

const MODEL: &str = "whisper-tiny-ggml";
const SAMPLE_RATE: u32 = 16_000;
const FEED_CHUNK_SAMPLES: usize = SAMPLE_RATE as usize;

#[test]
fn facade_session_streams_real_whisper_and_reuses_the_loaded_model() {
    let Some(model_dir) = fixtures::model_for_test(MODEL) else {
        eprintln!("Skipping: {MODEL} not downloaded. Run: ./integration-tests/download.sh {MODEL}");
        return;
    };

    let loader = ModelLoader::from_directory(model_dir.to_string_lossy().into_owned())
        .expect("the Whisper fixture directory should be valid");
    let model = loader.load().expect("the Whisper fixture should load");
    let session = model
        .stream(AsrStreamConfig {
            language: Some("en".into()),
            ..AsrStreamConfig::default()
        })
        .expect("the facade should open a live ASR worker");

    let partial_session = Arc::clone(&session);
    let partials = std::thread::spawn(move || collect_partials(partial_session));

    let wav_path = fixtures::input_dir().join("jfk.wav");
    let wav = std::fs::read(&wav_path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", wav_path.display()));
    let pcm = decode_wav_audio(&wav, SAMPLE_RATE, 1)
        .unwrap_or_else(|error| panic!("decoding {}: {error}", wav_path.display()));

    for chunk in pcm.chunks(FEED_CHUNK_SAMPLES) {
        session
            .feed(chunk.to_vec())
            .expect("one second of microphone PCM should queue");
    }

    let final_result = session
        .flush()
        .expect("flush should drain every queued PCM chunk");
    assert!(
        !final_result.text.trim().is_empty(),
        "real Whisper should produce a final transcript"
    );
    assert!(final_result.duration_ms >= 5_000);
    assert!(final_result.chunks_processed > 0);

    // The foreign session is intentionally reusable: flush completes one
    // utterance, reset clears it, and neither operation reloads the model.
    session
        .reset()
        .expect("reset after flush should keep the worker alive");
    session.close().expect("close should terminate the worker");

    let partials = partials
        .join()
        .expect("the partial-result collector should not panic");
    assert!(!partials.is_empty(), "real Whisper should emit partials");
    assert!(
        partials
            .windows(2)
            .all(|pair| pair[0].chunk_index < pair[1].chunk_index),
        "the facade must expose distinct, ordered partials"
    );

    println!("partials: {}", partials.len());
    println!("final: {:?}", final_result.text);
}

fn collect_partials(session: Arc<AsrStreamingSession>) -> Vec<xybrid_ffi_facade::AsrPartialResult> {
    let mut partials = Vec::new();
    while let Some(partial) = session
        .next()
        .expect("the real ASR worker should not return an inference error")
    {
        println!("partial {}: {:?}", partial.chunk_index, partial.text);
        partials.push(partial);
    }
    partials
}
