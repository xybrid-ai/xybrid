//! Real Phonon speech through public batch and audio-streaming APIs.
#![cfg(all(feature = "tts-zzz", unix))]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use xybrid_core::device::{MemoryPressure, ResourceSnapshot, ResourceSnapshotProvider};
use xybrid_core::execution::{TtsStatus, TtsStreamResult};
use xybrid_sdk::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::{
    AbortPolicy, AbortSignal, CancellationToken, GenerationConfig, ModelLoader, RunOptions,
    SdkError, XybridModel,
};

const CURRENT: &str = "Phonon-7e71a02d-200.gguf";

fn load_model(file: &str, overrides: Value) -> Option<(tempfile::TempDir, XybridModel)> {
    let fixture = PathBuf::from(std::env::var_os("HOME")?)
        .join(".zzz/models")
        .join(file);
    if !fixture.is_file() {
        assert_ne!(
            std::env::var("ZZZ_REQUIRE_FIXTURES").as_deref(),
            Ok("1"),
            "required SDK Phonon fixture missing"
        );
        eprintln!("skipping SDK Phonon test: model fixture unavailable");
        return None;
    }
    let directory = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(fixture, directory.path().join("model.gguf")).unwrap();
    let mut template = json!({"type":"ZzzPhonon", "model_file":"model.gguf"});
    template
        .as_object_mut()
        .unwrap()
        .extend(overrides.as_object().unwrap().clone());
    let metadata = json!({
        "model_id":"phonon-sdk-test", "version":"1.0", "execution_template":template,
        "files":["model.gguf"], "metadata":{"task":"text-to-speech"}
    });
    std::fs::write(
        directory.path().join("model_metadata.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
    let model = ModelLoader::from_directory(directory.path())
        .unwrap()
        .load()
        .unwrap();
    Some((directory, model))
}

fn text() -> Envelope {
    Envelope::new(EnvelopeKind::Text("Hello there.".into()))
}

fn collect(
    model: &XybridModel,
    input: &Envelope,
    options: &RunOptions,
) -> (Vec<u8>, TtsStreamResult) {
    let mut pcm = Vec::new();
    let mut packets = 0;
    let summary = model
        .run_tts_streaming(input, options, |packet| {
            assert_eq!(packet.sample_rate, 24_000);
            assert_eq!(packet.channels, 1);
            assert_eq!(packet.first_sample, pcm.len() as u64 / 2);
            pcm.extend(packet.pcm);
            packets += 1;
            true
        })
        .unwrap();
    assert!(packets > 1, "native audio arrives incrementally");
    assert!(pcm
        .chunks_exact(2)
        .any(|sample| i16::from_le_bytes([sample[0], sample[1]]) != 0));
    assert_eq!(summary.samples, pcm.len() as u64 / 2);
    (pcm, summary)
}

#[test]
fn zzz_phonon_every_bundle_batch_stream_defaults_and_reuse() {
    for (file, voice, temperature) in [
        (CURRENT, "freya", 0.3),
        ("Phonon-202605-beta1.gguf", "marlowe", 0.4),
        ("Phonon-31m-en.gguf", "default", 0.3),
    ] {
        let Some((_directory, model)) = load_model(file, json!({})) else {
            continue;
        };
        let batch = model.run(&text(), None).unwrap();
        let expected = batch.audio_bytes().unwrap().to_vec();
        assert!(expected.len() > 4800);
        let (streamed, summary) = collect(&model, &text(), &RunOptions::default());
        assert_eq!(summary.status, TtsStatus::Completed);
        assert_eq!(expected, streamed);
        let controlled = model
            .run_with_options(&text(), &RunOptions::default())
            .unwrap();
        assert_eq!(expected, controlled.audio_bytes().unwrap());
        let mut explicit = text();
        explicit.metadata.insert("voice_id".into(), voice.into());
        explicit.metadata.insert("language".into(), "en".into());
        explicit
            .metadata
            .insert("temperature".into(), temperature.to_string());
        assert_eq!(
            expected,
            model.run(&explicit, None).unwrap().audio_bytes().unwrap()
        );
    }
}

#[test]
fn zzz_phonon_request_precedence_and_overrides_do_not_leak() {
    let Some((_directory, model)) = load_model(CURRENT, json!({"temperature":0.3,"seed":42}))
    else {
        return;
    };
    let baseline = model
        .run(&text(), None)
        .unwrap()
        .audio_bytes()
        .unwrap()
        .to_vec();
    let mut input = text();
    input.metadata.insert("temperature".into(), "0.8".into());
    input.metadata.insert("seed".into(), "0".into());
    let changed = model
        .run(&input, None)
        .unwrap()
        .audio_bytes()
        .unwrap()
        .to_vec();
    assert_ne!(baseline, changed);
    let mut speaker = text();
    speaker.metadata.insert("voice_id".into(), "harper".into());
    speaker.metadata.insert("language".into(), "fr".into());
    let spoken = model.run(&speaker, None).unwrap();
    assert_ne!(baseline, spoken.audio_bytes().unwrap());
    speaker
        .metadata
        .insert("voice_id".into(), "voice-not-in-gguf".into());
    assert!(model.run(&speaker, None).is_err());
    assert_eq!(
        baseline,
        model.run(&text(), None).unwrap().audio_bytes().unwrap()
    );
    let mut invalid_temperature = text();
    invalid_temperature
        .metadata
        .insert("temperature".into(), "invalid".into());
    let config = GenerationConfig {
        temperature: 0.3,
        ..GenerationConfig::default()
    };
    let options = RunOptions::new().with_generation_config(config.clone());
    assert_eq!(
        baseline,
        model
            .run(&invalid_temperature, Some(&config))
            .unwrap()
            .audio_bytes()
            .unwrap()
    );
    assert_eq!(baseline, collect(&model, &invalid_temperature, &options).0);
    assert!(model.run(&invalid_temperature, None).is_err());
    let mut bad_seed = text();
    bad_seed.metadata.insert("seed".into(), "invalid".into());
    assert!(model.run(&bad_seed, None).is_err());
    assert_eq!(
        baseline,
        model.run(&text(), None).unwrap().audio_bytes().unwrap()
    );
}

#[test]
fn zzz_phonon_cancellation_limits_and_reuse() {
    let Some((_directory, model)) = load_model(CURRENT, json!({})) else {
        return;
    };
    let token = CancellationToken::new();
    let options = RunOptions::new().with_cancellation_token(token.clone());
    token.cancel();
    let summary = model
        .run_tts_streaming(&text(), &options, |_| {
            panic!("pre-cancelled run delivered audio")
        })
        .unwrap();
    assert_eq!(summary.status, TtsStatus::Cancelled);
    assert_eq!(summary.samples, 0);
    let token = CancellationToken::new();
    let options = RunOptions::new().with_cancellation_token(token.clone());
    let summary = model
        .run_tts_streaming(&text(), &options, |_| {
            token.cancel();
            true
        })
        .unwrap();
    assert_eq!(summary.status, TtsStatus::Cancelled);
    assert!(summary.samples > 0);
    assert_eq!(
        collect(&model, &text(), &RunOptions::default()).1.status,
        TtsStatus::Completed
    );
    let summary = model
        .run_tts_streaming(&text(), &RunOptions::default(), |_| false)
        .unwrap();
    assert_eq!(summary.status, TtsStatus::Cancelled);
    let Some((_limited_directory, limited)) =
        load_model(CURRENT, json!({"max_frames_per_chunk":1}))
    else {
        return;
    };
    let result = limited
        .run_with_options(&text(), &RunOptions::default())
        .unwrap();
    assert_eq!(result.envelope().metadata["tts_status"], "limited");
    assert!(!result.audio_bytes().unwrap().is_empty());
}

#[test]
fn zzz_phonon_sync_and_async_share_request_controls() {
    let Some((_directory, model)) = load_model(CURRENT, json!({"temperature":0.8})) else {
        return;
    };
    let options = RunOptions::new().with_generation_config(GenerationConfig {
        temperature: 0.3,
        ..GenerationConfig::default()
    });
    let expected = collect(&model, &text(), &options).0;
    let pcm = Arc::new(Mutex::new(Vec::new()));
    let captured = pcm.clone();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let batch = model
            .run_async(&text(), options.generation_config.as_ref())
            .await
            .unwrap();
        assert_eq!(expected, batch.audio_bytes().unwrap());
        let summary = model
            .run_tts_streaming_async(&text(), &options, move |packet| {
                captured.lock().unwrap().extend(packet.pcm);
                true
            })
            .await
            .unwrap();
        assert_eq!(summary.status, TtsStatus::Completed);
    });
    assert_eq!(expected, *pcm.lock().unwrap());
}

#[derive(Debug, Default)]
struct Pressure(AtomicBool);

impl ResourceSnapshotProvider for Pressure {
    fn current_snapshot(&self, _: Duration) -> ResourceSnapshot {
        ResourceSnapshot {
            memory_pressure: if self.0.load(Ordering::Acquire) {
                MemoryPressure::Critical
            } else {
                MemoryPressure::Normal
            },
            ..ResourceSnapshot::unknown()
        }
    }
}

#[test]
fn zzz_phonon_resource_abort_preserves_fallback_and_session_reuse() {
    let Some((_directory, model)) = load_model(CURRENT, json!({})) else {
        return;
    };
    let input = Envelope::new(EnvelopeKind::Text(
        "Hello there. Please tell me your name and explain what brought you to the village today, before the gate closes for the night.".into(),
    ));
    for fallback in [false, true] {
        let pressure = Arc::new(Pressure::default());
        let options = RunOptions::new()
            .with_abort_policy(
                AbortPolicy::default()
                    .stop_on(AbortSignal::MemoryPressureCritical)
                    .with_cloud_fallback(fallback),
            )
            .with_resource_provider(pressure.clone());
        let mut packets = 0;
        let error = model
            .run_tts_streaming(&input, &options, |_| {
                packets += 1;
                pressure.0.store(true, Ordering::Release);
                // Cross the resource poll interval while native delivery is active.
                std::thread::sleep(Duration::from_millis(150));
                true
            })
            .unwrap_err();
        assert_eq!(packets, 1);
        if fallback {
            assert!(matches!(error, SdkError::AbortedForCloudFallback { .. }));
        } else {
            assert!(error.to_string().contains("memory_pressure_critical"));
        }
        assert_eq!(
            collect(&model, &text(), &RunOptions::default()).1.status,
            TtsStatus::Completed
        );
    }
}
