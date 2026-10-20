//! Phonon ownership, defaults and callbacks against the pinned native archive.
#![cfg(feature = "bindings")]

use std::path::PathBuf;
use xybrid_zzz_sys::{default_options, PhononSession, PhononSettings, SynthesisStatus, ZzzError};

fn model(file: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("HOME")?)
        .join(".zzz/models")
        .join(file);
    if path.is_file() {
        return Some(path);
    }
    assert_ne!(
        std::env::var("ZZZ_REQUIRE_FIXTURES").as_deref(),
        Ok("1"),
        "required Phonon fixture missing"
    );
    eprintln!("skipping Phonon native test: model fixture unavailable");
    None
}

fn open(file: &str, voice: Option<&str>, settings: PhononSettings) -> Option<PhononSession> {
    Some(PhononSession::open_phonon(&model(file)?, voice, settings).expect("open Phonon"))
}

const CURRENT: &str = "Phonon-7e71a02d-200.gguf";

#[test]
fn defaults_come_from_engine_and_zero_seed_is_valid() {
    let native = default_options();
    let settings = PhononSettings::default();
    assert_eq!(settings.max_frames_per_chunk, native.max_frames_per_chunk);
    assert_eq!(settings.max_chunk_tokens, native.max_chunk_tokens);
    assert_eq!(settings.temperature, 0.0);
    assert_eq!(settings.seed, native.seed);
    PhononSettings {
        seed: 0,
        ..settings
    }
    .validate()
    .unwrap();
}

#[test]
fn invalid_settings_and_nul_are_rejected_before_open() {
    let defaults = PhononSettings::default();
    for settings in [
        PhononSettings {
            max_frames_per_chunk: 0,
            ..defaults
        },
        PhononSettings {
            max_chunk_tokens: 0,
            ..defaults
        },
        PhononSettings {
            temperature: -0.1,
            ..defaults
        },
        PhononSettings {
            temperature: f32::NAN,
            ..defaults
        },
        PhononSettings {
            temperature: f32::INFINITY,
            ..defaults
        },
    ] {
        assert!(matches!(
            PhononSession::open_phonon(std::path::Path::new("missing.gguf"), None, settings),
            Err(ZzzError::InvalidArgument { .. })
        ));
    }
    assert!(matches!(
        PhononSession::open_phonon(std::path::Path::new("missing\0.gguf"), None, defaults),
        Err(ZzzError::InvalidArgument { .. })
    ));
    assert!(matches!(
        PhononSession::open_phonon(
            std::path::Path::new("missing.gguf"),
            Some("bad\0voice"),
            defaults
        ),
        Err(ZzzError::InvalidArgument { .. })
    ));
}

#[test]
fn both_generations_use_bundle_defaults_and_reset_seed_on_reuse() {
    for (file, voice, temperature) in [
        (CURRENT, "freya", 0.3),
        ("Phonon-202605-beta1.gguf", "marlowe", 0.4),
    ] {
        let Some(mut session) = open(file, None, PhononSettings::default()) else {
            continue;
        };
        let first = session.synthesize("Hello there.", None).unwrap();
        assert_eq!(first.result.sample_rate, 24_000);
        assert_eq!(first.result.channels, 1);
        assert!(first.samples.len() > 2400);
        assert!(first.samples.iter().all(|value| value.is_finite()));
        assert!(first.samples.iter().any(|value| value.abs() > 0.001));
        let second = session.synthesize("Hello there.", Some("en")).unwrap();
        assert_eq!(first.samples, second.samples);
        drop(session);
        let mut explicit = open(
            file,
            Some(voice),
            PhononSettings {
                temperature,
                ..PhononSettings::default()
            },
        )
        .unwrap();
        assert_eq!(
            first.samples,
            explicit.synthesize("Hello there.", None).unwrap().samples
        );
    }
}

#[test]
fn callback_panic_cancellation_and_input_errors_leave_session_reusable() {
    let Some(mut session) = open(CURRENT, None, PhononSettings::default()) else {
        return;
    };
    let panic = session
        .synthesize_stream_outcome("Hello there.", None, &mut |_, _| panic!("test callback"));
    assert!(matches!(panic, Err(ZzzError::SynthesisFailed { .. })));
    let mut end = 0;
    let result = session
        .synthesize_stream_outcome("Hello there.", None, &mut |samples, info| {
            assert_eq!(info.first_sample, end);
            end += samples.len() as u64;
            false
        })
        .unwrap();
    assert_eq!(result.status, SynthesisStatus::Cancelled);
    assert!(end > 0);
    assert!(session.synthesize("bad\0text", None).is_err());
    assert!(session
        .synthesize("Hello there.", Some("unsupported"))
        .is_err());
    assert!(!session
        .synthesize("Hello there.", None)
        .unwrap()
        .samples
        .is_empty());
}

#[test]
fn concurrent_cancellation_is_lifetime_safe_and_not_carried_into_next_run() {
    let Some(mut session) = open(CURRENT, None, PhononSettings::default()) else {
        return;
    };
    let mut cancel = Some(session.cancellation_handle());
    let result = session
        .synthesize_stream_outcome("Hello there.", None, &mut |_, _| {
            if let Some(handle) = cancel.take() {
                std::thread::spawn(move || handle.cancel()).join().unwrap();
            }
            true
        })
        .unwrap();
    assert_eq!(result.status, SynthesisStatus::Cancelled);
    session.cancellation_handle().cancel(); // idle cancellation is ignored
    assert!(!session
        .synthesize("Hello there.", None)
        .unwrap()
        .samples
        .is_empty());
    let alive = session.cancellation_handle();
    drop(session);
    alive.cancel(); // the final handle still anchors native close
}

#[test]
fn frame_limits_preserve_partial_audio_and_reuse() {
    let Some(mut session) = open(
        CURRENT,
        None,
        PhononSettings {
            max_frames_per_chunk: 1,
            ..PhononSettings::default()
        },
    ) else {
        return;
    };
    for _ in 0..2 {
        let mut samples = 0;
        let outcome = session
            .synthesize_stream_outcome("Hello there.", None, &mut |audio, _| {
                samples += audio.len();
                true
            })
            .unwrap();
        assert_eq!(outcome.status, SynthesisStatus::Limited);
        assert!(samples > 0);
        assert!(outcome.result.limited_chunks > 0);
    }
}

#[test]
fn oversized_input_reports_a_limit_without_audio_then_reuses_the_session() {
    let Some(mut session) = open(CURRENT, None, PhononSettings::default()) else {
        return;
    };
    let text = "word ".repeat(14_000);
    let outcome = session
        .synthesize_stream_outcome(&text, None, &mut |_, _| {
            panic!("oversized text must not deliver audio")
        })
        .unwrap();
    assert_eq!(outcome.status, SynthesisStatus::Limited);
    assert_eq!(outcome.result.samples, 0);
    assert!(!session
        .synthesize("Hello there.", None)
        .unwrap()
        .samples
        .is_empty());
}
