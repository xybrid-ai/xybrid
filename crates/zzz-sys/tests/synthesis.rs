//! Real-engine tests linking the staged slice. They exercise the ABI/struct
//! layout and a real synthesis when local model assets are available.
//!
//! Scheduled like other engine tests: runs under the `bindings` feature
//! (which links the pinned slice) and skips cleanly — with a loud note —
//! when the optional local weights are absent, so the plain workspace test
//! suite never fails on machines without staged engine models.

#![cfg(feature = "bindings")]

use std::env;
use std::path::PathBuf;

use xybrid_zzz_sys::{
    abi_version, default_options, supports_model, KittenSession, KittenSettings,
    SYNTHESIS_CHANNELS, SYNTHESIS_SAMPLE_RATE, ZZZ_EMBED_ABI_VERSION, ZZZ_EMBED_INVALID_ARGUMENT,
    ZZZ_EMBED_MODEL_NOT_COMPILED, ZZZ_EMBED_OK,
};

/// Resolve the four Kitten TTS 2 assets from the environment or the local
/// default paths; `None` data skips the real-model tests loudly.
fn kitten_assets() -> Option<[(PathBuf, &'static str); 4]> {
    let env_or =
        |var: &str, default: PathBuf| env::var_os(var).map(PathBuf::from).or(Some(default));
    let home = env::var_os("HOME").map(PathBuf::from)?;
    let language_model = env_or(
        "ZZZ_TEST_LANGUAGE_MODEL",
        home.join(".zzz/models/kitten-tts-2-q2_0.gguf"),
    )?;
    let decoder_model = env_or(
        "ZZZ_TEST_DECODER_MODEL",
        home.join(".zzz/models/kitten-s3-meanflow-f32.gguf"),
    )?;
    let zzz_root = env::var_os("ZZZ_TEST_ZZZ_ROOT").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        PathBuf::from,
    );
    let language_voice = env_or(
        "ZZZ_TEST_LANGUAGE_VOICE",
        zzz_root.join("fixtures/reference/kitten-tts-2/bruno-short.json"),
    )?;
    let decoder_voice = env_or(
        "ZZZ_TEST_DECODER_VOICE",
        zzz_root.join("fixtures/reference/kitten-s3/bruno-voice.json"),
    )?;
    // The zzz repository's fixtures symlink into the shared model store;
    // resolve only at test time and skip with a note when missing.
    let missing: Vec<&str> = [
        (&language_model, "language model"),
        (&decoder_model, "decoder model"),
        (&language_voice, "language voice"),
        (&decoder_voice, "decoder voice"),
    ]
    .into_iter()
    .filter(|(path, _)| !path.exists())
    .map(|(_, label)| label)
    .collect();
    if !missing.is_empty() {
        eprintln!("skipping real Kitten synthesis, missing {missing:?} locally");
        return None;
    }
    Some([
        (language_model, "language model"),
        (decoder_model, "decoder model"),
        (language_voice, "language voice"),
        (decoder_voice, "decoder voice"),
    ])
}

fn open_session() -> Option<KittenSession> {
    let assets = kitten_assets()?;
    let [language_model, decoder_model, language_voice, decoder_voice] = assets;
    let session = KittenSession::open_kitten(
        &language_model.0,
        &decoder_model.0,
        &language_voice.0,
        &decoder_voice.0,
        KittenSettings {
            threads: 6,
            ..KittenSettings::default()
        },
    );
    Some(session.expect("opening the verified engine with local assets"))
}

#[test]
fn abi_and_capability_match_the_pinned_manifest() {
    assert_eq!(abi_version(), ZZZ_EMBED_ABI_VERSION, "engine ABI drift");
    assert!(supports_model("kitten-tts2"));
    assert!(!supports_model("another-engine"));
}

#[test]
fn engine_defaults_fill_positive_and_finite_fields() {
    let options = default_options();
    assert_eq!(
        options.struct_size,
        size_of::<xybrid_zzz_sys::EmbedOptions>() as u32
    );
    assert_eq!(options.max_frames_per_chunk, 512);
    assert_eq!(options.max_chunk_tokens, 256);
    assert!(options.temperature > 0.0 && options.temperature.is_finite());
}

#[test]
fn collected_synthesis_yields_nonsilent_mono_audio() {
    let Some(mut session) = open_session() else {
        return;
    };
    let outcome = session
        .synthesize("Your name sounds like trouble.", None)
        .expect("synthesis of a short utterance");
    assert_eq!(outcome.result.sample_rate, SYNTHESIS_SAMPLE_RATE);
    assert_eq!(outcome.result.channels, SYNTHESIS_CHANNELS);
    assert_eq!(outcome.samples.len(), outcome.result.samples as usize);
    assert_eq!(
        outcome.result.limited_chunks, 0,
        "short text stays under the cap"
    );
    assert!(outcome
        .samples
        .iter()
        .any(|sample| sample.abs() > 1.0 / 128.0));
    assert!(outcome.samples.iter().all(|sample| sample.is_finite()));
}

#[test]
fn streaming_chunks_walk_sample_offsets_in_order() {
    let Some(mut session) = open_session() else {
        return;
    };
    let deliveries: std::sync::Mutex<Vec<(u64, usize)>> = std::sync::Mutex::new(Vec::new());
    let result = session
        .synthesize_stream(
            "The village gate stays closed until you tell me your true business.",
            None,
            &mut |chunk, info| {
                deliveries
                    .lock()
                    .unwrap()
                    .push((info.first_sample, chunk.len()));
                true
            },
        )
        .expect("streaming synthesis");
    let deliveries = deliveries.into_inner().unwrap();
    assert!(deliveries.len() > 1, "the engine chunks bounded deliveries");
    // One utterance's chunks carry continuous sample offsets.
    let mut expected = deliveries[0].0;
    for (first_sample, len) in &deliveries {
        assert_eq!(
            (first_sample, len),
            (&expected, len),
            "chunk offsets must walk the sample timeline without gaps"
        );
        expected += *len as u64;
    }
    assert_eq!(
        result.samples as usize,
        deliveries.iter().map(|(_, len)| len).sum::<usize>()
    );
}

#[test]
fn cancellation_stops_delivery_and_reports_cancelled() {
    let Some(mut session) = open_session() else {
        return;
    };
    let long_text = "Tell me your name. Then explain your business. The village gate stays closed until I know who you are and what you want here today.".to_string();
    let deliveries = std::cell::Cell::new(0);
    let error = session
        .synthesize_stream(&long_text, None, &mut |_chunk, _info| {
            deliveries.set(deliveries.get() + 1);
            // Cancel within the first two delivered chunks.
            deliveries.get() < 2
        })
        .expect_err("cancellation must surface as an error");
    assert!(matches!(error, xybrid_zzz_sys::ZzzError::Cancelled { .. }));
    assert!(deliveries.get() >= 1 && deliveries.get() <= 2);
    // The session remains reusable after the cancelled call.
    let again = session
        .synthesize("Your name sounds like trouble.", None)
        .expect("resident reuse must succeed");
    assert!(!again.samples.is_empty());
}

#[test]
fn bad_asset_paths_fail_with_typed_errors() {
    let missing = PathBuf::from("definitely/not/a/model.gguf");
    let error = KittenSession::open_kitten(
        &missing,
        &missing,
        &missing,
        &missing,
        KittenSettings::default(),
    )
    .expect_err("missing model assets must fail");
    assert!(
        matches!(
            error,
            xybrid_zzz_sys::ZzzError::LoadFailed { .. }
                | xybrid_zzz_sys::ZzzError::InvalidArgument { .. }
        ),
        "unexpected: {error:?}"
    );
    // And the engine's error storage stays empty-safe; no message leak of
    // private receipt or API values in a type that must not carry any:
    let storage = [xybrid_zzz_sys::EmbedError::zeroed(); 2];
    assert_eq!(storage[0].code, ZZZ_EMBED_OK);
    assert!(storage[0].is_ok());
    let bad = xybrid_zzz_sys::EmbedError {
        code: ZZZ_EMBED_INVALID_ARGUMENT,
        message: [0; 192],
    };
    assert_eq!(bad.code, ZZZ_EMBED_INVALID_ARGUMENT);
    assert!(!zzz_code_matches(
        &[ZZZ_EMBED_MODEL_NOT_COMPILED],
        ZZZ_EMBED_INVALID_ARGUMENT
    ));
}

/// Helper asserted above without importing every code constant publicly.
fn zzz_code_matches(codes: &[i32], probe: i32) -> bool {
    codes.contains(&probe)
}

#[test]
fn final_callback_cancel_and_idle_cancel_preserve_resident_session() {
    let Some(mut session) = open_session() else {
        return;
    };
    let cancel = session.cancellation_handle();
    cancel.cancel(); // native ignores idle requests
    let baseline = session
        .synthesize("Your name sounds like trouble.", None)
        .expect("idle cancellation ignored");
    let total = baseline.result.samples;
    let outcome = session
        .synthesize_stream_outcome(
            "Your name sounds like trouble.",
            None,
            &mut |samples, info| {
                if info.first_sample + samples.len() as u64 == total {
                    cancel.cancel();
                }
                true
            },
        )
        .expect("native returned an outcome");
    assert_eq!(outcome.status, xybrid_zzz_sys::SynthesisStatus::Cancelled);
    assert!(!session
        .synthesize("Your name sounds like trouble.", None)
        .expect("reuse after final cancellation")
        .samples
        .is_empty());
    // The cancellation handle owns the lifetime even after the synthesis owner drops.
    drop(session);
    cancel.cancel();
    drop(cancel);
}

#[test]
fn explicit_token_limit_retains_native_partial_counters() {
    let Some(assets) = kitten_assets() else {
        return;
    };
    let mut session = KittenSession::open_kitten(
        &assets[0].0,
        &assets[1].0,
        &assets[2].0,
        &assets[3].0,
        KittenSettings {
            max_tokens: 8,
            ..KittenSettings::default()
        },
    )
    .expect("open capped session");
    let mut count = 0;
    let outcome = session
        .synthesize_stream_outcome(
            "This is a much longer sentence than eight speech tokens can express.",
            None,
            &mut |samples, _| {
                count += samples.len();
                true
            },
        )
        .expect("limited outcome");
    assert_eq!(outcome.status, xybrid_zzz_sys::SynthesisStatus::Limited);
    assert!(outcome.result.limited_chunks > 0);
    assert!(count > 0);
    assert_eq!(count as u64, outcome.result.samples);
}

#[test]
fn callback_panic_is_contained_and_reported_as_failure() {
    let Some(mut session) = open_session() else {
        return;
    };
    let error = session
        .synthesize_stream("Your name sounds like trouble.", None, &mut |_, _| {
            panic!("test callback panic")
        })
        .expect_err("panic becomes failure, not successful cancellation");
    assert!(matches!(
        error,
        xybrid_zzz_sys::ZzzError::SynthesisFailed { .. }
    ));
}
