//! Real `ZzzEmbed` execution through the [`TemplateExecutor`].
//!
//! Skips loudly — like `tts_smoke` — without the local Kitten TTS 2 assets:
//! these tests exercise the pinned engine slice (`tts-zzz`) with real
//! bundle files, so they build a synthetic bundle on a tempdir whose four
//! asset entries symlink to the model store and the zzz reference fixtures.

#![cfg(feature = "tts-zzz")]

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

use xybrid_core::execution::{ExecutionTemplate, ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};

struct BundleAssets {
    language_model: PathBuf,
    decoder_model: PathBuf,
    language_voice: PathBuf,
    decoder_voice: PathBuf,
}

/// Resolve the four Kitten TTS 2 assets from the environment or the shared
/// model store plus the zzz reference fixtures; `None` skips the tests.
fn resolved_assets() -> Option<BundleAssets> {
    let home = env::var_os("HOME").map(PathBuf::from)?;
    let language_model = env::var_os("ZZZ_TEST_LANGUAGE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".zzz/models/kitten-tts-2-q2_0.gguf"));
    let decoder_model = env::var_os("ZZZ_TEST_DECODER_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".zzz/models/kitten-s3-meanflow-f32.gguf"));
    let zzz_root = env::var_os("ZZZ_TEST_ZZZ_ROOT").map(PathBuf::from)?;
    let language_voice = env::var_os("ZZZ_TEST_LANGUAGE_VOICE")
        .map(PathBuf::from)
        .or_else(|| Some(zzz_root.join("fixtures/reference/kitten-tts-2/bruno-short.json")))?;
    let decoder_voice = env::var_os("ZZZ_TEST_DECODER_VOICE")
        .map(PathBuf::from)
        .or_else(|| Some(zzz_root.join("fixtures/reference/kitten-s3/bruno-voice.json")))?;
    let bundle = BundleAssets {
        language_model,
        decoder_model,
        language_voice,
        decoder_voice,
    };
    let missing: Vec<&str> = [
        (&bundle.language_model, "language model"),
        (&bundle.decoder_model, "decoder model"),
        (&bundle.language_voice, "language voice"),
        (&bundle.decoder_voice, "decoder voice"),
    ]
    .into_iter()
    .filter(|(path, _)| !path.exists())
    .map(|(_, label)| label)
    .collect();
    if !missing.is_empty() {
        eprintln!("skipping real zzz TTS execution, missing {missing:?} locally");
        return None;
    }
    Some(bundle)
}

/// A synthetic bundle whose four asset entries are symlinks.
struct SyntheticBundle {
    _dir: tempfile::TempDir,
    metadata: ModelMetadata,
}

fn synthetic_bundle() -> Option<SyntheticBundle> {
    let assets = resolved_assets()?;
    let dir = tempfile::TempDir::new().expect("bundle tempdir");
    let alias = |target: &std::path::Path, name: &str| -> String {
        std::os::unix::fs::symlink(target, dir.path().join(name)).expect("symlink into bundle");
        name.to_string()
    };
    let metadata = ModelMetadata {
        model_id: "zzz-kitten-tts2-smoke".to_string(),
        version: "0.1.0".to_string(),
        execution_template: ExecutionTemplate::ZzzEmbed {
            model_file: alias(&assets.language_model, "kitten-tts-2.gguf"),
            decoder_file: alias(&assets.decoder_model, "kitten-s3-meanflow.gguf"),
            language_voice_file: alias(&assets.language_voice, "lm-voice.json"),
            decoder_voice_file: alias(&assets.decoder_voice, "s3-voice.json"),
            language: Some("en".into()),
            threads: 0,
            max_tokens: 0,
            seed: 0,
        },
        preprocessing: Vec::new(),
        postprocessing: Vec::new(),
        files: Vec::new(),
        vision_encoder: None,
        description: Some("pinned zzz Kitten TTS 2 slice".to_string()),
        metadata: HashMap::new(),
        voices: None,
        max_chunk_chars: None,
        trim_trailing_samples: None,
    };
    Some(SyntheticBundle {
        _dir: dir,
        metadata,
    })
}

#[test]
fn executor_routes_zzz_embed_to_non_silent_24k_audio() {
    // The engine's PoC accepts prepared English only, matching the bundle.
    let Some(bundle) = synthetic_bundle() else {
        return;
    };
    let Some(base_path) = bundle
        ._dir
        .path()
        .to_str()
        .map(std::string::ToString::to_string)
    else {
        panic!("temp bundle path is not valid UTF-8");
    };
    let mut executor = TemplateExecutor::new(&base_path);
    let output = executor
        .execute(
            &bundle.metadata,
            &Envelope::new(EnvelopeKind::Text(
                "Hello from the xybrid zzz engine.".to_string(),
            )),
            None,
        )
        .expect("zzz execution");
    let EnvelopeKind::Audio(wav) = output.kind else {
        panic!(
            "zzz execution must yield an audio envelope, got {:?}",
            output.kind_str()
        );
    };
    // WAV container: "RIFF" header, "WAVE" form, fmt chunk, 24 kHz.
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(
        u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
        24000
    );
    assert!(wav.len() > 8000, "utterance above trivial length");
    assert_eq!(
        output.metadata.get("sample_rate").map(String::as_str),
        Some("24000")
    );
    assert_eq!(
        output.metadata.get("channels").map(String::as_str),
        Some("1")
    );
}

#[test]
fn executor_rejects_non_text_input_for_zzz_embed() {
    let Some(bundle) = synthetic_bundle() else {
        return;
    };
    let Some(base_path) = bundle
        ._dir
        .path()
        .to_str()
        .map(std::string::ToString::to_string)
    else {
        panic!("temp bundle path is not valid UTF-8");
    };
    let mut executor = TemplateExecutor::new(&base_path);
    let error = executor
        .execute(
            &bundle.metadata,
            &Envelope::new(EnvelopeKind::Embedding(vec![0.0])),
            None,
        )
        .expect_err("embedding input cannot drive zzz TTS");
    assert!(
        error.to_string().to_lowercase().contains("text"),
        "unexpected error: {error}"
    );
}
