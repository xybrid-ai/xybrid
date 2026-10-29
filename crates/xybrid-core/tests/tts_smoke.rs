//! TTS model smoke tests — verify all TTS fixtures execute through the generic path.
//!
//! This test iterates over all model directories in integration-tests/fixtures/models/
//! (or `$XYBRID_TEST_MODELS`), identifies TTS models by their
//! `metadata.task == "text-to-speech"` field, and runs each through
//! `TemplateExecutor`: a short text several times, and a text split into
//! several chunks through the chunked batch and streaming paths. Every result
//! must be speech-length audio.
//!
//! Marked `#[ignore]` because it requires downloaded ONNX model files.
//!
//! ```bash
//! # Download TTS models first
//! ./integration-tests/download.sh kitten-tts-nano-0.2 kokoro-82m
//!
//! # Run smoke tests
//! cargo test -p xybrid-core --test tts_smoke -- --ignored
//!
//! # Or run every TTS model the CLI and SDK have already cached
//! XYBRID_TEST_MODELS=~/.xybrid/cache/extracted \
//!     cargo test -p xybrid-core --test tts_smoke -- --ignored
//! ```

use std::collections::HashMap;
use xybrid_core::execution::{ModelMetadata, PostprocessingStep, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

/// Short text, synthesized in one chunk.
const TEXT: &str = "Hello world";

/// Three short sentences. With the chunk limit lowered to [`CHUNK_CHARS`],
/// each is synthesized as its own chunk, which exercises the chunked batch and
/// streaming paths without the minute of audio a 350-character chunk takes.
const MULTI_CHUNK_TEXT: &str =
    "The morning report is read aloud. The farmers listen. Then the station is quiet.";

/// Chunk limit for [`MULTI_CHUNK_TEXT`]: longer than any one of its sentences,
/// shorter than any two together.
const CHUNK_CHARS: usize = 40;

/// Shortest audio a model may return for [`TEXT`], or for each chunk of
/// [`MULTI_CHUNK_TEXT`]. Speech runs about a second; a non-audio tensor encoded
/// by mistake is far shorter (KittenTTS's 15 per-phoneme durations came out as
/// 0.6 ms).
const MIN_AUDIO_SECS: f64 = 0.3;

/// Runs of [`TEXT`] per model. Output selection once depended on hash-map
/// order, which changes from run to run, so a single run could pass by luck.
const RUNS_PER_MODEL: usize = 3;

/// Discover all TTS model directories that have both model_metadata.json and
/// the ONNX model file present.
fn discover_tts_models() -> Vec<(String, std::path::PathBuf)> {
    let Some(models_dir) = model_fixtures::models_dir() else {
        return vec![];
    };

    let Ok(entries) = std::fs::read_dir(models_dir) else {
        return vec![];
    };

    let mut tts_models = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let metadata_path = path.join("model_metadata.json");
        if !metadata_path.exists() {
            continue;
        }

        // Parse metadata to check task type
        let Ok(content) = std::fs::read_to_string(&metadata_path) else {
            continue;
        };
        let Ok(metadata) = serde_json::from_str::<ModelMetadata>(&content) else {
            continue;
        };

        // Check if this is a TTS model
        let is_tts = metadata
            .metadata
            .get("task")
            .and_then(|v| v.as_str())
            .map(|t| t == "text-to-speech")
            .unwrap_or(false);

        if !is_tts {
            continue;
        }

        // Check if any model file from the files list exists (skip if not downloaded)
        let has_model_file = metadata.files.iter().any(|f| {
            let ext = f.rsplit('.').next().unwrap_or("");
            matches!(ext, "onnx" | "safetensors" | "gguf") && path.join(f).exists()
        });

        if !has_model_file {
            eprintln!(
                "Skipping {}: model file not downloaded",
                entry.file_name().to_string_lossy(),
            );
            continue;
        }

        let name = entry.file_name().to_string_lossy().to_string();
        tts_models.push((name, path));
    }

    tts_models
}

/// Sample rate of the model's `TTSAudioEncode` step, or 24 kHz when it has
/// none (the executor's default).
fn sample_rate(metadata: &ModelMetadata) -> u32 {
    metadata
        .postprocessing
        .iter()
        .find_map(|step| match step {
            PostprocessingStep::TTSAudioEncode { sample_rate, .. } => Some(*sample_rate),
            _ => None,
        })
        .unwrap_or(24_000)
}

fn text_input(text: &str) -> Envelope {
    Envelope {
        kind: EnvelopeKind::Text(text.to_string()),
        metadata: HashMap::new(),
    }
}

/// Seconds of 16-bit mono PCM.
fn pcm16_secs(bytes: usize, sample_rate: u32) -> f64 {
    bytes as f64 / 2.0 / f64::from(sample_rate)
}

/// `(bytes, seconds)` of the audio in `result`, or why there is none.
fn audio_len<E: std::fmt::Display>(
    result: Result<Envelope, E>,
    sample_rate: u32,
) -> Result<(usize, f64), String> {
    match result {
        Ok(Envelope {
            kind: EnvelopeKind::Audio(bytes),
            ..
        }) => Ok((bytes.len(), pcm16_secs(bytes.len(), sample_rate))),
        Ok(other) => Err(format!(
            "produced {:?} instead of Audio",
            std::mem::discriminant(&other.kind)
        )),
        Err(e) => Err(format!("execution failed: {}", e)),
    }
}

#[test]
#[ignore]
fn test_all_tts_models_produce_audio() {
    let tts_models = discover_tts_models();

    if tts_models.is_empty() {
        eprintln!("No TTS models found with ONNX files. Download with:");
        eprintln!("  ./integration-tests/download.sh kitten-tts-nano-0.2 kokoro-82m");
        return;
    }

    println!("Found {} TTS model(s) to test", tts_models.len());

    let mut passed = 0;
    let mut failed = Vec::new();

    for (name, model_dir) in &tts_models {
        println!("\n--- Testing: {} ---", name);

        // Load metadata
        let metadata_path = model_dir.join("model_metadata.json");
        let content = std::fs::read_to_string(&metadata_path)
            .unwrap_or_else(|e| panic!("Failed to read metadata for {}: {}", name, e));
        let metadata: ModelMetadata = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("Failed to parse metadata for {}: {}", name, e));

        println!("  model_id: {}", metadata.model_id);
        println!("  preprocessing: {} step(s)", metadata.preprocessing.len());
        println!(
            "  postprocessing: {} step(s)",
            metadata.postprocessing.len()
        );

        let sample_rate = sample_rate(&metadata);
        let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
        let mut failures = Vec::new();

        // Short text: one chunk through the batch path, several times.
        for run in 1..=RUNS_PER_MODEL {
            match audio_len(
                executor.execute(&metadata, &text_input(TEXT), None),
                sample_rate,
            ) {
                Ok((bytes, secs)) => {
                    println!("  output: Audio ({} bytes, {:.3} s)", bytes, secs);
                    if secs < MIN_AUDIO_SECS {
                        failures.push(format!(
                            "run {} produced {:.4} s of audio ({} bytes)",
                            run, secs, bytes
                        ));
                    }
                }
                Err(e) => failures.push(format!("run {} {}", run, e)),
            }
        }

        // Several chunks: every streamed chunk must be speech, and the chunked
        // batch result must hold about as much audio as the streamed chunks.
        let mut chunked_metadata = metadata.clone();
        chunked_metadata.max_chunk_chars = Some(CHUNK_CHARS);
        let mut chunk_secs = Vec::new();
        let streamed = executor.execute_tts_streaming(
            &chunked_metadata,
            &text_input(MULTI_CHUNK_TEXT),
            &mut |pcm: Vec<u8>, rate: u32| {
                chunk_secs.push(pcm16_secs(pcm.len(), rate));
                true
            },
        );
        println!(
            "  streamed: {} chunk(s), {:.3?} s",
            chunk_secs.len(),
            chunk_secs
        );
        if let Err(e) = streamed {
            failures.push(format!("streaming failed: {}", e));
        }
        if chunk_secs.len() < 2 {
            failures.push(format!(
                "text streamed as {} chunk(s), expected several",
                chunk_secs.len()
            ));
        }
        for (i, secs) in chunk_secs.iter().enumerate() {
            if *secs < MIN_AUDIO_SECS {
                failures.push(format!(
                    "streamed chunk {} has {:.4} s of audio",
                    i + 1,
                    secs
                ));
            }
        }
        let streamed_secs: f64 = chunk_secs.iter().sum();
        match audio_len(
            executor.execute(&chunked_metadata, &text_input(MULTI_CHUNK_TEXT), None),
            sample_rate,
        ) {
            Ok((bytes, secs)) => {
                println!("  chunked: Audio ({} bytes, {:.3} s)", bytes, secs);
                if secs < 0.8 * streamed_secs {
                    failures.push(format!(
                        "chunked run produced {:.3} s of audio, streaming {:.3} s",
                        secs, streamed_secs
                    ));
                }
            }
            Err(e) => failures.push(format!("chunked run {}", e)),
        }

        for failure in &failures {
            eprintln!("  FAIL: {}", failure);
        }
        if failures.is_empty() {
            passed += 1;
        }
        failed.extend(
            failures
                .into_iter()
                .map(|failure| format!("Model {}: {}", name, failure)),
        );
    }

    println!("\n=== Results: {}/{} passed ===", passed, tts_models.len());

    if !failed.is_empty() {
        panic!(
            "TTS smoke test failures:\n{}",
            failed
                .iter()
                .map(|f| format!("  - {}", f))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
