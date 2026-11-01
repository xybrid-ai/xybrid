//! TTS model smoke tests — verify all TTS fixtures execute through the generic path.
//!
//! This test iterates over all model directories in integration-tests/fixtures/models/
//! (or `$XYBRID_TEST_MODELS`), identifies TTS models by their
//! `metadata.task == "text-to-speech"` field, and runs each through
//! `TemplateExecutor`: a short text several times, and a text split into
//! several chunks through the chunked batch and streaming paths. Every result
//! must be speech-length audio.
//!
//! A model whose ONNX graph has a per-token duration output (`duration`,
//! `durations` or `pred_dur`) must also report phoneme timings on its batch
//! results: in order, within the audio, and ending at most an audible end pad
//! before it does. Any other model must report none.
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
use std::path::Path;
use xybrid_core::execution::{
    ExecutionTemplate, ModelMetadata, PostprocessingStep, TemplateExecutor,
};
use xybrid_core::ir::{Envelope, EnvelopeKind, PhonemeTiming};
use xybrid_core::runtime_adapter::{ExecutionProviderKind, ONNXSession, SessionOptions};
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

/// Output names the executor reads per-token durations from.
const DURATION_OUTPUT_NAMES: [&str; 3] = ["duration", "durations", "pred_dur"];

/// Longest a timed model's audio may run on after its last phoneme ends: the
/// frames of the end padding token, which timings never list, that the
/// encoder's -40 dB trim keeps. Kokoro's end pad is silent and trimmed away,
/// so its last phoneme ends with the audio. KittenTTS 0.8's is audible (the
/// release of the last sound) and kept: Nano runs on for about 50 ms, Mini
/// 78 ms and Micro up to 100 ms, four of its 25 ms frames. The bound allows
/// six.
const MAX_TAIL_SECS: f64 = 0.15;

/// Slack for comparing `f32` timings with the audio length in `f64` seconds.
const TIMING_EPSILON_SECS: f64 = 1e-4;

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
    result: &Result<Envelope, E>,
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

/// Whether the model's ONNX graph has a per-token duration output next to
/// its waveform, so its batch results must carry phoneme timings.
fn declares_duration_output(metadata: &ModelMetadata, model_dir: &Path) -> Result<bool, String> {
    let ExecutionTemplate::Onnx { model_file } = &metadata.execution_template else {
        return Ok(false);
    };
    let path = model_dir.join(model_file);
    let path = path
        .to_str()
        .ok_or_else(|| format!("model path {} is not UTF-8", path.display()))?;
    let session = ONNXSession::build(path, ExecutionProviderKind::Cpu, SessionOptions::default())
        .map_err(|e| format!("cannot open {}: {}", path, e))?;
    let names = session.output_names();
    Ok(names.len() > 1
        && names
            .iter()
            .any(|name| DURATION_OUTPUT_NAMES.contains(&name.as_str())))
}

/// The phoneme timings on `envelope`, checked against its `audio_secs` of
/// audio, or what is wrong with them.
///
/// With `timed`, they must be present, non-empty spans in order that never
/// overlap, lie within the audio, and end no more than [`MAX_TAIL_SECS`]
/// before it does. Without, the metadata key must be absent.
fn checked_timings(
    envelope: &Envelope,
    audio_secs: f64,
    timed: bool,
) -> Result<Vec<PhonemeTiming>, String> {
    let raw = envelope
        .metadata
        .get(Envelope::PHONEME_TIMINGS_METADATA_KEY);
    let raw = match (raw, timed) {
        (None, false) => return Ok(Vec::new()),
        (Some(_), false) => {
            return Err("reports phoneme timings without a duration output".to_string())
        }
        (None, true) => return Err("has a duration output but no phoneme timings".to_string()),
        (Some(raw), true) => raw,
    };
    let timings: Vec<PhonemeTiming> =
        serde_json::from_str(raw).map_err(|e| format!("unparseable phoneme timings: {}", e))?;
    let (Some(first), Some(last)) = (timings.first(), timings.last()) else {
        return Err("reports an empty list of phoneme timings".to_string());
    };
    if first.start < 0.0 {
        return Err(format!(
            "first phoneme starts before the audio: {:?}",
            first
        ));
    }
    if let Some(empty) = timings.iter().find(|t| t.start >= t.end) {
        return Err(format!("empty phoneme span {:?}", empty));
    }
    if let Some(pair) = timings.windows(2).find(|pair| pair[0].end > pair[1].start) {
        return Err(format!("phoneme spans overlap: {:?}", pair));
    }
    let last_end = f64::from(last.end);
    if last_end > audio_secs + TIMING_EPSILON_SECS {
        return Err(format!(
            "last phoneme ends at {:.4} s, after the {:.4} s of audio",
            last_end, audio_secs
        ));
    }
    if audio_secs - last_end > MAX_TAIL_SECS + TIMING_EPSILON_SECS {
        return Err(format!(
            "last phoneme ends at {:.4} s, {:.0} ms before the end of the {:.4} s of audio",
            last_end,
            (audio_secs - last_end) * 1000.0,
            audio_secs
        ));
    }
    Ok(timings)
}

/// Runs [`checked_timings`] on a batch `result` with `audio_secs` of audio,
/// printing a summary under `label` or recording what is wrong in `failures`.
fn check_timings<E>(
    label: &str,
    result: &Result<Envelope, E>,
    audio_secs: f64,
    timed: bool,
    failures: &mut Vec<String>,
) {
    let Ok(envelope) = result else {
        return; // Already reported as an execution failure.
    };
    match checked_timings(envelope, audio_secs, timed) {
        Ok(timings) if timings.is_empty() => {}
        Ok(timings) => {
            let sample: Vec<String> = timings
                .iter()
                .take(6)
                .map(|t| format!("{} {:.3}-{:.3}", t.phoneme, t.start, t.end))
                .collect();
            println!(
                "  {}: {} phoneme timing(s), last ends {:.4} s of {:.4} s [{} ...]",
                label,
                timings.len(),
                timings.last().map_or(0.0, |t| t.end),
                audio_secs,
                sample.join(", ")
            );
        }
        Err(e) => failures.push(format!("{} {}", label, e)),
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
        let timed = match declares_duration_output(&metadata, model_dir) {
            Ok(timed) => timed,
            Err(e) => {
                failures.push(e);
                false
            }
        };
        println!("  duration output: {}", timed);

        // Short text: one chunk through the batch path, several times.
        for run in 1..=RUNS_PER_MODEL {
            let result = executor.execute(&metadata, &text_input(TEXT), None);
            match audio_len(&result, sample_rate) {
                Ok((bytes, secs)) => {
                    println!("  output: Audio ({} bytes, {:.3} s)", bytes, secs);
                    if secs < MIN_AUDIO_SECS {
                        failures.push(format!(
                            "run {} produced {:.4} s of audio ({} bytes)",
                            run, secs, bytes
                        ));
                    }
                    check_timings(&format!("run {}", run), &result, secs, timed, &mut failures);
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
        let chunked = executor.execute(&chunked_metadata, &text_input(MULTI_CHUNK_TEXT), None);
        match audio_len(&chunked, sample_rate) {
            Ok((bytes, secs)) => {
                println!("  chunked: Audio ({} bytes, {:.3} s)", bytes, secs);
                if secs < 0.8 * streamed_secs {
                    failures.push(format!(
                        "chunked run produced {:.3} s of audio, streaming {:.3} s",
                        secs, streamed_secs
                    ));
                }
                // Timings stay in order across the crossfaded chunk joins.
                check_timings("chunked run", &chunked, secs, timed, &mut failures);
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
