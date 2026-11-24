//! Voice Assistant Demo — real ASR → LLM → TTS with rich spans
//!
//! Each stage runs through `ModelLoader` and the SDK's observed run path.
//! Explicit observation parents connect the ASR, LLM and TTS inference spans
//! under one workflow. Adapter timing and token counts are recorded on the
//! corresponding spans automatically.
//!
//! ## Run
//!
//! ```bash
//! cd repos/xybrid
//!
//! # One-time: warm the three extracted bundles
//! ./target/release/xybrid -q run -m wav2vec2-base-960h \
//!     --input-audio integration-tests/fixtures/input/jfk.wav
//! ./target/release/xybrid -q run -m qwen2.5-0.5b-instruct \
//!     --input-text hi >/dev/null
//! ./target/release/xybrid -q run -m kokoro-82m \
//!     --input-text hi -o /tmp/warm.wav
//!
//! OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
//!   cargo run --release --example voice_assistant_demo \
//!     -p xybrid-sdk --features platform-macos -- \
//!     integration-tests/fixtures/input/jfk.wav \
//!     /tmp/voice-assistant-reply.wav
//! ```
//!
//! Set `XYBRID_API_KEY` for Xybrid or `OTEL_EXPORTER_OTLP_ENDPOINT` for a
//! Collector. With neither set, inference stays local without exporting traces.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::observability::{
    self, KeyValue, ObservationContext, ObservationOptions, StartOptions,
};
use xybrid_sdk::{ModelLoader, RunOptions};

#[path = "support/observability.rs"]
mod observability_support;

const DEFAULT_INPUT: &str = "integration-tests/fixtures/input/jfk.wav";
const DEFAULT_OUTPUT: &str = "/tmp/voice-assistant-reply.wav";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input_wav = PathBuf::from(args.get(1).map(String::as_str).unwrap_or(DEFAULT_INPUT));
    let output_wav = PathBuf::from(args.get(2).map(String::as_str).unwrap_or(DEFAULT_OUTPUT));

    println!("═══════════════════════════════════════════════════════");
    println!("  Voice Assistant Demo — real ASR → LLM → TTS");
    println!("═══════════════════════════════════════════════════════");
    println!("Input  : {}", input_wav.display());
    println!("Output : {}", output_wav.display());
    println!();

    let export = observability_support::configure("voice-assistant-demo")?;
    let workflow =
        observability::observer().start_workflow("voice-assistant-demo", StartOptions::default());

    let extracted_root = extracted_cache_root()?;
    let asr_dir = stage_dir(&extracted_root, "wav2vec2-base-960h")?;
    let llm_dir = stage_dir(&extracted_root, "qwen2.5-0.5b-instruct")?;
    let tts_dir = stage_dir(&extracted_root, "kokoro-82m")?;

    let run_start = std::time::Instant::now();

    // ────────────── 1. ASR ──────────────
    let asr_audio = std::fs::read(&input_wav)?;
    let asr_input = Envelope {
        kind: EnvelopeKind::Audio(asr_audio),
        metadata: HashMap::new(),
    };
    let asr_output = run_stage("asr", &asr_dir, &asr_input, workflow.context())?;
    let transcript = match &asr_output.kind {
        EnvelopeKind::Text(t) => t.clone(),
        other => return Err(format!("ASR returned unexpected kind: {other:?}").into()),
    };
    println!("🎙  ASR → {transcript}");

    // ────────────── 2. LLM ──────────────
    let mut llm_meta = HashMap::new();
    llm_meta.insert(
        "system_prompt".to_string(),
        "You are the Xybrid voice assistant. Reply in one or two short \
         sentences — the answer is read back through TTS, so keep it \
         concise and natural."
            .to_string(),
    );
    llm_meta.insert("max_tokens".to_string(), "80".to_string());
    llm_meta.insert("temperature".to_string(), "0.7".to_string());
    let llm_input = Envelope {
        kind: EnvelopeKind::Text(transcript),
        metadata: llm_meta,
    };
    let llm_output = run_stage("llm", &llm_dir, &llm_input, workflow.context())?;
    let reply = match &llm_output.kind {
        EnvelopeKind::Text(t) => t.clone(),
        other => return Err(format!("LLM returned unexpected kind: {other:?}").into()),
    };
    println!("🤖 LLM → {reply}");

    // ────────────── 3. TTS ──────────────
    let mut tts_meta = HashMap::new();
    tts_meta.insert("voice".to_string(), "af_bella".to_string());
    let tts_input = Envelope {
        kind: EnvelopeKind::Text(reply.clone()),
        metadata: tts_meta,
    };
    let tts_output = run_stage("tts", &tts_dir, &tts_input, workflow.context())?;
    match &tts_output.kind {
        EnvelopeKind::Audio(bytes) => {
            std::fs::write(&output_wav, bytes)?;
            println!(
                "🔊 TTS → {} bytes of audio saved to {}",
                bytes.len(),
                output_wav.display()
            );
        }
        other => return Err(format!("TTS returned unexpected kind: {other:?}").into()),
    };

    let total_latency_ms = run_start.elapsed().as_millis() as u64;
    println!();
    println!("✅ Pipeline done in {total_latency_ms} ms");

    let reference = workflow.observation_ref();
    workflow.end();
    if let Some(reference) = reference {
        println!("ASR → LLM → TTS trace: {}", reference.trace_id);
    }
    export.finish()?;

    Ok(())
}

fn run_stage(
    id: &str,
    model_dir: &Path,
    input: &Envelope,
    parent: Option<ObservationContext>,
) -> Result<Envelope, Box<dyn std::error::Error>> {
    let model = ModelLoader::from_directory(model_dir)?.load()?;
    let observation = ObservationOptions {
        parent,
        attributes: vec![KeyValue::new("demo.stage", id.to_owned())],
        ..ObservationOptions::default()
    };
    let result =
        model.run_with_options(input, &RunOptions::default().with_observation(observation))?;
    Ok(result.envelope().clone())
}

fn extracted_cache_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let home =
        std::env::var_os("HOME").ok_or("HOME environment variable must be set for cache lookup")?;
    Ok(PathBuf::from(home)
        .join(".xybrid")
        .join("cache")
        .join("extracted"))
}

fn stage_dir(root: &Path, model_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dir = root.join(model_id);
    if !dir.join("model_metadata.json").exists() {
        return Err(format!(
            "{} is not extracted at {} — run `xybrid -q run -m {} …` once to warm the cache.",
            model_id,
            dir.display(),
            model_id
        )
        .into());
    }
    Ok(dir)
}
