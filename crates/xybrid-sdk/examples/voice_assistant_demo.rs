//! Voice Assistant Demo — real ASR → LLM → TTS with rich spans
//!
//! The SDK's high-level `PipelineRef::from_yaml().load().run()` path
//! currently falls back to mock stage outputs when the orchestrator
//! can't wire adapters to extracted bundles
//! (see `xybrid-core/src/executor.rs:557` — the `ModelNotLoaded`
//! fallback). Until that's fixed, the working way to get a real
//! 3-stage flamegraph is to drive `TemplateExecutor` directly for
//! each stage and wrap each call in a `xybrid_core::tracing` span.
//! The LLM adapter already emits `ttft_ms`, `tokens_generated`,
//! `decode_tps`, `prefill_tps`, `mean_itl_ms`, `p95_itl_ms` via
//! `tracing::add_metadata` (see `xybrid-core/src/runtime_adapter/
//! llm.rs:409-475`), so those land on the LLM span automatically.
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
//! XYBRID_API_KEY=sk_test_u9Y5WQeT6SfMmvdj2iwJ6dYdzehclUIs \
//! XYBRID_PLATFORM_URL=http://localhost:8081 \
//!   cargo run --release --example voice_assistant_demo \
//!     -p xybrid-sdk --features platform-macos -- \
//!     integration-tests/fixtures/input/jfk.wav \
//!     /tmp/voice-assistant-reply.wav
//! ```
//!
//! Leave `XYBRID_API_KEY` unset to run locally without telemetry.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::tracing as core_tracing;
use xybrid_sdk::{
    flush_platform_telemetry, init_platform_telemetry, publish_telemetry_event,
    set_telemetry_pipeline_context, shutdown_platform_telemetry, TelemetryConfig, TelemetryEvent,
};

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

    let tel_enabled = configure_telemetry();
    core_tracing::init_tracing(true);
    // Share one trace_id across every event this run emits (the ASR/LLM/TTS
    // ExecutionStarted+Completed pairs + the final PipelineComplete). Without
    // this the traces_list pipe falls back to grouping by the per-event `id`
    // so each run shows up as 7 separate rows.
    let pipeline_id = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, b"voice-assistant-demo");
    let trace_id = uuid::Uuid::new_v4();
    set_telemetry_pipeline_context(Some(pipeline_id), Some(trace_id));

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
    let asr_output = run_stage("asr", &asr_dir, &asr_input)?;
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
    let llm_output = run_stage("llm", &llm_dir, &llm_input)?;
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
    let tts_output = run_stage("tts", &tts_dir, &tts_input)?;
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

    // Dump the span tree we captured — this is literally what gets
    // POSTed to the platform as `stages` on the PipelineComplete event.
    // Look for `ttft_ms`, `decode_tps`, `prefill_tps`, `tokens_generated`
    // on the `llm:…` span: those are the fields the console flamegraph
    // renders as prefill/decode bars and token counters.
    let spans = core_tracing::get_stages_json();
    println!();
    println!("── spans (preview) ──────────────────────────────────────");
    println!("{}", serde_json::to_string_pretty(&spans)?);

    if tel_enabled {
        publish_telemetry_event(TelemetryEvent {
            event_type: "PipelineComplete".to_string(),
            stage_name: Some("voice-assistant-demo".to_string()),
            target: Some("local".to_string()),
            latency_ms: Some(total_latency_ms as u32),
            error: None,
            data: Some(
                serde_json::json!({
                    // `model_id` deliberately omitted — a pipeline has no
                    // single model. Per-stage models live on their spans
                    // and are what `top_models` / `model_performance`
                    // now aggregate.
                    "pipeline": "voice-assistant-demo",
                    "stages": ["asr", "llm", "tts"],
                    "target": "local",
                    "status": "success"
                })
                .to_string(),
            ),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis() as u64,
        });

        // Let the SDK's worker thread drain the channel into the exporter
        // buffer before we flush. Without this delay the main thread can
        // call `shutdown_platform_telemetry` (which resets tracing state)
        // before the worker has a chance to call `convert_to_platform_event`,
        // causing the `stages` JSON to be emitted as empty.
        std::thread::sleep(std::time::Duration::from_millis(500));

        println!("📡 Flushing platform telemetry …");
        flush_platform_telemetry();
        // Second sleep + flush catches any events that arrived between the
        // first flush and now.
        std::thread::sleep(std::time::Duration::from_millis(250));
        flush_platform_telemetry();
        shutdown_platform_telemetry();
        println!("   Open /traces to see the ASR → LLM → TTS flamegraph.");
    }

    core_tracing::reset_tracing();
    Ok(())
}

fn run_stage(
    id: &str,
    model_dir: &Path,
    input: &Envelope,
) -> Result<Envelope, Box<dyn std::error::Error>> {
    let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(
        model_dir.join("model_metadata.json"),
    )?)?;

    // Each stage gets a top-level span. The LLM adapter appends its own
    // metadata (ttft_ms, decode_tps, tokens_generated, …) to the
    // currently-active span, so they land under the llm id.
    let span_id = core_tracing::start_span(format!("execute:{}:{id}", metadata.model_id));

    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let start = std::time::Instant::now();
    let result = executor.execute(&metadata, input, None);
    let latency_ms = start.elapsed().as_millis() as u64;

    core_tracing::add_metadata("latency_ms", latency_ms.to_string());
    core_tracing::add_metadata("target", "local");
    core_tracing::add_metadata("stage_id", id);
    core_tracing::end_span_by_id(span_id);

    Ok(result?)
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

fn configure_telemetry() -> bool {
    let Ok(api_key) = std::env::var("XYBRID_API_KEY") else {
        println!(
            "ℹ  XYBRID_API_KEY not set — running without platform telemetry.\n   \
             Set XYBRID_API_KEY + XYBRID_PLATFORM_URL to ship the trace."
        );
        return false;
    };
    let endpoint = std::env::var("XYBRID_PLATFORM_URL")
        .unwrap_or_else(|_| "https://ingest.xybrid.dev".to_string());

    println!("📡 Platform ingest : {}", endpoint);

    let config = TelemetryConfig::new(&endpoint, &api_key)
        .with_app_version(env!("CARGO_PKG_VERSION"))
        .with_device_label("voice-assistant-demo")
        .with_batch_size(1)
        .with_flush_interval(1);

    init_platform_telemetry(config);
    true
}
