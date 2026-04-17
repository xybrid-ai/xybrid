//! LLM streaming telemetry metrics — TTFT, TPS, ITL.
//!
//! Runs a local GGUF LLM through the standard `TemplateExecutor` path and
//! reports the streaming-derived telemetry fields (time-to-first-token,
//! mean / p95 inter-chunk latency, emitted chunk count) alongside the
//! existing aggregate metrics.
//!
//! Run with:
//!   cargo run --example llm_streaming_metrics -p xybrid-core --features llm-mistral --release
//!
//! To also export telemetry to a local ingest service, set:
//!   XYBRID_API_KEY=<your-api-key> \
//!   XYBRID_INGEST_URL=http://localhost:8081 \
//!   cargo run --example llm_streaming_metrics -p xybrid-core --features llm-mistral --release
//!
//! Reuses the `integration-tests/fixtures/models/qwen2.5-0.5b-instruct/`
//! fixture. See `integration-tests/download.sh` for download instructions
//! if the model isn't present locally.

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::template_executor::TemplateExecutor;

// For telemetry export and trace ID generation.
extern crate serde_json;
extern crate uuid;
extern crate xybrid_sdk;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  LLM Streaming Metrics — TTFT / TPS / ITL");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Initialize platform telemetry if XYBRID_API_KEY is set.
    // This sends telemetry events to the ingest service (local or prod)
    // so the dashboard can display the LLM metrics.
    let telemetry_enabled = xybrid_sdk::init_platform_telemetry_from_env();
    if telemetry_enabled {
        println!(
            "📡 Platform telemetry enabled (sending to {})",
            std::env::var("XYBRID_INGEST_URL").unwrap_or_else(|_| "default endpoint".into())
        );
        // Set a trace ID so the exporter can group the event.
        let trace_id = uuid::Uuid::new_v4();
        xybrid_sdk::set_telemetry_pipeline_context(None, Some(trace_id));
    } else {
        println!("ℹ️  Platform telemetry disabled (set XYBRID_API_KEY to enable)");
    }
    println!();

    let model_dir = PathBuf::from("integration-tests/fixtures/models/qwen2.5-0.5b-instruct");
    if !model_dir.exists() {
        eprintln!("❌ Model directory not found: {}", model_dir.display());
        eprintln!("   Run `integration-tests/download.sh qwen2.5-0.5b-instruct` first.");
        return Err("Model not found".into());
    }

    let metadata_path = model_dir.join("model_metadata.json");
    let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;

    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Force `max_tokens` high enough that multi-chunk output is the common
    // case — otherwise a too-short reply can legitimately arrive as a single
    // chunk, in which case ITL summaries are correctly `None`.
    let mut envelope_metadata = HashMap::new();
    envelope_metadata.insert(
        "system_prompt".to_string(),
        "You are a helpful assistant.".to_string(),
    );
    envelope_metadata.insert("max_tokens".to_string(), "64".to_string());
    envelope_metadata.insert("temperature".to_string(), "0.7".to_string());

    let prompt = "List five interesting facts about the city of Lisbon.";
    let input = Envelope {
        kind: EnvelopeKind::Text(prompt.to_string()),
        metadata: envelope_metadata,
    };

    println!("💬 Prompt: {}", prompt);
    println!("🔄 Running streaming inference (first run loads the model)...");
    println!();

    let output = executor.execute(&metadata, &input, None)?;

    let response_text = match &output.kind {
        EnvelopeKind::Text(t) => t.clone(),
        _ => return Err("expected text output from LLM".into()),
    };

    println!("═══════════════════════════════════════════════════════");
    println!("📤 Response:");
    println!("═══════════════════════════════════════════════════════");
    println!("{}", response_text);
    println!("═══════════════════════════════════════════════════════");
    println!();

    let m = &output.metadata;

    println!("📊 Aggregate metrics:");
    println!(
        "   tokens_generated   : {}",
        m.get("tokens_generated").map(|s| s.as_str()).unwrap_or("?")
    );
    println!(
        "   generation_time_ms : {}",
        m.get("generation_time_ms")
            .map(|s| s.as_str())
            .unwrap_or("?")
    );
    println!(
        "   tokens_per_second  : {}",
        m.get("tokens_per_second")
            .map(|s| s.as_str())
            .unwrap_or("?")
    );
    println!(
        "   finish_reason      : {}",
        m.get("finish_reason").map(|s| s.as_str()).unwrap_or("?")
    );
    println!();

    println!("📊 Streaming metrics:");
    let ttft = m.get("ttft_ms");
    let mean_itl = m.get("mean_itl_ms");
    let p95_itl = m.get("p95_itl_ms");
    let emitted_chunks: Option<u32> = m.get("emitted_chunks").and_then(|s| s.parse().ok());
    println!(
        "   ttft_ms            : {}",
        ttft.map(|s| s.as_str()).unwrap_or("(unset)")
    );
    println!(
        "   emitted_chunks     : {}",
        emitted_chunks
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(unset)".into())
    );
    println!(
        "   mean_itl_ms        : {}",
        mean_itl.map(|s| s.as_str()).unwrap_or("(none)")
    );
    println!(
        "   p95_itl_ms         : {}",
        p95_itl.map(|s| s.as_str()).unwrap_or("(none)")
    );
    println!();

    // Soft assertions (per plan): TTFT must be present whenever at least one
    // chunk arrived; ITL summaries are only required when multiple chunks
    // were emitted (a one-chunk reply is a legitimate degenerate case).
    assert!(
        ttft.is_some(),
        "TTFT must be measured whenever at least one chunk arrives"
    );
    if emitted_chunks.unwrap_or(0) > 1 {
        assert!(
            mean_itl.is_some(),
            "mean_itl_ms must be present when multiple chunks were emitted"
        );
        assert!(
            p95_itl.is_some(),
            "p95_itl_ms must be present when multiple chunks were emitted"
        );
    }

    println!("🎯 VALIDATION:");
    println!("   ✅ TTFT measured");
    if emitted_chunks.unwrap_or(0) > 1 {
        println!("   ✅ Mean / p95 ITL measured");
    } else {
        println!("   ℹ️  Single-chunk reply — ITL summaries legitimately absent");
    }
    println!();

    // Publish a ModelComplete event so the exporter converts it to a
    // PlatformEvent with stages (including LLM span metadata).
    // The exporter calls core_tracing::get_stages_json() on
    // PipelineComplete / ModelComplete events to attach span data.
    if telemetry_enabled {
        let elapsed_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        xybrid_sdk::publish_telemetry_event(xybrid_sdk::TelemetryEvent {
            event_type: "ModelComplete".to_string(),
            stage_name: Some("qwen2.5-0.5b-instruct".to_string()),
            target: Some("local".to_string()),
            latency_ms: m.get("generation_time_ms").and_then(|s| s.parse().ok()),
            error: None,
            data: Some(
                serde_json::json!({
                    "model_id": "qwen2.5-0.5b-instruct",
                    "output_type": "Text",
                })
                .to_string(),
            ),
            timestamp_ms: elapsed_ms,
        });

        println!("📡 Flushing telemetry...");
        // Small delay to let the exporter batch pick up the event.
        std::thread::sleep(std::time::Duration::from_millis(500));
        xybrid_sdk::flush_platform_telemetry();
        xybrid_sdk::shutdown_platform_telemetry();
        println!("   ✅ Telemetry sent to ingest");
    }

    Ok(())
}
