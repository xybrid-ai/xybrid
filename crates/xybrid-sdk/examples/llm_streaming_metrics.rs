//! LLM streaming telemetry metrics — TTFT, TPS, ITL.
//!
//! Runs a local GGUF LLM through the SDK streaming path and
//! reports the streaming-derived telemetry fields (time-to-first-token,
//! mean / p95 inter-chunk latency, emitted chunk count) alongside the
//! existing aggregate metrics.
//!
//! Run with:
//!   cargo run --example llm_streaming_metrics -p xybrid-sdk --features llm-mistral --release
//!
//! To also export traces and metrics to a local Collector, set:
//!   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
//!   cargo run --example llm_streaming_metrics -p xybrid-sdk --features llm-mistral --release
//!
//! Reuses the `integration-tests/fixtures/models/qwen2.5-0.5b-instruct/`
//! fixture. See `integration-tests/download.sh` for download instructions
//! if the model isn't present locally.

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::ir::{Envelope, EnvelopeKind};

#[path = "support/observability.rs"]
mod observability_support;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  LLM Streaming Metrics — TTFT / TPS / ITL");
    println!("═══════════════════════════════════════════════════════");
    println!();

    let export = observability_support::configure("llm-streaming-metrics")?;

    let model_dir = PathBuf::from("integration-tests/fixtures/models/qwen2.5-0.5b-instruct");
    if !model_dir.exists() {
        eprintln!("❌ Model directory not found: {}", model_dir.display());
        eprintln!("   Run `integration-tests/download.sh qwen2.5-0.5b-instruct` first.");
        return Err("Model not found".into());
    }

    let model = xybrid_sdk::ModelLoader::from_directory(&model_dir)?.load()?;

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

    let result = model.run_streaming(&input, None, |_token| Ok(()))?;
    let output = result.envelope();

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

    if let Some(reference) = result.observation_ref() {
        println!("Inference trace: {}", reference.trace_id);
    }
    export.finish()?;

    Ok(())
}
