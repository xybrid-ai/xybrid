//! Single Model Inference Example
//!
//! Demonstrates running a single model (Local LLM) using the SDK's
//! PipelineRef API. This is the simplest way to run inference.
//!
//! Run with:
//!   cargo run --example single_model_inference -p xybrid-sdk --features llm-mistral

use std::collections::HashMap;
use xybrid_sdk::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::{DownloadProgress, PipelineRef};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Single Model Inference via SDK");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Define a single-stage pipeline inline
    let yaml = r#"
name: single-llm
stages:
  - qwen2.5-0.5b-instruct
"#;

    println!("📋 Loading pipeline from YAML...");
    let pipeline_ref = PipelineRef::from_yaml(yaml)?;

    println!("   Name: {:?}", pipeline_ref.name());
    println!("   Stages: {:?}", pipeline_ref.stage_ids());
    println!();

    // Load the pipeline (resolves models via registry)
    println!("🔄 Loading pipeline (resolving models)...");
    let pipeline = pipeline_ref.load()?;

    println!("   Download size: {} bytes", pipeline.download_size());
    println!("   Ready: {}", pipeline.is_ready());
    println!();

    // Preload models with progress
    println!("📦 Preloading models...");
    pipeline.load_models_with_progress(|progress: DownloadProgress| {
        println!(
            "   [{}/{}] {} - {}%",
            progress.stage_index + 1,
            progress.total_stages,
            progress.model_id,
            progress.percent
        );
    })?;
    println!("✅ Models loaded");
    println!();

    // Create input envelope
    let prompt = "What is the capital of France? Answer in one sentence.";
    println!("💬 Prompt: {}", prompt);
    println!();

    let mut metadata = HashMap::new();
    metadata.insert(
        "system_prompt".to_string(),
        "You are a helpful assistant.".to_string(),
    );
    metadata.insert("max_tokens".to_string(), "128".to_string());

    let input = Envelope {
        kind: EnvelopeKind::Text(prompt.to_string()),
        metadata,
    };

    // Run inference
    println!("🔄 Running inference...");
    let start = std::time::Instant::now();
    let result = pipeline.run(&input)?;
    let elapsed = start.elapsed();

    println!("✅ Inference complete! ({:.2}s)", elapsed.as_secs_f32());
    println!();

    // Display results
    println!("═══════════════════════════════════════════════════════");
    println!("📊 Pipeline Results:");
    println!("═══════════════════════════════════════════════════════");

    if let Some(name) = &result.name {
        println!("   Pipeline: {}", name);
    }
    println!("   Total latency: {}ms", result.total_latency_ms);
    println!();

    println!("   Stages:");
    for (i, stage) in result.stages.iter().enumerate() {
        println!(
            "     {}. {} - {}ms ({})",
            i + 1,
            stage.name,
            stage.latency_ms,
            stage.target
        );
    }
    println!();

    println!("📤 Response:");
    println!("═══════════════════════════════════════════════════════");
    if let Some(text) = result.text() {
        println!("{}", text);
    } else {
        println!("(Output type: {:?})", result.output_type);
    }
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}
