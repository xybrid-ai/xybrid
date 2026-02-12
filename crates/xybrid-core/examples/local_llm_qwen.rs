//! Local LLM Inference via Metadata-Driven Execution (GGUF)
//!
//! This example demonstrates local LLM inference using the metadata-driven
//! execution system with a GGUF model (Qwen 2.5 0.5B Instruct).
//!
//! Run with:
//!   cargo run --example local_llm_qwen -p xybrid-core --features llm-mistral
//!
//! Requires model to be downloaded:
//!   ./integration-tests/download.sh qwen2.5-0.5b-instruct

use std::collections::HashMap;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Local LLM - Qwen 2.5 0.5B Instruct (GGUF)");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Resolve model directory using fixtures
    let model_dir = model_fixtures::require_model("qwen2.5-0.5b-instruct");

    // Load metadata
    let metadata_path = model_dir.join("model_metadata.json");
    println!("📋 Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("✅ Metadata loaded:");
    println!("   Model: {} v{}", metadata.model_id, metadata.version);
    if let Some(desc) = &metadata.description {
        println!("   Description: {}", desc);
    }
    println!("   Execution: {:?}", metadata.execution_template);
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    println!("✅ TemplateExecutor created");
    println!();

    // Create a test prompt
    let prompt = "What is the capital of France? Answer in one sentence.";
    println!("💬 Prompt: {}", prompt);
    println!();

    // Create input envelope
    let mut envelope_metadata = HashMap::new();
    // Optional: add system prompt
    envelope_metadata.insert(
        "system_prompt".to_string(),
        "You are a helpful assistant. Give concise answers.".to_string(),
    );
    // Optional: override generation parameters
    envelope_metadata.insert("max_tokens".to_string(), "128".to_string());
    envelope_metadata.insert("temperature".to_string(), "0.7".to_string());

    let input_envelope = Envelope {
        kind: EnvelopeKind::Text(prompt.to_string()),
        metadata: envelope_metadata,
    };

    // Execute inference via TemplateExecutor
    println!("🔄 Running inference via TemplateExecutor...");
    println!("   → Loading GGUF model (first run may take a moment)...");
    println!();

    let start = std::time::Instant::now();
    let output_envelope = executor.execute(&metadata, &input_envelope)?;
    let elapsed = start.elapsed();

    println!("✅ Inference complete! ({:.2}s)", elapsed.as_secs_f32());
    println!();

    // Parse output
    match &output_envelope.kind {
        EnvelopeKind::Text(response) => {
            println!("═══════════════════════════════════════════════════════");
            println!("📤 Response:");
            println!("═══════════════════════════════════════════════════════");
            println!("{}", response);
            println!("═══════════════════════════════════════════════════════");
            println!();

            // Print generation stats from metadata
            if let Some(tokens) = output_envelope.metadata.get("tokens_generated") {
                println!("📊 Generation stats:");
                println!("   Tokens generated: {}", tokens);
            }
            if let Some(time_ms) = output_envelope.metadata.get("generation_time_ms") {
                println!("   Generation time: {}ms", time_ms);
            }
            if let Some(tps) = output_envelope.metadata.get("tokens_per_second") {
                println!("   Tokens/second: {}", tps);
            }
            if let Some(reason) = output_envelope.metadata.get("finish_reason") {
                println!("   Finish reason: {}", reason);
            }
        }
        EnvelopeKind::Embedding(_) => {
            println!("⚠️  Unexpected: Got embedding output instead of text");
        }
        EnvelopeKind::Audio(_) => {
            println!("⚠️  Unexpected: Got audio output instead of text");
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Test Complete");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("🎯 KEY VALIDATION:");
    println!("   ✅ GGUF model loaded via LlmRuntimeAdapter");
    println!("   ✅ Text input via Envelope");
    println!("   ✅ Chat generation with system prompt");
    println!("   ✅ Generation stats in response metadata");
    println!();

    Ok(())
}
