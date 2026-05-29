//! Local LLM Inference - Ministral 3 3B (GGUF)
//!
//! Run with:
//!   cargo run --example local_llm_ministral -p xybrid-core --features llm-llamacpp

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Local LLM - Ministral 3 3B Instruct (GGUF)");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Resolve model directory - use integration-tests fixtures
    let model_dir = PathBuf::from("repos/xybrid/integration-tests/fixtures/models/ministral-3-3b");
    if !model_dir.exists() {
        eprintln!("Model directory not found: {}", model_dir.display());
        eprintln!("Run: cd repos/xybrid/integration-tests && ./download.sh ministral-3-3b");
        return Err("Model not found".into());
    }

    // Load metadata
    let metadata_path = model_dir.join("model_metadata.json");
    println!("Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("Metadata loaded:");
    println!("   Model: {} v{}", metadata.model_id, metadata.version);
    println!("   Execution: {:?}", metadata.execution_template);
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    println!("TemplateExecutor created");
    println!();

    // Create a test prompt
    let prompt = "What is the capital of France? Answer in one sentence.";
    println!("Prompt: {}", prompt);
    println!();

    // Create input envelope
    let mut envelope_metadata = HashMap::new();
    envelope_metadata.insert(
        "system_prompt".to_string(),
        "You are a helpful assistant. Give concise answers.".to_string(),
    );
    envelope_metadata.insert("max_tokens".to_string(), "128".to_string());
    envelope_metadata.insert("temperature".to_string(), "0.7".to_string());

    let input_envelope = Envelope {
        kind: EnvelopeKind::Text(prompt.to_string()),
        metadata: envelope_metadata,
    };

    // Execute inference
    println!("Running inference...");
    println!("   Loading GGUF model (first run may take a moment)...");
    println!();

    let start = std::time::Instant::now();
    let output_envelope = executor.execute(&metadata, &input_envelope, None)?;
    let elapsed = start.elapsed();

    println!("Inference complete! ({:.2}s)", elapsed.as_secs_f32());
    println!();

    // Parse output
    match &output_envelope.kind {
        EnvelopeKind::Text(response) => {
            println!("═══════════════════════════════════════════════════════");
            println!("Response:");
            println!("═══════════════════════════════════════════════════════");
            println!("{}", response);
            println!("═══════════════════════════════════════════════════════");
        }
        _ => {
            println!("Unexpected output type");
        }
    }

    Ok(())
}
