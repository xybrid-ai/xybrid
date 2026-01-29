//! Local LLM Streaming Inference - Gemma 3 1B (GGUF)
//!
//! This example demonstrates token streaming for LLM inference.
//!
//! Run with:
//!   cargo run --example streaming_llm -p xybrid-core --features llm-llamacpp

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;

#[cfg(feature = "llm-llamacpp")]
use xybrid_core::execution::ModelMetadata;
#[cfg(feature = "llm-llamacpp")]
use xybrid_core::execution::TemplateExecutor;
#[cfg(feature = "llm-llamacpp")]
use xybrid_core::ir::{Envelope, EnvelopeKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "llm-llamacpp"))]
    {
        eprintln!("This example requires the llm-llamacpp feature.");
        eprintln!("Run with: cargo run --example streaming_llm -p xybrid-core --features llm-llamacpp");
        return Ok(());
    }

    #[cfg(feature = "llm-llamacpp")]
    {
        run_streaming_example()
    }
}

#[cfg(feature = "llm-llamacpp")]
fn run_streaming_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Streaming LLM - Gemma 3 1B Instruct (GGUF)");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Resolve model directory - use integration-tests fixtures
    let model_dir = PathBuf::from("integration-tests/fixtures/models/gemma-3-1b");
    if !model_dir.exists() {
        // Try alternative path (running from workspace root)
        let alt_path = PathBuf::from("repos/xybrid/integration-tests/fixtures/models/gemma-3-1b");
        if alt_path.exists() {
            return run_with_model_dir(&alt_path);
        }
        eprintln!("Model directory not found: {}", model_dir.display());
        eprintln!("Please ensure the gemma-3-1b model is downloaded.");
        return Err("Model not found".into());
    }

    run_with_model_dir(&model_dir)
}

#[cfg(feature = "llm-llamacpp")]
fn run_with_model_dir(model_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Load metadata
    let metadata_path = model_dir.join("model_metadata.json");
    println!("Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("Model: {} v{}", metadata.model_id, metadata.version);
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Create a test prompt
    let prompt = "Write a haiku about programming.";
    println!("Prompt: {}", prompt);
    println!();

    // Create input envelope
    let mut envelope_metadata = HashMap::new();
    envelope_metadata.insert(
        "system_prompt".to_string(),
        "You are a creative assistant. Give poetic answers.".to_string(),
    );
    envelope_metadata.insert("max_tokens".to_string(), "64".to_string());
    envelope_metadata.insert("temperature".to_string(), "0.8".to_string());

    let input_envelope = Envelope {
        kind: EnvelopeKind::Text(prompt.to_string()),
        metadata: envelope_metadata,
    };

    // Execute with streaming
    println!("Streaming response:");
    println!("───────────────────────────────────────────────────────");

    let mut token_count = 0;
    let start = std::time::Instant::now();

    let output_envelope = executor.execute_streaming(
        &metadata,
        &input_envelope,
        Box::new(|token| {
            // Print each token as it arrives
            print!("{}", token.token);
            io::stdout().flush()?;
            token_count += 1;

            // Show token ID for debugging (optional)
            if let Some(id) = token.token_id {
                // Uncomment to see token IDs:
                // eprint!("[{}]", id);
                let _ = id; // silence unused warning
            }

            Ok(())
        }),
    )?;

    let elapsed = start.elapsed();
    println!();
    println!("───────────────────────────────────────────────────────");

    // Show stats
    println!();
    println!("Stats:");
    println!("  Tokens streamed: {}", token_count);
    println!("  Total time: {:.2}s", elapsed.as_secs_f32());
    if token_count > 0 {
        let tps = token_count as f32 / elapsed.as_secs_f32();
        println!("  Tokens/sec: {:.1}", tps);
    }

    // Show final output metadata
    if let Some(tps) = output_envelope.metadata.get("tokens_per_second") {
        println!("  Backend tokens/sec: {}", tps);
    }
    if let Some(reason) = output_envelope.metadata.get("finish_reason") {
        println!("  Finish reason: {}", reason);
    }

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Streaming test completed successfully!");
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}
