//! Candle Whisper Bundle Example
//!
//! This example demonstrates using the Candle Whisper model through the
//! metadata-driven execution system, which is how bundled models are executed.
//!
//! # Running
//!
//! ```bash
//! cargo run --example candle_whisper_bundle --features candle
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(feature = "candle")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use xybrid_core::execution_template::ModelMetadata;
    use xybrid_core::ir::{Envelope, EnvelopeKind};
    use xybrid_core::template_executor::TemplateExecutor;

    println!("=== Candle Whisper Bundle Example ===\n");

    let model_dir = PathBuf::from("test_models/whisper-tiny-candle");

    // 1. Load model metadata
    println!("1. Loading model metadata...");
    let metadata_path = model_dir.join("model_metadata.json");
    if !metadata_path.exists() {
        println!("   Model metadata not found at {:?}", metadata_path);
        return Ok(());
    }

    let metadata_json = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_json)?;
    println!("   Model: {} v{}", metadata.model_id, metadata.version);
    println!("   Template: {:?}", metadata.execution_template);

    // 2. Load test audio file
    let audio_path = model_dir.join("jfk.wav");
    if !audio_path.exists() {
        println!("\n2. No test audio file found at {:?}", audio_path);
        return Ok(());
    }

    println!("\n2. Loading audio file: {:?}", audio_path);
    let audio_bytes = std::fs::read(&audio_path)?;
    println!("   Audio size: {} bytes", audio_bytes.len());

    // 3. Create input envelope
    println!("\n3. Creating input envelope...");
    let input = Envelope {
        kind: EnvelopeKind::Audio(audio_bytes),
        metadata: HashMap::new(),
    };

    // 4. Create executor and run
    println!("\n4. Running transcription via TemplateExecutor...");
    let mut executor = xybrid_core::template_executor::TemplateExecutor::with_base_path(
        model_dir.to_str().unwrap()
    );

    let start = std::time::Instant::now();
    let output = executor.execute(&metadata, &input)?;
    let elapsed = start.elapsed();

    // 5. Print result
    println!("\n=== Transcription Result ===");
    match &output.kind {
        EnvelopeKind::Text(text) => {
            println!("{}", text);
        }
        other => {
            println!("Unexpected output type: {:?}", other);
        }
    }
    println!("============================");
    println!("\nTime: {:.2}s", elapsed.as_secs_f32());

    println!("\n=== Example complete ===");
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    println!("This example requires the 'candle' feature.");
    println!("Run with: cargo run --example candle_whisper_bundle --features candle");
}
