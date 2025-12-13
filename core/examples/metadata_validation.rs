//! Example: Validate model metadata JSON files
//!
//! This example demonstrates loading and validating model metadata
//! for different execution templates (SimpleMode and Pipeline).

use xybrid_core::execution_template::ModelMetadata;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Model Metadata Validation ===\n");

    // Test 1: Whisper Pipeline Metadata
    println!("1. Loading Whisper-tiny metadata (Pipeline)...");
    let whisper_json = fs::read_to_string("../examples/whisper-tiny-metadata.json")?;
    let whisper_metadata: ModelMetadata = serde_json::from_str(&whisper_json)?;

    println!("   ✓ Model: {} v{}", whisper_metadata.model_id, whisper_metadata.version);
    println!("   ✓ Description: {}", whisper_metadata.description.as_ref().unwrap_or(&"N/A".to_string()));
    println!("   ✓ Files: {:?}", whisper_metadata.files);
    println!("   ✓ Preprocessing steps: {}", whisper_metadata.preprocessing.len());
    println!("   ✓ Postprocessing steps: {}", whisper_metadata.postprocessing.len());

    match &whisper_metadata.execution_template {
        xybrid_core::execution_template::ExecutionTemplate::Pipeline { stages, config } => {
            println!("   ✓ Execution: Pipeline with {} stages", stages.len());
            for stage in stages {
                println!("      - Stage '{}': {} (mode: {:?})",
                    stage.name,
                    stage.model_file,
                    stage.execution_mode
                );
            }
            println!("   ✓ Config keys: {:?}", config.keys().collect::<Vec<_>>());
        }
        _ => println!("   ✗ Unexpected execution template type!"),
    }
    println!();

    // Test 2: MNIST SimpleMode Metadata
    println!("2. Loading MNIST metadata (SimpleMode)...");
    let mnist_json = fs::read_to_string("../examples/mnist-metadata.json")?;
    let mnist_metadata: ModelMetadata = serde_json::from_str(&mnist_json)?;

    println!("   ✓ Model: {} v{}", mnist_metadata.model_id, mnist_metadata.version);
    println!("   ✓ Description: {}", mnist_metadata.description.as_ref().unwrap_or(&"N/A".to_string()));
    println!("   ✓ Files: {:?}", mnist_metadata.files);
    println!("   ✓ Preprocessing steps: {}", mnist_metadata.preprocessing.len());
    println!("   ✓ Postprocessing steps: {}", mnist_metadata.postprocessing.len());

    match &mnist_metadata.execution_template {
        xybrid_core::execution_template::ExecutionTemplate::SimpleMode { model_file } => {
            println!("   ✓ Execution: SimpleMode ({})", model_file);
        }
        _ => println!("   ✗ Unexpected execution template type!"),
    }
    println!();

    // Test 3: Round-trip serialization
    println!("3. Testing round-trip serialization...");
    let whisper_reserialized = serde_json::to_string_pretty(&whisper_metadata)?;
    let whisper_reparsed: ModelMetadata = serde_json::from_str(&whisper_reserialized)?;
    assert_eq!(whisper_metadata.model_id, whisper_reparsed.model_id);
    println!("   ✓ Whisper round-trip successful");

    let mnist_reserialized = serde_json::to_string_pretty(&mnist_metadata)?;
    let mnist_reparsed: ModelMetadata = serde_json::from_str(&mnist_reserialized)?;
    assert_eq!(mnist_metadata.model_id, mnist_reparsed.model_id);
    println!("   ✓ MNIST round-trip successful");
    println!();

    println!("=== All Validation Tests Passed ✓ ===");

    Ok(())
}
