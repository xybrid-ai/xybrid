//! Example: Validate model metadata JSON files
//!
//! This example demonstrates loading and validating model metadata
//! for different execution templates (Onnx, SafeTensors, and ModelGraph).

use std::fs;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Model Metadata Validation ===\n");

    // Test 1: Whisper Metadata (could be SafeTensors with Candle or ONNX)
    println!("1. Loading Whisper-tiny metadata...");
    let whisper_dir = model_fixtures::require_model("whisper-tiny");
    let whisper_json = fs::read_to_string(whisper_dir.join("model_metadata.json"))?;
    let whisper_metadata: ModelMetadata = serde_json::from_str(&whisper_json)?;

    println!(
        "   ✓ Model: {} v{}",
        whisper_metadata.model_id, whisper_metadata.version
    );
    println!(
        "   ✓ Description: {}",
        whisper_metadata
            .description
            .as_ref()
            .unwrap_or(&"N/A".to_string())
    );
    println!("   ✓ Files: {:?}", whisper_metadata.files);
    println!(
        "   ✓ Preprocessing steps: {}",
        whisper_metadata.preprocessing.len()
    );
    println!(
        "   ✓ Postprocessing steps: {}",
        whisper_metadata.postprocessing.len()
    );

    match &whisper_metadata.execution_template {
        xybrid_core::execution_template::ExecutionTemplate::ModelGraph { stages, config } => {
            println!("   ✓ Execution: ModelGraph with {} stages", stages.len());
            for stage in stages {
                println!(
                    "      - Stage '{}': {} (mode: {:?})",
                    stage.name, stage.model_file, stage.execution_mode
                );
            }
            println!("   ✓ Config keys: {:?}", config.keys().collect::<Vec<_>>());
        }
        xybrid_core::execution_template::ExecutionTemplate::Onnx { model_file } => {
            println!("   ✓ Execution: Onnx ({})", model_file);
        }
        xybrid_core::execution_template::ExecutionTemplate::SafeTensors {
            model_file,
            architecture,
            ..
        } => {
            println!(
                "   ✓ Execution: SafeTensors ({}, arch: {:?})",
                model_file, architecture
            );
        }
        _ => println!("   ✓ Execution: Other template type"),
    }
    println!();

    // Test 2: MNIST ONNX Metadata
    println!("2. Loading MNIST metadata (Onnx)...");
    let mnist_dir = model_fixtures::require_model("mnist");
    let mnist_json = fs::read_to_string(mnist_dir.join("model_metadata.json"))?;
    let mnist_metadata: ModelMetadata = serde_json::from_str(&mnist_json)?;

    println!(
        "   ✓ Model: {} v{}",
        mnist_metadata.model_id, mnist_metadata.version
    );
    println!(
        "   ✓ Description: {}",
        mnist_metadata
            .description
            .as_ref()
            .unwrap_or(&"N/A".to_string())
    );
    println!("   ✓ Files: {:?}", mnist_metadata.files);
    println!(
        "   ✓ Preprocessing steps: {}",
        mnist_metadata.preprocessing.len()
    );
    println!(
        "   ✓ Postprocessing steps: {}",
        mnist_metadata.postprocessing.len()
    );

    match &mnist_metadata.execution_template {
        xybrid_core::execution_template::ExecutionTemplate::Onnx { model_file } => {
            println!("   ✓ Execution: Onnx ({})", model_file);
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
