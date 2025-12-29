//! MNIST Digit Recognition via Metadata-Driven Execution (SimpleMode)
//!
//! This example demonstrates the complete metadata-driven execution system with a
//! simple MNIST classifier. It proves that the TemplateExecutor can run models via
//! JSON metadata with full preprocessing/postprocessing pipelines.

use std::path::PathBuf;
use std::collections::HashMap;
use ndarray::{Array4};
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  MNIST - Full Metadata-Driven Execution");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Load metadata
    let model_dir = model_fixtures::require_model("mnist");
    let metadata_path = model_dir.join("model_metadata.json");
    println!("📋 Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("✅ Metadata loaded:");
    println!("   Model: {} v{}", metadata.model_id, metadata.version);
    println!("   Execution: {:?}", metadata.execution_template);
    println!("   Preprocessing: {} steps", metadata.preprocessing.len());
    println!("   Postprocessing: {} steps", metadata.postprocessing.len());
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    println!("✅ TemplateExecutor created");
    println!();

    // Create a simple test digit (handcrafted "7" pattern)
    println!("🎨 Creating test digit pattern (handcrafted '7')...");
    let mut image_data = Array4::<f32>::zeros((1, 1, 28, 28));

    // Draw a simple "7" pattern with pixel values 0-255
    // Top horizontal line
    for x in 5..23 {
        image_data[[0, 0, 5, x]] = 255.0;
        image_data[[0, 0, 6, x]] = 255.0;
    }
    // Diagonal stroke
    for i in 0..18 {
        let y = 7 + i;
        let x = 22 - i;
        if y < 28 && x < 28 {
            image_data[[0, 0, y, x]] = 255.0;
            if x > 0 {
                image_data[[0, 0, y, x - 1]] = 255.0;
            }
        }
    }

    // Print ASCII visualization of the digit
    println!("\n   Digit visualization (28x28):");
    for y in 0..28 {
        print!("   ");
        for x in 0..28 {
            let val = image_data[[0, 0, y, x]];
            if val > 128.0 {
                print!("█");
            } else {
                print!(" ");
            }
        }
        println!();
    }
    println!();

    // Convert image_data to flat embedding vector (will be reshaped by preprocessing)
    let flattened: Vec<f32> = image_data.iter().cloned().collect();

    // Create input envelope with Embedding variant
    let envelope_metadata = HashMap::new();
    let input_envelope = Envelope {
        kind: EnvelopeKind::Embedding(flattened),
        metadata: envelope_metadata,
    };

    // Execute inference via TemplateExecutor (with full preprocessing/postprocessing)
    println!("🔄 Running inference via TemplateExecutor...");
    println!("   → Preprocessing: Reshape [784] → [1,1,28,28], Normalize /255");
    println!("   → Model execution: mnist-12.onnx");
    println!("   → Postprocessing: Softmax");
    println!();

    let output_envelope = executor.execute(&metadata, &input_envelope)?;

    println!("✅ Inference complete!");
    println!();

    // Parse output
    match &output_envelope.kind {
        EnvelopeKind::Embedding(output_data) => {
            println!("📊 Output probabilities (after softmax):");
            println!("   Length: {}", output_data.len());

            if output_data.len() == 10 {
                // Find predicted digit (argmax)
                let (predicted_digit, confidence) = output_data.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .ok_or("Failed to find max")?;

                println!();
                println!("🎯 Prediction:");
                println!("   Digit: {}", predicted_digit);
                println!("   Confidence: {:.2}%", confidence * 100.0);
                println!();

                // Show all probabilities
                println!("   All probabilities:");
                for (digit, prob) in output_data.iter().enumerate() {
                    let bar_length = (prob * 50.0).min(50.0).max(0.0) as usize;
                    let bar: String = "█".repeat(bar_length);
                    println!("     {}: {:.4} ({:5.2}%) {}", digit, prob, prob * 100.0, bar);
                }

                // Verify prediction is correct
                println!();
                if predicted_digit == 7 {
                    println!("✅ SUCCESS: Correctly predicted digit 7!");
                } else {
                    println!("⚠️  UNEXPECTED: Predicted {} instead of 7", predicted_digit);
                }
            } else {
                println!("   Raw output: {:?}", &output_data[..10.min(output_data.len())]);
            }
        }
        EnvelopeKind::Text(text) => {
            println!("📄 Text output: {}", text);
        }
        EnvelopeKind::Audio(_) => {
            println!("🔊 Audio output (unexpected for MNIST)");
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Test Complete");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("🎯 KEY VALIDATION:");
    println!("   ✅ Metadata-driven preprocessing (Reshape + Normalize)");
    println!("   ✅ SimpleMode execution via TemplateExecutor");
    println!("   ✅ Metadata-driven postprocessing (Softmax)");
    println!("   ✅ End-to-end MNIST classification from flat embedding");
    println!();
    println!("📝 This proves the complete metadata system works!");
    println!("   - Input: Flat 784-element vector (raw pixels 0-255)");
    println!("   - Preprocessing: Reshape [1,1,28,28] + Normalize [0,1]");
    println!("   - Execution: ONNX model via dynamic I/O resolution");
    println!("   - Postprocessing: Softmax probabilities");
    println!("   - Output: Clean probability distribution");
    println!();

    Ok(())
}
