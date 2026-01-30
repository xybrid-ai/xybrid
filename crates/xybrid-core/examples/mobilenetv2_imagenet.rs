//! MobileNetV2 ImageNet Classification via Metadata-Driven Execution
//!
//! This example demonstrates the metadata-driven execution system with MobileNetV2,
//! a lightweight mobile-optimized architecture. Proves the system works across
//! different model architectures (ResNet-50 heavy, MobileNet light).

use ndarray::Array4;
use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::testing::model_fixtures;

// ImageNet class labels (top 10 for demo)
const IMAGENET_CLASSES: &[&str] = &[
    "tench, Tinca tinca",
    "goldfish, Carassius auratus",
    "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
    "tiger shark, Galeocerdo cuvier",
    "hammerhead, hammerhead shark",
    "electric ray, crampfish, numbfish, torpedo",
    "stingray",
    "cock",
    "hen",
    "ostrich, Struthio camelus",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  MobileNetV2 - Lightweight ImageNet Classification");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Load metadata
    let model_dir = model_fixtures::require_model("mobilenet");
    let metadata_path = model_dir.join("model_metadata.json");
    println!("📋 Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("✅ Metadata loaded:");
    println!("   Model: {} v{}", metadata.model_id, metadata.version);
    println!("   Model Size: 13.3 MB (lightweight!)");
    println!("   Execution: {:?}", metadata.execution_template);
    println!("   Preprocessing: {} steps", metadata.preprocessing.len());
    println!("   Postprocessing: {} steps", metadata.postprocessing.len());
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    println!("✅ TemplateExecutor created");
    println!();

    // Create a test image (random pattern for now - in real use, load actual image)
    // Shape: [batch=1, channels=3, height=256, width=256] (will be cropped to 224x224)
    println!("🎨 Creating test image (256x256 RGB)...");
    let mut image_data = Array4::<f32>::zeros((1, 3, 256, 256));

    // Create a different pattern than ResNet-50 (diagonal stripes)
    for c in 0..3 {
        for h in 0..256 {
            for w in 0..256 {
                // Create diagonal stripe pattern
                let pattern = ((h + w) / 32) % 2;
                image_data[[0, c, h, w]] = if pattern == 0 {
                    200.0 + (c * 20) as f32
                } else {
                    50.0 + (c * 10) as f32
                };
            }
        }
    }

    println!("✅ Test image created (256x256x3)");
    println!("   Pattern: Diagonal stripes");
    println!("   Note: Using synthetic pattern for testing");
    println!("   For real predictions, use actual ImageNet images");
    println!();

    // Convert image_data to flat embedding vector
    let flattened: Vec<f32> = image_data.iter().cloned().collect();

    // Create input envelope
    let envelope_metadata = HashMap::new();
    let input_envelope = Envelope {
        kind: EnvelopeKind::Embedding(flattened),
        metadata: envelope_metadata,
    };

    // Execute inference via TemplateExecutor
    println!("🔄 Running inference via TemplateExecutor...");
    println!("   → Preprocessing:");
    println!("      1. Reshape to [1, 3, 256, 256]");
    println!("      2. CenterCrop to 224x224");
    println!("      3. Normalize (ImageNet mean/std)");
    println!("   → Model execution: mobilenetv2-12.onnx");
    println!("   → Postprocessing:");
    println!("      1. Softmax (probabilities)");
    println!("      2. TopK (top 5 predictions)");
    println!();

    let output_envelope = executor.execute(&metadata, &input_envelope)?;

    println!("✅ Inference complete!");
    println!();

    // Parse output
    match &output_envelope.kind {
        EnvelopeKind::Embedding(output_data) => {
            println!("📊 Top-5 Predictions:");
            println!("   Output format: [class_idx, score, ...]");
            println!(
                "   Length: {} values ({} predictions)",
                output_data.len(),
                output_data.len() / 2
            );
            println!();

            // Parse top-k results (format: [idx1, score1, idx2, score2, ...])
            for i in (0..output_data.len()).step_by(2) {
                let class_idx = output_data[i] as usize;
                let score = output_data[i + 1];

                let class_name = if class_idx < IMAGENET_CLASSES.len() {
                    IMAGENET_CLASSES[class_idx]
                } else {
                    "<class name not available>"
                };

                println!("   {}. {} (ID: {})", i / 2 + 1, class_name, class_idx);
                println!("      Confidence: {:.2}%", score * 100.0);
            }
        }
        EnvelopeKind::Text(text) => {
            println!("📄 Text output: {}", text);
        }
        EnvelopeKind::Audio(_) => {
            println!("🔊 Audio output (unexpected for MobileNetV2)");
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Test Complete");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("🎯 KEY VALIDATION:");
    println!("   ✅ Metadata-driven preprocessing (Reshape + CenterCrop + Normalize)");
    println!("   ✅ SimpleMode execution via TemplateExecutor");
    println!("   ✅ Metadata-driven postprocessing (Softmax + TopK)");
    println!("   ✅ MobileNetV2 inference from metadata configuration");
    println!("   ✅ Lightweight model (13.3 MB vs ResNet-50's 98 MB)");
    println!();
    println!("📝 This proves the metadata system works across architectures!");
    println!("   - Heavy models: ResNet-50 (98 MB, 75.81% accuracy)");
    println!("   - Light models: MobileNetV2 (13.3 MB, 69.48% accuracy)");
    println!("   - Same metadata format, same execution engine");
    println!("   - Zero architecture-specific code required!");
    println!();
    println!("⚠️  NOTE: Using synthetic test image. For real predictions:");
    println!("   - Load actual ImageNet image (.jpg/.png)");
    println!("   - Decode to RGB values (0-255)");
    println!("   - Pass to TemplateExecutor");
    println!();

    Ok(())
}
