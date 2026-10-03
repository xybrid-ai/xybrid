//! ResNet CIFAR-10 Classification
//! The model originally comes from MLCommons (ML Perf Tiny ). Extracted to ONNX by ketiswp/mlcommons-ResNet8-CIFAR10-fp32-onnx HuggingFace.
//! This is a lightweight model at only 320KB (0.32 MB) suitable for embedded devices
//! The input image is resized to a 32x32 NHWC, raw 0-255 pixels

use xybrid_core::execution::ModelMetadata;
use xybrid_core::execution::TemplateExecutor;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

// CIFAR-10 class labels (all 10 labels)
const CIFAR10_CLASSES: &[&str] = &[
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  CIFAR-10 - Lightweight Classification on 10 classes");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Load metadata
    let model_dir = model_fixtures::require_model("cifar10-resnet");
    let metadata_path = model_dir.join("model_metadata.json");
    println!("📋 Loading metadata from: {}", metadata_path.display());

    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

    println!("✅ Metadata loaded:");
    println!("   Model: {} v{}", metadata.model_id, metadata.version);
    println!("   Model Size: 320 KB (lightweight!)");
    println!("   Execution: {:?}", metadata.execution_template);
    println!("   Preprocessing: {} steps", metadata.preprocessing.len());
    println!("   Postprocessing: {} steps", metadata.postprocessing.len());
    println!();

    // Create TemplateExecutor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    println!("✅ TemplateExecutor created");
    println!();

    // Create an encoded test image. The metadata pipeline decodes, resizes,
    // normalizes it before ONNX Runtime sees the tensor.
    println!("🎨 Creating encoded test image (320x256 PNG)...");
    // let image_bytes = create_test_image_png()?;

    // accept an image path
    let image_arg = std::env::args().nth(1);
    let (image_bytes, image_format) = match &image_arg {
        Some(path) => {
            println!("🖼️ Loading image: {}", path);
            let format = std::path::Path::new(path)
                .extension()
                .and_then(|ext| ext.to_str())
                .ok_or("image file has no extension")?;
            (std::fs::read(path)?, format)
        }
        None => {
            println!("🎨 No image given - using synthetic test image (320x256 PNG)");
            println!("   Pass a path to classify a real image:");
            println!("   cargo run --example cifar10_classification -- <image-path>");
            println!("   Pattern: Diagonal stripes");
            println!("   Note: Using synthetic pattern for testing");
            println!("   For real predictions, use actual CIFAR-10 images");
            (create_test_image_png()?, "png")
        }
    };

    let input_envelope = Envelope::image(image_bytes, image_format)?;

    // Execute inference via TemplateExecutor
    println!("🔄 Running inference via TemplateExecutor...");
    println!("   → Preprocessing:");
    println!("      1. ImageDecode PNG to RGB NHWC tensor");
    println!("      2. ImageResize to 32x32");
    println!("      3. ImageNormalize (Custom)");
    println!("   → Model execution: model.onnx");
    println!("   → Postprocessing:");
    println!("      1. TopK (top 3 predictions)");
    println!();

    let output_envelope = executor.execute(&metadata, &input_envelope, None)?;

    println!("✅ Inference complete!");
    println!();

    // Parse output
    match &output_envelope.kind {
        EnvelopeKind::Embedding(output_data) => {
            println!("📊 Top-3 Predictions:");
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

                let class_name = if class_idx < CIFAR10_CLASSES.len() {
                    CIFAR10_CLASSES[class_idx]
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
            println!("🔊 Audio output (unexpected for CIFAR10-ResNet)");
        }
        EnvelopeKind::Image { .. } => {
            println!("🖼️ Image output (unexpected for CIFAR10-ResNet)");
        }
        EnvelopeKind::MultiPart(_) => {
            println!("📦 Multipart output (unexpected for CIFAR10-ResNet)");
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Test Complete");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("🎯 KEY VALIDATION:");
    println!("   ✅ Metadata-driven preprocessing (ImageDecode + ImageResize + ImageNormalize)");
    println!("   ✅ ONNX execution via TemplateExecutor");
    println!("   ✅ Metadata-driven postprocessing (TopK)");
    println!("   ✅ ResNet-CIFAR-10 inference from metadata configuration");
    println!("   ✅ Lightweight model (~320 KB)");
    println!();
    println!("⚠️  NOTE: Using synthetic test image. For real predictions:");
    println!("   - Load actual CIFAR-10 image (.jpg/.png)");
    println!("   - Pass encoded bytes to Envelope::image");
    println!("   - Let model_metadata.json drive decode/resize/normalize");
    println!();

    Ok(())
}

fn create_test_image_png() -> Result<Vec<u8>, image::ImageError> {
    let mut image = image::RgbImage::new(320, 256);

    for y in 0..256 {
        for x in 0..320 {
            let pattern = ((x + y) / 32) % 2;
            let pixel = if pattern == 0 {
                image::Rgb([200, 220, 240])
            } else {
                image::Rgb([50, 60, 70])
            };
            image.put_pixel(x, y, pixel);
        }
    }

    let mut encoded = std::io::Cursor::new(Vec::new());
    image.write_to(&mut encoded, image::ImageFormat::Png)?;
    Ok(encoded.into_inner())
}
