//! MobileNetV2 ImageNet Classification via Metadata-Driven Execution
//!
//! This example demonstrates the metadata-driven execution system with MobileNetV2,
//! a lightweight mobile-optimized architecture. Proves the system works across
//! different model architectures (ResNet-50 heavy, MobileNet light).

#[cfg(feature = "vision")]
use xybrid_core::execution::ModelMetadata;
#[cfg(feature = "vision")]
use xybrid_core::execution::TemplateExecutor;
#[cfg(feature = "vision")]
use xybrid_core::ir::{Envelope, EnvelopeKind};
#[cfg(feature = "vision")]
use xybrid_core::testing::model_fixtures;

// ImageNet class labels (top 10 for demo)
#[cfg(feature = "vision")]
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

#[cfg(feature = "vision")]
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

    // Create an encoded test image. The metadata pipeline decodes, resizes,
    // crops, and normalizes it before ONNX Runtime sees the tensor.
    println!("🎨 Creating encoded test image (320x256 PNG)...");
    let image_bytes = create_test_image_png()?;
    println!("✅ Test image created");
    println!("   Pattern: Diagonal stripes");
    println!("   Note: Using synthetic pattern for testing");
    println!("   For real predictions, use actual ImageNet images");
    println!();

    let input_envelope = Envelope::image(image_bytes, "png")?;

    // Execute inference via TemplateExecutor
    println!("🔄 Running inference via TemplateExecutor...");
    println!("   → Preprocessing:");
    println!("      1. ImageDecode PNG to RGB NCHW tensor");
    println!("      2. ImageResize to 256x256");
    println!("      3. CenterCrop to 224x224");
    println!("      4. ImageNormalize (ImageNet preset)");
    println!("   → Model execution: mobilenetv2-12.onnx");
    println!("   → Postprocessing:");
    println!("      1. Softmax (probabilities)");
    println!("      2. TopK (top 5 predictions)");
    println!();

    let output_envelope = executor.execute(&metadata, &input_envelope, None)?;

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
        EnvelopeKind::Image { .. } => {
            println!("🖼️ Image output (unexpected for MobileNetV2)");
        }
        EnvelopeKind::MultiPart(_) => {
            println!("📦 Multipart output (unexpected for MobileNetV2)");
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Test Complete");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("🎯 KEY VALIDATION:");
    println!("   ✅ Metadata-driven preprocessing (ImageDecode + ImageResize + CenterCrop + ImageNormalize)");
    println!("   ✅ ONNX execution via TemplateExecutor");
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
    println!("   - Pass encoded bytes to Envelope::image");
    println!("   - Let model_metadata.json drive decode/resize/crop/normalize");
    println!();

    Ok(())
}

#[cfg(not(feature = "vision"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("This example requires the `vision` feature.");
    eprintln!("Run: cargo run --example mobilenetv2_imagenet --features vision");
    Ok(())
}

#[cfg(feature = "vision")]
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
