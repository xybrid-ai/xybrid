//! Gemma 4 E2B vision caption example.
//!
//! Builds a text+image user message and runs it through the metadata-driven
//! `VisionLanguage` path. The checked-in fixture metadata is small, but the
//! Q8 language model plus Q8 mmproj pair are about 5.5 GB and are not checked
//! into git.
//!
//! Run after downloading the fixture:
//!   ./integration-tests/download.sh gemma-4-e2b
//!   cargo run -p xybrid-core --example gemma4_vision_caption \
//!     --features llm-llamacpp-vision -- \
//!     --prompt "Describe this image in one short sentence."

#[cfg(feature = "llm-llamacpp-vision")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;
    use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
    use xybrid_core::ir::{Envelope, EnvelopeKind};
    use xybrid_core::runtime_adapter::llm::GenerationConfig;
    use xybrid_core::testing::model_fixtures;

    const MODEL_ID: &str = "gemma-4-e2b";

    let args = ExampleArgs::parse()?;
    let model_dir = args.model_dir.unwrap_or_else(|| {
        model_fixtures::model_path(MODEL_ID)
            .unwrap_or_else(|| PathBuf::from("integration-tests/fixtures/models").join(MODEL_ID))
    });

    println!("Gemma 4 E2B vision caption example");
    println!("Model directory: {}", model_dir.display());

    let metadata_path = model_dir.join("model_metadata.json");
    if !metadata_path.exists() {
        println!("SKIP: metadata not found at {}", metadata_path.display());
        println!("Run: ./integration-tests/download.sh {MODEL_ID}");
        return Ok(());
    }

    let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;
    let required_files = required_vision_files(&metadata)?;
    let missing: Vec<_> = required_files
        .iter()
        .filter(|file| !model_dir.join(file).exists())
        .collect();
    if !missing.is_empty() {
        println!("SKIP: Gemma 4 E2B artifacts are not downloaded.");
        for file in missing {
            println!("Missing: {}", model_dir.join(file).display());
        }
        println!("Run: ./integration-tests/download.sh {MODEL_ID}");
        return Ok(());
    }

    let (image_bytes, image_format) = match args.image {
        Some(path) => {
            let format = image_format_from_path(&path)?;
            (std::fs::read(&path)?, format)
        }
        None => (synthetic_png()?, "png"),
    };

    let image = Envelope::image(image_bytes, image_format)?;
    let input = Envelope::user_message(args.prompt, vec![image])?;
    let config = GenerationConfig {
        max_tokens: 128,
        temperature: 0.1,
        top_p: 0.9,
        ..Default::default()
    };

    let mut executor = TemplateExecutor::with_base_path(
        model_dir
            .to_str()
            .ok_or("model directory path is not valid UTF-8")?,
    );

    println!("Running local Gemma 4 E2B vision inference...");
    let start = std::time::Instant::now();
    let output = executor.execute(&metadata, &input, Some(&config))?;
    let elapsed = start.elapsed();

    match output.kind {
        EnvelopeKind::Text(text) => {
            println!();
            println!("Caption:");
            println!("{}", text.trim());
        }
        other => {
            println!("Unexpected output kind: {other:?}");
        }
    }

    if let Some(ms) = output.metadata.get("image_preprocess_ms") {
        println!("Image preprocess: {ms} ms");
    }
    println!("Elapsed: {:.2}s", elapsed.as_secs_f32());

    Ok(())
}

#[cfg(not(feature = "llm-llamacpp-vision"))]
fn main() {
    eprintln!("This example requires the `llm-llamacpp-vision` feature.");
    eprintln!(
        "Run: cargo run -p xybrid-core --example gemma4_vision_caption --features llm-llamacpp-vision"
    );
}

#[cfg(feature = "llm-llamacpp-vision")]
struct ExampleArgs {
    model_dir: Option<std::path::PathBuf>,
    image: Option<std::path::PathBuf>,
    prompt: String,
}

#[cfg(feature = "llm-llamacpp-vision")]
impl ExampleArgs {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            model_dir: None,
            image: None,
            prompt: "Describe this image in one short sentence.".to_string(),
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-dir" => {
                    parsed.model_dir = Some(next_path(&mut args, "--model-dir")?);
                }
                "--image" => {
                    parsed.image = Some(next_path(&mut args, "--image")?);
                }
                "--prompt" => {
                    parsed.prompt = next_value(&mut args, "--prompt")?;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => {
                    return Err(format!("unknown argument: {other}").into());
                }
            }
        }

        Ok(parsed)
    }
}

#[cfg(feature = "llm-llamacpp-vision")]
fn next_path(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    Ok(std::path::PathBuf::from(next_value(args, flag)?))
}

#[cfg(feature = "llm-llamacpp-vision")]
fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

#[cfg(feature = "llm-llamacpp-vision")]
fn print_usage() {
    println!("Usage:");
    println!("  cargo run -p xybrid-core --example gemma4_vision_caption --features llm-llamacpp-vision -- [options]");
    println!();
    println!("Options:");
    println!("  --model-dir <path>  Directory containing model_metadata.json and Gemma artifacts");
    println!("  --image <path>      PNG, JPEG, or WebP image. Defaults to a synthetic PNG");
    println!("  --prompt <text>     Prompt sent with the image");
}

#[cfg(feature = "llm-llamacpp-vision")]
fn required_vision_files(
    metadata: &xybrid_core::execution::ModelMetadata,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    match &metadata.execution_template {
        xybrid_core::execution::ExecutionTemplate::VisionLanguage { model_file, .. } => {
            files.push(model_file.clone());
        }
        other => {
            return Err(format!("expected VisionLanguage metadata, got {other:?}").into());
        }
    }

    let vision_encoder = metadata
        .vision_encoder
        .as_ref()
        .ok_or("Gemma 4 E2B metadata is missing vision_encoder")?;
    files.push(vision_encoder.file.clone());
    Ok(files)
}

#[cfg(feature = "llm-llamacpp-vision")]
fn image_format_from_path(
    path: &std::path::Path,
) -> Result<&'static str, Box<dyn std::error::Error>> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => Ok("png"),
        Some("jpg" | "jpeg") => Ok("jpeg"),
        Some("webp") => Ok("webp"),
        _ => Err(format!(
            "cannot infer image format from {}; use png, jpg, jpeg, or webp",
            path.display()
        )
        .into()),
    }
}

#[cfg(feature = "llm-llamacpp-vision")]
fn synthetic_png() -> Result<Vec<u8>, image::ImageError> {
    let mut image = image::RgbImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            let pixel = if x < 32 {
                image::Rgb([220, 60, 40])
            } else if y < 32 {
                image::Rgb([40, 160, 220])
            } else {
                image::Rgb([40, 180, 80])
            };
            image.put_pixel(x, y, pixel);
        }
    }

    let mut encoded = std::io::Cursor::new(Vec::new());
    image.write_to(&mut encoded, image::ImageFormat::Png)?;
    Ok(encoded.into_inner())
}
