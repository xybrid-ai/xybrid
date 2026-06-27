#![cfg(feature = "llm-llamacpp")]

use std::path::Path;
use xybrid_core::{
    execution::{ExecutionTemplate, ModelMetadata, TemplateExecutor, VisionPreprocessingPreset},
    ir::{Envelope, EnvelopeKind},
    runtime_adapter::GenerationConfig,
    testing::model_fixtures,
};

const MODEL_ID: &str = "lfm2-vl-450m";
const MODEL_FILE: &str = "LFM2-VL-450M-Q4_0.gguf";
const MMPROJ_FILE: &str = "mmproj-LFM2-VL-450M-Q8_0.gguf";

fn encoded_test_image(width: u32, height: u32) -> Vec<u8> {
    let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        width,
        height,
        image::Rgb([17, 34, 51]),
    ));
    let mut encoded = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("test image encodes");
    encoded.into_inner()
}

fn load_metadata(model_dir: &Path) -> ModelMetadata {
    let metadata_path = model_dir.join("model_metadata.json");
    let content = std::fs::read_to_string(&metadata_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", metadata_path.display()));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", metadata_path.display()))
}

#[test]
fn lfm2_vl_fixture_metadata_is_ready_for_runtime_smoke() {
    let model_dir = model_fixtures::model_path(MODEL_ID)
        .unwrap_or_else(|| panic!("{} fixture metadata directory must exist", MODEL_ID));
    let metadata = load_metadata(&model_dir);

    metadata.validate().expect("metadata validates");
    assert_eq!(metadata.model_id, MODEL_ID);
    assert!(metadata.files.iter().any(|file| file == MODEL_FILE));
    assert!(metadata.files.iter().any(|file| file == MMPROJ_FILE));
    assert_eq!(
        metadata
            .metadata
            .get("task")
            .and_then(|value| value.as_str()),
        Some("vlm")
    );

    match &metadata.execution_template {
        ExecutionTemplate::VisionLanguage {
            model_file,
            context_length,
            ..
        } => {
            assert_eq!(model_file, MODEL_FILE);
            assert_eq!(*context_length, 4096);
        }
        other => panic!("expected VisionLanguage execution template, got {other:?}"),
    }

    let vision = metadata
        .vision_encoder
        .as_ref()
        .expect("VisionLanguage fixture declares vision_encoder");
    assert_eq!(vision.file, MMPROJ_FILE);
    assert_eq!(
        vision.preprocessing_preset,
        VisionPreprocessingPreset::SigLip
    );
    assert_eq!(vision.image_size, 512);
}

#[test]
fn lfm2_vl_runtime_caption_smoke_when_fixture_is_downloaded() {
    let Some(model_dir) = model_fixtures::model_or_skip(MODEL_ID) else {
        return;
    };
    if !model_dir.join(MODEL_FILE).exists() || !model_dir.join(MMPROJ_FILE).exists() {
        eprintln!(
            "Skipping: {} requires both {} and {}",
            MODEL_ID, MODEL_FILE, MMPROJ_FILE
        );
        return;
    }

    let metadata = load_metadata(&model_dir);
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let image = Envelope::image(encoded_test_image(32, 32), "png").expect("valid image");
    let input = Envelope::user_message("Describe the image in one short sentence.", vec![image])
        .expect("multipart VLM input");
    let config = GenerationConfig::greedy().with_max_tokens(24);

    xybrid_core::tracing::init_tracing(true);
    xybrid_core::tracing::reset_tracing();
    let output = executor
        .execute(&metadata, &input, Some(&config))
        .expect("LFM2-VL runtime caption smoke succeeds");
    let spans = xybrid_core::tracing::get_stages_json();
    xybrid_core::tracing::reset_tracing();

    match &output.kind {
        EnvelopeKind::Text(text) => {
            assert!(!text.trim().is_empty(), "VLM output must not be empty");
        }
        other => panic!("expected text output, got {other:?}"),
    }
    assert!(
        output.metadata.contains_key("tokens_generated"),
        "caption smoke should expose LLM generation metadata"
    );

    let span_items = spans["spans"]
        .as_array()
        .expect("runtime smoke should capture tracing spans");
    let execute_metadata = span_items
        .iter()
        .find(|span| span["name"].as_str() == Some("execute:lfm2-vl-450m"))
        .and_then(|span| span["metadata"].as_object())
        .expect("execute span should carry metadata");
    assert_eq!(
        execute_metadata
            .get("task")
            .and_then(|value| value.as_str()),
        Some("vlm"),
        "real VLM inference should carry task=vlm metadata: {spans}"
    );

    let image_preprocess_ms = span_items
        .iter()
        .filter_map(|span| span["metadata"].get("image_preprocess_ms"))
        .filter_map(|value| value.as_str())
        .filter_map(|value| value.parse::<u64>().ok())
        .find(|value| *value > 0);
    assert!(
        image_preprocess_ms.is_some(),
        "real VLM inference should carry positive image_preprocess_ms metadata: {spans}"
    );
}
