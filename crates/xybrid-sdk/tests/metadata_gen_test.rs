//! Integration tests for auto-generation of model_metadata.json from HuggingFace repos.
//!
//! Tests the metadata_gen module's ability to produce valid ModelMetadata
//! from GGUF files and HF model cards without requiring network access.

use std::io::Write;
use tempfile::TempDir;
use xybrid_core::execution::ModelMetadata;

/// Helper: write a minimal valid GGUF v3 file header.
fn write_gguf_v3_header(path: &std::path::Path, architecture: &str, context_length: u32) {
    let mut f = std::fs::File::create(path).unwrap();

    // Magic: "GGUF"
    f.write_all(b"GGUF").unwrap();
    // Version: 3
    f.write_all(&3u32.to_le_bytes()).unwrap();
    // Tensor count: 0
    f.write_all(&0u64.to_le_bytes()).unwrap();
    // Metadata KV count: 2
    f.write_all(&2u64.to_le_bytes()).unwrap();

    // KV 1: general.architecture = architecture
    write_gguf_string(&mut f, "general.architecture");
    f.write_all(&8u32.to_le_bytes()).unwrap(); // STRING type
    write_gguf_string(&mut f, architecture);

    // KV 2: {architecture}.context_length = context_length (UINT32 = type 4)
    let ctx_key = format!("{}.context_length", architecture);
    write_gguf_string(&mut f, &ctx_key);
    f.write_all(&4u32.to_le_bytes()).unwrap(); // UINT32 type
    f.write_all(&context_length.to_le_bytes()).unwrap();
}

fn write_gguf_string(f: &mut std::fs::File, s: &str) {
    f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
    f.write_all(s.as_bytes()).unwrap();
}

#[test]
fn test_gguf_model_produces_valid_metadata() {
    let dir = TempDir::new().unwrap();

    // Simulate a HuggingFace repo with a GGUF model and README
    std::fs::write(
        dir.path().join("README.md"),
        r#"---
pipeline_tag: text-generation
library_name: gguf
language:
  - en
  - zh
license: apache-2.0
tags:
  - gguf
  - llama
---
# Test GGUF Model
A test model for metadata generation.
"#,
    )
    .unwrap();

    // Write a minimal GGUF file
    write_gguf_v3_header(&dir.path().join("model-Q4_K_M.gguf"), "llama", 8192);

    // Generate metadata
    let metadata =
        xybrid_sdk::metadata_gen::generate_metadata(dir.path(), "test-org/test-llama-model")
            .expect("generate_metadata should succeed");

    // Validate core fields
    assert_eq!(metadata.model_id, "test-llama-model");
    assert_eq!(metadata.version, "1.0");

    // Validate execution template is GGUF
    match &metadata.execution_template {
        xybrid_core::execution::ExecutionTemplate::Gguf {
            model_file,
            context_length,
            ..
        } => {
            assert_eq!(model_file, "model-Q4_K_M.gguf");
            assert_eq!(*context_length, 8192);
        }
        other => panic!("Expected Gguf template, got {:?}", other),
    }

    // GGUF models should have no preprocessing/postprocessing
    assert!(metadata.preprocessing.is_empty());
    assert!(metadata.postprocessing.is_empty());

    // Validate metadata fields from model card + GGUF header
    assert_eq!(
        metadata.metadata.get("task").and_then(|v| v.as_str()),
        Some("text-generation")
    );
    assert_eq!(
        metadata
            .metadata
            .get("architecture")
            .and_then(|v| v.as_str()),
        Some("llama")
    );
    assert_eq!(
        metadata.metadata.get("backend").and_then(|v| v.as_str()),
        Some("llamacpp")
    );
    assert_eq!(
        metadata
            .metadata
            .get("auto_generated")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        metadata
            .metadata
            .get("quantization")
            .and_then(|v| v.as_str()),
        Some("Q4_K_M")
    );

    // Validate languages from model card
    let languages = metadata.metadata.get("languages").unwrap();
    assert_eq!(languages, &serde_json::json!(["en", "zh"]));

    // Validate model_metadata.json was written to disk
    let metadata_path = dir.path().join("model_metadata.json");
    assert!(
        metadata_path.exists(),
        "model_metadata.json should be written to disk"
    );

    // Validate it round-trips through JSON
    let json = std::fs::read_to_string(&metadata_path).unwrap();
    let parsed: ModelMetadata =
        serde_json::from_str(&json).expect("Generated model_metadata.json should be valid JSON");
    assert_eq!(parsed.model_id, "test-llama-model");
    assert_eq!(parsed.files, vec!["model-Q4_K_M.gguf"]);
}

#[test]
fn test_gguf_model_without_readme_uses_defaults() {
    let dir = TempDir::new().unwrap();

    // Only a GGUF file, no README
    write_gguf_v3_header(&dir.path().join("model.gguf"), "qwen2", 4096);

    let metadata = xybrid_sdk::metadata_gen::generate_metadata(dir.path(), "someone/qwen2-model")
        .expect("should succeed even without README");

    assert_eq!(metadata.model_id, "qwen2-model");

    // Task should be "unknown" without a model card
    assert_eq!(
        metadata.metadata.get("task").and_then(|v| v.as_str()),
        Some("unknown")
    );

    // Architecture should still be detected from GGUF header
    assert_eq!(
        metadata
            .metadata
            .get("architecture")
            .and_then(|v| v.as_str()),
        Some("qwen2")
    );
}

#[test]
fn test_no_model_files_returns_error() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("README.md"), "# No model files here").unwrap();

    let result = xybrid_sdk::metadata_gen::generate_metadata(dir.path(), "test/empty-repo");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("No model files"), "Error: {}", err);
}

#[test]
fn test_onnx_model_with_task_produces_correct_steps() {
    let dir = TempDir::new().unwrap();

    // Model card says image classification
    std::fs::write(
        dir.path().join("README.md"),
        "---\npipeline_tag: image-classification\n---\n# Vision model\n",
    )
    .unwrap();

    // Dummy ONNX file
    std::fs::write(dir.path().join("model.onnx"), b"dummy onnx data").unwrap();

    let metadata = xybrid_sdk::metadata_gen::generate_metadata(dir.path(), "test-org/vision-model")
        .expect("should succeed");

    assert_eq!(metadata.model_id, "vision-model");

    // Should have Normalize preprocessing for image classification
    assert!(
        !metadata.preprocessing.is_empty(),
        "Should have preprocessing steps"
    );

    // Should have Argmax postprocessing
    assert!(
        !metadata.postprocessing.is_empty(),
        "Should have postprocessing steps"
    );

    // Should be an ONNX template
    match &metadata.execution_template {
        xybrid_core::execution::ExecutionTemplate::Onnx { model_file } => {
            assert_eq!(model_file, "model.onnx");
        }
        other => panic!("Expected Onnx template, got {:?}", other),
    }
}
