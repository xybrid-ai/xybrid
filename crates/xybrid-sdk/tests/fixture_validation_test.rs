//! Integration tests validating `inspect_and_generate()` against real fixture models.
//!
//! These tests copy fixture model files to temporary directories (to avoid writing
//! model_metadata.json into the fixture dirs) and verify the generated metadata
//! matches expected patterns.
//!
//! Run with: `cargo test -p xybrid-sdk --features onnx-inspect -- fixture_validation`

use std::path::PathBuf;
use xybrid_core::execution::{ExecutionTemplate, PostprocessingStep, PreprocessingStep};

/// Path to integration-tests/fixtures/models/ from the workspace root.
fn fixtures_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // crates/xybrid-sdk/ → ../../integration-tests/fixtures/models/
    manifest
        .join("../../integration-tests/fixtures/models")
        .canonicalize()
        .expect("fixtures/models/ directory must exist")
}

/// Copy specific files from a fixture directory to a temp directory.
fn copy_fixture_to_temp(fixture_name: &str, files: &[&str]) -> tempfile::TempDir {
    let src = fixtures_dir().join(fixture_name);
    let tmp = tempfile::TempDir::new().expect("create temp dir");
    for file in files {
        let src_file = src.join(file);
        if src_file.exists() {
            std::fs::copy(&src_file, tmp.path().join(file))
                .unwrap_or_else(|e| panic!("Failed to copy {}: {}", src_file.display(), e));
        }
    }
    tmp
}

// ---------------------------------------------------------------------------
// US-007 Acceptance Criterion 1:
// MNIST → Onnx template, image-classification with Normalize + Softmax
// ---------------------------------------------------------------------------

#[test]
fn test_mnist_fixture_generates_image_classification() {
    let tmp = copy_fixture_to_temp("mnist", &["model.onnx"]);

    let (metadata, task_inference) =
        xybrid_sdk::metadata_gen::inspect_and_generate(tmp.path(), "", None)
            .expect("inspect_and_generate should succeed for MNIST");

    // Template should be Onnx
    match &metadata.execution_template {
        ExecutionTemplate::Onnx { model_file } => {
            assert_eq!(model_file, "model.onnx");
        }
        other => panic!("Expected Onnx template, got {:?}", other),
    }

    // Task should be image-classification (inferred from 4D input + small output)
    let task = metadata
        .metadata
        .get("task")
        .and_then(|v| v.as_str())
        .expect("task metadata should exist");
    assert_eq!(task, "image-classification");

    // Preprocessing should include Normalize
    assert!(
        metadata
            .preprocessing
            .iter()
            .any(|s| matches!(s, PreprocessingStep::Normalize { .. })),
        "Expected Normalize preprocessing, got: {:?}",
        metadata.preprocessing
    );

    // Postprocessing should include Softmax
    assert!(
        metadata
            .postprocessing
            .iter()
            .any(|s| matches!(s, PostprocessingStep::Softmax { .. })),
        "Expected Softmax postprocessing, got: {:?}",
        metadata.postprocessing
    );

    // Files should include model.onnx
    assert!(metadata.files.contains(&"model.onnx".to_string()));

    // Model ID should be derived from directory name
    // (temp dir name varies, but should be sanitized)
    assert!(!metadata.model_id.is_empty());

    // Task inference should be present with Medium confidence (from output shapes)
    let ti = task_inference.expect("TaskInference should be Some for ONNX models");
    assert_eq!(ti.task, "image-classification");
}

// ---------------------------------------------------------------------------
// US-007 Acceptance Criterion 2:
// GGUF → Gguf template, empty pre/post, correct context_length
// ---------------------------------------------------------------------------

#[test]
fn test_gguf_fixture_generates_correct_template() {
    use std::io::Write;

    let tmp = tempfile::TempDir::new().unwrap();

    // Write a minimal GGUF v3 file with known architecture and context_length
    let gguf_path = tmp.path().join("test-model-Q4_K_M.gguf");
    let mut f = std::fs::File::create(&gguf_path).unwrap();

    // Magic: "GGUF"
    f.write_all(b"GGUF").unwrap();
    // Version: 3
    f.write_all(&3u32.to_le_bytes()).unwrap();
    // Tensor count: 0
    f.write_all(&0u64.to_le_bytes()).unwrap();
    // Metadata KV count: 2
    f.write_all(&2u64.to_le_bytes()).unwrap();

    // KV 1: general.architecture = "llama"
    write_gguf_string(&mut f, "general.architecture");
    f.write_all(&8u32.to_le_bytes()).unwrap(); // STRING type
    write_gguf_string(&mut f, "llama");

    // KV 2: llama.context_length = 4096
    write_gguf_string(&mut f, "llama.context_length");
    f.write_all(&4u32.to_le_bytes()).unwrap(); // UINT32 type
    f.write_all(&4096u32.to_le_bytes()).unwrap();

    drop(f);

    let (metadata, _task_inference) =
        xybrid_sdk::metadata_gen::inspect_and_generate(tmp.path(), "", None)
            .expect("inspect_and_generate should succeed for GGUF");

    // Template should be Gguf with correct context_length
    match &metadata.execution_template {
        ExecutionTemplate::Gguf {
            model_file,
            context_length,
            ..
        } => {
            assert_eq!(model_file, "test-model-Q4_K_M.gguf");
            assert_eq!(*context_length, 4096);
        }
        other => panic!("Expected Gguf template, got {:?}", other),
    }

    // GGUF models should have empty preprocessing and postprocessing
    assert!(
        metadata.preprocessing.is_empty(),
        "GGUF should have no preprocessing, got: {:?}",
        metadata.preprocessing
    );
    assert!(
        metadata.postprocessing.is_empty(),
        "GGUF should have no postprocessing, got: {:?}",
        metadata.postprocessing
    );

    // Architecture should be detected from GGUF header
    assert_eq!(
        metadata
            .metadata
            .get("architecture")
            .and_then(|v| v.as_str()),
        Some("llama")
    );

    // Quantization should be inferred from filename
    assert_eq!(
        metadata
            .metadata
            .get("quantization")
            .and_then(|v| v.as_str()),
        Some("Q4_K_M")
    );
}

fn write_gguf_string(f: &mut std::fs::File, s: &str) {
    use std::io::Write;
    f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
    f.write_all(s.as_bytes()).unwrap();
}

// ---------------------------------------------------------------------------
// US-007 Acceptance Criterion 3:
// all-MiniLM (sentence embeddings) → Tokenize + MeanPool
// ---------------------------------------------------------------------------

#[test]
fn test_all_minilm_fixture_generates_tokenize_and_meanpool() {
    let tmp = copy_fixture_to_temp(
        "all-minilm",
        &["model.onnx", "tokenizer.json", "config.json", "vocab.txt"],
    );

    let (metadata, task_inference) =
        xybrid_sdk::metadata_gen::inspect_and_generate(tmp.path(), "", None)
            .expect("inspect_and_generate should succeed for all-MiniLM");

    // Template should be Onnx
    match &metadata.execution_template {
        ExecutionTemplate::Onnx { model_file } => {
            assert_eq!(model_file, "model.onnx");
        }
        other => panic!("Expected Onnx template, got {:?}", other),
    }

    // Preprocessing should include Tokenize
    let has_tokenize = metadata
        .preprocessing
        .iter()
        .any(|s| matches!(s, PreprocessingStep::Tokenize { .. }));
    assert!(
        has_tokenize,
        "Expected Tokenize preprocessing, got: {:?}",
        metadata.preprocessing
    );

    // Verify tokenizer parameters when available
    for step in &metadata.preprocessing {
        if let PreprocessingStep::Tokenize {
            vocab_file,
            max_length,
            ..
        } = step
        {
            // Should use tokenizer.json (since it exists in the directory)
            assert!(
                vocab_file == "tokenizer.json" || vocab_file == "vocab.txt",
                "Unexpected vocab_file: {}",
                vocab_file
            );
            // max_length should come from config.json max_position_embeddings (512)
            assert_eq!(
                *max_length,
                Some(512),
                "max_length should be 512 from config.json max_position_embeddings"
            );
        }
    }

    // Postprocessing should include MeanPool (for feature-extraction/sentence-similarity)
    let has_meanpool = metadata
        .postprocessing
        .iter()
        .any(|s| matches!(s, PostprocessingStep::MeanPool { .. }));
    assert!(
        has_meanpool,
        "Expected MeanPool postprocessing, got: {:?}",
        metadata.postprocessing
    );

    // Files should include all copied files
    assert!(metadata.files.contains(&"model.onnx".to_string()));
    assert!(metadata.files.contains(&"tokenizer.json".to_string()));
    assert!(metadata.files.contains(&"config.json".to_string()));

    // Task inference should be present
    let ti = task_inference.expect("TaskInference should be Some for ONNX models");
    // Should be feature-extraction (NLP model with 3D output → embeddings)
    assert!(
        ti.task == "feature-extraction" || ti.task == "sentence-similarity",
        "Expected feature-extraction or sentence-similarity task, got: {}",
        ti.task
    );
}

// ---------------------------------------------------------------------------
// US-007 Acceptance Criterion 4:
// Integration test comparing generate_metadata() output for MNIST
// ---------------------------------------------------------------------------

#[test]
fn test_generate_metadata_writes_valid_json_for_mnist() {
    use xybrid_core::execution::ModelMetadata;

    let tmp = copy_fixture_to_temp("mnist", &["model.onnx"]);

    // generate_metadata writes model_metadata.json to disk
    let (metadata, _) = xybrid_sdk::metadata_gen::generate_metadata(tmp.path(), "")
        .expect("generate_metadata should succeed for MNIST");

    // Verify model_metadata.json was written
    let metadata_path = tmp.path().join("model_metadata.json");
    assert!(
        metadata_path.exists(),
        "model_metadata.json should be written"
    );

    // Round-trip: read back and parse
    let json = std::fs::read_to_string(&metadata_path).unwrap();
    let parsed: ModelMetadata =
        serde_json::from_str(&json).expect("Written model_metadata.json should be valid JSON");

    // Verify it matches the returned metadata
    assert_eq!(parsed.model_id, metadata.model_id);
    assert_eq!(parsed.version, metadata.version);
    assert_eq!(parsed.files, metadata.files);
    assert_eq!(parsed.preprocessing.len(), metadata.preprocessing.len());
    assert_eq!(parsed.postprocessing.len(), metadata.postprocessing.len());
}

// ---------------------------------------------------------------------------
// US-007 Acceptance Criterion 4 (continued):
// Verify generate_metadata() on GGUF produces expected output
// ---------------------------------------------------------------------------

#[test]
fn test_generate_metadata_writes_valid_json_for_gguf() {
    use std::io::Write;
    use xybrid_core::execution::ModelMetadata;

    let tmp = tempfile::TempDir::new().unwrap();

    // Synthetic GGUF
    let gguf_path = tmp.path().join("model-Q8_0.gguf");
    let mut f = std::fs::File::create(&gguf_path).unwrap();
    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&2u64.to_le_bytes()).unwrap();
    write_gguf_string(&mut f, "general.architecture");
    f.write_all(&8u32.to_le_bytes()).unwrap();
    write_gguf_string(&mut f, "qwen2");
    write_gguf_string(&mut f, "qwen2.context_length");
    f.write_all(&4u32.to_le_bytes()).unwrap();
    f.write_all(&32768u32.to_le_bytes()).unwrap();
    drop(f);

    let (metadata, _) = xybrid_sdk::metadata_gen::generate_metadata(tmp.path(), "")
        .expect("generate_metadata should succeed for GGUF");

    // Verify file was written and round-trips
    let metadata_path = tmp.path().join("model_metadata.json");
    assert!(metadata_path.exists());
    let parsed: ModelMetadata =
        serde_json::from_str(&std::fs::read_to_string(&metadata_path).unwrap())
            .expect("Written model_metadata.json should parse");

    match &parsed.execution_template {
        ExecutionTemplate::Gguf { context_length, .. } => {
            assert_eq!(*context_length, 32768);
        }
        other => panic!("Expected Gguf template, got {:?}", other),
    }

    assert!(parsed.preprocessing.is_empty());
    assert!(parsed.postprocessing.is_empty());
    assert_eq!(parsed.model_id, metadata.model_id);
}

// ---------------------------------------------------------------------------
// Additional: Verify model_id sanitization from directory name
// ---------------------------------------------------------------------------

#[test]
fn test_model_id_derived_from_directory_name() {
    // Copy mnist model.onnx to a temp dir with a specific name
    let parent = tempfile::TempDir::new().unwrap();
    let model_dir = parent.path().join("My Custom_Model.v2");
    std::fs::create_dir_all(&model_dir).unwrap();

    let src = fixtures_dir().join("mnist/model.onnx");
    std::fs::copy(&src, model_dir.join("model.onnx")).unwrap();

    let (metadata, _) = xybrid_sdk::metadata_gen::inspect_and_generate(&model_dir, "", None)
        .expect("inspect_and_generate should succeed");

    // model_id should be sanitized: lowercase, kebab-case
    assert_eq!(metadata.model_id, "my-custom-model.v2");
}

// ---------------------------------------------------------------------------
// Additional: Verify --model-id override works
// ---------------------------------------------------------------------------

#[test]
fn test_model_id_override() {
    let tmp = copy_fixture_to_temp("mnist", &["model.onnx"]);

    let (metadata, _) =
        xybrid_sdk::metadata_gen::inspect_and_generate(tmp.path(), "", Some("custom-id"))
            .expect("inspect_and_generate with model_id override should succeed");

    assert_eq!(metadata.model_id, "custom-id");
}

// ---------------------------------------------------------------------------
// Additional: Verify no model files returns error
// ---------------------------------------------------------------------------

#[test]
fn test_empty_directory_returns_error() {
    let tmp = tempfile::TempDir::new().unwrap();
    // Create a non-model file
    std::fs::write(tmp.path().join("readme.txt"), "not a model").unwrap();

    let result = xybrid_sdk::metadata_gen::inspect_and_generate(tmp.path(), "", None);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("No model files"),
        "Expected 'No model files' error, got: {}",
        err
    );
}
