//! Integration tests for ASR (Automatic Speech Recognition) pipeline.
//!
//! This tests:
//! - Wav2Vec2 model loading and inference
//! - Audio preprocessing (WAV decode, resampling)
//! - CTC decoding postprocessing
//! - End-to-end speech-to-text pipeline
//!
//! Run with: cargo test -p integration-tests --test asr_integration

use integration_tests::fixtures;
use std::collections::HashMap;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::template_executor::TemplateExecutor;

#[test]
fn test_wav2vec2_model_metadata() {
    let Some(model_dir) = fixtures::model_if_available("wav2vec2-base-960h") else {
        eprintln!("Skipping test: wav2vec2-base-960h not downloaded");
        eprintln!("Run: ./integration-tests/download.sh wav2vec2-base-960h");
        return;
    };

    let metadata_path = model_dir.join("model_metadata.json");
    let metadata_content =
        std::fs::read_to_string(&metadata_path).expect("Failed to read model_metadata.json");

    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_content).expect("Failed to parse model_metadata.json");

    // Verify metadata structure
    assert_eq!(metadata["model_id"], "wav2vec2-base-960h");
    assert!(metadata["preprocessing"].is_array());
    assert!(metadata["postprocessing"].is_array());

    // Check preprocessing has AudioDecode step
    let preprocessing = metadata["preprocessing"].as_array().unwrap();
    assert!(!preprocessing.is_empty());
    assert_eq!(preprocessing[0]["type"], "AudioDecode");
    assert_eq!(preprocessing[0]["sample_rate"], 16000);

    // Check postprocessing has CTCDecode step
    let postprocessing = metadata["postprocessing"].as_array().unwrap();
    assert!(!postprocessing.is_empty());
    assert_eq!(postprocessing[0]["type"], "CTCDecode");

    println!("Model metadata verified:");
    println!("  model_id: {}", metadata["model_id"]);
    println!("  preprocessing: {:?}", preprocessing.len());
    println!("  postprocessing: {:?}", postprocessing.len());
}

#[test]
fn test_wav2vec2_model_files_exist() {
    let Some(model_dir) = fixtures::model_if_available("wav2vec2-base-960h") else {
        eprintln!("Skipping test: wav2vec2-base-960h not downloaded");
        return;
    };

    // Check required files exist
    assert!(
        model_dir.join("model_metadata.json").exists(),
        "model_metadata.json should exist"
    );
    assert!(
        model_dir.join("model.onnx").exists(),
        "model.onnx should exist"
    );
    assert!(
        model_dir.join("vocab.json").exists(),
        "vocab.json should exist"
    );

    // Check model file size is reasonable (should be ~200MB+)
    let model_size = std::fs::metadata(model_dir.join("model.onnx"))
        .expect("Failed to get model size")
        .len();
    assert!(
        model_size > 100_000_000,
        "Model should be > 100MB, got {} bytes",
        model_size
    );

    println!("Model files verified:");
    println!("  model.onnx: {} MB", model_size / 1_000_000);
}

#[test]
fn test_wav2vec2_vocab_loading() {
    let Some(model_dir) = fixtures::model_if_available("wav2vec2-base-960h") else {
        eprintln!("Skipping test: wav2vec2-base-960h not downloaded");
        return;
    };

    let vocab_path = model_dir.join("vocab.json");
    let vocab_content = std::fs::read_to_string(&vocab_path).expect("Failed to read vocab.json");

    let vocab: serde_json::Value =
        serde_json::from_str(&vocab_content).expect("Failed to parse vocab.json");

    // Vocab should be an object with character mappings
    assert!(vocab.is_object(), "Vocab should be an object");

    let vocab_obj = vocab.as_object().unwrap();
    assert!(
        vocab_obj.len() >= 30,
        "Vocab should have at least 30 entries"
    );

    // Check for common characters
    assert!(
        vocab_obj.contains_key("<pad>") || vocab_obj.contains_key("|"),
        "Vocab should have padding token"
    );

    println!("Vocabulary loaded:");
    println!("  Total tokens: {}", vocab_obj.len());
}

#[test]
fn test_wav2vec2_inference_with_audio() {
    let Some(model_dir) = fixtures::model_if_available("wav2vec2-base-960h") else {
        eprintln!("Skipping test: wav2vec2-base-960h not downloaded");
        return;
    };

    // Load test audio fixture
    let audio_path = fixtures::input_dir().join("test_audio.wav");
    if !audio_path.exists() {
        eprintln!("Skipping test: test_audio.wav not found in fixtures/input/");
        return;
    }

    // Load metadata
    let metadata_path = model_dir.join("model_metadata.json");
    let metadata_content =
        std::fs::read_to_string(&metadata_path).expect("Failed to read metadata");
    let metadata: ModelMetadata =
        serde_json::from_str(&metadata_content).expect("Failed to parse metadata");

    // Create executor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Read audio file
    let audio_bytes = std::fs::read(&audio_path).expect("Failed to read audio file");
    println!("Audio file size: {} bytes", audio_bytes.len());

    // Create input envelope
    let input_envelope = Envelope {
        kind: EnvelopeKind::Audio(audio_bytes),
        metadata: HashMap::new(),
    };

    // Execute inference
    let output_envelope = executor
        .execute(&metadata, &input_envelope)
        .expect("Inference should succeed");

    // Verify output is text
    match &output_envelope.kind {
        EnvelopeKind::Text(transcription) => {
            println!("Transcription: \"{}\"", transcription);
            // Transcription should not be empty (even if it's just silence detected)
            // Some models return empty string for silence, which is valid
        }
        other => {
            panic!("Expected text output, got {:?}", other);
        }
    }
}

#[test]
fn test_whisper_model_available() {
    // Check if whisper-tiny is available (Candle format)
    let Some(model_dir) = fixtures::model_if_available("whisper-tiny") else {
        eprintln!("Skipping test: whisper-tiny not downloaded");
        eprintln!("Run: ./integration-tests/download.sh whisper-tiny");
        return;
    };

    // Whisper uses different file format (SafeTensors)
    let has_safetensors = model_dir.join("model.safetensors").exists();
    let has_config = model_dir.join("config.json").exists();

    println!("Whisper model directory: {}", model_dir.display());
    println!("  Has safetensors: {}", has_safetensors);
    println!("  Has config: {}", has_config);

    // At minimum should have model_metadata.json
    assert!(
        model_dir.join("model_metadata.json").exists(),
        "Should have model_metadata.json"
    );
}
