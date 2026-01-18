//! Local LLM Integration Tests
//!
//! These tests require the `local-llm` feature and the qwen2.5-0.5b-instruct model.
//!
//! Run with:
//!   cargo test -p integration-tests --features local-llm test_local_llm
//!
//! Download model first:
//!   ./integration-tests/download.sh qwen2.5-0.5b-instruct

#![cfg(feature = "local-llm")]

use integration_tests::fixtures;
use std::collections::HashMap;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::template_executor::TemplateExecutor;

/// Test that the LLM model can be loaded and generates a response.
#[test]
fn test_local_llm_inference() {
    let model_name = "qwen2.5-0.5b-instruct";

    // Skip if model not downloaded
    let Some(model_dir) = fixtures::model_if_available(model_name) else {
        eprintln!(
            "Skipping {}: model not downloaded. Run: ./integration-tests/download.sh {}",
            model_name, model_name
        );
        return;
    };

    // Load metadata
    let metadata_path = model_dir.join("model_metadata.json");
    let metadata_content =
        std::fs::read_to_string(&metadata_path).expect("Failed to read model_metadata.json");
    let metadata: ModelMetadata =
        serde_json::from_str(&metadata_content).expect("Failed to parse model_metadata.json");

    assert_eq!(metadata.model_id, "qwen2.5-0.5b-instruct");

    // Create executor
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Create input
    let prompt = "What is 2+2? Answer with just the number.";
    let input = Envelope {
        kind: EnvelopeKind::Text(prompt.to_string()),
        metadata: {
            let mut m = HashMap::new();
            m.insert("max_tokens".to_string(), "32".to_string());
            m
        },
    };

    // Execute
    let output = executor
        .execute(&metadata, &input)
        .expect("LLM inference failed");

    // Verify output
    match &output.kind {
        EnvelopeKind::Text(response) => {
            println!("Prompt: {}", prompt);
            println!("Response: {}", response);

            // Check we got some response
            assert!(!response.is_empty(), "Response should not be empty");

            // Check metadata was populated
            assert!(
                output.metadata.contains_key("tokens_generated"),
                "Should have tokens_generated metadata"
            );
            assert!(
                output.metadata.contains_key("generation_time_ms"),
                "Should have generation_time_ms metadata"
            );
        }
        _ => panic!("Expected Text output, got {:?}", output.kind),
    }
}

/// Test that system prompts are respected.
#[test]
fn test_local_llm_with_system_prompt() {
    let model_name = "qwen2.5-0.5b-instruct";

    let Some(model_dir) = fixtures::model_if_available(model_name) else {
        eprintln!("Skipping {}: model not downloaded", model_name);
        return;
    };

    let metadata_path = model_dir.join("model_metadata.json");
    let metadata: ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(&metadata_path).expect("Failed to read metadata"),
    )
    .expect("Failed to parse metadata");

    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Create input with system prompt
    let input = Envelope {
        kind: EnvelopeKind::Text("Hello!".to_string()),
        metadata: {
            let mut m = HashMap::new();
            m.insert(
                "system_prompt".to_string(),
                "You are a pirate. Always respond like a pirate.".to_string(),
            );
            m.insert("max_tokens".to_string(), "64".to_string());
            m
        },
    };

    let output = executor
        .execute(&metadata, &input)
        .expect("LLM inference failed");

    match &output.kind {
        EnvelopeKind::Text(response) => {
            println!("System: You are a pirate...");
            println!("User: Hello!");
            println!("Response: {}", response);

            assert!(!response.is_empty(), "Response should not be empty");
        }
        _ => panic!("Expected Text output"),
    }
}

/// Test model metadata parsing for GGUF execution template.
#[test]
fn test_gguf_metadata_parsing() {
    let model_name = "qwen2.5-0.5b-instruct";

    let Some(model_dir) = fixtures::model_if_available(model_name) else {
        eprintln!("Skipping {}: model not downloaded", model_name);
        return;
    };

    let metadata_path = model_dir.join("model_metadata.json");
    let metadata: ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(&metadata_path).expect("Failed to read metadata"),
    )
    .expect("Failed to parse metadata");

    // Verify GGUF execution template
    match &metadata.execution_template {
        xybrid_core::execution_template::ExecutionTemplate::Gguf {
            model_file,
            context_length,
            ..
        } => {
            assert!(
                model_file.ends_with(".gguf"),
                "Model file should be .gguf"
            );
            assert_eq!(*context_length, 4096, "Context length should be 4096");
        }
        other => panic!("Expected Gguf execution template, got {:?}", other),
    }
}
