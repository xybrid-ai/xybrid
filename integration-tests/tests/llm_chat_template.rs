//! Chat Template Integration Tests
//!
//! Tests that verify LLM chat template handling is correct.
//! These tests target the root cause of template tokens (e.g., <|im_start|>,
//! <end_of_turn>) appearing in generated text.
//!
//! Run with:
//!   cargo test -p integration-tests --features llm-llamacpp -- llm_chat_template --nocapture
//!
//! Requires gemma-3-1b model:
//!   ./integration-tests/download.sh gemma-3-1b

#![cfg(feature = "llm-llamacpp")]

use integration_tests::fixtures;
use std::collections::HashMap;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::runtime_adapter::llama_cpp::LlamaCppBackend;
use xybrid_core::runtime_adapter::llm::{LlmBackend, LlmConfig};
use xybrid_core::runtime_adapter::{ChatMessage, GenerationConfig};
use xybrid_core::template_executor::TemplateExecutor;

// =============================================================================
// Bug #1: Wrong chat template — model's GGUF template is ignored
// =============================================================================

/// Test that the model's chat template is extracted from GGUF metadata.
///
/// BUG: llama_format_chat_with_model_c() passes nullptr to
/// llama_chat_apply_template(), which defaults to "chatml" for ALL models.
/// It should use llama_model_chat_template(model) to get the actual template.
///
/// This test loads a Gemma model and verifies that llama_format_chat() uses
/// Gemma's template format (containing <start_of_turn>) instead of ChatML
/// (containing <|im_start|>).
#[test]
fn test_gemma_uses_correct_chat_template() {
    let model_name = "gemma-3-1b";

    let Some(model_dir) = fixtures::model_if_available(model_name) else {
        eprintln!(
            "Skipping {}: model not downloaded. Run: ./integration-tests/download.sh {}",
            model_name, model_name
        );
        return;
    };

    // Find the GGUF file
    let gguf_path = std::fs::read_dir(&model_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().map(|ext| ext == "gguf").unwrap_or(false))
        .expect("No .gguf file found in model directory")
        .path();

    // Load model directly
    let mut backend = LlamaCppBackend::new().expect("Failed to create backend");
    let config = LlmConfig::new(gguf_path.to_str().unwrap()).with_context_length(2048);
    backend.load(&config).expect("Failed to load model");

    // Generate a simple response using the LlmBackend trait directly
    let messages = vec![
        ChatMessage::system("You are helpful."),
        ChatMessage::user("Say hi."),
    ];
    let gen_config = GenerationConfig {
        max_tokens: 32,
        temperature: 0.1,
        ..Default::default()
    };

    let output = backend
        .generate(&messages, &gen_config)
        .expect("Generation failed");

    println!("Gemma generate() output: {:?}", output.text);

    // BUG INDICATOR: If the wrapper applies ChatML template to a Gemma model,
    // the model may generate ChatML markers as text (since it doesn't recognize
    // them as special tokens). We also check that the output is short and clean
    // — a long output with conversation turns indicates the model didn't stop.
    //
    // A properly handled Gemma model should produce a short, clean response
    // because it recognizes <end_of_turn> as its EOG token.
    assert!(
        !output.text.contains("<|im_start|>"),
        "Gemma output contains ChatML marker <|im_start|>! Wrong template is being used.\n\
         Output: {:?}",
        output.text
    );
    assert!(
        !output.text.contains("<|im_end|>"),
        "Gemma output contains ChatML marker <|im_end|>! Wrong template is being used.\n\
         Output: {:?}",
        output.text
    );

    // The finish reason should be "stop" (model generated EOG token), not "length"
    // (model ran out of max_tokens without finding a stop point).
    // With the wrong template, the model often hits max_tokens.
    assert_eq!(
        output.finish_reason, "stop",
        "Gemma should stop naturally via EOG token, not by hitting max_tokens.\n\
         finish_reason='{}' suggests the model doesn't recognize the template boundaries.\n\
         Output: {:?}",
        output.finish_reason, output.text
    );
}

// =============================================================================
// Bug #2: parse_special=false when tokenizing chat prompts
// =============================================================================

/// Test that tokenization correctly handles special tokens in chat templates.
///
/// BUG: The generate() function calls llama_tokenize() with parse_special=false.
/// This means special tokens in the chat template (like <end_of_turn> for Gemma
/// or <|im_start|> for ChatML) are tokenized as individual characters instead
/// of their special token IDs.
///
/// This test verifies that when generating with Gemma, the model receives
/// proper special tokens — evidenced by it stopping correctly at its
/// end-of-turn boundary.
#[test]
fn test_gemma_tokenization_produces_correct_eog() {
    let model_name = "gemma-3-1b";

    let Some(model_dir) = fixtures::model_if_available(model_name) else {
        eprintln!("Skipping {}: model not downloaded", model_name);
        return;
    };

    let gguf_path = std::fs::read_dir(&model_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().map(|ext| ext == "gguf").unwrap_or(false))
        .expect("No .gguf file found")
        .path();

    let mut backend = LlamaCppBackend::new().expect("Failed to create backend");
    let config = LlmConfig::new(gguf_path.to_str().unwrap()).with_context_length(2048);
    backend.load(&config).expect("Failed to load model");

    // Use a prompt that should produce a definitive, bounded response
    let messages = vec![
        ChatMessage::system("Answer in exactly one word."),
        ChatMessage::user("What color is the sky?"),
    ];
    let gen_config = GenerationConfig {
        max_tokens: 64,
        temperature: 0.0, // greedy = deterministic
        ..Default::default()
    };

    let output = backend
        .generate(&messages, &gen_config)
        .expect("Generation failed");

    println!("Gemma one-word test: {:?}", output.text);
    println!("  finish_reason: {}", output.finish_reason);
    println!("  tokens_generated: {}", output.tokens_generated);

    // With correct tokenization (parse_special=true), the model receives
    // proper <start_of_turn>/<end_of_turn> tokens and knows when to stop.
    // It should generate a short response and stop naturally.
    //
    // With broken tokenization (parse_special=false), the model sees the
    // template markers as character sequences, doesn't understand them,
    // and may generate many tokens before hitting max_tokens.
    assert!(
        output.tokens_generated < 30,
        "Model generated {} tokens for a one-word answer — it probably isn't \
         recognizing the template boundaries due to incorrect tokenization.\n\
         Output: {:?}",
        output.tokens_generated, output.text
    );

    assert_eq!(
        output.finish_reason, "stop",
        "Expected stop via EOG token, got finish_reason='{}'. \
         The model isn't stopping at its end-of-turn boundary.\n\
         Output: {:?}",
        output.finish_reason, output.text
    );
}

// =============================================================================
// Regression: Output must be clean of ALL template markers
// =============================================================================

/// Common chat template markers that should NEVER appear in generated output.
const TEMPLATE_MARKERS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "im_start",
    "im_end",
];

/// Test that Gemma model output through TemplateExecutor has no template markers.
#[test]
fn test_gemma_executor_output_clean() {
    let model_name = "gemma-3-1b";

    let Some(model_dir) = fixtures::model_if_available(model_name) else {
        eprintln!("Skipping {}: model not downloaded", model_name);
        return;
    };

    let metadata_path = model_dir.join("model_metadata.json");
    let metadata: ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(&metadata_path).expect("Failed to read model_metadata.json"),
    )
    .expect("Failed to parse model_metadata.json");

    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    let input = Envelope {
        kind: EnvelopeKind::Text("Say hello in one sentence.".to_string()),
        metadata: {
            let mut m = HashMap::new();
            m.insert("max_tokens".to_string(), "64".to_string());
            m.insert("temperature".to_string(), "0.1".to_string());
            m
        },
    };

    let output = executor
        .execute(&metadata, &input)
        .expect("LLM inference failed");

    let response = match &output.kind {
        EnvelopeKind::Text(text) => text.clone(),
        _ => panic!("Expected Text output, got {:?}", output.kind),
    };

    println!("Gemma executor response: {:?}", response);

    for marker in TEMPLATE_MARKERS {
        assert!(
            !response.contains(marker),
            "Output contains template marker '{}' — template tokens are leaking!\n\
             Full output: {:?}",
            marker,
            response
        );
    }

    assert!(!response.trim().is_empty(), "Response should not be empty");
}
