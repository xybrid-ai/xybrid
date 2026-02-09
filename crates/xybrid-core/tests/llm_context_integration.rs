//! Integration tests for ConversationContext + LLM inference.
//!
//! These tests verify that `execute_with_context` and `execute_streaming_with_context`
//! produce valid output when used with a real GGUF model.
//!
//! Regression: Previously, `execute_with_context` double-formatted the chat template
//! (once in Rust, again in llama.cpp), causing heap corruption / SIGSEGV.
//!
//! Run with:
//!   cargo test -p xybrid-core --test llm_context_integration --features llm-llamacpp

use std::path::PathBuf;
use xybrid_core::conversation::ConversationContext;
use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
use xybrid_core::testing::model_fixtures;

/// Helper to load model metadata from a fixture directory.
fn load_metadata(model_dir: &PathBuf) -> ModelMetadata {
    let metadata_path = model_dir.join("model_metadata.json");
    let content = std::fs::read_to_string(&metadata_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", metadata_path.display(), e));
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", metadata_path.display(), e))
}

/// Regression test: execute_with_context must not crash with conversation history.
///
/// This was the primary crash scenario: system prompt + user message passed through
/// ConversationContext caused double chat template formatting → heap corruption.
#[test]
fn test_execute_with_context_no_crash() {
    if !model_fixtures::model_available("gemma-3-1b") {
        eprintln!("Skipping: gemma-3-1b model not downloaded");
        return;
    }

    let model_dir = model_fixtures::require_model("gemma-3-1b");
    let metadata = load_metadata(&model_dir);
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Build conversation context (matches the crash reproduction from the bug report)
    let ctx = ConversationContext::new()
        .with_system(
            Envelope::new(EnvelopeKind::Text(
                "You are a helpful NPC assistant in a fantasy game. Keep responses under 50 words."
                    .to_string(),
            ))
            .with_role(MessageRole::System),
        )
        .with_max_history_len(20);

    let input = Envelope::new(EnvelopeKind::Text(
        "What is the capital of France? One word.".to_string(),
    ))
    .with_role(MessageRole::User);

    // This previously crashed with SIGSEGV or heap corruption
    let result = executor.execute_with_context(&metadata, &input, &ctx);

    assert!(result.is_ok(), "execute_with_context failed: {:?}", result.err());

    let output = result.unwrap();
    assert!(output.is_assistant_message());

    if let EnvelopeKind::Text(text) = &output.kind {
        assert!(!text.is_empty(), "Response should not be empty");
        // Sanity check: response should not contain raw chat template markers
        // (which would indicate the double-formatting bug)
        assert!(
            !text.contains("<|im_start|>"),
            "Response contains raw ChatML markers — double-formatting may still be occurring: {}",
            text
        );
        assert!(
            !text.contains("<start_of_turn>"),
            "Response contains raw Gemma markers — double-formatting may still be occurring: {}",
            text
        );
        println!("Response: {}", text);
    } else {
        panic!("Expected text output, got: {:?}", output.kind);
    }
}

/// Regression test: execute_with_context with multi-turn history must not crash.
///
/// Tests the scenario with system prompt + conversation history + new user message.
#[test]
fn test_execute_with_context_multi_turn_no_crash() {
    if !model_fixtures::model_available("gemma-3-1b") {
        eprintln!("Skipping: gemma-3-1b model not downloaded");
        return;
    }

    let model_dir = model_fixtures::require_model("gemma-3-1b");
    let metadata = load_metadata(&model_dir);
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Build multi-turn context
    let mut ctx = ConversationContext::new()
        .with_system(
            Envelope::new(EnvelopeKind::Text(
                "You are a concise assistant.".to_string(),
            ))
            .with_role(MessageRole::System),
        )
        .with_max_history_len(20);

    // Push a previous turn
    ctx.push(
        Envelope::new(EnvelopeKind::Text("What is 2+2?".to_string()))
            .with_role(MessageRole::User),
    );
    ctx.push(
        Envelope::new(EnvelopeKind::Text("4".to_string()))
            .with_role(MessageRole::Assistant),
    );

    // New user message
    let input = Envelope::new(EnvelopeKind::Text("And 3+3?".to_string()))
        .with_role(MessageRole::User);

    let result = executor.execute_with_context(&metadata, &input, &ctx);

    assert!(
        result.is_ok(),
        "Multi-turn execute_with_context failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    if let EnvelopeKind::Text(text) = &output.kind {
        assert!(!text.is_empty());
        println!("Multi-turn response: {}", text);
    }
}

/// Regression test: streaming with context must not SIGSEGV.
#[test]
fn test_execute_streaming_with_context_no_crash() {
    if !model_fixtures::model_available("gemma-3-1b") {
        eprintln!("Skipping: gemma-3-1b model not downloaded");
        return;
    }

    let model_dir = model_fixtures::require_model("gemma-3-1b");
    let metadata = load_metadata(&model_dir);
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    let ctx = ConversationContext::new()
        .with_system(
            Envelope::new(EnvelopeKind::Text(
                "You are a helpful assistant. Keep responses brief.".to_string(),
            ))
            .with_role(MessageRole::System),
        )
        .with_max_history_len(20);

    let input = Envelope::new(EnvelopeKind::Text("Say hello.".to_string()))
        .with_role(MessageRole::User);

    let mut tokens_received = 0u32;

    // This previously caused SIGSEGV in llama_generate_streaming_c
    let result = executor.execute_streaming_with_context(
        &metadata,
        &input,
        &ctx,
        Box::new(|_token| {
            tokens_received += 1;
            Ok(())
        }),
    );

    assert!(
        result.is_ok(),
        "Streaming with context failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    assert!(output.is_assistant_message());

    if let EnvelopeKind::Text(text) = &output.kind {
        assert!(!text.is_empty());
        assert!(tokens_received > 0, "Should have received streaming tokens");
        println!(
            "Streaming response ({} tokens): {}",
            tokens_received, text
        );
    }
}
