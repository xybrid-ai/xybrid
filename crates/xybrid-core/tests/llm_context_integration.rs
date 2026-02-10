//! Integration tests for ConversationContext + LLM inference.
//!
//! These tests verify that `execute_with_context` and `execute_streaming_with_context`
//! produce valid output when used with a real GGUF model.
//!
//! Regressions covered:
//! - Double chat template formatting → heap corruption / SIGSEGV
//! - Prompt size exceeding context window → heap corruption (no bounds check)
//! - Batch buffer overflow when input > 512 tokens → heap corruption
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

/// Regression test: prompt exceeding context window must return error, not crash.
///
/// Previously, there was no validation that input tokens fit within the KV cache
/// context window (n_ctx). Oversized prompts caused heap corruption / SIGSEGV.
/// Additionally, the C batch was fixed at 512 tokens, causing overflow for any
/// prompt > 512 tokens.
#[test]
fn test_oversized_prompt_returns_error_not_crash() {
    if !model_fixtures::model_available("gemma-3-1b") {
        eprintln!("Skipping: gemma-3-1b model not downloaded");
        return;
    }

    let model_dir = model_fixtures::require_model("gemma-3-1b");
    let metadata = load_metadata(&model_dir);
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Build a conversation context with a massive history that will exceed context window.
    // gemma-3-1b has context_length 4096 tokens ≈ ~16K chars.
    // We generate ~30K chars of history to guarantee overflow.
    let mut ctx = ConversationContext::new()
        .with_system(
            Envelope::new(EnvelopeKind::Text(
                "You are a helpful assistant.".to_string(),
            ))
            .with_role(MessageRole::System),
        )
        .with_max_history_len(500); // Allow large history

    // Fill history with long messages to exceed context window
    for i in 0..100 {
        ctx.push(
            Envelope::new(EnvelopeKind::Text(format!(
                "This is user message number {} with padding text to consume tokens. \
                 The quick brown fox jumps over the lazy dog. {}",
                i,
                "Lorem ipsum dolor sit amet. ".repeat(10)
            )))
            .with_role(MessageRole::User),
        );
        ctx.push(
            Envelope::new(EnvelopeKind::Text(format!(
                "Response to message {}. Here is some additional text to consume tokens. \
                 The answer to your question involves many considerations. {}",
                i,
                "Sed do eiusmod tempor incididunt. ".repeat(10)
            )))
            .with_role(MessageRole::Assistant),
        );
    }

    let input = Envelope::new(EnvelopeKind::Text(
        "What was my first question?".to_string(),
    ))
    .with_role(MessageRole::User);

    // This previously caused SIGSEGV — now it should return a clean error
    let result = executor.execute_with_context(&metadata, &input, &ctx);

    assert!(
        result.is_err(),
        "Oversized prompt should return an error, not succeed or crash"
    );

    let err = result.unwrap_err();
    let err_msg = format!("{}", err);
    println!("Got expected error: {}", err_msg);
    assert!(
        err_msg.contains("too long") || err_msg.contains("exceeds context window"),
        "Error should mention input being too long, got: {}",
        err_msg
    );
}

/// Regression test: moderately large prompt (>512 tokens but within context window) must work.
///
/// Previously, llama_batch_init(512, ...) in the C layer caused heap corruption
/// when the input prompt had more than 512 tokens. The batch now dynamically sizes
/// to fit the input.
#[test]
fn test_moderate_prompt_over_512_tokens_works() {
    if !model_fixtures::model_available("gemma-3-1b") {
        eprintln!("Skipping: gemma-3-1b model not downloaded");
        return;
    }

    let model_dir = model_fixtures::require_model("gemma-3-1b");
    let metadata = load_metadata(&model_dir);
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // Build a conversation with enough history to exceed 512 tokens but stay
    // well within the 4096 context window (~1000-2000 tokens).
    let mut ctx = ConversationContext::new()
        .with_system(
            Envelope::new(EnvelopeKind::Text(
                "You are a helpful assistant that gives concise answers.".to_string(),
            ))
            .with_role(MessageRole::System),
        )
        .with_max_history_len(50);

    // ~8 turns of moderate length should produce ~800-1200 tokens total
    for i in 0..8 {
        ctx.push(
            Envelope::new(EnvelopeKind::Text(format!(
                "Tell me an interesting fact number {}. I want to learn something new \
                 about science, history, or nature. Please be detailed.",
                i + 1
            )))
            .with_role(MessageRole::User),
        );
        ctx.push(
            Envelope::new(EnvelopeKind::Text(format!(
                "Here is fact number {}: The mitochondria is the powerhouse of the cell. \
                 This organelle generates most of the cell's supply of adenosine triphosphate, \
                 used as a source of chemical energy. It was first discovered in the 1840s.",
                i + 1
            )))
            .with_role(MessageRole::Assistant),
        );
    }

    let input = Envelope::new(EnvelopeKind::Text(
        "Summarize the facts you shared. One sentence.".to_string(),
    ))
    .with_role(MessageRole::User);

    // This previously crashed when input > 512 tokens due to batch overflow
    let result = executor.execute_with_context(&metadata, &input, &ctx);

    assert!(
        result.is_ok(),
        "Moderate prompt (>512 tokens, <4096) should succeed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    if let EnvelopeKind::Text(text) = &output.kind {
        assert!(!text.is_empty(), "Response should not be empty");
        println!("Moderate prompt response: {}", text);
    }
}
