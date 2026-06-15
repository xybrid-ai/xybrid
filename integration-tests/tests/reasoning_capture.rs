//! Chain-of-thought (`reasoning_content`) capture — end-to-end.
//!
//! Exercises the full path a thinking model takes: the llama.cpp backend
//! parses `<think>...</think>` blocks, the answer text is stripped clean, and
//! the reasoning is surfaced on the envelope metadata under the
//! `reasoning_content` key (the same value the SDK exposes via
//! `InferenceResult::reasoning_content()`).
//!
//! Requires the `llm-llamacpp` feature and a thinking-capable model.
//!
//! Run with:
//!   cargo test -p integration-tests --features llm-llamacpp reasoning_capture -- --nocapture
//!
//! Download model first:
//!   ./integration-tests/download.sh qwen3.5-0.8b

#![cfg(feature = "llm-llamacpp")]

use integration_tests::fixtures;
use std::collections::HashMap;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::template_executor::TemplateExecutor;

/// A thinking model's answer text must never contain `<think>` markers, and
/// its reasoning must be captured separately on the envelope metadata.
#[test]
fn test_reasoning_content_captured_and_stripped() {
    let model_name = "qwen3.5-0.8b";

    // Skip cleanly if the model isn't downloaded — matches the rest of the
    // integration suite, so CI without fixtures stays green.
    let Some(model_dir) = fixtures::model_if_available(model_name) else {
        eprintln!(
            "Skipping {}: model not downloaded. Run: ./integration-tests/download.sh {}",
            model_name, model_name
        );
        return;
    };

    let metadata_path = model_dir.join("model_metadata.json");
    let metadata_content =
        std::fs::read_to_string(&metadata_path).expect("Failed to read model_metadata.json");
    let metadata: ModelMetadata =
        serde_json::from_str(&metadata_content).expect("Failed to parse model_metadata.json");
    assert_eq!(metadata.model_id, model_name);

    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    // A small reasoning task — reliably triggers a `<think>` block on Qwen 3.5.
    // Give decode enough room for reasoning + answer.
    let prompt = "If a train travels 60 km in 45 minutes, what is its speed in km/h? \
                  Reason step by step, then give the final answer.";
    let input = Envelope {
        kind: EnvelopeKind::Text(prompt.to_string()),
        metadata: {
            let mut m = HashMap::new();
            m.insert("max_tokens".to_string(), "256".to_string());
            m
        },
    };

    let output = executor
        .execute(&metadata, &input, None)
        .expect("LLM inference failed");

    let EnvelopeKind::Text(answer) = &output.kind else {
        panic!("Expected Text output, got {:?}", output.kind);
    };

    let reasoning = output.metadata.get("reasoning_content");
    println!("Prompt:    {prompt}");
    println!("Answer:    {answer}");
    println!("Reasoning: {reasoning:?}");

    // 1. Stripping invariant — ALWAYS holds, regardless of whether the model
    //    chose to think. The answer the user sees is free of reasoning markup.
    assert!(
        !answer.contains("<think>") && !answer.contains("</think>"),
        "answer text must not leak <think> markers: {answer:?}"
    );

    // 2. Capture — Qwen 3.5 runs in thinking mode by default, so a reasoning
    //    prompt should yield a captured, non-empty reasoning block that is
    //    distinct from the answer text.
    let reasoning = reasoning.expect(
        "expected reasoning_content metadata from a thinking model; \
         none was captured (did the model emit <think> blocks?)",
    );
    assert!(
        !reasoning.is_empty(),
        "captured reasoning_content should be non-empty"
    );
    assert!(
        !reasoning.contains("<think>") && !reasoning.contains("</think>"),
        "captured reasoning should hold inner text only, not the tags: {reasoning:?}"
    );
    assert!(
        !answer.is_empty(),
        "answer text should be non-empty alongside the reasoning"
    );
}
