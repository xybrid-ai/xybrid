//! Reasoning-model budget regression coverage.
//!
//! Pins the #355 invariant that a thinking model with too little shared
//! reasoning-plus-answer budget reports an empty answer with captured
//! reasoning, while an ample budget still produces visible output.
//!
//! Run with:
//!   cargo test -p integration-tests --features llm-llamacpp reasoning_budget -- --nocapture
//!
//! Download model first:
//!   ./integration-tests/download.sh qwen3-0.6b

#![cfg(feature = "llm-llamacpp")]

use integration_tests::fixtures;
use std::collections::HashMap;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::runtime_adapter::llama_cpp::LlamaCppBackend;
use xybrid_core::runtime_adapter::llm::LlmBackend;
use xybrid_core::runtime_adapter::{ChatMessage, GenerationConfig, LlmConfig, PartialToken};
use xybrid_core::template_executor::TemplateExecutor;

const ROOK_PROMPT: &str = "How many ways can you place 3 rooks on a 3x3 chessboard so that none attack each other? Think through every case carefully before answering.";

#[test]
fn reasoning_budget_exhaustion_recovers_with_ample_budget() {
    let model_name = "qwen3-0.6b";
    let Some(model_dir) = fixtures::model_if_available(model_name) else {
        eprintln!(
            "Skipping {}: model not downloaded. Run: ./integration-tests/download.sh {}",
            model_name, model_name
        );
        return;
    };

    let metadata_path = model_dir.join("model_metadata.json");
    let metadata_content =
        std::fs::read_to_string(&metadata_path).expect("failed to read model_metadata.json");
    let metadata: ModelMetadata =
        serde_json::from_str(&metadata_content).expect("failed to parse model_metadata.json");
    assert_eq!(metadata.model_id, model_name);

    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());

    let failure_input = Envelope {
        kind: EnvelopeKind::Text(ROOK_PROMPT.to_string()),
        metadata: HashMap::from([(String::from("max_tokens"), String::from("32"))]),
    };
    let exhausted = executor
        .execute(&metadata, &failure_input, None)
        .expect("LLM inference failed for tiny reasoning budget");

    let EnvelopeKind::Text(answer) = &exhausted.kind else {
        panic!("Expected Text output, got {:?}", exhausted.kind);
    };
    assert!(
        answer.trim().is_empty(),
        "tiny budget should leave no answer: {answer:?}"
    );
    assert_eq!(
        exhausted.metadata.get("finish_reason").map(String::as_str),
        Some("length")
    );
    assert!(
        exhausted
            .metadata
            .get("reasoning_content")
            .is_some_and(|reasoning| !reasoning.trim().is_empty()),
        "tiny budget should preserve partial reasoning"
    );

    let recovery_input = Envelope {
        kind: EnvelopeKind::Text("What is 2+2? Answer with just the number.".to_string()),
        metadata: HashMap::from([(String::from("max_tokens"), String::from("2048"))]),
    };
    let recovered = executor
        .execute(&metadata, &recovery_input, None)
        .expect("LLM inference failed for ample reasoning budget");

    let EnvelopeKind::Text(answer) = recovered.kind else {
        panic!("Expected Text output, got non-text recovery output");
    };
    assert!(
        !answer.trim().is_empty(),
        "ample budget should produce an answer"
    );
}

/// A thinking model's output starts inside the `<think>` channel primed into
/// its prompt, and the stream filter suppresses the thought. When the budget
/// ends the turn mid-thought, every delta is suppressed and the terminal token
/// is the only chunk left; it must still reach the callback, carrying
/// `length`. The grammar keeps the single budgeted token inside the thought.
///
/// No workflow downloads qwen3-0.6b, so this runs only where the fixture is
/// staged. CI runs the same adapter path through its FunctionGemma twin,
/// `a_fully_suppressed_stream_still_sends_its_terminal_token` in
/// `llamacpp_release.rs`: the priming comes from `LlmConfig::reasoning`, not
/// from the model.
#[test]
fn reasoning_cut_off_mid_thought_still_sends_the_terminal_token() {
    let model_name = "qwen3-0.6b";
    let Some(model_dir) = fixtures::model_for_test(model_name) else {
        eprintln!(
            "Skipping {}: model not downloaded. Run: ./integration-tests/download.sh {}",
            model_name, model_name
        );
        return;
    };
    let metadata: ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("model_metadata.json"))
            .expect("failed to read model_metadata.json"),
    )
    .expect("failed to parse model_metadata.json");
    assert_eq!(
        metadata
            .metadata
            .get("reasoning")
            .and_then(|value| value.as_bool()),
        Some(true),
        "the fixture must be configured as a thinking model"
    );

    let model_file = model_dir.join("Qwen3-0.6B-Q8_0.gguf");
    let mut backend = LlamaCppBackend::new().expect("llama.cpp backend initializes");
    backend
        .load(
            &LlmConfig::new(model_file.to_str().expect("fixture path is UTF-8"))
                .with_context_length(512)
                .with_reasoning(true),
        )
        .expect("qwen3-0.6b loads");

    let mut chunks: Vec<PartialToken> = Vec::new();
    let output = backend
        .generate_streaming(
            &[ChatMessage::user("Is 17 a prime number?")],
            &GenerationConfig::greedy()
                .with_max_tokens(1)
                .with_grammar(r#"root ::= "Let""#),
            Box::new(|chunk| {
                chunks.push(chunk);
                Ok(())
            }),
        )
        .expect("streaming generation succeeds");

    assert_eq!(output.finish_reason, "length");
    assert_eq!(output.text, "", "the whole output is reasoning");
    assert_eq!(output.reasoning_content.as_deref(), Some("Let"));
    assert_eq!(chunks.len(), 1, "only the terminal token: {chunks:?}");
    assert_eq!(chunks[0].token, "");
    assert_eq!(chunks[0].finish_reason.as_deref(), Some("length"));
}
