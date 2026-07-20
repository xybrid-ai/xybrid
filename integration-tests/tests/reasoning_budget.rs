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
