//! Real constrained generation required before preparing a llama.cpp update release.
//!
//! Locally this skips without the FunctionGemma fixture. CI downloads it and
//! sets `XYBRID_REQUIRE_MODELS`, so missing weights fail instead of skipping.

#![cfg(feature = "llm-llamacpp")]

use integration_tests::fixtures;
use xybrid_core::{
    execution::{ModelMetadata, TemplateExecutor},
    ir::{Envelope, EnvelopeKind},
    runtime_adapter::GenerationConfig,
};

#[test]
fn text_generation_obeys_grammar() {
    let Some(model_dir) = fixtures::model_for_test("functiongemma-270m-it") else {
        return;
    };
    let model_file = model_dir.join("functiongemma-270m-it-q8_0.gguf");
    if !model_file.is_file() {
        assert!(
            std::env::var_os(fixtures::REQUIRE_MODELS_ENV).is_none(),
            "required GGUF is missing: {}",
            model_file.display()
        );
        eprintln!("Skipping: FunctionGemma weights are not downloaded");
        return;
    }
    let metadata: ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("model_metadata.json"))
            .expect("fixture metadata is readable"),
    )
    .expect("fixture metadata is valid");
    let mut executor =
        TemplateExecutor::with_base_path(model_dir.to_str().expect("fixture path is UTF-8"));
    let config = GenerationConfig::greedy()
        .with_max_tokens(16)
        .with_grammar("root ::= \"ready\"");
    let output = executor
        .execute(
            &metadata,
            &Envelope::new(EnvelopeKind::Text("Say ready.".to_string())),
            Some(&config),
        )
        .expect("constrained inference succeeds");
    match output.kind {
        EnvelopeKind::Text(text) => assert_eq!(text.trim(), "ready"),
        other => panic!("expected constrained text, got {other:?}"),
    }
}
