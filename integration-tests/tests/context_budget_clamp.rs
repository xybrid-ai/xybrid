#![cfg(feature = "llm-llamacpp")]
#![doc = "Context-budget regression coverage for llama.cpp prompt and generation boundaries."]
#![doc = "An oversized requested budget must return a length-truncated result, not native -4."]
#![doc = "Run with: cargo test -p integration-tests --features llm-llamacpp context_budget_clamp -- --nocapture"]
#![doc = "Download model first: ./integration-tests/download.sh qwen3-0.6b"]

use integration_tests::fixtures;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::template_executor::TemplateExecutor;

const MODEL_NAME: &str = "qwen3-0.6b";
const MODEL_FILE: &str = "Qwen3-0.6B-Q8_0.gguf";

struct MaterializedFixture {
    path: PathBuf,
}

impl Drop for MaterializedFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn link_or_copy_fixture_file(source: &Path, destination: &Path) {
    if std::fs::hard_link(source, destination).is_ok() {
        return;
    }

    #[cfg(unix)]
    if std::os::unix::fs::symlink(source, destination).is_ok() {
        return;
    }

    std::fs::copy(source, destination)
        .unwrap_or_else(|err| panic!("failed to materialize {}: {err}", source.display()));
}

fn materialize_fixture(source_dir: &Path) -> MaterializedFixture {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after the unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("xybrid-context-budget-{suffix}"));
    std::fs::create_dir(&path).expect("create temporary context-budget fixture");

    let metadata_path = source_dir.join("model_metadata.json");
    let mut metadata: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&metadata_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", metadata_path.display())),
    )
    .unwrap_or_else(|err| panic!("failed to parse {}: {err}", metadata_path.display()));
    metadata["execution_template"]["context_length"] = serde_json::json!(256);
    metadata["metadata"]["context_length"] = serde_json::json!(256);
    metadata["metadata"]["reasoning"] = serde_json::json!(true);
    std::fs::write(
        path.join("model_metadata.json"),
        serde_json::to_vec_pretty(&metadata).expect("serialize fixture metadata"),
    )
    .expect("write temporary fixture metadata");

    link_or_copy_fixture_file(&source_dir.join(MODEL_FILE), &path.join(MODEL_FILE));

    MaterializedFixture { path }
}

#[test]
fn oversized_generation_budget_is_clamped_to_context_and_finishes_by_length() {
    let Some(source_dir) = fixtures::model_if_available(MODEL_NAME) else {
        eprintln!(
            "Skipping {}: model not downloaded. Run: ./integration-tests/download.sh {}",
            MODEL_NAME, MODEL_NAME
        );
        return;
    };

    let fixture = materialize_fixture(&source_dir);
    let metadata_path = fixture.path.join("model_metadata.json");
    let metadata_content =
        std::fs::read_to_string(&metadata_path).expect("failed to read rewritten metadata");
    let metadata: ModelMetadata =
        serde_json::from_str(&metadata_content).expect("failed to parse rewritten metadata");
    let mut executor = TemplateExecutor::with_base_path(
        fixture
            .path
            .to_str()
            .expect("temporary fixture path must be valid utf-8"),
    );
    let input = Envelope {
        kind: EnvelopeKind::Text(
            "Explain why context windows matter for generation. Use a concise answer.".to_string(),
        ),
        metadata: HashMap::from([(String::from("max_tokens"), String::from("2048"))]),
    };

    let result = executor.execute(&metadata, &input, None);
    assert!(
        result.is_ok(),
        "generation must truncate normally instead of returning native -4: {result:?}"
    );
    let output = result.expect("clamped generation should return an output envelope");
    assert_eq!(
        output.metadata.get("finish_reason").map(String::as_str),
        Some("length")
    );
}
