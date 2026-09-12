//! Model-enabled regression: the SDK streaming fast path must honor the same
//! YAML generation options as batch execution.
//!
//! Requires the real `functiongemma-270m-it` fixture. With
//! `XYBRID_REQUIRE_MODELS=1` a missing fixture fails; otherwise the tests
//! skip. Run explicitly:
//!
//! ```bash
//! XYBRID_REQUIRE_MODELS=1 cargo test -p xybrid-sdk --features llm-llamacpp \
//!   --test streaming_fast_path -- --nocapture
//! ```
//!
//! The registry stub is explicitly loopback; the fixture is staged into an
//! SDK cache directory configured for the test process, so no network access
//! happens beyond the loopback registry probe.
#![cfg(all(feature = "llm-llamacpp", unix))]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use httpmock::prelude::*;
use xybrid_core::runtime_adapter::types::{PartialToken, StreamingCallback};
use xybrid_core::testing::model_fixtures;
use xybrid_sdk::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::run_options::RunOptions;
use xybrid_sdk::Xybrid;

const MODEL_ID: &str = "functiongemma-270m-it";
const PROMPT: &str = "Reply with the single word hello.";

fn fixture_dir_or_skip() -> Option<PathBuf> {
    match model_fixtures::model_or_skip(MODEL_ID) {
        Some(dir) => Some(dir),
        None => {
            assert!(
                std::env::var_os("XYBRID_REQUIRE_MODELS").is_none(),
                "XYBRID_REQUIRE_MODELS is set but model '{MODEL_ID}' is not available"
            );
            None
        }
    }
}

/// One SDK cache for the whole test binary: `init_sdk_cache_dir` is a
/// process-global `OnceLock`, so every test in this file must share the root.
fn shared_cache_root() -> &'static Path {
    static CACHE: OnceLock<tempfile::TempDir> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let temp = tempfile::tempdir().expect("temp cache dir");
            xybrid_sdk::init_sdk_cache_dir(temp.path().join("models"));
            temp
        })
        .path()
}

fn link_or_copy(source: &Path, destination: &Path) {
    if std::fs::hard_link(source, destination).is_ok() {
        return;
    }
    std::fs::copy(source, destination)
        .unwrap_or_else(|err| panic!("failed to materialize {}: {err}", source.display()));
}

/// Stage the real fixture into `<cache>/extracted/<id>/` once.
fn staged_model_dir() -> PathBuf {
    static STAGED: OnceLock<PathBuf> = OnceLock::new();
    STAGED
        .get_or_init(|| {
            let fixture = fixture_dir_or_skip().expect("fixture checked before staging");
            let extracted = shared_cache_root().join("extracted").join(MODEL_ID);
            std::fs::create_dir_all(&extracted).unwrap();
            link_or_copy(
                &fixture.join("model_metadata.json"),
                &extracted.join("model_metadata.json"),
            );
            let metadata: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(fixture.join("model_metadata.json")).unwrap(),
            )
            .unwrap();
            for file in metadata["files"].as_array().expect("metadata.files") {
                let name = file.as_str().expect("file name");
                link_or_copy(&fixture.join(name), &extracted.join(name));
            }
            extracted
        })
        .clone()
}

/// Loopback registry stub: makes `is_cached` / `resolve` local. The model is
/// already staged in the SDK cache, so `fetch_extracted` short-circuits to the
/// extracted directory and the download URL is never followed.
fn registry_resolve_mock(server: &MockServer) -> httpmock::Mock<'_> {
    server.mock(|when, then| {
        when.method(GET)
            .path_contains(format!("/v1/models/{MODEL_ID}/resolve"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(serde_json::json!({
                "mask": MODEL_ID,
                "platform": "test",
                "resolved": {
                    "hf_repo": "ggml-org/functiongemma-270m-it-GGUF",
                    "file": "functiongemma-270m-it-q8_0.gguf",
                    "download_url": "http://127.0.0.1:9/never-followed.gguf",
                    "format": "gguf",
                    "quantization": "Q8_0",
                    "size_bytes": 1,
                    "sha256": ""
                }
            }));
    })
}

fn pipeline_yaml(server: &MockServer, extra_stage: &str) -> String {
    format!(
        r#"name: sdk-streaming-options
registry: {base}
stages:
  - id: llm
    model: {MODEL_ID}
    target: device
{extra_stage}"#,
        base = server.base_url()
    )
}

fn error_chain(err: &dyn std::error::Error) -> String {
    let mut message = err.to_string();
    let mut current = err.source();
    while let Some(source) = current {
        message.push_str(": ");
        message.push_str(&source.to_string());
        current = source.source();
    }
    message
}

fn run_streaming(yaml: &str, envelope: &Envelope) -> (Vec<PartialToken>, String) {
    let tokens: Arc<Mutex<Vec<PartialToken>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = tokens.clone();
    let callback: StreamingCallback<'_> = Box::new(move |token| {
        sink.lock().unwrap().push(token);
        Ok(())
    });

    let result = Xybrid::run_pipeline_streaming_with_options(
        yaml,
        envelope,
        &RunOptions::default(),
        callback,
    )
    .expect("streaming fast path runs");

    let collected = tokens.lock().unwrap().clone();
    (collected, result.text().unwrap_or_default().to_string())
}

fn generated_tokens(tokens: &[PartialToken]) -> Vec<&PartialToken> {
    tokens
        .iter()
        .filter(|token| !token.token.is_empty())
        .collect()
}

#[test]
fn yaml_max_tokens_caps_streaming_generation() {
    let Some(_) = fixture_dir_or_skip() else {
        return;
    };
    let _ = staged_model_dir();
    let server = MockServer::start();
    let _registry = registry_resolve_mock(&server);

    let yaml = pipeline_yaml(
        &server,
        "    max_tokens: 1\n    temperature: 0\n    top_p: 0.9\n    system_prompt: \"Be terse.\"\n",
    );
    let envelope = Envelope::new(EnvelopeKind::Text(PROMPT.to_string()));

    let (tokens, text) = run_streaming(&yaml, &envelope);
    let generated = generated_tokens(&tokens);

    assert_eq!(
        generated.len(),
        1,
        "max_tokens: 1 must stop at one token: {tokens:?}"
    );
    assert!(!text.is_empty(), "generation must produce output");
}

#[test]
fn envelope_metadata_overrides_yaml_in_streaming() {
    let Some(_) = fixture_dir_or_skip() else {
        return;
    };
    let _ = staged_model_dir();
    let server = MockServer::start();
    let _registry = registry_resolve_mock(&server);

    let yaml = pipeline_yaml(&server, "    max_tokens: 4\n");
    let mut envelope = Envelope::new(EnvelopeKind::Text(PROMPT.to_string()));
    envelope
        .metadata
        .insert("max_tokens".to_string(), "1".to_string());

    let (tokens, _) = run_streaming(&yaml, &envelope);
    let generated = generated_tokens(&tokens);

    assert_eq!(
        generated.len(),
        1,
        "input metadata must beat the YAML cap: {tokens:?}"
    );
}

#[test]
fn invalid_yaml_option_fails_before_any_token() {
    let Some(_) = fixture_dir_or_skip() else {
        return;
    };
    let _ = staged_model_dir();
    let server = MockServer::start();
    let _registry = registry_resolve_mock(&server);

    let yaml = pipeline_yaml(&server, "    max_tokens: 0\n");
    let envelope = Envelope::new(EnvelopeKind::Text(PROMPT.to_string()));

    let tokens: Arc<Mutex<Vec<PartialToken>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = tokens.clone();
    let callback: StreamingCallback<'_> = Box::new(move |token| {
        sink.lock().unwrap().push(token);
        Ok(())
    });

    let err = Xybrid::run_pipeline_streaming_with_options(
        &yaml,
        &envelope,
        &RunOptions::default(),
        callback,
    )
    .expect_err("invalid option must fail before generation");

    let chain = error_chain(&err);
    assert!(chain.contains("positive integer"), "{chain}");
    assert!(
        tokens.lock().unwrap().is_empty(),
        "no token callback may fire for an invalid option"
    );
}
