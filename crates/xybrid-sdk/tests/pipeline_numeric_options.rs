//! SDK pipeline numeric-option regression: YAML integers must survive option
//! conversion and reach the provider as integers.
//!
//! Model-free: a cloud-only DeepSeek stage is pointed at a loopback httpmock
//! server that serves both the registry resolve response and the fake
//! completion endpoint. The assertions are about the wire body: `max_tokens`
//! must be the integer `16`, never the float `16.0`.
//!
//! The registry stub and completion endpoint are explicitly loopback; captured
//! request assertions verify those endpoints, not process-wide isolation.

use httpmock::prelude::*;
use httpmock::Mock;
use xybrid_sdk::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::Xybrid;

const FAKE_REPLY: &str = "FAKE_NUMERIC_REPLY";

fn completion_body() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-1",
        "model": "deepseek-flash",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": FAKE_REPLY },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 }
    })
}

/// Loopback registry stub. The stage is cloud-only (`provider` set), so the
/// only registry call is the `is_cached` probe; this keeps that probe local.
fn registry_resolve_mock(server: &MockServer) -> Mock<'_> {
    server.mock(|when, then| {
        when.method(GET)
            .path_contains("/v1/models/sdk-numeric-model/resolve");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(serde_json::json!({
                "mask": "sdk-numeric-model",
                "platform": "test",
                "resolved": {
                    "hf_repo": "xybrid-test/sdk-numeric-model",
                    "file": "model.xyb",
                    "download_url": "http://127.0.0.1:9/model.xyb",
                    "format": "xyb",
                    "quantization": "none",
                    "size_bytes": 1,
                    "sha256": ""
                }
            }));
    })
}

/// Flatten an error and its sources into one string, so SDK wrappers that
/// keep the detail in `source()` are still assertable.
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

fn pipeline_yaml(server: &MockServer, extra_stage: &str) -> String {
    format!(
        r#"name: sdk-numeric-options
registry: {base}
stages:
  - id: llm
    model: sdk-numeric-model
    target: cloud
    provider: deepseek
    gateway_url: {base}/v1
    api_key: dummy-key
{extra_stage}"#,
        base = server.base_url()
    )
}

/// Completion mock that only matches an integer `max_tokens`.
fn integer_max_tokens_mock(server: &MockServer, tokens: u64) -> Mock<'_> {
    server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .json_body_partial(format!(r#"{{"max_tokens":{tokens}}}"#));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(completion_body());
    })
}

/// Guard mock that fires only if `max_tokens` was serialized as a float.
fn float_max_tokens_guard(server: &MockServer) -> Mock<'_> {
    server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .json_body_partial(r#"{"max_tokens":16.0}"#);
        then.status(599).body("max_tokens must remain an integer");
    })
}

#[test]
fn run_pipeline_sends_integer_max_tokens() {
    let server = MockServer::start();
    let _registry = registry_resolve_mock(&server);
    let integer_mock = integer_max_tokens_mock(&server, 16);
    let float_guard = float_max_tokens_guard(&server);

    let yaml = pipeline_yaml(&server, "    max_tokens: 16\n");
    let envelope = Envelope::new(EnvelopeKind::Text("hello".to_string()));

    let result = Xybrid::run_pipeline(&yaml, &envelope).expect("pipeline runs");

    assert_eq!(result.text(), Some(FAKE_REPLY));
    assert_eq!(integer_mock.hits(), 1, "integer 16 must reach the wire");
    assert_eq!(
        float_guard.hits(),
        0,
        "max_tokens must not be serialized as 16.0"
    );
}

#[test]
fn input_metadata_override_reaches_the_wire() {
    let server = MockServer::start();
    let _registry = registry_resolve_mock(&server);
    let yaml_mock = integer_max_tokens_mock(&server, 16);
    let override_mock = integer_max_tokens_mock(&server, 7);

    let yaml = pipeline_yaml(&server, "    max_tokens: 16\n");
    let mut envelope = Envelope::new(EnvelopeKind::Text("hello".to_string()));
    envelope
        .metadata
        .insert("max_tokens".to_string(), "7".to_string());

    let result = Xybrid::run_pipeline(&yaml, &envelope).expect("pipeline runs");

    assert_eq!(result.text(), Some(FAKE_REPLY));
    assert_eq!(override_mock.hits(), 1, "input override must win");
    assert_eq!(
        yaml_mock.hits(),
        0,
        "YAML value must not override input metadata"
    );
}

#[test]
fn invalid_options_fail_before_any_completion_request() {
    let cases = [
        ("    max_tokens: 0\n", "positive integer"),
        ("    max_tokens: 1.5\n", "positive integer"),
        ("    max_tokens: \"16\"\n", "positive integer"),
        ("    system_prompt: 3\n", "a string"),
    ];

    for (extra_stage, expected) in cases {
        let server = MockServer::start();
        let _registry = registry_resolve_mock(&server);
        let completion = server.mock(|when, then| {
            when.method(POST).path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(completion_body());
        });

        let yaml = pipeline_yaml(&server, extra_stage);
        let envelope = Envelope::new(EnvelopeKind::Text("hello".to_string()));

        let err = Xybrid::run_pipeline(&yaml, &envelope).expect_err("invalid option must fail");
        let chain = error_chain(&err);

        assert!(
            chain.contains(expected),
            "{extra_stage}: expected '{expected}' in: {chain}"
        );
        assert_eq!(
            completion.hits(),
            0,
            "{extra_stage}: no completion request may be made"
        );
    }
}

#[test]
fn async_run_uses_the_same_prepared_options() {
    let server = MockServer::start();
    let _registry = registry_resolve_mock(&server);
    let integer_mock = integer_max_tokens_mock(&server, 16);
    let float_guard = float_max_tokens_guard(&server);

    let yaml = pipeline_yaml(&server, "    max_tokens: 16\n");
    let envelope = Envelope::new(EnvelopeKind::Text("hello".to_string()));

    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let result = runtime
        .block_on(async {
            let pipeline = Xybrid::pipeline(&yaml)
                .expect("pipeline parses")
                .load()
                .expect("pipeline loads");
            pipeline.run_async(&envelope).await
        })
        .expect("async pipeline runs");

    assert_eq!(result.text(), Some(FAKE_REPLY));
    assert_eq!(integer_mock.hits(), 1, "async path sends integer 16");
    assert_eq!(float_guard.hits(), 0, "async path must not send 16.0");
}
