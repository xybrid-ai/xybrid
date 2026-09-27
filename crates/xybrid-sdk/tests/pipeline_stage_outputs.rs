//! A pipeline run keeps every stage's output, end to end.
//!
//! Two cloud stages run through the real gateway adapter against a loopback
//! mock, so the test needs no model download, no external service and no
//! process-global credentials. Adapted from the loopback test in #552.

use httpmock::{Method::POST, MockServer};
use serde_json::json;
use std::time::Duration;
use xybrid_sdk::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::PipelineRef;

fn pipeline_yaml(server: &MockServer) -> String {
    // Each stage carries its own gateway and key, so a machine with production
    // credentials or gateway environment variables still stays on loopback.
    format!(
        r#"
name: two-stage-outputs
registry: "{base}"
stages:
  - id: draft
    model: first-model
    target: cloud
    provider: openai
    gateway_url: "{base}/v1"
    api_key: local-test-only
    timeout_ms: 2000
  - id: polish
    model: second-model
    target: cloud
    provider: openai
    gateway_url: "{base}/v1"
    api_key: local-test-only
    timeout_ms: 2000
"#,
        base = server.base_url()
    )
}

fn completion(text: &str) -> serde_json::Value {
    json!({
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }]
    })
}

fn text_of(envelope: &Envelope) -> Option<&str> {
    match &envelope.kind {
        EnvelopeKind::Text(text) => Some(text),
        _ => None,
    }
}

#[test]
fn every_stage_output_survives_the_run() {
    let server = MockServer::start();
    let first = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .json_body_partial(r#"{"model":"first-model"}"#)
            .body_contains("original input");
        then.status(200)
            .json_body(completion("first-stage output"))
            .delay(Duration::from_millis(20));
    });
    let second = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .json_body_partial(r#"{"model":"second-model"}"#)
            .body_contains("first-stage output");
        then.status(200)
            .json_body(completion("final output"))
            .delay(Duration::from_millis(20));
    });

    let pipeline = PipelineRef::from_yaml(&pipeline_yaml(&server))
        .expect("pipeline YAML should parse")
        .load()
        .expect("a cloud pipeline should load without downloading any model");
    let input = Envelope::new(EnvelopeKind::Text("original input".into()));

    let result = pipeline.run(&input).expect("both stages should succeed");

    assert_eq!(first.hits(), 1, "the first stage must call the gateway");
    assert_eq!(
        second.hits(),
        1,
        "the second stage must consume the first's output"
    );

    let ids: Vec<&str> = result.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        ids,
        ["draft", "polish"],
        "stages are named by YAML id, not model id"
    );

    // The point of the test: the intermediate output is not discarded.
    assert_eq!(
        text_of(&result.stages[0].output),
        Some("first-stage output")
    );
    assert_eq!(text_of(&result.stages[1].output), Some("final output"));
    assert_eq!(text_of(&result.output), Some("final output"));

    // The mock delay only guarantees a nonzero duration; nothing here depends
    // on scheduler speed.
    assert!(result.stages.iter().all(|s| s.latency_ms >= 10));
}

#[test]
fn a_later_stage_failure_is_an_error_not_partial_success() {
    let server = MockServer::start();
    let first = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .json_body_partial(r#"{"model":"first-model"}"#);
        then.status(200).json_body(completion("first-stage output"));
    });
    let second = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .json_body_partial(r#"{"model":"second-model"}"#);
        // Non-retryable, so the test never sleeps through retry backoff.
        then.status(400)
            .json_body(json!({"error": {"message": "second stage rejected input"}}));
    });

    let pipeline = PipelineRef::from_yaml(&pipeline_yaml(&server))
        .expect("pipeline YAML should parse")
        .load()
        .expect("a cloud pipeline should load without downloading any model");
    let input = Envelope::new(EnvelopeKind::Text("original input".into()));

    let err = pipeline
        .run(&input)
        .expect_err("a failed second stage must not return the first stage's output");

    // The top-level message is generic; the HTTP status sits in the source
    // chain.
    let chain: Vec<String> =
        std::iter::successors(Some(&err as &dyn std::error::Error), |e| e.source())
            .map(ToString::to_string)
            .collect();
    assert!(chain.iter().any(|m| m.contains("400")), "{chain:?}");
    assert_eq!(first.hits(), 1);
    assert_eq!(second.hits(), 1);
}
