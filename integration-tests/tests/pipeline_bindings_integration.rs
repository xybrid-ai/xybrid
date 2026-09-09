//! Exercise the Bolt pipeline surface through the real gateway adapter on loopback.
//! No model downloads, external services, or process-global credentials are needed.

use httpmock::{Method::POST, MockServer};
use serde_json::json;
use std::time::Duration;
use xybrid_bolt::{
    XybridEnvelope, XybridEnvelopeKind, XybridError, XybridExecutionTarget, XybridPipeline,
};

fn pipeline(server: &MockServer) -> XybridPipeline {
    // Each stage carries its own gateway/key override, so even machines with
    // production credentials or gateway environment variables stay on loopback.
    XybridPipeline::from_yaml(format!(
        r#"
name: two-stage-binding-test
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
    ))
    .expect("a cloud pipeline should load without downloading any models")
}

fn input() -> XybridEnvelope {
    XybridEnvelope {
        kind: XybridEnvelopeKind::Text {
            text: "original input".into(),
        },
        metadata: Vec::new(),
    }
}

fn completion(text: &str) -> serde_json::Value {
    json!({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})
}

#[test]
fn bolt_pipeline_chains_real_gateway_requests_and_preserves_metrics() {
    let server = MockServer::start();
    let first = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .header("authorization", "Bearer local-test-only")
            .json_body_partial(r#"{"model":"first-model","messages":[{"role":"user","content":"original input"}]}"#);
        then.status(200).json_body(completion("first-stage output")).delay(Duration::from_millis(20));
    });
    let second = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .header("authorization", "Bearer local-test-only")
            .json_body_partial(r#"{"model":"second-model","messages":[{"role":"user","content":"first-stage output"}]}"#);
        then.status(200).json_body(completion("final output")).delay(Duration::from_millis(20));
    });
    let pipeline = pipeline(&server);
    assert_eq!(pipeline.name().as_deref(), Some("two-stage-binding-test"));
    assert_eq!(pipeline.stage_names(), ["draft", "polish"]);
    assert_eq!(pipeline.stage_count(), 2);

    let result = pipeline.run(input()).expect("both stages should succeed");
    assert_eq!(
        first.hits(),
        1,
        "the first stage must actually call the gateway"
    );
    assert_eq!(
        second.hits(),
        1,
        "the second stage must consume the first response"
    );
    assert!(
        matches!(result.envelope.kind, XybridEnvelopeKind::Text { ref text } if text == "final output")
    );
    assert_eq!(result.model_id, "second-model");
    assert_eq!(result.execution_target, XybridExecutionTarget::Cloud);
    let stages = &result.metrics.stage_latencies_ms;
    assert_eq!(
        stages
            .iter()
            .map(|s| s.stage_id.as_str())
            .collect::<Vec<_>>(),
        ["draft", "polish"]
    );
    // Delay only establishes a nonzero duration; no upper timing bound depends
    // on scheduler speed. Stage IDs must not be replaced by the model IDs.
    assert!(stages.iter().all(|s| s.latency_ms >= 10));
    assert!(result.metrics.total_ms >= stages.iter().map(|s| s.latency_ms).sum::<u32>());
    assert_eq!(result.latency_ms, result.metrics.total_ms);
}

#[test]
fn bolt_pipeline_propagates_a_later_stage_failure_instead_of_partial_success() {
    let server = MockServer::start();
    let first = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .json_body_partial(r#"{"model":"first-model"}"#);
        then.status(200).json_body(completion("first-stage output"));
    });
    let second = server.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions").json_body_partial(r#"{"model":"second-model","messages":[{"role":"user","content":"first-stage output"}]}"#);
        // Non-retryable, so this test never sleeps through retry backoff.
        then.status(400).json_body(json!({"error": {"message": "second stage rejected input"}}));
    });
    match pipeline(&server).run(input()) {
        Err(XybridError::PipelineError { message }) => {
            assert!(message.contains("400"), "{message}")
        }
        Err(error) => panic!("expected a typed pipeline error, got {error:?}"),
        Ok(_) => panic!("a failed second stage must not return the first stage's output"),
    }
    assert_eq!(first.hits(), 1);
    assert_eq!(second.hits(), 1);
}
