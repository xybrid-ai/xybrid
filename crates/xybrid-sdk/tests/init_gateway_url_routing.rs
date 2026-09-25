//! Regression: the gateway configured through `xybrid_sdk::init().gateway_url(..)`
//! must be the endpoint an SDK pipeline's cloud stage actually calls.
//!
//! The builder used to write an SDK-local cell that nothing on the pipeline
//! path read, while the orchestrator's cloud adapter resolved its destination
//! from the core cell. A stage without a stage-level `gateway_url` was therefore
//! dispatched to the ambient/production gateway — carrying the Xybrid bearer
//! key — and the configured gateway never saw a request.
//!
//! Model-free: a cloud-only DeepSeek stage with neither a stage-level
//! `gateway_url` nor `api_key`, so destination and credential come from process
//! configuration alone. Two loopback httpmock servers stand in for "the
//! configured gateway" and "everywhere else" (installed as the platform URL,
//! the next source in the resolution order). The assertions are about which
//! server received the completion request, and with which bearer token.
//!
//! This binary holds a single test on purpose: it configures process-global
//! state and must not share a process with tests that read it.

use httpmock::prelude::*;
use httpmock::Mock;
use xybrid_sdk::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::Xybrid;

const FAKE_REPLY: &str = "FAKE_GATEWAY_REPLY";
const API_KEY: &str = "xy_test_configured_gateway";
const MODEL_ID: &str = "sdk-gateway-model";

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
            .path_contains(format!("/v1/models/{MODEL_ID}/resolve"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(serde_json::json!({
                "mask": MODEL_ID,
                "platform": "test",
                "resolved": {
                    "hf_repo": format!("xybrid-test/{MODEL_ID}"),
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

/// Completion mock on the configured gateway that only matches a request
/// carrying the SDK's platform key, so one hit proves both the destination
/// and that the credential followed it.
fn authenticated_completion_mock(server: &MockServer) -> Mock<'_> {
    server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .header("authorization", format!("Bearer {API_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(completion_body());
    })
}

/// Any completion request at all on the wrong server, headers or not, so a
/// misrouted request is counted even if it arrives without a credential.
fn wrong_destination_mock(server: &MockServer) -> Mock<'_> {
    server.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(599).body("request reached the wrong gateway");
    })
}

#[test]
fn init_gateway_url_is_where_sdk_pipeline_cloud_stages_are_sent() {
    let configured = MockServer::start();
    let elsewhere = MockServer::start();
    let _registry = registry_resolve_mock(&configured);
    let configured_completion = authenticated_completion_mock(&configured);
    let elsewhere_completion = wrong_destination_mock(&elsewhere);

    // "Elsewhere" is what core resolves when the builder's gateway is not
    // honoured: the platform URL is the next source in the resolution order
    // (normally the production gateway).
    xybrid_sdk::set_platform_url(&elsewhere.base_url());
    xybrid_sdk::init()
        .gateway_url(format!("{}/v1", configured.base_url()))
        .run();
    // Same cell the builder's `.api_key(..)` writes, minus the telemetry
    // exporter that `.api_key(..)` also starts.
    xybrid_sdk::set_api_key(API_KEY);

    let yaml = format!(
        r#"name: sdk-init-gateway
registry: {base}
stages:
  - id: llm
    model: {MODEL_ID}
    target: cloud
    provider: deepseek
"#,
        base = configured.base_url()
    );
    let envelope = Envelope::new(EnvelopeKind::Text("hello".to_string()));

    let result = Xybrid::run_pipeline(&yaml, &envelope).expect("pipeline runs");

    assert_eq!(result.text(), Some(FAKE_REPLY));
    assert_eq!(
        configured_completion.hits(),
        1,
        "the configured gateway must receive the request, with the Xybrid bearer key"
    );
    assert_eq!(
        elsewhere_completion.hits(),
        0,
        "no request may reach the platform-URL fallback"
    );
}
