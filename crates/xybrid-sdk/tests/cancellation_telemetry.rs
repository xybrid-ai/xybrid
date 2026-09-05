//! Cancellable batch calls preserve the logical run shape and resource telemetry.
#![cfg(feature = "llm-llamacpp")]

use std::sync::mpsc;
use xybrid_core::testing::model_fixtures;
use xybrid_sdk::{
    init_platform_telemetry, register_telemetry_sender, shutdown_platform_telemetry,
    CancellationToken, ConversationContext, GenerationConfig, ModelLoader, ResourceTelemetryMode,
    RunOptions, TelemetryConfig,
};

#[test]
fn cancellation_preserves_batch_context_and_streaming_telemetry() {
    let Some(path) = model_fixtures::model_or_skip("functiongemma-270m-it") else {
        return;
    };
    // Keep the exporter on loopback even if the machine has production credentials.
    let server = httpmock::MockServer::start();
    let _ingest = server.mock(|when, then| {
        when.method(httpmock::Method::POST);
        then.status(200);
    });
    init_platform_telemetry(
        TelemetryConfig::new(server.base_url(), "local-test-only")
            .with_resource_telemetry(ResourceTelemetryMode::Boundary),
    );
    struct Shutdown;
    impl Drop for Shutdown {
        fn drop(&mut self) {
            shutdown_platform_telemetry();
        }
    }
    let _shutdown = Shutdown;
    assert_eq!(
        xybrid_sdk::telemetry::resource_telemetry_mode(),
        ResourceTelemetryMode::Boundary
    );

    let model = ModelLoader::from_directory(path).unwrap().load().unwrap();
    let config = GenerationConfig {
        max_tokens: 16,
        temperature: 0.0,
        ..Default::default()
    };
    let input = xybrid_sdk::ir::Envelope::new(xybrid_sdk::ir::EnvelopeKind::Text(
        "Say hello in English.".into(),
    ));
    let context = ConversationContext::new();
    let (sender, events) = mpsc::channel();
    register_telemetry_sender(sender);

    for streaming in [false, true] {
        for with_context in [false, true] {
            for cancellable in [false, true] {
                let mut options = RunOptions::new().with_generation_config(config.clone());
                if cancellable {
                    options = options.with_cancellation_token(CancellationToken::new());
                }
                let result = match (streaming, with_context) {
                    (false, false) => model.run_with_options(&input, &options),
                    (false, true) => model.run_with_context_options(&input, &context, &options),
                    (true, false) => model.run_streaming_with_options(&input, &options, |_| Ok(())),
                    (true, true) => {
                        model.run_streaming_with_context_options(&input, &context, &options, |_| {
                            Ok(())
                        })
                    }
                }
                .unwrap();
                assert!(!result.text().unwrap().is_empty());
                let completed: Vec<_> = events
                    .try_iter()
                    .filter(|event| event.event_type == "ModelComplete")
                    .collect();
                assert_eq!(completed.len(), 1, "expected one completion for streaming={streaming}, context={with_context}, token={cancellable}");
                let event = &completed[0];
                assert_eq!(event.target.as_deref(), Some("local"));
                let data: serde_json::Value =
                    serde_json::from_str(event.data.as_ref().unwrap()).unwrap();
                assert!(
                    data["resource_summary"].is_object(),
                    "resource summary missing: {data}"
                );
                assert_eq!(
                    data["streaming"].as_bool().unwrap_or(false),
                    streaming,
                    "wrong run shape: {data}"
                );
                if with_context {
                    assert_eq!(data["context_messages"], 0);
                }
            }
        }
    }
}
