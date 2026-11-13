//! Demonstrates local telemetry events and explicit OpenTelemetry export.
//!
//! Set `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` for a Collector,
//! or `XYBRID_API_KEY` for the Xybrid dashboard. With neither set the example
//! stays local. A manual span always runs; the TTS inference additionally runs
//! when its model fixture is available.
//!
//! ```bash
//! cargo run -p xybrid-sdk --example telemetry_test
//! ```

use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;
use xybrid_sdk::observability::{self, KeyValue, StartOptions};
use xybrid_sdk::{publish_telemetry_event, register_telemetry_sender, ModelLoader, TelemetryEvent};

#[path = "support/observability.rs"]
mod observability_support;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let export = observability_support::configure("telemetry-test")?;
    let (tx, rx) = std::sync::mpsc::channel();
    register_telemetry_sender(tx);

    // Local event delivery is available independently of trace export.
    publish_telemetry_event(TelemetryEvent {
        event_type: "TestEvent".to_owned(),
        stage_name: Some("telemetry_test".to_owned()),
        target: Some("local".to_owned()),
        latency_ms: Some(42),
        error: None,
        data: Some(serde_json::json!({"test": true}).to_string()),
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as u64,
    });
    let span = observability::observer().start_span(
        "telemetry_test",
        vec![KeyValue::new("demo.target_mode", "privacy-strict")],
        StartOptions::default(),
    );
    span.end();

    if let Some(model_dir) = model_fixtures::model_or_skip("kitten-tts-nano-0.2") {
        let model = ModelLoader::from_directory(&model_dir)?.load()?;
        let input = Envelope::new(EnvelopeKind::Text(
            "Hello, this is an observability test.".to_owned(),
        ));
        let result = model.run(&input, None)?;
        if let Some(audio) = result.audio_bytes() {
            println!("Generated {} bytes of audio", audio.len());
        }
        if let Some(reference) = result.observation_ref() {
            println!("Inference trace: {}", reference.trace_id);
        }
    }
    for event in rx.try_iter() {
        println!("Local event: {}", event.event_type);
    }
    export.finish()?;
    println!("Observability shutdown complete");
    Ok(())
}
