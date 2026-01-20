//! Telemetry Integration Test
//!
//! This example demonstrates the SDK telemetry integration with the Xybrid Platform.
//! It runs a simple TTS pipeline and sends telemetry events to the platform API.
//!
//! ## Prerequisites
//! 1. Start the local backend: `cd repos/xybrid-platform/backend && cargo shuttle run`
//! 2. Create an API key in the console (or use a test key)
//! 3. Set environment variables:
//!    - XYBRID_API_KEY=your-api-key
//!    - XYBRID_PLATFORM_URL=http://localhost:8000 (for local testing)
//!
//! ## Usage
//! ```bash
//! cargo run --example telemetry_test
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::{
    flush_platform_telemetry, init_platform_telemetry, publish_telemetry_event,
    shutdown_platform_telemetry, TelemetryConfig, TelemetryEvent,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Xybrid Telemetry Integration Test ===\n");

    // Check for API key
    let api_key = std::env::var("XYBRID_API_KEY").unwrap_or_else(|_| {
        println!("Note: XYBRID_API_KEY not set, using test key");
        "sk_test_telemetry_demo".to_string()
    });

    let endpoint = std::env::var("XYBRID_PLATFORM_URL")
        .unwrap_or_else(|_| "http://localhost:8000".to_string());

    println!("Platform endpoint: {}", endpoint);
    println!("API key: {}...\n", &api_key[..api_key.len().min(20)]);

    // Initialize telemetry with explicit config
    let config = TelemetryConfig::new(&endpoint, &api_key)
        .with_device("test-device-001", "macos")
        .with_app_version("0.0.13-test")
        .with_batch_size(1) // Flush immediately for testing
        .with_flush_interval(1);

    println!("Initializing platform telemetry...");
    init_platform_telemetry(config);

    // Emit a test event manually
    println!("Sending test telemetry event...");
    let test_event = TelemetryEvent {
        event_type: "TestEvent".to_string(),
        stage_name: Some("telemetry_test".to_string()),
        target: Some("local".to_string()),
        latency_ms: Some(42),
        error: None,
        data: Some(r#"{"test": true, "message": "Hello from SDK telemetry test"}"#.to_string()),
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    };
    publish_telemetry_event(test_event);

    // Try to run a simple TTS inference (if model is available)
    let model_dir = PathBuf::from("test_models/kitten-tts-nano");
    if model_dir.exists() {
        println!("\nRunning TTS inference with telemetry...");

        let metadata_path = model_dir.join("model_metadata.json");
        if metadata_path.exists() {
            let metadata: ModelMetadata =
                serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;

            let mut executor = xybrid_core::template_executor::TemplateExecutor::with_base_path(
                model_dir.to_str().unwrap(),
            );

            let input = Envelope {
                kind: EnvelopeKind::Text("Hello, this is a telemetry test.".to_string()),
                metadata: HashMap::new(),
            };

            // Emit pipeline start event
            let start_event = TelemetryEvent {
                event_type: "PipelineStart".to_string(),
                stage_name: None,
                target: None,
                latency_ms: None,
                error: None,
                data: Some(r#"{"stages": ["tts"]}"#.to_string()),
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            };
            publish_telemetry_event(start_event);

            let start = std::time::Instant::now();
            match executor.execute(&metadata, &input) {
                Ok(output) => {
                    let latency = start.elapsed().as_millis() as u32;
                    println!("TTS completed in {}ms", latency);

                    if let EnvelopeKind::Audio(audio) = &output.kind {
                        println!("Generated {} bytes of audio", audio.len());
                    }

                    // Emit success event
                    let complete_event = TelemetryEvent {
                        event_type: "PipelineComplete".to_string(),
                        stage_name: Some("tts".to_string()),
                        target: Some("device".to_string()),
                        latency_ms: Some(latency),
                        error: None,
                        data: None,
                        timestamp_ms: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64,
                    };
                    publish_telemetry_event(complete_event);
                }
                Err(e) => {
                    println!("TTS failed: {}", e);

                    // Emit error event
                    let error_event = TelemetryEvent {
                        event_type: "PipelineError".to_string(),
                        stage_name: Some("tts".to_string()),
                        target: Some("device".to_string()),
                        latency_ms: None,
                        error: Some(e.to_string()),
                        data: None,
                        timestamp_ms: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64,
                    };
                    publish_telemetry_event(error_event);
                }
            }
        } else {
            println!("Model metadata not found at {:?}", metadata_path);
        }
    } else {
        println!("\nNote: kitten-tts-nano model not found, skipping TTS test");
        println!("Only the test event will be sent.");
    }

    // Flush and shutdown
    println!("\nFlushing telemetry...");
    flush_platform_telemetry();

    // Give time for HTTP request to complete
    std::thread::sleep(std::time::Duration::from_secs(2));

    println!("Shutting down telemetry...");
    shutdown_platform_telemetry();

    println!("\n=== Test Complete ===");
    println!(
        "Check the platform console at {}/ to verify events arrived",
        endpoint.replace("8000", "5173")
    ); // Assuming console runs on 5173

    Ok(())
}
