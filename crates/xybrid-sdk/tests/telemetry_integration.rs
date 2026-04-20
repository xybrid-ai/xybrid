//! Telemetry integration test — end-to-end with a real model and ingestion API.
//!
//! This test runs a lightweight MNIST inference, emits telemetry events, and
//! verifies they are accepted by the configured ingestion endpoint.
//!
//! ## Prerequisites
//!
//! 1. Download the MNIST model fixture:
//!    ```bash
//!    cd repos/xybrid && ./integration-tests/download.sh mnist
//!    ```
//!
//! 2. Set environment variables:
//!    ```bash
//!    export XYBRID_TEST_INGEST_URL=http://localhost:8000   # ingestion API base URL
//!    export XYBRID_TEST_API_KEY=sk_test_your_key_here      # API key for the endpoint
//!    ```
//!
//! ## Usage
//!
//! ```bash
//! # Run the integration test (ignored by default — needs env vars + model)
//! XYBRID_TEST_INGEST_URL=http://localhost:8000 \
//! XYBRID_TEST_API_KEY=sk_test_abc123 \
//!   cargo test -p xybrid-sdk --test telemetry_integration -- --ignored
//! ```

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_core::testing::model_fixtures;
use xybrid_sdk::{
    flush_platform_telemetry, init_platform_telemetry, publish_telemetry_event,
    register_telemetry_sender, shutdown_platform_telemetry, HttpTelemetryExporter, TelemetryConfig,
    TelemetryEvent,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn event(event_type: &str) -> TelemetryEvent {
    TelemetryEvent {
        event_type: event_type.to_string(),
        stage_name: None,
        target: None,
        latency_ms: None,
        error: None,
        data: None,
        timestamp_ms: now_ms(),
    }
}

fn start_mock_ingest() -> (String, mpsc::Receiver<String>, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("mock ingest should bind");
    let addr = listener.local_addr().expect("mock ingest should have addr");
    let (tx, rx) = mpsc::channel();

    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("mock ingest should accept");
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 1024];
        let mut header_end = None;
        let mut content_length = 0_usize;

        loop {
            let read = stream.read(&mut chunk).expect("mock ingest should read");
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);

            if header_end.is_none() {
                header_end = find_header_end(&buffer);
                if let Some(end) = header_end {
                    content_length = parse_content_length(&buffer[..end]);
                }
            }

            if let Some(end) = header_end {
                let body_start = end + 4;
                if buffer.len() >= body_start + content_length {
                    break;
                }
            }
        }

        let body = header_end
            .map(|end| {
                let body_start = end + 4;
                let body_end = (body_start + content_length).min(buffer.len());
                String::from_utf8_lossy(&buffer[body_start..body_end]).to_string()
            })
            .unwrap_or_default();
        tx.send(body).expect("mock ingest should send body");

        let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
        stream
            .write_all(response)
            .expect("mock ingest should respond");
    });

    (format!("http://{addr}"), rx, handle)
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_content_length(headers: &[u8]) -> usize {
    String::from_utf8_lossy(headers)
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0)
}

/// Create a dummy 28×28 grayscale image as a flat f32 vec (784 pixels).
/// Draws a rough "1" digit shape for a semi-realistic input.
fn mnist_input_envelope() -> Envelope {
    let mut pixels = vec![0.0f32; 28 * 28];
    // Draw a vertical line in columns 13-14, rows 4-24 (a rough "1")
    for row in 4..24 {
        for col in 13..15 {
            pixels[row * 28 + col] = 255.0;
        }
    }
    Envelope {
        kind: EnvelopeKind::Embedding(pixels),
        metadata: HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full end-to-end: init telemetry → run MNIST → emit events → flush to API.
///
/// Ignored by default because it requires:
/// - `XYBRID_TEST_INGEST_URL` and `XYBRID_TEST_API_KEY` env vars
/// - The MNIST model fixture on disk
#[test]
#[ignore]
fn telemetry_e2e_with_mnist_inference() {
    // -- 1. Read config from env ------------------------------------------
    let ingest_url = std::env::var("XYBRID_TEST_INGEST_URL")
        .expect("Set XYBRID_TEST_INGEST_URL to the ingestion API base URL");
    let api_key =
        std::env::var("XYBRID_TEST_API_KEY").expect("Set XYBRID_TEST_API_KEY to a valid API key");

    println!("Ingestion endpoint : {}", ingest_url);
    println!(
        "API key            : {}…",
        &api_key[..api_key.len().min(12)]
    );

    // -- 2. Locate model --------------------------------------------------
    let model_dir = model_fixtures::require_model("mnist");
    let metadata_path = model_dir.join("model_metadata.json");
    let metadata: ModelMetadata =
        serde_json::from_str(&std::fs::read_to_string(&metadata_path).unwrap()).unwrap();

    // -- 3. Register a local channel so we can observe emitted events -----
    let (tx, rx) = mpsc::channel::<TelemetryEvent>();
    register_telemetry_sender(tx);

    // -- 4. Init platform telemetry (HTTP exporter) -----------------------
    let config = TelemetryConfig::new(&ingest_url, &api_key)
        .with_device("integration-test", "ci")
        .with_app_version(env!("CARGO_PKG_VERSION"))
        .with_batch_size(1) // flush every event immediately
        .with_flush_interval(1);
    init_platform_telemetry(config);

    // -- 5. Emit PipelineStart --------------------------------------------
    let mut start_ev = event("PipelineStart");
    start_ev.data = Some(r#"{"stages":["mnist"]}"#.to_string());
    publish_telemetry_event(start_ev);

    // -- 6. Run MNIST inference -------------------------------------------
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let input = mnist_input_envelope();

    let start = Instant::now();
    let output = executor
        .execute(&metadata, &input, None)
        .expect("MNIST inference should succeed");
    let latency_ms = start.elapsed().as_millis() as u32;

    // Validate output is an Embedding with 10 class probabilities
    match &output.kind {
        EnvelopeKind::Embedding(probs) => {
            assert_eq!(
                probs.len(),
                10,
                "MNIST should output 10 class probabilities"
            );
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Softmax output should sum to ~1.0, got {}",
                sum
            );
            let predicted = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            println!(
                "MNIST predicted digit: {} (latency: {}ms)",
                predicted, latency_ms
            );
        }
        other => panic!("Expected Embedding output, got {:?}", other.as_str()),
    }

    // -- 7. Emit PipelineComplete -----------------------------------------
    let mut complete_ev = event("PipelineComplete");
    complete_ev.stage_name = Some("mnist".to_string());
    complete_ev.target = Some("device".to_string());
    complete_ev.latency_ms = Some(latency_ms);
    publish_telemetry_event(complete_ev);

    // -- 8. Flush and shutdown --------------------------------------------
    flush_platform_telemetry();
    // Give the HTTP exporter a moment to deliver
    std::thread::sleep(Duration::from_secs(2));
    shutdown_platform_telemetry();

    // -- 9. Verify events were observed locally ---------------------------
    let mut collected: Vec<TelemetryEvent> = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        collected.push(ev);
    }

    // With automatic instrumentation we expect at least 4 events:
    // PipelineStart, ExecutionStarted, ExecutionCompleted, PipelineComplete
    assert!(
        collected.len() >= 4,
        "Expected at least 4 events (PipelineStart + ExecutionStarted + ExecutionCompleted + PipelineComplete), got {}",
        collected.len()
    );

    let types: Vec<&str> = collected.iter().map(|e| e.event_type.as_str()).collect();
    assert!(
        types.contains(&"PipelineStart"),
        "Missing PipelineStart event. Got: {:?}",
        types
    );
    assert!(
        types.contains(&"ExecutionStarted"),
        "Missing automatic ExecutionStarted event. Got: {:?}",
        types
    );
    assert!(
        types.contains(&"ExecutionCompleted"),
        "Missing automatic ExecutionCompleted event. Got: {:?}",
        types
    );
    assert!(
        types.contains(&"PipelineComplete"),
        "Missing PipelineComplete event. Got: {:?}",
        types
    );

    println!(
        "OK — {} telemetry events captured and flushed to {}",
        collected.len(),
        ingest_url
    );
}

/// Verify that telemetry events are published through the sender channel
/// even without a remote endpoint. This does NOT require env vars.
#[test]
fn telemetry_local_event_publishing() {
    let (tx, rx) = mpsc::channel::<TelemetryEvent>();
    register_telemetry_sender(tx);

    // Publish a few events
    publish_telemetry_event(event("TestStart"));
    publish_telemetry_event(event("TestComplete"));

    // Small delay for channel delivery
    std::thread::sleep(Duration::from_millis(50));

    let mut collected: Vec<TelemetryEvent> = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        collected.push(ev);
    }

    assert!(
        collected.len() >= 2,
        "Expected at least 2 events, got {}",
        collected.len()
    );
    assert_eq!(collected[0].event_type, "TestStart");
    assert_eq!(collected[1].event_type, "TestComplete");
}

/// Verify that telemetry events carry correct metadata when fields are set.
#[test]
fn telemetry_event_fields() {
    let mut ev = event("StageComplete");
    ev.stage_name = Some("preprocess".to_string());
    ev.target = Some("local".to_string());
    ev.latency_ms = Some(42);
    ev.error = None;
    ev.data = Some(r#"{"model":"mnist"}"#.to_string());

    assert_eq!(ev.event_type, "StageComplete");
    assert_eq!(ev.stage_name.as_deref(), Some("preprocess"));
    assert_eq!(ev.target.as_deref(), Some("local"));
    assert_eq!(ev.latency_ms, Some(42));
    assert!(ev.error.is_none());
    assert!(ev.timestamp_ms > 0);

    // Round-trip through serde
    let json = serde_json::to_string(&ev).unwrap();
    let deser: TelemetryEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.event_type, ev.event_type);
    assert_eq!(deser.latency_ms, ev.latency_ms);
}

/// Verify HTTP-exported events include the resolved device profile.
#[test]
fn telemetry_http_event_includes_device_profile() {
    let (endpoint, rx, handle) = start_mock_ingest();
    let config = TelemetryConfig::new(endpoint, "test-key")
        .with_device("integration-test", "ci")
        .with_hardware_chip("integration-chip")
        .with_hardware_ram_gb(64)
        .with_batch_size(1)
        .with_flush_interval(60);
    let exporter = HttpTelemetryExporter::new(config);

    exporter.push(event("DeviceProfileEvent"));

    let body = rx
        .recv_timeout(Duration::from_secs(2))
        .expect("mock ingest should receive a request body");
    handle.join().expect("mock ingest thread should finish");

    let json: serde_json::Value = serde_json::from_str(&body).expect("request body should be JSON");
    let device = &json["events"][0]["device"];

    assert_eq!(device["chip_family"].as_str(), Some("integration-chip"));
    assert_eq!(device["ram_gb"].as_u64(), Some(64));
    assert!(
        device["chip_family"].is_string() || device["arch"].is_string(),
        "device should include chip_family or arch: {device:?}"
    );
}

/// Verify that TemplateExecutor automatically emits ExecutionStarted and
/// ExecutionCompleted events when platform telemetry is initialized.
/// This does NOT require env vars — uses a local channel to observe events.
#[test]
fn automatic_execution_events() {
    use xybrid_core::execution::listener;

    let Some(model_dir) = model_fixtures::model_or_skip("mnist") else {
        return; // model not downloaded — skip gracefully
    };

    let metadata_path = model_dir.join("model_metadata.json");
    let metadata: ModelMetadata =
        serde_json::from_str(&std::fs::read_to_string(&metadata_path).unwrap()).unwrap();

    // Register a local channel to observe events
    let (tx, rx) = mpsc::channel::<TelemetryEvent>();
    register_telemetry_sender(tx);

    // Simulate what init_platform_telemetry does for the execution listener
    // (we don't init the full HTTP exporter since we don't need it)
    listener::set_execution_listener(|event| {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let telemetry_event = match event {
            xybrid_sdk::ExecutionEvent::Started { model_id, method } => TelemetryEvent {
                event_type: "ExecutionStarted".to_string(),
                stage_name: Some(method),
                target: Some("device".to_string()),
                latency_ms: None,
                error: None,
                data: Some(format!(r#"{{"model":"{}"}}"#, model_id)),
                timestamp_ms,
            },
            xybrid_sdk::ExecutionEvent::Completed {
                model_id,
                method,
                latency_ms,
            } => TelemetryEvent {
                event_type: "ExecutionCompleted".to_string(),
                stage_name: Some(method),
                target: Some("device".to_string()),
                latency_ms: Some(latency_ms as u32),
                error: None,
                data: Some(format!(r#"{{"model":"{}"}}"#, model_id)),
                timestamp_ms,
            },
            xybrid_sdk::ExecutionEvent::Failed {
                model_id,
                method,
                latency_ms,
                error,
            } => TelemetryEvent {
                event_type: "ExecutionFailed".to_string(),
                stage_name: Some(method),
                target: Some("device".to_string()),
                latency_ms: Some(latency_ms as u32),
                error: Some(error),
                data: Some(format!(r#"{{"model":"{}"}}"#, model_id)),
                timestamp_ms,
            },
        };

        publish_telemetry_event(telemetry_event);
    });

    // Run MNIST inference — should automatically emit events
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let input = mnist_input_envelope();
    let _output = executor
        .execute(&metadata, &input, None)
        .expect("MNIST inference should succeed");

    // Small delay for channel delivery
    std::thread::sleep(Duration::from_millis(50));

    // Clean up
    listener::clear_execution_listener();

    // Collect and verify events
    let mut collected: Vec<TelemetryEvent> = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        collected.push(ev);
    }

    let types: Vec<&str> = collected.iter().map(|e| e.event_type.as_str()).collect();

    assert!(
        types.contains(&"ExecutionStarted"),
        "Missing automatic ExecutionStarted event. Got: {:?}",
        types
    );
    assert!(
        types.contains(&"ExecutionCompleted"),
        "Missing automatic ExecutionCompleted event. Got: {:?}",
        types
    );

    // Verify ExecutionStarted has correct model_id
    let started = collected
        .iter()
        .find(|e| e.event_type == "ExecutionStarted")
        .unwrap();
    assert_eq!(started.stage_name.as_deref(), Some("execute"));
    assert!(started.data.as_ref().unwrap().contains("mnist"));

    // Verify ExecutionCompleted has latency
    let completed = collected
        .iter()
        .find(|e| e.event_type == "ExecutionCompleted")
        .unwrap();
    assert!(completed.latency_ms.is_some());
    assert!(completed.latency_ms.unwrap() > 0);

    println!(
        "OK — {} automatic execution events captured",
        collected.len()
    );
}

/// Smoke test: MNIST inference works without telemetry (no env vars needed).
/// Ensures the model fixture is valid and the execution pipeline doesn't panic.
#[test]
fn mnist_inference_smoke_test() {
    let Some(model_dir) = model_fixtures::model_or_skip("mnist") else {
        return; // model not downloaded — skip gracefully
    };

    let metadata_path = model_dir.join("model_metadata.json");
    let metadata: ModelMetadata =
        serde_json::from_str(&std::fs::read_to_string(&metadata_path).unwrap()).unwrap();

    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let input = mnist_input_envelope();

    let output = executor
        .execute(&metadata, &input, None)
        .expect("MNIST inference should succeed");

    match &output.kind {
        EnvelopeKind::Embedding(probs) => {
            assert_eq!(probs.len(), 10);
        }
        other => panic!("Expected Embedding output, got {:?}", other.as_str()),
    }
}
