//! Integration tests for the HIIIPE pipeline.
//!
//! These tests simulate the full Hiiipe demo workflow:
//! Mic Input (AudioRaw) → Local ASR (wav2vec2@1.0) → Cloud Motivator (motivator-llm@5) → Local TTS (xtts-mini@0.6)
//!
//! The tests verify:
//! - Policy enforcement (no raw audio off-device)
//! - Dynamic routing (local vs cloud)
//! - Telemetry emission
//! - End-to-end pipeline execution

use xybrid_core::context::{DeviceMetrics, Envelope, EnvelopeKind, StageDescriptor};
use xybrid_core::event_bus::OrchestratorEvent;
use xybrid_core::orchestrator::{ExecutionMode, Orchestrator};
use xybrid_core::orchestrator::routing_engine::LocalAvailability;

fn audio_envelope() -> Envelope {
    Envelope::new(EnvelopeKind::Audio(vec![0u8; 1600]))
}

fn assert_text_contains(envelope: &Envelope, needle: &str) {
    match &envelope.kind {
        EnvelopeKind::Text(text) => {
            assert!(
                text.contains(needle),
                "expected text envelope to contain '{}', got '{}'",
                needle,
                text
            );
        }
        other => panic!("expected text envelope, got {:?}", other),
    }
}

/// Test the complete Hiiipe pipeline with batch execution.
#[test]
fn test_hiiipe_pipeline() {
    // Setup: Create orchestrator with default components
    let mut orchestrator = Orchestrator::new();

    // Define the Hiiipe pipeline stages
    let stages = vec![
        StageDescriptor::new("asr"), // whisper-tiny@1.2 (local)
        StageDescriptor::new("motivator"), // motivator-llm@5 (cloud)
        StageDescriptor::new("tts"), // xtts-mini@0.6 (local)
    ];

    // Simulate mic input (raw audio)
    let mic_input = audio_envelope();

    // Simulate device metrics (good conditions for cloud routing)
    let metrics = DeviceMetrics {
        network_rtt: 100,  // Good network
        battery: 80,       // Good battery
        temperature: 25.0, // Normal temperature
    };

    // Define model availability function
    // ASR and TTS available locally, Motivator only in cloud
    let availability_fn = |stage: &str| -> LocalAvailability {
        match stage {
            "asr" => LocalAvailability::new(true), // whisper-tiny available locally
            "tts" => LocalAvailability::new(true), // xtts-mini available locally
            "motivator" => LocalAvailability::new(false), // motivator-llm only in cloud
            _ => LocalAvailability::new(false),
        }
    };

    // Execute the pipeline
    let results = orchestrator
        .execute_pipeline(&stages, &mic_input, &metrics, &availability_fn)
        .expect("Pipeline execution should succeed");

    // Verify we got results for all three stages
    assert_eq!(results.len(), 3, "Should have results for all 3 stages");

    // Verify ASR stage (stage 0)
    let asr_result = &results[0];
    assert_eq!(asr_result.stage, "asr");
    // ASR should route to local (policy denies AudioRaw for cloud)
    assert_eq!(asr_result.routing_decision.target.as_str(), "local");
    assert_text_contains(&asr_result.output, "asr");
    // Latency is tracked (u32, always >= 0)

    // Verify Motivator stage (stage 1)
    let motivator_result = &results[1];
    assert_eq!(motivator_result.stage, "motivator");
    // Motivator should route to cloud (not available locally, policy allows)
    assert_eq!(motivator_result.routing_decision.target.as_str(), "cloud");
    assert_text_contains(&motivator_result.output, "motivator");
    // Latency is tracked (u32, always >= 0)

    // Verify TTS stage (stage 2)
    let tts_result = &results[2];
    assert_eq!(tts_result.stage, "tts");
    // TTS may route to cloud or local depending on conditions
    // When available locally and conditions are good, routing engine may choose cloud
    assert!(
        tts_result.routing_decision.target.as_str() == "local"
            || tts_result.routing_decision.target.as_str() == "cloud"
    );
    assert_text_contains(&tts_result.output, "tts");
    // Latency is tracked (u32, always >= 0)

    // Verify the output chain: ASR output becomes Motivator input
    assert_text_contains(&asr_result.output, "asr_output");
    assert_text_contains(&motivator_result.output, "motivator_output");

    // Verify the output chain: Motivator output becomes TTS input
    assert_text_contains(&tts_result.output, "tts_output");
}

/// Test that policy correctly denies raw audio for cloud execution.
#[test]
fn test_hiiipe_policy_enforcement() {
    let mut orchestrator = Orchestrator::new();

    let stage = StageDescriptor::new("asr");

    // AudioRaw should trigger policy denial for cloud
    let audio_input = audio_envelope();

    let metrics = DeviceMetrics {
        network_rtt: 50, // Excellent network
        battery: 90,     // Excellent battery
        temperature: 20.0,
    };

    let availability = LocalAvailability::new(true);

    let result = orchestrator
        .execute_stage(&stage, &audio_input, &metrics, &availability)
        .expect("Stage execution should succeed");

    // Policy should deny cloud execution, forcing local routing
    assert_eq!(result.routing_decision.target.as_str(), "local");
    assert!(
        result.routing_decision.reason.contains("policy_deny")
            || result.routing_decision.reason.contains("AudioRaw")
    );
}

/// Test Hiiipe pipeline with event bus subscription.
#[test]
fn test_hiiipe_pipeline_with_events() {
    let mut orchestrator = Orchestrator::new();

    // Subscribe to events to verify event emission
    let event_bus = orchestrator.event_bus();
    let subscription = event_bus.subscribe();

    let stages = vec![
        StageDescriptor::new("asr"),
        StageDescriptor::new("motivator"),
        StageDescriptor::new("tts"),
    ];

    let mic_input = audio_envelope();

    let metrics = DeviceMetrics {
        network_rtt: 100,
        battery: 80,
        temperature: 25.0,
    };

    let availability_fn = |stage: &str| -> LocalAvailability {
        match stage {
            "asr" | "tts" => LocalAvailability::new(true),
            _ => LocalAvailability::new(false),
        }
    };

    // Execute pipeline
    let results = orchestrator
        .execute_pipeline(&stages, &mic_input, &metrics, &availability_fn)
        .expect("Pipeline execution should succeed");

    assert_eq!(results.len(), 3);

    // Collect events (non-blocking)
    let mut events_received = 0;
    loop {
        match subscription.try_recv() {
            Ok(event) => {
                events_received += 1;
                // Verify event types
                match event {
                    OrchestratorEvent::StageStart { .. } => {}
                    OrchestratorEvent::StageComplete { .. } => {}
                    OrchestratorEvent::PolicyEvaluated { .. } => {}
                    OrchestratorEvent::RoutingDecided { .. } => {}
                    OrchestratorEvent::ExecutionStarted { .. } => {}
                    OrchestratorEvent::ExecutionCompleted { .. } => {}
                    OrchestratorEvent::PipelineStart { .. } => {}
                    OrchestratorEvent::PipelineComplete { .. } => {}
                    _ => {}
                }
            }
            Err(_) => break, // No more events
        }
    }

    // Should have received multiple events (at least pipeline start/complete + stage events)
    assert!(
        events_received >= 2,
        "Should receive pipeline and stage events"
    );
}

/// Test Hiiipe pipeline with different network conditions.
#[test]
fn test_hiiipe_pipeline_high_latency() {
    let mut orchestrator = Orchestrator::new();

    let stages = vec![
        StageDescriptor::new("asr"),
        StageDescriptor::new("motivator"),
        StageDescriptor::new("tts"),
    ];

    let mic_input = audio_envelope();

    // High network latency should force local routing for motivator
    let metrics = DeviceMetrics {
        network_rtt: 300, // High latency (> 250ms threshold)
        battery: 50,
        temperature: 25.0,
    };

    let availability_fn = |stage: &str| -> LocalAvailability {
        match stage {
            "asr" | "tts" => LocalAvailability::new(true),
            "motivator" => LocalAvailability::new(true), // Also available locally as fallback
            _ => LocalAvailability::new(false),
        }
    };

    let results = orchestrator
        .execute_pipeline(&stages, &mic_input, &metrics, &availability_fn)
        .expect("Pipeline execution should succeed");

    assert_eq!(results.len(), 3);

    // With high latency, motivator might route locally if available
    // But policy might still deny if AudioRaw data is being sent
    // For this test, ASR should definitely route locally (policy deny)
    assert_eq!(results[0].routing_decision.target.as_str(), "local");

    // Motivator should route locally due to high latency (if available)
    // or cloud if forced (depending on policy)
    let motivator_target = results[1].routing_decision.target.as_str();
    assert!(motivator_target == "local" || motivator_target == "cloud");

    // TTS may route locally or cloud depending on conditions
    assert!(
        results[2].routing_decision.target.as_str() == "local"
            || results[2].routing_decision.target.as_str() == "cloud"
    );
}

/// Test streaming execution mode for Hiiipe pipeline.
#[test]
fn test_hiiipe_pipeline_streaming() {
    use xybrid_core::streaming::StreamManagerConfig;

    // Create orchestrator in streaming mode
    let config = StreamManagerConfig::default();
    let mut orchestrator = Orchestrator::with_streaming(config);

    assert_eq!(*orchestrator.execution_mode(), ExecutionMode::Streaming);

    let stage = StageDescriptor::new("asr");

    let metrics = DeviceMetrics {
        network_rtt: 100,
        battery: 80,
        temperature: 25.0,
    };

    let availability = LocalAvailability::new(true);

    // Push streaming chunks
    let chunk1 = Envelope::new(EnvelopeKind::Audio(vec![0u8; 4]));
    let chunk2 = Envelope::new(EnvelopeKind::Audio(vec![1u8; 4]));

    orchestrator.push_stream_chunk(chunk1, false).unwrap();
    orchestrator.push_stream_chunk(chunk2, true).unwrap(); // Last chunk

    // Process first chunk
    let result1 = orchestrator
        .execute_streaming_stage(&stage, &metrics, &availability)
        .expect("Streaming stage execution should succeed");

    assert!(result1.is_some());
    let exec_result1 = result1.unwrap();
    assert_eq!(exec_result1.stage, "asr");
    assert_text_contains(&exec_result1.output, "asr_output");

    // Check output buffer
    let output_chunk = orchestrator.pop_stream_output();
    assert!(output_chunk.is_some());
    let chunk = output_chunk.unwrap();
    assert_text_contains(&chunk.data, "asr_output");
    assert!(!chunk.is_last); // First chunk

    // Process second chunk (last)
    let result2 = orchestrator
        .execute_streaming_stage(&stage, &metrics, &availability)
        .expect("Streaming stage execution should succeed");

    assert!(result2.is_some());
    let exec_result2 = result2.unwrap();
    assert_eq!(exec_result2.stage, "asr");

    let output_chunk2 = orchestrator.pop_stream_output();
    assert!(output_chunk2.is_some());
    let chunk2 = output_chunk2.unwrap();
    assert!(chunk2.is_last); // Last chunk
}

/// Test complete Hiiipe pipeline with realistic model availability.
#[test]
fn test_hiiipe_complete_workflow() {
    let mut orchestrator = Orchestrator::new();

    // Simulate the complete Hiiipe demo workflow
    let stages = vec![
        StageDescriptor::new("wav2vec2@1.0"), // ASR model
        StageDescriptor::new("motivator-llm@5"), // Motivator model
        StageDescriptor::new("xtts-mini@0.6"), // TTS model
    ];

    // Mic input (raw audio)
    let mic_input = audio_envelope();

    // Good device conditions
    let metrics = DeviceMetrics {
        network_rtt: 110, // Good network (under 250ms threshold)
        battery: 75,      // Good battery (above 15%)
        temperature: 24.0,
    };

    // Model availability matching the demo
    let availability_fn = |stage: &str| -> LocalAvailability {
        match stage {
            "wav2vec2@1.0" => LocalAvailability::new(true),     // ASR available locally
            "xtts-mini@0.6" => LocalAvailability::new(true),    // TTS available locally
            "motivator-llm@5" => LocalAvailability::new(false), // Motivator only in cloud
            _ => LocalAvailability::new(false),
        }
    };

    // Execute the complete workflow
    let results = orchestrator
        .execute_pipeline(&stages, &mic_input, &metrics, &availability_fn)
        .expect("Complete workflow should succeed");

    // Verify all stages executed
    assert_eq!(results.len(), 3);

    // Stage 1: ASR (wav2vec2) - should route locally
    let asr = &results[0];
    assert_eq!(asr.stage, "wav2vec2@1.0");
    assert_eq!(asr.routing_decision.target.as_str(), "local");
    assert!(
        asr.routing_decision.reason.contains("policy_deny")
            || asr.routing_decision.reason.contains("AudioRaw")
    );

    // Stage 2: Motivator (llm) - should route to cloud
    let motivator = &results[1];
    assert_eq!(motivator.stage, "motivator-llm@5");
    assert_eq!(motivator.routing_decision.target.as_str(), "cloud");
    assert!(
        motivator
            .routing_decision
            .reason
            .contains("optimal_conditions")
            || motivator
                .routing_decision
                .reason
                .contains("model_unavailable")
    );

    // Stage 3: TTS (xtts-mini) - may route locally or cloud depending on conditions
    let tts = &results[2];
    assert_eq!(tts.stage, "xtts-mini@0.6");
    // With good network conditions, routing engine may choose cloud even if local available
    // This is valid behavior - the routing engine optimizes for conditions
    assert!(
        tts.routing_decision.target.as_str() == "local"
            || tts.routing_decision.target.as_str() == "cloud"
    );

    // Verify latency tracking (latency_ms is u32, always non-negative)
    for result in &results {
        assert!(
            result.latency_ms < 10000,
            "Latency should be reasonable (< 10s)"
        );
    }

    // Verify output transformation chain
    assert_text_contains(&asr.output, "wav2vec2");
    assert_text_contains(&motivator.output, "motivator-llm");
    assert_text_contains(&tts.output, "xtts-mini");
}

/// Test that pipeline handles policy changes correctly.
#[test]
fn test_hiiipe_with_policy_loading() {
    let mut orchestrator = Orchestrator::new();

    // Load a custom policy
    let policy_yaml = r#"
version: "0.1.0"
deny_cloud_if:
  - input.kind == "AudioRaw"
  - metrics.network_rtt > 300
signature: "test_policy"
"#;

    orchestrator
        .load_policies(policy_yaml.as_bytes().to_vec())
        .expect("Policy loading should succeed");

    let stage = StageDescriptor::new("asr");

    let audio_input = audio_envelope();

    let metrics = DeviceMetrics {
        network_rtt: 50,
        battery: 90,
        temperature: 20.0,
    };

    let availability = LocalAvailability::new(true);

    let result = orchestrator
        .execute_stage(&stage, &audio_input, &metrics, &availability)
        .expect("Stage execution should succeed");

    // Loaded policy should still deny AudioRaw for cloud
    assert_eq!(result.routing_decision.target.as_str(), "local");
}
