//! Integration tests for the HIIIPE pipeline.
//!
//! These tests simulate the Hiiipe demo workflow:
//! Mic Input (audio) → Local ASR → Cloud Motivator → Local TTS
//!
//! They drive the public `Orchestrator` with named mock adapters for both legs
//! and a fixed, quiet device snapshot, so every assertion is about routing and
//! policy — not about the machine the test happens to run on and not about any
//! network. The tests verify:
//! - Policy enforcement (raw audio never leaves the device)
//! - Dynamic routing (local vs cloud) and the adapter that actually ran
//! - Event emission
//! - End-to-end pipeline execution, including the default bootstrapped
//!   orchestrator honouring `load_policies`

use std::sync::Arc;
use std::time::Duration;
use xybrid_core::context::{DeviceMetrics, Envelope, EnvelopeKind, StageDescriptor};
use xybrid_core::device::{ResourceSnapshot, ResourceSnapshotProvider};
use xybrid_core::event_bus::OrchestratorEvent;
use xybrid_core::orchestrator::routing_engine::LocalAvailability;
use xybrid_core::orchestrator::{ExecutionMode, LocalAuthority, Orchestrator};
use xybrid_core::runtime_adapter::RuntimeAdapter;
use xybrid_core::testing::mocks::MockRuntimeAdapter;

/// Raw audio must stay on the device.
const AUDIO_STAYS_LOCAL_POLICY: &[u8] = b"deny_cloud_if:\n  - 'input.kind == \"audio\"'\n";

/// A device with no observed stress, so the default-local rule applies.
#[derive(Debug)]
struct QuietDevice;

impl ResourceSnapshotProvider for QuietDevice {
    fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
        ResourceSnapshot::unknown()
    }
}

fn audio_envelope() -> Envelope {
    Envelope::new(EnvelopeKind::Audio(vec![0u8; 1600]))
}

fn text_of(envelope: &Envelope) -> &str {
    envelope
        .as_text()
        .unwrap_or_else(|| panic!("expected text envelope, got {:?}", envelope.kind))
}

fn loaded_mock(name: &str, output: &str) -> Arc<MockRuntimeAdapter> {
    let mut adapter = MockRuntimeAdapter::with_text_output(output).with_name(name);
    adapter.load_model("/mock/model").unwrap();
    Arc::new(adapter)
}

struct Harness {
    orchestrator: Orchestrator,
    local: Arc<MockRuntimeAdapter>,
    cloud: Arc<MockRuntimeAdapter>,
}

/// Orchestrator on a quiet device with the audio-stays-local policy loaded and
/// one named mock adapter per leg.
fn hiiipe_harness() -> Harness {
    let authority = LocalAuthority::new().with_resource_provider(Arc::new(QuietDevice));
    let mut orchestrator = Orchestrator::with_authority(Box::new(authority));
    orchestrator
        .load_policies(AUDIO_STAYS_LOCAL_POLICY.to_vec())
        .expect("policy loads");
    let local = loaded_mock("onnx", "local output");
    let cloud = loaded_mock("cloud", "cloud output");
    orchestrator.executor_mut().register_adapter(local.clone());
    orchestrator.executor_mut().register_adapter(cloud.clone());
    Harness {
        orchestrator,
        local,
        cloud,
    }
}

fn hiiipe_stages() -> Vec<StageDescriptor> {
    vec![
        StageDescriptor::new("asr"),
        StageDescriptor::new("motivator"),
        StageDescriptor::new("tts"),
    ]
}

/// ASR and TTS are cached locally; the motivator LLM exists only in the cloud.
fn hiiipe_availability(stage: &str) -> LocalAvailability {
    LocalAvailability::new(matches!(stage, "asr" | "tts"))
}

#[test]
fn test_hiiipe_pipeline() {
    let Harness {
        mut orchestrator,
        local,
        cloud,
    } = hiiipe_harness();

    let results = orchestrator
        .execute_pipeline(
            &hiiipe_stages(),
            &audio_envelope(),
            &DeviceMetrics::default(),
            &hiiipe_availability,
        )
        .expect("pipeline executes");

    assert_eq!(results.len(), 3);

    // ASR: raw audio in, policy forbids cloud → local.
    let asr = &results[0];
    assert_eq!(asr.stage, "asr");
    assert_eq!(asr.routing_decision.target.as_str(), "local");
    assert!(
        asr.routing_decision.reason.contains("policy_deny"),
        "{}",
        asr.routing_decision.reason
    );
    assert_eq!(asr.adapter, "onnx");
    assert_eq!(text_of(&asr.output), "local output");

    // Motivator: text in, no local model → cloud.
    let motivator = &results[1];
    assert_eq!(motivator.stage, "motivator");
    assert_eq!(motivator.routing_decision.target.as_str(), "cloud");
    assert!(
        motivator
            .routing_decision
            .reason
            .contains("model_unavailable"),
        "{}",
        motivator.routing_decision.reason
    );
    assert_eq!(motivator.adapter, "cloud");
    assert_eq!(text_of(&motivator.output), "cloud output");

    // TTS: text in, local model, quiet device → default local.
    let tts = &results[2];
    assert_eq!(tts.stage, "tts");
    assert_eq!(tts.routing_decision.target.as_str(), "local");
    assert!(
        tts.routing_decision.reason.contains("default_local"),
        "{}",
        tts.routing_decision.reason
    );
    assert_eq!(tts.adapter, "onnx");

    assert_eq!(local.call_count(), 2);
    assert_eq!(cloud.call_count(), 1);
}

#[test]
fn test_hiiipe_policy_enforcement() {
    let Harness {
        mut orchestrator,
        local,
        cloud,
    } = hiiipe_harness();

    // Even when the ASR model is NOT available locally, raw audio must not go
    // to the cloud: the stage is routed local and served by the local adapter
    // (a stage with a provider and no bundle would fail locally instead).
    for available in [true, false] {
        let result = orchestrator
            .execute_stage(
                &StageDescriptor::new("asr"),
                &audio_envelope(),
                &DeviceMetrics::default(),
                &LocalAvailability::new(available),
            )
            .expect("stage executes locally");

        assert_eq!(result.routing_decision.target.as_str(), "local");
        assert!(
            result.routing_decision.reason.contains("policy_deny"),
            "{}",
            result.routing_decision.reason
        );
        assert_eq!(result.adapter, "onnx");
    }
    assert_eq!(local.call_count(), 2);
    assert_eq!(
        cloud.call_count(),
        0,
        "raw audio must never reach the cloud adapter"
    );
}

#[test]
fn test_hiiipe_pipeline_with_events() {
    let Harness {
        mut orchestrator, ..
    } = hiiipe_harness();
    let subscription = orchestrator.event_bus().subscribe();

    let results = orchestrator
        .execute_pipeline(
            &hiiipe_stages(),
            &audio_envelope(),
            &DeviceMetrics::default(),
            &hiiipe_availability,
        )
        .expect("pipeline executes");
    assert_eq!(results.len(), 3);

    let mut policy_events = Vec::new();
    let mut routing_events = Vec::new();
    let mut pipeline_start = 0;
    let mut pipeline_complete = 0;
    while let Ok(event) = subscription.try_recv() {
        match event {
            OrchestratorEvent::PolicyEvaluated {
                stage_name,
                allowed,
                ..
            } => policy_events.push((stage_name, allowed)),
            OrchestratorEvent::RoutingDecided {
                stage_name, target, ..
            } => routing_events.push((stage_name, target)),
            OrchestratorEvent::PipelineStart { .. } => pipeline_start += 1,
            OrchestratorEvent::PipelineComplete { .. } => pipeline_complete += 1,
            _ => {}
        }
    }

    assert_eq!(pipeline_start, 1);
    assert_eq!(pipeline_complete, 1);
    assert_eq!(
        policy_events,
        vec![
            ("asr".to_string(), false),
            ("motivator".to_string(), true),
            ("tts".to_string(), true),
        ]
    );
    assert_eq!(
        routing_events,
        vec![
            ("asr".to_string(), "local".to_string()),
            ("motivator".to_string(), "cloud".to_string()),
            ("tts".to_string(), "local".to_string()),
        ]
    );
}

#[test]
fn test_hiiipe_pipeline_streaming() {
    let Harness {
        mut orchestrator,
        local,
        cloud,
    } = hiiipe_harness();
    orchestrator.set_execution_mode(ExecutionMode::Streaming);
    assert_eq!(*orchestrator.execution_mode(), ExecutionMode::Streaming);

    let stage = StageDescriptor::new("asr");
    let metrics = DeviceMetrics::default();
    let availability = LocalAvailability::new(true);

    orchestrator
        .push_stream_chunk(Envelope::new(EnvelopeKind::Audio(vec![0u8; 4])), false)
        .unwrap();
    orchestrator
        .push_stream_chunk(Envelope::new(EnvelopeKind::Audio(vec![1u8; 4])), true)
        .unwrap();

    let first = orchestrator
        .execute_streaming_stage(&stage, &metrics, &availability)
        .expect("first chunk executes")
        .expect("a chunk was queued");
    assert_eq!(first.stage, "asr");
    assert_eq!(first.routing_decision.target.as_str(), "local");
    assert_eq!(text_of(&first.output), "local output");
    let chunk = orchestrator
        .pop_stream_output()
        .expect("first output chunk");
    assert!(!chunk.is_last);

    let second = orchestrator
        .execute_streaming_stage(&stage, &metrics, &availability)
        .expect("second chunk executes")
        .expect("a chunk was queued");
    assert_eq!(second.routing_decision.target.as_str(), "local");
    let chunk = orchestrator
        .pop_stream_output()
        .expect("second output chunk");
    assert!(chunk.is_last);

    assert_eq!(local.call_count(), 2);
    assert_eq!(cloud.call_count(), 0);
}

#[test]
fn test_hiiipe_complete_workflow() {
    let Harness {
        mut orchestrator, ..
    } = hiiipe_harness();

    let stages = vec![
        StageDescriptor::new("wav2vec2@1.0"),
        StageDescriptor::new("motivator-llm@5"),
        StageDescriptor::new("xtts-mini@0.6"),
    ];
    let availability = |stage: &str| -> LocalAvailability {
        LocalAvailability::new(matches!(stage, "wav2vec2@1.0" | "xtts-mini@0.6"))
    };

    let results = orchestrator
        .execute_pipeline(
            &stages,
            &audio_envelope(),
            &DeviceMetrics::default(),
            &availability,
        )
        .expect("complete workflow executes");

    let targets: Vec<&str> = results
        .iter()
        .map(|r| r.routing_decision.target.as_str())
        .collect();
    assert_eq!(targets, vec!["local", "cloud", "local"]);
    let adapters: Vec<&str> = results.iter().map(|r| r.adapter.as_str()).collect();
    assert_eq!(adapters, vec!["onnx", "cloud", "onnx"]);
    for result in &results {
        assert!(result.latency_ms < 10_000, "latency should be reasonable");
    }
}

/// The default bootstrapped orchestrator must honour `load_policies` — this
/// used to be silently ignored because the bundle went to an engine nothing
/// consulted.
#[test]
fn test_hiiipe_with_policy_loading() {
    let mut orchestrator = Orchestrator::new();
    // Replace the bootstrapped adapters with loaded mocks under the same names.
    let local = loaded_mock("onnx", "local output");
    let cloud = loaded_mock("cloud", "cloud output");
    orchestrator.executor_mut().register_adapter(local.clone());
    orchestrator.executor_mut().register_adapter(cloud.clone());

    // Legacy label still accepted as an alias for `audio`.
    orchestrator
        .load_policies(b"version: \"0.1.0\"\ndeny_cloud_if:\n  - 'input.kind == \"AudioRaw\"'\nsignature: \"test_policy\"\n".to_vec())
        .expect("policy loads");

    // Audio with no local model would route to cloud without the policy.
    let denied = orchestrator
        .execute_stage(
            &StageDescriptor::new("asr"),
            &audio_envelope(),
            &DeviceMetrics::default(),
            &LocalAvailability::new(false),
        )
        .expect("local mock serves the stage");
    assert_eq!(denied.routing_decision.target.as_str(), "local");
    assert!(
        denied.routing_decision.reason.contains("policy_deny"),
        "{}",
        denied.routing_decision.reason
    );
    assert_eq!(cloud.call_count(), 0);

    // Text is not covered by the policy: no local model → cloud.
    let allowed = orchestrator
        .execute_stage(
            &StageDescriptor::new("motivator"),
            &Envelope::new(EnvelopeKind::Text("hello".to_string())),
            &DeviceMetrics::default(),
            &LocalAvailability::new(false),
        )
        .expect("cloud mock serves the stage");
    assert_eq!(allowed.routing_decision.target.as_str(), "cloud");
    assert_eq!(cloud.call_count(), 1);

    // An invalid reload is rejected and the audio denial still holds.
    let err = orchestrator
        .load_policies(b"deny_cloud_if:\n  - 'metrics.network_rtt > 300'\n".to_vec())
        .expect_err("unknown operand is rejected");
    assert!(err.to_string().contains("unknown operand"), "{err}");
    let still_denied = orchestrator
        .execute_stage(
            &StageDescriptor::new("asr"),
            &audio_envelope(),
            &DeviceMetrics::default(),
            &LocalAvailability::new(false),
        )
        .expect("local mock serves the stage");
    assert_eq!(still_denied.routing_decision.target.as_str(), "local");
    assert_eq!(cloud.call_count(), 1);
}
