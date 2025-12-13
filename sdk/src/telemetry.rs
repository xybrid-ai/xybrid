//! Telemetry event bridge - Converts OrchestratorEvent to TelemetryEvent
//!
//! This module bridges events from the orchestrator's event bus to the
//! telemetry stream used by Flutter and other consumers.

use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use xybrid_core::event_bus::OrchestratorEvent;

/// Telemetry event type (simplified for FFI)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Event type name
    pub event_type: String,
    /// Stage name (if applicable)
    pub stage_name: Option<String>,
    /// Target (local/cloud/fallback)
    pub target: Option<String>,
    /// Latency in milliseconds (if applicable)
    pub latency_ms: Option<u32>,
    /// Error message (if applicable)
    pub error: Option<String>,
    /// Additional event data as JSON string
    pub data: Option<String>,
    /// Timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
}

/// Global telemetry event channel for streaming
pub type TelemetrySender = mpsc::Sender<TelemetryEvent>;

static TELEMETRY_SENDERS: Mutex<Vec<TelemetrySender>> = Mutex::new(Vec::new());

/// Register a telemetry event sender
pub fn register_telemetry_sender(sender: TelemetrySender) {
    let mut senders = TELEMETRY_SENDERS.lock().unwrap();
    senders.push(sender);
}

/// Convert OrchestratorEvent to TelemetryEvent
pub fn convert_orchestrator_event(event: &OrchestratorEvent) -> TelemetryEvent {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    match event {
        OrchestratorEvent::PipelineStart { stages } => TelemetryEvent {
            event_type: "PipelineStart".to_string(),
            stage_name: None,
            target: None,
            latency_ms: None,
            error: None,
            data: Some(serde_json::json!({"stages": stages}).to_string()),
            timestamp_ms,
        },
        OrchestratorEvent::PipelineComplete { total_latency_ms } => TelemetryEvent {
            event_type: "PipelineComplete".to_string(),
            stage_name: None,
            target: None,
            latency_ms: Some(*total_latency_ms),
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::StageStart { stage_name } => TelemetryEvent {
            event_type: "StageStart".to_string(),
            stage_name: Some(stage_name.clone()),
            target: None,
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::StageComplete {
            stage_name,
            target,
            latency_ms,
        } => TelemetryEvent {
            event_type: "StageComplete".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: Some(*latency_ms),
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::StageError { stage_name, error } => TelemetryEvent {
            event_type: "StageError".to_string(),
            stage_name: Some(stage_name.clone()),
            target: None,
            latency_ms: None,
            error: Some(error.clone()),
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::RoutingDecided {
            stage_name,
            target,
            reason,
        } => TelemetryEvent {
            event_type: "RoutingDecided".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: None,
            error: None,
            data: Some(serde_json::json!({"reason": reason}).to_string()),
            timestamp_ms,
        },
        OrchestratorEvent::ExecutionStarted { stage_name, target } => TelemetryEvent {
            event_type: "ExecutionStarted".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::ExecutionCompleted {
            stage_name,
            target,
            execution_time_ms,
        } => TelemetryEvent {
            event_type: "ExecutionCompleted".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: Some(*execution_time_ms),
            error: None,
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::ExecutionFailed {
            stage_name,
            target,
            error,
        } => TelemetryEvent {
            event_type: "ExecutionFailed".to_string(),
            stage_name: Some(stage_name.clone()),
            target: Some(target.clone()),
            latency_ms: None,
            error: Some(error.clone()),
            data: None,
            timestamp_ms,
        },
        OrchestratorEvent::PolicyEvaluated {
            stage_name,
            allowed,
            reason,
        } => TelemetryEvent {
            event_type: "PolicyEvaluated".to_string(),
            stage_name: Some(stage_name.clone()),
            target: None,
            latency_ms: None,
            error: if *allowed {
                None
            } else {
                reason.clone().or(Some("Policy violation".to_string()))
            },
            data: Some(
                serde_json::json!({
                    "allowed": allowed,
                    "reason": reason
                })
                .to_string(),
            ),
            timestamp_ms,
        },
        _ => TelemetryEvent {
            event_type: format!("{:?}", event),
            stage_name: None,
            target: None,
            latency_ms: None,
            error: None,
            data: Some(format!("{:?}", event)),
            timestamp_ms,
        },
    }
}

/// Publish a telemetry event to all registered subscribers
pub fn publish_telemetry_event(event: TelemetryEvent) {
    let senders = TELEMETRY_SENDERS.lock().unwrap();
    let mut dead_senders = Vec::new();

    for (idx, sender) in senders.iter().enumerate() {
        if sender.send(event.clone()).is_err() {
            dead_senders.push(idx);
        }
    }

    // Remove dead senders
    drop(senders);
    if !dead_senders.is_empty() {
        let mut senders = TELEMETRY_SENDERS.lock().unwrap();
        for idx in dead_senders.iter().rev() {
            senders.remove(*idx);
        }
    }
}

/// Bridge orchestrator events to telemetry stream
///
/// This function subscribes to orchestrator events and converts them
/// to telemetry events, publishing them to all registered subscribers.
pub fn bridge_orchestrator_events(orchestrator: &xybrid_core::orchestrator::Orchestrator) {
    let event_bus = orchestrator.event_bus();
    let subscription = event_bus.subscribe();

    thread::spawn(move || {
        loop {
            match subscription.recv() {
                Ok(event) => {
                    let telemetry_event = convert_orchestrator_event(&event);
                    publish_telemetry_event(telemetry_event);
                }
                Err(_) => break, // Event bus closed
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_stage_start_event() {
        let event = OrchestratorEvent::StageStart {
            stage_name: "asr".to_string(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "StageStart");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert!(telemetry.target.is_none());
        assert!(telemetry.latency_ms.is_none());
        assert!(telemetry.error.is_none());
        assert!(telemetry.timestamp_ms > 0);
    }

    #[test]
    fn test_convert_stage_complete_event() {
        let event = OrchestratorEvent::StageComplete {
            stage_name: "tts".to_string(),
            target: "local".to_string(),
            latency_ms: 150,
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "StageComplete");
        assert_eq!(telemetry.stage_name, Some("tts".to_string()));
        assert_eq!(telemetry.target, Some("local".to_string()));
        assert_eq!(telemetry.latency_ms, Some(150));
        assert!(telemetry.error.is_none());
    }

    #[test]
    fn test_convert_stage_error_event() {
        let event = OrchestratorEvent::StageError {
            stage_name: "asr".to_string(),
            error: "Model not found".to_string(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "StageError");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.error, Some("Model not found".to_string()));
    }

    #[test]
    fn test_convert_pipeline_start_event() {
        let event = OrchestratorEvent::PipelineStart {
            stages: vec!["asr".to_string(), "llm".to_string(), "tts".to_string()],
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PipelineStart");
        assert!(telemetry.stage_name.is_none());
        assert!(telemetry.data.is_some());
        let data = telemetry.data.unwrap();
        assert!(data.contains("asr"));
        assert!(data.contains("llm"));
        assert!(data.contains("tts"));
    }

    #[test]
    fn test_convert_pipeline_complete_event() {
        let event = OrchestratorEvent::PipelineComplete {
            total_latency_ms: 500,
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PipelineComplete");
        assert_eq!(telemetry.latency_ms, Some(500));
    }

    #[test]
    fn test_convert_routing_decided_event() {
        let event = OrchestratorEvent::RoutingDecided {
            stage_name: "asr".to_string(),
            target: "cloud".to_string(),
            reason: "network_optimal".to_string(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "RoutingDecided");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.target, Some("cloud".to_string()));
        assert!(telemetry.data.is_some());
        let data = telemetry.data.unwrap();
        assert!(data.contains("network_optimal"));
    }

    #[test]
    fn test_convert_execution_started_event() {
        let event = OrchestratorEvent::ExecutionStarted {
            stage_name: "asr".to_string(),
            target: "local".to_string(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "ExecutionStarted");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.target, Some("local".to_string()));
    }

    #[test]
    fn test_convert_execution_completed_event() {
        let event = OrchestratorEvent::ExecutionCompleted {
            stage_name: "asr".to_string(),
            target: "local".to_string(),
            execution_time_ms: 75,
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "ExecutionCompleted");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert_eq!(telemetry.target, Some("local".to_string()));
        assert_eq!(telemetry.latency_ms, Some(75));
    }

    #[test]
    fn test_convert_execution_failed_event() {
        let event = OrchestratorEvent::ExecutionFailed {
            stage_name: "tts".to_string(),
            target: "cloud".to_string(),
            error: "Timeout".to_string(),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "ExecutionFailed");
        assert_eq!(telemetry.stage_name, Some("tts".to_string()));
        assert_eq!(telemetry.target, Some("cloud".to_string()));
        assert_eq!(telemetry.error, Some("Timeout".to_string()));
    }

    #[test]
    fn test_convert_policy_evaluated_allowed() {
        let event = OrchestratorEvent::PolicyEvaluated {
            stage_name: "asr".to_string(),
            allowed: true,
            reason: Some("All conditions met".to_string()),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PolicyEvaluated");
        assert_eq!(telemetry.stage_name, Some("asr".to_string()));
        assert!(telemetry.error.is_none()); // No error when allowed
        assert!(telemetry.data.is_some());
    }

    #[test]
    fn test_convert_policy_evaluated_denied() {
        let event = OrchestratorEvent::PolicyEvaluated {
            stage_name: "llm".to_string(),
            allowed: false,
            reason: Some("Privacy policy violation".to_string()),
        };
        let telemetry = convert_orchestrator_event(&event);

        assert_eq!(telemetry.event_type, "PolicyEvaluated");
        assert_eq!(telemetry.stage_name, Some("llm".to_string()));
        assert_eq!(telemetry.error, Some("Privacy policy violation".to_string()));
    }

    #[test]
    fn test_telemetry_event_serialization() {
        let event = TelemetryEvent {
            event_type: "StageStart".to_string(),
            stage_name: Some("asr".to_string()),
            target: None,
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms: 1234567890,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("StageStart"));
        assert!(json.contains("asr"));

        let deserialized: TelemetryEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.event_type, "StageStart");
        assert_eq!(deserialized.stage_name, Some("asr".to_string()));
    }

    #[test]
    fn test_register_and_publish() {
        let (tx, rx) = mpsc::channel();
        register_telemetry_sender(tx);

        let event = TelemetryEvent {
            event_type: "TestEvent".to_string(),
            stage_name: Some("test".to_string()),
            target: None,
            latency_ms: None,
            error: None,
            data: None,
            timestamp_ms: 0,
        };

        publish_telemetry_event(event.clone());

        // Should receive the event
        let received = rx.recv_timeout(std::time::Duration::from_millis(100));
        assert!(received.is_ok());
        let received_event = received.unwrap();
        assert_eq!(received_event.event_type, "TestEvent");
    }
}
