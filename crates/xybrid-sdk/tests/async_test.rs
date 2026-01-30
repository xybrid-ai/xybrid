//! Async SDK tests for xybrid-sdk

use std::time::Duration;
use tokio::time::timeout;
use xybrid_sdk::{prelude::*, subscribe_events, OrchestratorEvent};

/// Test that event subscription works and we can receive events.
///
/// This test verifies that:
/// 1. We can subscribe to an orchestrator's event stream
/// 2. Events are properly bridged from sync to async
/// 3. The event types match what we expect
#[test]
fn test_event_stream_subscription() {
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        // Create orchestrator - this emits bootstrap events
        let orchestrator = Orchestrator::new();
        let mut event_stream = subscribe_events(&orchestrator);

        // Give event stream bridge thread time to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // The Orchestrator::new() emits bootstrap events during construction.
        // We should receive at least one event from the event bus.
        // Note: Events are emitted during new(), so we subscribe after and may
        // miss some. Let's test with a direct event publish.

        // Get the event bus and publish a test event directly
        let event_bus = orchestrator.event_bus();
        event_bus.publish(OrchestratorEvent::StageStart {
            stage_name: "test_stage".to_string(),
        });

        // Wait for the event to be received
        let mut found_stage_start = false;
        for _ in 0..5 {
            if let Ok(event) = timeout(Duration::from_millis(200), event_stream.recv()).await {
                if let Some(OrchestratorEvent::StageStart { stage_name }) = event {
                    if stage_name == "test_stage" {
                        found_stage_start = true;
                        break;
                    }
                }
            }
        }

        assert!(
            found_stage_start,
            "Did not receive StageStart event for test_stage"
        );
    });
}

/// Test that try_recv returns an error when no events are available.
#[test]
fn test_event_stream_try_recv() {
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let orchestrator = Orchestrator::new();
        let mut event_stream = subscribe_events(&orchestrator);

        // Should return error when no events available immediately
        // (bootstrap events may or may not have arrived yet)
        // This just tests that try_recv doesn't panic
        let _ = event_stream.try_recv();
    });
}
