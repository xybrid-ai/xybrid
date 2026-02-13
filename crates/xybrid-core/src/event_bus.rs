//! Event Bus module - Event-driven communication between orchestrator components.
//!
//! The Event Bus provides a pub/sub mechanism for components to communicate asynchronously,
//! enabling loose coupling and reactive behavior throughout the orchestrator.
//!
//! For MVP, this implements a simple event enum with a synchronous broadcast channel.
//! Future versions will support async channels (Tokio) and more sophisticated event routing.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

/// Event types that can be published to the event bus.
#[derive(Debug, Clone)]
pub enum OrchestratorEvent {
    /// Stage execution started.
    StageStart { stage_name: String },
    /// Stage execution completed.
    StageComplete {
        stage_name: String,
        target: String,
        latency_ms: u32,
    },
    /// Stage execution failed.
    StageError { stage_name: String, error: String },
    /// Policy evaluation occurred.
    PolicyEvaluated {
        stage_name: String,
        allowed: bool,
        reason: Option<String>,
    },
    /// Routing decision was made.
    RoutingDecided {
        stage_name: String,
        target: String,
        reason: String,
    },
    /// Execution started.
    ExecutionStarted { stage_name: String, target: String },
    /// Execution completed.
    ExecutionCompleted {
        stage_name: String,
        target: String,
        execution_time_ms: u32,
    },
    /// Execution failed.
    ExecutionFailed {
        stage_name: String,
        target: String,
        error: String,
    },
    /// Pipeline started.
    PipelineStart { stages: Vec<String> },
    /// Pipeline completed.
    PipelineComplete { total_latency_ms: u32 },
    /// Bootstrap process started.
    BootstrapStart,
    /// Component initialized during bootstrap.
    ComponentInitialized { component: String },
    /// Adapter registered during bootstrap.
    AdapterRegistered { name: String },
    /// Executor is ready with registered adapters.
    ExecutorReady,
    /// Orchestrator is fully initialized and ready.
    OrchestratorReady,
}

/// Event subscription handle for managing subscriptions.
#[derive(Debug)]
pub struct Subscription {
    id: usize,
    receiver: Receiver<OrchestratorEvent>,
}

impl Subscription {
    /// Try to receive the next event without blocking.
    pub fn try_recv(&self) -> Result<OrchestratorEvent, mpsc::TryRecvError> {
        self.receiver.try_recv()
    }

    /// Receive the next event, blocking until one is available.
    pub fn recv(&self) -> Result<OrchestratorEvent, mpsc::RecvError> {
        self.receiver.recv()
    }

    /// Get the subscription ID.
    pub fn id(&self) -> usize {
        self.id
    }
}

/// Internal structure for managing subscribers.
struct Subscriber {
    sender: Sender<OrchestratorEvent>,
}

/// Event bus for component communication.
///
/// The EventBus implements a publish-subscribe pattern where components can:
/// - Publish events to notify other components
/// - Subscribe to events of interest
/// - Handle events synchronously or asynchronously
pub struct EventBus {
    subscribers: Arc<Mutex<HashMap<usize, Subscriber>>>,
    next_id: Arc<Mutex<usize>>,
}

impl EventBus {
    /// Creates a new event bus instance.
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(0)),
        }
    }

    /// Publish an event to all subscribers.
    pub fn publish(&self, event: OrchestratorEvent) {
        let subscribers = self.subscribers.lock().unwrap();
        let mut failed_ids = Vec::new();

        for (id, subscriber) in subscribers.iter() {
            if subscriber.sender.send(event.clone()).is_err() {
                // Receiver was dropped, mark for removal
                failed_ids.push(*id);
            }
        }

        // Remove dead subscribers
        if !failed_ids.is_empty() {
            drop(subscribers);
            let mut subscribers = self.subscribers.lock().unwrap();
            for id in failed_ids {
                subscribers.remove(&id);
            }
        }
    }

    /// Subscribe to events, returning a subscription handle.
    ///
    /// The subscription allows receiving events manually via `recv()` or `try_recv()`.
    pub fn subscribe(&self) -> Subscription {
        let (sender, receiver) = mpsc::channel();
        let mut next_id = self.next_id.lock().unwrap();
        let id = *next_id;
        *next_id += 1;
        drop(next_id);

        let mut subscribers = self.subscribers.lock().unwrap();
        subscribers.insert(id, Subscriber { sender });

        Subscription { id, receiver }
    }

    /// Subscribe with an event handler that will be called automatically.
    ///
    /// The handler will be invoked whenever an event is published.
    /// Note: For MVP, handlers are called synchronously in a background thread.
    /// Future versions will support async handlers.
    pub fn subscribe_with_handler<F>(&self, handler: F) -> usize
    where
        F: Fn(&OrchestratorEvent) + Send + Sync + 'static,
    {
        let (sender, receiver) = mpsc::channel();
        let mut next_id = self.next_id.lock().unwrap();
        let id = *next_id;
        *next_id += 1;
        drop(next_id);

        let mut subscribers = self.subscribers.lock().unwrap();
        subscribers.insert(id, Subscriber { sender });
        drop(subscribers);

        // Spawn a thread to handle events for this subscription
        let handler_box = Box::new(handler);
        std::thread::spawn(move || loop {
            match receiver.recv() {
                Ok(event) => {
                    handler_box(&event);
                }
                Err(_) => break,
            }
        });

        id
    }

    /// Unsubscribe by subscription ID.
    pub fn unsubscribe(&self, subscription_id: usize) {
        let mut subscribers = self.subscribers.lock().unwrap();
        subscribers.remove(&subscription_id);
    }

    /// Get the number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        let subscribers = self.subscribers.lock().unwrap();
        subscribers.len()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper trait for easy event publishing.
pub trait EventPublisher {
    fn publish_event(&self, event: OrchestratorEvent);
}

impl EventPublisher for EventBus {
    fn publish_event(&self, event: OrchestratorEvent) {
        self.publish(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_event_bus_creation() {
        let bus = EventBus::new();
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn test_publish_and_receive() {
        let bus = EventBus::new();
        let subscription = bus.subscribe();

        bus.publish(OrchestratorEvent::StageStart {
            stage_name: "test_stage".to_string(),
        });

        let event = subscription.recv().unwrap();
        match event {
            OrchestratorEvent::StageStart { stage_name } => {
                assert_eq!(stage_name, "test_stage");
            }
            _ => panic!("Unexpected event type"),
        }
    }

    #[test]
    fn test_multiple_subscribers() {
        let bus = EventBus::new();
        let sub1 = bus.subscribe();
        let sub2 = bus.subscribe();

        bus.publish(OrchestratorEvent::PipelineStart {
            stages: vec!["stage1".to_string(), "stage2".to_string()],
        });

        let event1 = sub1.recv().unwrap();
        let event2 = sub2.recv().unwrap();

        match (event1, event2) {
            (
                OrchestratorEvent::PipelineStart { stages: s1 },
                OrchestratorEvent::PipelineStart { stages: s2 },
            ) => {
                assert_eq!(s1, s2);
                assert_eq!(s1.len(), 2);
            }
            _ => panic!("Unexpected event types"),
        }

        assert_eq!(bus.subscriber_count(), 2);
    }

    #[test]
    fn test_subscribe_with_handler() {
        let bus = EventBus::new();
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        let _subscription_id = bus.subscribe_with_handler(move |event| {
            if let OrchestratorEvent::StageComplete { .. } = event {
                count_clone.fetch_add(1, Ordering::SeqCst);
            }
        });

        bus.publish(OrchestratorEvent::StageComplete {
            stage_name: "test".to_string(),
            target: "local".to_string(),
            latency_ms: 100,
        });

        // Give handler thread time to process
        std::thread::sleep(std::time::Duration::from_millis(10));

        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_unsubscribe() {
        let bus = EventBus::new();
        let subscription = bus.subscribe();
        let id = subscription.id();

        assert_eq!(bus.subscriber_count(), 1);

        bus.unsubscribe(id);

        // Give a moment for cleanup
        std::thread::sleep(std::time::Duration::from_millis(10));

        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn test_try_recv() {
        let bus = EventBus::new();
        let subscription = bus.subscribe();

        // Should return error when no events available
        assert!(subscription.try_recv().is_err());

        bus.publish(OrchestratorEvent::PolicyEvaluated {
            stage_name: "test".to_string(),
            allowed: true,
            reason: None,
        });

        // Should receive event now
        let event = subscription.try_recv().unwrap();
        match event {
            OrchestratorEvent::PolicyEvaluated { allowed, .. } => {
                assert!(allowed);
            }
            _ => panic!("Unexpected event type"),
        }
    }

    #[test]
    fn test_event_cloning() {
        let event = OrchestratorEvent::RoutingDecided {
            stage_name: "test".to_string(),
            target: "cloud".to_string(),
            reason: "optimal".to_string(),
        };

        // Verify Clone works
        let cloned = event.clone();
        match (event, cloned) {
            (
                OrchestratorEvent::RoutingDecided {
                    stage_name: s1,
                    target: t1,
                    reason: r1,
                },
                OrchestratorEvent::RoutingDecided {
                    stage_name: s2,
                    target: t2,
                    reason: r2,
                },
            ) => {
                assert_eq!(s1, s2);
                assert_eq!(t1, t2);
                assert_eq!(r1, r2);
            }
            _ => panic!("Unexpected event types"),
        }
    }

    #[test]
    fn test_all_event_types() {
        let bus = EventBus::new();
        let subscription = bus.subscribe();

        // Test all event types
        bus.publish(OrchestratorEvent::StageStart {
            stage_name: "test".to_string(),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::StageComplete {
            stage_name: "test".to_string(),
            target: "local".to_string(),
            latency_ms: 100,
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::StageError {
            stage_name: "test".to_string(),
            error: "test error".to_string(),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::PolicyEvaluated {
            stage_name: "test".to_string(),
            allowed: true,
            reason: Some("policy passed".to_string()),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::RoutingDecided {
            stage_name: "test".to_string(),
            target: "cloud".to_string(),
            reason: "optimal".to_string(),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::ExecutionStarted {
            stage_name: "test".to_string(),
            target: "local".to_string(),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::ExecutionCompleted {
            stage_name: "test".to_string(),
            target: "local".to_string(),
            execution_time_ms: 50,
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::ExecutionFailed {
            stage_name: "test".to_string(),
            target: "cloud".to_string(),
            error: "execution error".to_string(),
        });
        let _ = subscription.recv().unwrap();
    }
}
