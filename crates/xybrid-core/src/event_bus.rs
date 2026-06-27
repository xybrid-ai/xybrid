//! Event Bus module - Event-driven communication between orchestrator components.
//!
//! The Event Bus provides a pub/sub mechanism for components to communicate asynchronously,
//! enabling loose coupling and reactive behavior throughout the orchestrator.
//!
//! For MVP, this implements a simple event enum with a synchronous broadcast channel.
//! Future versions will support async channels (Tokio) and more sophisticated event routing.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use uuid::Uuid;

thread_local! {
    static CURRENT_EVENT_CONTEXT: RefCell<EventContext> = RefCell::new(EventContext::default());
}

/// Producer-side context attached to orchestrator events before they cross
/// thread/task boundaries.
///
/// Consumers should read this struct from the event payload. They must not
/// assume telemetry task-local or thread-local state is still available when
/// a bridge drains the event later.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EventContext {
    pub pipeline_id: Option<Uuid>,
    pub trace_id: Option<Uuid>,
    pub correlation_id: Option<String>,
    pub request_id: Option<String>,
    pub model_id: Option<String>,
    pub span_id: Option<String>,
}

impl EventContext {
    pub fn current() -> Self {
        CURRENT_EVENT_CONTEXT.with(|context| context.borrow().clone())
    }

    pub fn is_empty(&self) -> bool {
        self.pipeline_id.is_none()
            && self.trace_id.is_none()
            && self.correlation_id.is_none()
            && self.request_id.is_none()
            && self.model_id.is_none()
            && self.span_id.is_none()
    }

    pub fn with_pipeline_id(mut self, pipeline_id: Uuid) -> Self {
        self.pipeline_id = Some(pipeline_id);
        self
    }

    pub fn with_trace_id(mut self, trace_id: Uuid) -> Self {
        self.trace_id = Some(trace_id);
        self
    }

    pub fn with_correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }

    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    pub fn with_span_id(mut self, span_id: impl Into<String>) -> Self {
        self.span_id = Some(span_id.into());
        self
    }

    fn merge_missing(mut self, fallback: EventContext) -> Self {
        if self.pipeline_id.is_none() {
            self.pipeline_id = fallback.pipeline_id;
        }
        if self.trace_id.is_none() {
            self.trace_id = fallback.trace_id;
        }
        if self.correlation_id.is_none() {
            self.correlation_id = fallback.correlation_id;
        }
        if self.request_id.is_none() {
            self.request_id = fallback.request_id;
        }
        if self.model_id.is_none() {
            self.model_id = fallback.model_id;
        }
        if self.span_id.is_none() {
            self.span_id = fallback.span_id;
        }
        self
    }
}

/// RAII guard for temporarily installing a producer event context.
pub struct EventContextGuard {
    previous: EventContext,
}

impl EventContextGuard {
    pub fn install(context: EventContext) -> Self {
        let previous = CURRENT_EVENT_CONTEXT.with(|slot| slot.replace(context));
        Self { previous }
    }
}

impl Drop for EventContextGuard {
    fn drop(&mut self) {
        CURRENT_EVENT_CONTEXT.with(|slot| {
            slot.replace(self.previous.clone());
        });
    }
}

pub fn set_current_event_context(context: EventContext) {
    CURRENT_EVENT_CONTEXT.with(|slot| {
        slot.replace(context);
    });
}

pub fn clear_current_event_context() {
    set_current_event_context(EventContext::default());
}

/// Event types that can be published to the event bus.
#[derive(Debug, Clone)]
pub enum OrchestratorEvent {
    /// Stage execution started.
    StageStart {
        stage_name: String,
        context: EventContext,
    },
    /// Stage execution completed.
    StageComplete {
        stage_name: String,
        target: String,
        latency_ms: u32,
        context: EventContext,
    },
    /// Stage execution failed.
    StageError {
        stage_name: String,
        error: String,
        context: EventContext,
    },
    /// Policy evaluation occurred.
    PolicyEvaluated {
        stage_name: String,
        allowed: bool,
        reason: Option<String>,
        context: EventContext,
    },
    /// Routing decision was made.
    ///
    /// `recent_abort_rate` and `sample_size` carry the rolling-window-derived
    /// local reliability hint that the authority computed for this
    /// `(model_id, signal_context)`. Empty-window decisions emit `(0.0, 0)`
    /// so consumers can distinguish "no history yet" from "field absent
    /// because of an older event shape". The SDK bridge hoists these into
    /// `event.data.local_reliability_hint` so the platform exporter can
    /// surface them at the payload top level.
    RoutingDecided {
        stage_name: String,
        target: String,
        reason: String,
        recent_abort_rate: f32,
        sample_size: u32,
        context: EventContext,
    },
    /// Execution started.
    ExecutionStarted {
        stage_name: String,
        target: String,
        context: EventContext,
    },
    /// Execution completed.
    ExecutionCompleted {
        stage_name: String,
        target: String,
        execution_time_ms: u32,
        context: EventContext,
    },
    /// Execution failed.
    ExecutionFailed {
        stage_name: String,
        target: String,
        error: String,
        context: EventContext,
    },
    /// Local execution was cooperatively aborted (e.g. resource pressure
    /// triggered an `AbortPolicy`). Distinct from `ExecutionFailed` because
    /// the run is expected to be retried on cloud by a higher layer; the
    /// `TemplateExecutor` deliberately suppresses its terminal listener
    /// event for these aborts because the orchestrator emits this richer
    /// `LocalAborted` event in its place. Without this variant, listener
    /// consumers see `ExecutionStarted` with no terminal counterpart on the
    /// orchestrator path.
    LocalAborted {
        stage_name: String,
        target: String,
        reason: String,
        context: EventContext,
    },
    /// Pipeline started.
    PipelineStart {
        stages: Vec<String>,
        context: EventContext,
    },
    /// Pipeline completed.
    PipelineComplete {
        total_latency_ms: u32,
        context: EventContext,
    },
    /// Bootstrap process started.
    BootstrapStart { context: EventContext },
    /// Component initialized during bootstrap.
    ComponentInitialized {
        component: String,
        context: EventContext,
    },
    /// Adapter registered during bootstrap.
    AdapterRegistered { name: String, context: EventContext },
    /// Executor is ready with registered adapters.
    ExecutorReady { context: EventContext },
    /// Orchestrator is fully initialized and ready.
    OrchestratorReady { context: EventContext },
}

impl OrchestratorEvent {
    pub fn context(&self) -> &EventContext {
        match self {
            Self::StageStart { context, .. }
            | Self::StageComplete { context, .. }
            | Self::StageError { context, .. }
            | Self::PolicyEvaluated { context, .. }
            | Self::RoutingDecided { context, .. }
            | Self::ExecutionStarted { context, .. }
            | Self::ExecutionCompleted { context, .. }
            | Self::ExecutionFailed { context, .. }
            | Self::LocalAborted { context, .. }
            | Self::PipelineStart { context, .. }
            | Self::PipelineComplete { context, .. }
            | Self::BootstrapStart { context }
            | Self::ComponentInitialized { context, .. }
            | Self::AdapterRegistered { context, .. }
            | Self::ExecutorReady { context }
            | Self::OrchestratorReady { context } => context,
        }
    }

    fn context_mut(&mut self) -> &mut EventContext {
        match self {
            Self::StageStart { context, .. }
            | Self::StageComplete { context, .. }
            | Self::StageError { context, .. }
            | Self::PolicyEvaluated { context, .. }
            | Self::RoutingDecided { context, .. }
            | Self::ExecutionStarted { context, .. }
            | Self::ExecutionCompleted { context, .. }
            | Self::ExecutionFailed { context, .. }
            | Self::LocalAborted { context, .. }
            | Self::PipelineStart { context, .. }
            | Self::PipelineComplete { context, .. }
            | Self::BootstrapStart { context }
            | Self::ComponentInitialized { context, .. }
            | Self::AdapterRegistered { context, .. }
            | Self::ExecutorReady { context }
            | Self::OrchestratorReady { context } => context,
        }
    }

    fn attach_context(mut self, context: EventContext) -> Self {
        let merged = self.context().clone().merge_missing(context);
        *self.context_mut() = merged;
        self
    }
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

    /// Receive the next event, blocking until one is available or timeout elapses.
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> Result<OrchestratorEvent, mpsc::RecvTimeoutError> {
        self.receiver.recv_timeout(timeout)
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
    next_id: AtomicUsize,
    context: EventContext,
}

impl EventBus {
    /// Creates a new event bus instance.
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicUsize::new(0),
            context: EventContext::current(),
        }
    }

    /// Publish an event to all subscribers.
    pub fn publish(&self, event: OrchestratorEvent) {
        let current_context = EventContext::current();
        let context = if current_context.is_empty() {
            self.context.clone()
        } else {
            current_context.merge_missing(self.context.clone())
        };
        let event = event.attach_context(context);
        // The subscriber registry is pure in-memory bookkeeping. `publish` runs
        // on every orchestrator event from multiple threads, so recover from a
        // poisoned lock (`into_inner`) rather than panicking — a single panicked
        // subscriber must not wedge all future event delivery.
        //
        // Take the lock once and drop dead subscribers (whose receiver hung up)
        // in place via `retain`. Sending to a live receiver returns `Ok`, so the
        // entry is kept; a dropped receiver returns `Err`, so it is removed.
        let mut subscribers = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
        subscribers.retain(|_, subscriber| subscriber.sender.send(event.clone()).is_ok());
    }

    /// Subscribe to events, returning a subscription handle.
    ///
    /// The subscription allows receiving events manually via `recv()` or `try_recv()`.
    pub fn subscribe(&self) -> Subscription {
        let (sender, receiver) = mpsc::channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let mut subscribers = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
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
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let mut subscribers = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
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
        let mut subscribers = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
        subscribers.remove(&subscription_id);
    }

    /// Get the number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        let subscribers = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
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

    fn empty_context() -> EventContext {
        EventContext::default()
    }

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
            context: empty_context(),
        });

        let event = subscription.recv().unwrap();
        match event {
            OrchestratorEvent::StageStart { stage_name, .. } => {
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
            context: empty_context(),
        });

        let event1 = sub1.recv().unwrap();
        let event2 = sub2.recv().unwrap();

        match (event1, event2) {
            (
                OrchestratorEvent::PipelineStart { stages: s1, .. },
                OrchestratorEvent::PipelineStart { stages: s2, .. },
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
            context: empty_context(),
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
            context: empty_context(),
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
            recent_abort_rate: 0.0,
            sample_size: 0,
            context: empty_context(),
        };

        // Verify Clone works
        let cloned = event.clone();
        match (event, cloned) {
            (
                OrchestratorEvent::RoutingDecided {
                    stage_name: s1,
                    target: t1,
                    reason: r1,
                    ..
                },
                OrchestratorEvent::RoutingDecided {
                    stage_name: s2,
                    target: t2,
                    reason: r2,
                    ..
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
            context: empty_context(),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::StageComplete {
            stage_name: "test".to_string(),
            target: "local".to_string(),
            latency_ms: 100,
            context: empty_context(),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::StageError {
            stage_name: "test".to_string(),
            error: "test error".to_string(),
            context: empty_context(),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::PolicyEvaluated {
            stage_name: "test".to_string(),
            allowed: true,
            reason: Some("policy passed".to_string()),
            context: empty_context(),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::RoutingDecided {
            stage_name: "test".to_string(),
            target: "cloud".to_string(),
            reason: "optimal".to_string(),
            recent_abort_rate: 0.0,
            sample_size: 0,
            context: empty_context(),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::ExecutionStarted {
            stage_name: "test".to_string(),
            target: "local".to_string(),
            context: empty_context(),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::ExecutionCompleted {
            stage_name: "test".to_string(),
            target: "local".to_string(),
            execution_time_ms: 50,
            context: empty_context(),
        });
        let _ = subscription.recv().unwrap();

        bus.publish(OrchestratorEvent::ExecutionFailed {
            stage_name: "test".to_string(),
            target: "cloud".to_string(),
            error: "execution error".to_string(),
            context: empty_context(),
        });
        let _ = subscription.recv().unwrap();
    }
}
