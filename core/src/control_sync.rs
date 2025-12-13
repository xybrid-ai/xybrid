//! Control Sync module - Synchronizes control plane state and configuration.
//!
//! The Control Sync module manages synchronization of policies, model configurations, and
//! runtime parameters between the control plane and the orchestrator runtime.

use crate::telemetry::{Severity, Telemetry};
use anyhow::Result as AnyResult;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::any::Any;
use std::borrow::Cow;
use std::cmp::min;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use thiserror::Error;

/// Control sync error type.
#[derive(Debug, Error)]
pub enum ControlSyncError {
    #[error("control sync channel error: {0}")]
    Channel(String),
    #[error("control sync provider error: {0}")]
    Provider(String),
    #[error("control sync handler error: {0}")]
    Handler(String),
    #[error("control sync worker spawn error: {0}")]
    Spawn(String),
    #[error("control sync worker join error: {0}")]
    Join(String),
}

/// Result type alias for control sync operations.
pub type ControlSyncResult<T> = Result<T, ControlSyncError>;

/// Control sync configuration.
#[derive(Debug, Clone)]
pub struct ControlSyncConfig {
    /// Interval between background polls.
    pub poll_interval: Duration,
    /// Initial retry backoff delay when sync fails.
    pub retry_backoff: Duration,
    /// Maximum backoff delay when sync continuously fails.
    pub max_retry_backoff: Duration,
    /// Whether to perform a sync immediately on startup.
    pub immediate_initial_sync: bool,
}

impl ControlSyncConfig {
    /// Creates a new config with the given parameters.
    pub fn new(
        poll_interval: Duration,
        retry_backoff: Duration,
        max_retry_backoff: Duration,
    ) -> Self {
        Self {
            poll_interval,
            retry_backoff,
            max_retry_backoff,
            immediate_initial_sync: true,
        }
    }
}

impl Default for ControlSyncConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(30),
            retry_backoff: Duration::from_secs(5),
            max_retry_backoff: Duration::from_secs(60),
            immediate_initial_sync: true,
        }
    }
}

/// Control message variants from the control plane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ControlMessageKind {
    /// Update policy bundles for runtime evaluation.
    PolicyUpdate { version: String, payload: Vec<u8> },
    /// Refresh registry entries or hydrate bundles.
    RegistryRefresh {
        bundle_id: String,
        version: Option<String>,
    },
    /// Control plane heartbeat signal.
    Heartbeat,
    /// Custom message with arbitrary payload.
    Custom {
        kind: String,
        payload: serde_json::Value,
    },
}

impl ControlMessageKind {
    fn kind_name(&self) -> Cow<'_, str> {
        match self {
            ControlMessageKind::PolicyUpdate { .. } => Cow::Borrowed("policy_update"),
            ControlMessageKind::RegistryRefresh { .. } => Cow::Borrowed("registry_refresh"),
            ControlMessageKind::Heartbeat => Cow::Borrowed("heartbeat"),
            ControlMessageKind::Custom { kind, .. } => Cow::Owned(kind.clone()),
        }
    }
}

/// Control plane message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlMessage {
    /// Unique message identifier for deduplication and acknowledgement.
    pub id: String,
    /// Message payload kind.
    pub kind: ControlMessageKind,
    /// Optional timestamp (milliseconds since epoch).
    #[serde(default)]
    pub timestamp: Option<u64>,
}

impl ControlMessage {
    fn kind_name(&self) -> Cow<'_, str> {
        self.kind.kind_name()
    }
}

/// Provider abstraction for fetching messages from a control plane.
pub trait ControlSyncProvider: Send + Sync {
    /// Fetch pending messages from the control plane.
    fn fetch_updates(&self) -> AnyResult<Vec<ControlMessage>>;

    /// Acknowledge that a message has been applied successfully.
    fn acknowledge(&self, _message_id: &str) -> AnyResult<()> {
        Ok(())
    }
}

/// Handler abstraction for applying control plane messages.
pub trait ControlSyncHandler: Send + Sync {
    /// Apply a single control message.
    fn handle(&self, message: &ControlMessage) -> AnyResult<()>;
}

/// No-op provider that always returns an empty update set.
pub struct NoopControlSyncProvider;

impl ControlSyncProvider for NoopControlSyncProvider {
    fn fetch_updates(&self) -> AnyResult<Vec<ControlMessage>> {
        Ok(vec![])
    }
}

/// No-op handler that treats all messages as successfully applied.
pub struct NoopControlSyncHandler;

impl ControlSyncHandler for NoopControlSyncHandler {
    fn handle(&self, _message: &ControlMessage) -> AnyResult<()> {
        Ok(())
    }
}

/// Commands sent to the control sync worker.
enum ControlCommand {
    TriggerSync,
    Shutdown,
}

/// Control plane synchronization manager.
pub struct ControlSync {
    sender: Sender<ControlCommand>,
    handle: Option<JoinHandle<()>>,
}

impl ControlSync {
    /// Spawns a control sync worker with the provided configuration, provider, and handler.
    pub fn new(
        config: ControlSyncConfig,
        provider: Arc<dyn ControlSyncProvider>,
        handler: Arc<dyn ControlSyncHandler>,
        telemetry: Arc<Telemetry>,
    ) -> ControlSyncResult<Self> {
        let (sender, receiver) = mpsc::channel();
        let worker = ControlWorker {
            config,
            provider,
            handler,
            telemetry,
        };

        let handle = thread::Builder::new()
            .name("control-sync-worker".to_string())
            .spawn(move || worker.run(receiver))
            .map_err(|err| ControlSyncError::Spawn(err.to_string()))?;

        Ok(Self {
            sender,
            handle: Some(handle),
        })
    }

    /// Manually trigger an immediate synchronization pass.
    pub fn trigger_sync(&self) -> ControlSyncResult<()> {
        self.send_command(ControlCommand::TriggerSync)
    }

    /// Gracefully shut down the control sync worker.
    pub fn shutdown(mut self) -> ControlSyncResult<()> {
        if self.handle.is_some() {
            let _ = self.send_command(ControlCommand::Shutdown);
            if let Some(handle) = self.handle.take() {
                handle
                    .join()
                    .map_err(|err| ControlSyncError::Join(format_join_error(err)))?;
            }
        }
        Ok(())
    }

    fn send_command(&self, command: ControlCommand) -> ControlSyncResult<()> {
        self.sender
            .send(command)
            .map_err(|err| ControlSyncError::Channel(err.to_string()))
    }
}

impl Drop for ControlSync {
    fn drop(&mut self) {
        if self.handle.is_some() {
            let _ = self.sender.send(ControlCommand::Shutdown);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

struct ControlWorker {
    config: ControlSyncConfig,
    provider: Arc<dyn ControlSyncProvider>,
    handler: Arc<dyn ControlSyncHandler>,
    telemetry: Arc<Telemetry>,
}

impl ControlWorker {
    fn run(self, receiver: Receiver<ControlCommand>) {
        let mut backoff = self.config.retry_backoff;
        let mut pending_sync = self.config.immediate_initial_sync;

        self.telemetry
            .log_control_sync_event(Severity::Info, "worker_start", json!({}));

        loop {
            if pending_sync {
                match self.sync_once() {
                    Ok(_) => {
                        backoff = self.config.retry_backoff;
                    }
                    Err(err) => {
                        self.telemetry.log_control_sync_event(
                            Severity::Warn,
                            "sync_failed",
                            json!({ "error": err.to_string() }),
                        );
                        thread::sleep(backoff);
                        backoff = increase_backoff(backoff, self.config.max_retry_backoff);
                    }
                }
                pending_sync = false;
                continue;
            }

            match receiver.recv_timeout(self.config.poll_interval) {
                Ok(ControlCommand::TriggerSync) => {
                    pending_sync = true;
                }
                Ok(ControlCommand::Shutdown) => {
                    self.telemetry.log_control_sync_event(
                        Severity::Info,
                        "worker_shutdown",
                        json!({}),
                    );
                    break;
                }
                Err(RecvTimeoutError::Timeout) => {
                    pending_sync = true;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    }

    fn sync_once(&self) -> ControlSyncResult<()> {
        self.telemetry
            .log_control_sync_event(Severity::Debug, "sync_start", json!({}));

        let messages = self
            .provider
            .fetch_updates()
            .map_err(|err| ControlSyncError::Provider(err.to_string()))?;

        if messages.is_empty() {
            self.telemetry
                .log_control_sync_event(Severity::Debug, "sync_idle", json!({}));
            return Ok(());
        }

        let mut applied_count = 0u64;

        for message in messages {
            let kind_name = message.kind_name().to_string();
            self.telemetry.log_control_sync_event(
                Severity::Info,
                "message_received",
                json!({
                    "message_id": message.id,
                    "kind": kind_name
                }),
            );

            match self.handler.handle(&message) {
                Ok(_) => {
                    applied_count += 1;
                    self.telemetry.log_control_sync_event(
                        Severity::Info,
                        "message_applied",
                        json!({
                            "message_id": message.id,
                            "kind": message.kind_name()
                        }),
                    );

                    if let Err(err) = self.provider.acknowledge(&message.id) {
                        self.telemetry.log_control_sync_event(
                            Severity::Warn,
                            "ack_failed",
                            json!({
                                "message_id": message.id,
                                "error": err.to_string()
                            }),
                        );
                        return Err(ControlSyncError::Provider(err.to_string()));
                    } else {
                        self.telemetry.log_control_sync_event(
                            Severity::Debug,
                            "acknowledged",
                            json!({
                                "message_id": message.id
                            }),
                        );
                    }
                }
                Err(err) => {
                    self.telemetry.log_control_sync_event(
                        Severity::Error,
                        "message_failed",
                        json!({
                            "message_id": message.id,
                            "kind": message.kind_name(),
                            "error": err.to_string()
                        }),
                    );
                    return Err(ControlSyncError::Handler(err.to_string()));
                }
            }
        }

        self.telemetry.log_control_sync_event(
            Severity::Debug,
            "sync_complete",
            json!({ "message_count": applied_count }),
        );

        Ok(())
    }
}

fn increase_backoff(current: Duration, max: Duration) -> Duration {
    let doubled = current.as_millis().saturating_mul(2);
    let clamped = min(doubled, max.as_millis());
    Duration::from_millis(clamped as u64)
}

fn format_join_error(err: Box<dyn Any + Send>) -> String {
    match err.downcast::<String>() {
        Ok(string) => *string,
        Err(err) => match err.downcast::<&str>() {
            Ok(str_ref) => (*str_ref).to_string(),
            Err(_) => "unknown panic".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    struct MockProvider {
        batches: Mutex<Vec<Vec<ControlMessage>>>,
        acknowledgements: Arc<Mutex<Vec<String>>>,
    }

    impl MockProvider {
        fn new(
            batches: Vec<Vec<ControlMessage>>,
            acknowledgements: Arc<Mutex<Vec<String>>>,
        ) -> Self {
            Self {
                batches: Mutex::new(batches),
                acknowledgements,
            }
        }
    }

    impl ControlSyncProvider for MockProvider {
        fn fetch_updates(&self) -> AnyResult<Vec<ControlMessage>> {
            let mut guard = self.batches.lock().unwrap();
            if guard.is_empty() {
                Ok(vec![])
            } else {
                Ok(guard.remove(0))
            }
        }

        fn acknowledge(&self, message_id: &str) -> AnyResult<()> {
            self.acknowledgements
                .lock()
                .unwrap()
                .push(message_id.to_string());
            Ok(())
        }
    }

    struct MockHandler {
        applied: Arc<Mutex<Vec<String>>>,
        should_fail: AtomicBool,
    }

    impl MockHandler {
        fn new(applied: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                applied,
                should_fail: AtomicBool::new(false),
            }
        }

        #[allow(dead_code)]
        fn fail_once(&self) {
            self.should_fail.store(true, Ordering::SeqCst);
        }
    }

    impl ControlSyncHandler for MockHandler {
        fn handle(&self, message: &ControlMessage) -> AnyResult<()> {
            if self.should_fail.swap(false, Ordering::SeqCst) {
                anyhow::bail!("forced failure");
            }

            self.applied.lock().unwrap().push(message.id.clone());
            Ok(())
        }
    }

    #[test]
    fn control_sync_processes_messages() {
        let applied = Arc::new(Mutex::new(Vec::new()));
        let acknowledgements = Arc::new(Mutex::new(Vec::new()));

        let messages = vec![vec![ControlMessage {
            id: "msg-1".to_string(),
            kind: ControlMessageKind::Heartbeat,
            timestamp: None,
        }]];

        let provider: Arc<dyn ControlSyncProvider> =
            Arc::new(MockProvider::new(messages, acknowledgements.clone()));
        let handler: Arc<dyn ControlSyncHandler> = Arc::new(MockHandler::new(applied.clone()));
        let telemetry = Arc::new(Telemetry::with_enabled(false));

        let mut config = ControlSyncConfig::default();
        config.poll_interval = Duration::from_millis(25);
        config.retry_backoff = Duration::from_millis(10);
        config.max_retry_backoff = Duration::from_millis(40);
        config.immediate_initial_sync = false;

        let control_sync =
            ControlSync::new(config, provider, handler, telemetry).expect("spawn control sync");
        control_sync.trigger_sync().expect("trigger sync");

        thread::sleep(Duration::from_millis(120));

        assert!(applied.lock().unwrap().contains(&"msg-1".to_string()));
        assert!(acknowledgements
            .lock()
            .unwrap()
            .contains(&"msg-1".to_string()));

        control_sync.shutdown().expect("shutdown");
    }
}
