//! Orchestrator Bootstrap module - Initializes the orchestration runtime environment.
//!
//! The bootstrap module provides a unified API for initializing an orchestrator
//! with all required components: adapters, executor, policy engine, routing engine,
//! event bus, and telemetry.
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_core::orchestrator::Orchestrator;
//!
//! let orchestrator = Orchestrator::bootstrap(None)?;
//! // orchestrator is ready to execute pipelines
//! ```

use crate::context::DeviceMetrics;
use crate::control_sync::{
    ControlSync, ControlSyncConfig, ControlSyncHandler, ControlSyncProvider,
    NoopControlSyncHandler, NoopControlSyncProvider,
};
use crate::device_adapter::DeviceAdapter;
use crate::device_adapter::LocalDeviceAdapter;
use crate::event_bus::{EventBus, OrchestratorEvent};
use crate::executor::Executor;
use crate::orchestrator::policy_engine::DefaultPolicyEngine;
use crate::orchestrator::routing_engine::DefaultRoutingEngine;
use crate::orchestrator::{
    ExecutionMode, LocalAuthority, OrchestrationAuthority, Orchestrator, OrchestratorError,
};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::runtime_adapter::CoreMLRuntimeAdapter;
#[cfg(any(target_os = "android"))]
use crate::runtime_adapter::ONNXMobileRuntimeAdapter;
use crate::runtime_adapter::{OnnxRuntimeAdapter, RuntimeAdapter};
use crate::streaming::manager::StreamManager;
use crate::telemetry::{Severity, Telemetry};
use anyhow::{Context, Result};
use serde_json::json;
use std::path::Path;
use std::sync::Arc;

/// Bootstrap configuration loaded from file.
#[derive(Debug, Clone, serde::Deserialize)]
struct BootstrapConfig {
    /// Execution mode (batch or streaming)
    #[serde(default)]
    execution_mode: Option<String>,
    /// Device metrics overrides
    #[serde(default)]
    metrics: Option<DeviceMetricsConfig>,
    /// Adapter configuration
    #[serde(default)]
    adapters: Option<AdapterConfig>,
}

/// Device metrics configuration.
#[derive(Debug, Clone, serde::Deserialize)]
struct DeviceMetricsConfig {
    /// Network RTT override (milliseconds)
    network_rtt: Option<u32>,
    /// Battery level override (0-100)
    battery: Option<u8>,
    /// Temperature override (Celsius)
    temperature: Option<f32>,
}

/// Adapter configuration.
#[derive(Debug, Clone, serde::Deserialize)]
struct AdapterConfig {
    /// Enable local adapter
    #[serde(default = "default_true")]
    local: bool,
    /// Enable cloud adapter
    #[serde(default = "default_true")]
    cloud: bool,
    /// Enable mock adapter
    #[serde(default = "default_false")]
    mock: bool,
}

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

impl Orchestrator {
    /// Bootstrap a new orchestrator instance with registered adapters and telemetry.
    ///
    /// This function initializes all orchestrator components:
    /// - Policy engine with default policies
    /// - Routing engine
    /// - Executor with registered runtime adapters
    /// - Event bus with subscription enabled
    /// - Telemetry for logging
    /// - Device metrics collection
    ///
    /// # Arguments
    ///
    /// * `config_path` - Optional path to configuration file (YAML format)
    ///
    /// # Returns
    ///
    /// A fully initialized `Orchestrator` ready to execute pipelines, or an error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use xybrid_core::orchestrator::Orchestrator;
    ///
    /// // Bootstrap with defaults
    /// let orchestrator = Orchestrator::bootstrap(None)?;
    ///
    /// // Bootstrap with configuration file
    /// let orchestrator = Orchestrator::bootstrap(Some("config/hiiipe.yml"))?;
    /// ```
    pub fn bootstrap(config_path: Option<&Path>) -> Result<Self, OrchestratorError> {
        // Emit bootstrap start event
        let event_bus = EventBus::new();
        event_bus.publish(OrchestratorEvent::BootstrapStart);

        // Load configuration if provided
        let config = if let Some(path) = config_path {
            load_config(path)
                .map_err(|e| OrchestratorError::Other(format!("Failed to load config: {}", e)))?
        } else {
            None
        };

        // Initialize telemetry
        let telemetry = Arc::new(Telemetry::new());
        telemetry.log_bootstrap_start();

        // Initialize policy engine
        let policy_engine = Box::new(DefaultPolicyEngine::with_default_policy());
        event_bus.publish(OrchestratorEvent::ComponentInitialized {
            component: "policy_engine".to_string(),
        });

        // Initialize routing engine
        let routing_engine = Box::new(DefaultRoutingEngine::new());
        event_bus.publish(OrchestratorEvent::ComponentInitialized {
            component: "routing_engine".to_string(),
        });

        // Initialize executor
        // Note: Model downloading is handled by the SDK's RegistryClient.
        // The executor works with already-downloaded models via bundle_path.
        let mut executor = Executor::new();

        event_bus.publish(OrchestratorEvent::ComponentInitialized {
            component: "executor".to_string(),
        });

        // Register adapters based on configuration
        let adapter_config = config
            .as_ref()
            .and_then(|c| c.adapters.as_ref())
            .cloned()
            .unwrap_or_else(|| AdapterConfig {
                local: true,
                cloud: true,
                mock: false,
            });

        // Register local adapters
        if adapter_config.local {
            // Prefer CoreML on macOS/iOS, fallback to ONNX
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                let adapter = Arc::new(CoreMLRuntimeAdapter::new());
                executor.register_adapter(adapter);
                event_bus.publish(OrchestratorEvent::AdapterRegistered {
                    name: "coreml".to_string(),
                });
            }

            // Prefer ONNX Mobile on Android
            #[cfg(target_os = "android")]
            {
                let adapter = Arc::new(ONNXMobileRuntimeAdapter::new());
                executor.register_adapter(adapter);
                event_bus.publish(OrchestratorEvent::AdapterRegistered {
                    name: "onnx-mobile".to_string(),
                });
            }

            // Register ONNX adapter (desktop/fallback)
            #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "android")))]
            {
                let adapter = Arc::new(OnnxRuntimeAdapter::new());
                executor.register_adapter(adapter);
                event_bus.publish(OrchestratorEvent::AdapterRegistered {
                    name: "onnx".to_string(),
                });
            }

            // On macOS/iOS, also register ONNX as fallback
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                let adapter = Arc::new(OnnxRuntimeAdapter::new());
                executor.register_adapter(adapter);
                event_bus.publish(OrchestratorEvent::AdapterRegistered {
                    name: "onnx".to_string(),
                });
            }

            // On Android, also register regular ONNX as fallback
            #[cfg(target_os = "android")]
            {
                let adapter = Arc::new(OnnxRuntimeAdapter::new());
                executor.register_adapter(adapter);
                event_bus.publish(OrchestratorEvent::AdapterRegistered {
                    name: "onnx".to_string(),
                });
            }
        }

        // Register cloud adapter (mock for now)
        if adapter_config.cloud {
            let adapter = Arc::new(CloudRuntimeAdapter::new());
            executor.register_adapter(adapter);
            event_bus.publish(OrchestratorEvent::AdapterRegistered {
                name: "cloud".to_string(),
            });
        }

        // Register mock adapter
        if adapter_config.mock {
            let adapter = Arc::new(MockRuntimeAdapter::new());
            executor.register_adapter(adapter);
            event_bus.publish(OrchestratorEvent::AdapterRegistered {
                name: "mock".to_string(),
            });
        }

        // Initialize stream manager
        let stream_manager = StreamManager::new();

        // Determine execution mode
        let execution_mode = config
            .as_ref()
            .and_then(|c| c.execution_mode.as_ref())
            .map(|m| match m.as_str() {
                "streaming" => ExecutionMode::Streaming,
                _ => ExecutionMode::Batch,
            })
            .unwrap_or(ExecutionMode::Batch);

        // Collect device metrics (for future use in routing decisions)
        let device_adapter = LocalDeviceAdapter::new();
        let _device_metrics =
            if let Some(metrics_config) = config.as_ref().and_then(|c| c.metrics.as_ref()) {
                DeviceMetrics {
                    network_rtt: metrics_config
                        .network_rtt
                        .unwrap_or_else(|| device_adapter.collect_metrics().network_rtt),
                    battery: metrics_config
                        .battery
                        .unwrap_or_else(|| device_adapter.collect_metrics().battery),
                    temperature: metrics_config
                        .temperature
                        .unwrap_or_else(|| device_adapter.collect_metrics().temperature),
                }
            } else {
                device_adapter.collect_metrics()
            };

        // Initialize control sync manager (noop defaults for now)
        let control_sync = {
            let provider: Arc<dyn ControlSyncProvider> = Arc::new(NoopControlSyncProvider);
            let handler: Arc<dyn ControlSyncHandler> = Arc::new(NoopControlSyncHandler);
            match ControlSync::new(
                ControlSyncConfig::default(),
                provider,
                handler,
                telemetry.clone(),
            ) {
                Ok(manager) => Some(manager),
                Err(err) => {
                    telemetry.log_control_sync_event(
                        Severity::Error,
                        "spawn_failed",
                        json!({ "error": err.to_string() }),
                    );
                    None
                }
            }
        };

        // Initialize authority (local by default - fully offline, no phone-home)
        let authority: Box<dyn OrchestrationAuthority> = Box::new(LocalAuthority::new());
        event_bus.publish(OrchestratorEvent::ComponentInitialized {
            component: "authority".to_string(),
        });

        event_bus.publish(OrchestratorEvent::ExecutorReady);
        event_bus.publish(OrchestratorEvent::OrchestratorReady);

        // Create orchestrator instance
        let orchestrator = Orchestrator::with_all(
            authority,
            policy_engine,
            routing_engine,
            executor,
            stream_manager,
            event_bus,
            telemetry.clone(),
            control_sync,
            execution_mode,
        );

        if orchestrator.control_sync.is_some() {
            orchestrator.telemetry.log_control_sync_event(
                Severity::Debug,
                "worker_ready",
                json!({}),
            );
        }

        // Log bootstrap completion
        orchestrator.telemetry.log_bootstrap_complete();

        Ok(orchestrator)
    }
}

/// Load bootstrap configuration from file.
fn load_config(path: &Path) -> Result<Option<BootstrapConfig>> {
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let config: BootstrapConfig = serde_yaml::from_str(&content)
        .with_context(|| format!("Failed to parse config file: {}", path.display()))?;

    Ok(Some(config))
}

/// Cloud runtime adapter (mock implementation).
///
/// This adapter simulates cloud inference execution by adding network latency
/// and returning mock outputs. Future implementations will integrate with
/// actual cloud inference services (gRPC, REST APIs, etc.).
struct CloudRuntimeAdapter {
    // Future: cloud endpoint configuration, auth tokens, etc.
}

impl CloudRuntimeAdapter {
    fn new() -> Self {
        Self {}
    }
}

impl RuntimeAdapter for CloudRuntimeAdapter {
    fn name(&self) -> &str {
        "cloud"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["onnx", "tensorflow", "pytorch"]
    }

    fn load_model(&mut self, _path: &str) -> crate::runtime_adapter::AdapterResult<()> {
        // Cloud models are loaded remotely, not from local files
        Ok(())
    }

    fn execute(
        &self,
        input: &crate::ir::Envelope,
    ) -> crate::runtime_adapter::AdapterResult<crate::ir::Envelope> {
        // Simulate cloud execution with network latency
        use crate::ir::EnvelopeKind;
        use std::thread;

        // Simulate network delay
        thread::sleep(std::time::Duration::from_millis(50));

        // Mock cloud inference
        let output = match &input.kind {
            EnvelopeKind::Audio(_) => EnvelopeKind::Text("cloud-output-transcribed".to_string()),
            EnvelopeKind::Text(t) => EnvelopeKind::Text(format!("cloud-output-{}", t)),
            EnvelopeKind::Embedding(_) => EnvelopeKind::Text("cloud-output".to_string()),
        };

        Ok(crate::ir::Envelope::new(output))
    }
}

/// Mock runtime adapter for testing.
///
/// This adapter always returns mock outputs without any real inference.
/// Useful for testing and development when models are not available.
struct MockRuntimeAdapter {
    // Future: mock response configuration
}

impl MockRuntimeAdapter {
    fn new() -> Self {
        Self {}
    }
}

impl RuntimeAdapter for MockRuntimeAdapter {
    fn name(&self) -> &str {
        "mock"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["*"] // Mock supports all formats
    }

    fn load_model(&mut self, _path: &str) -> crate::runtime_adapter::AdapterResult<()> {
        // Mock adapter doesn't need to load models
        Ok(())
    }

    fn execute(
        &self,
        input: &crate::ir::Envelope,
    ) -> crate::runtime_adapter::AdapterResult<crate::ir::Envelope> {
        // Return mock output
        use crate::ir::EnvelopeKind;

        let output = match &input.kind {
            EnvelopeKind::Audio(_) => EnvelopeKind::Text("mock-output-transcribed".to_string()),
            EnvelopeKind::Text(t) => EnvelopeKind::Text(format!("mock-output-{}", t)),
            EnvelopeKind::Embedding(_) => EnvelopeKind::Text("mock-output".to_string()),
        };

        Ok(crate::ir::Envelope::new(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_default() {
        let orchestrator = Orchestrator::bootstrap(None);
        assert!(orchestrator.is_ok());

        let orchestrator = orchestrator.unwrap();
        assert_eq!(*orchestrator.execution_mode(), ExecutionMode::Batch);
        assert!(orchestrator
            .executor
            .list_adapters()
            .contains(&"onnx".to_string()));
    }

    #[test]
    fn test_bootstrap_with_cloud_adapter() {
        // Create a temporary config file
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "adapters:\n  cloud: true").unwrap();
        let path = file.path();

        let orchestrator = Orchestrator::bootstrap(Some(path));
        assert!(orchestrator.is_ok());

        let orchestrator = orchestrator.unwrap();
        let adapters = orchestrator.executor.list_adapters();
        assert!(adapters.contains(&"cloud".to_string()));
    }

    #[test]
    fn test_bootstrap_with_mock_adapter() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "adapters:\n  mock: true").unwrap();
        let path = file.path();

        let orchestrator = Orchestrator::bootstrap(Some(path));
        assert!(orchestrator.is_ok());

        let orchestrator = orchestrator.unwrap();
        let adapters = orchestrator.executor.list_adapters();
        assert!(adapters.contains(&"mock".to_string()));
    }
}
