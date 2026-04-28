//! Context module - Shared data structures and types used across the orchestrator.
//!
//! This module defines common types such as `DeviceMetrics` and `StageDescriptor`
//! that are used throughout the orchestrator components. Data envelopes have
//! graduated into the IR layer (`crate::ir::Envelope`); we re-export them here
//! to maintain backwards compatibility while downstream code migrates.

use crate::device::{detect_capabilities, HardwareCapabilities, ResourceSnapshot};
pub use crate::ir::{Envelope, EnvelopeKind};
use crate::pipeline::{ExecutionTarget, IntegrationProvider, StageOptions};

/// Live device metrics (network, battery, temperature, etc.).
#[derive(Debug, Clone)]
pub struct DeviceMetrics {
    /// Legacy network RTT field. Kept for back-compat while richer routing
    /// signals live under `capabilities` and `resource`.
    pub network_rtt: u32,
    /// Legacy battery percentage. Prefer `capabilities.battery_level` for new
    /// routing decisions.
    pub battery: u8,
    /// Legacy temperature in Celsius. Prefer `capabilities.thermal_state` for
    /// new routing decisions.
    pub temperature: f32,
    /// Hardware capability snapshot used by the routing engine.
    pub capabilities: HardwareCapabilities,
    /// Live resource snapshot sampled by `ResourceMonitor`.
    pub resource: ResourceSnapshot,
}

impl DeviceMetrics {
    /// Return a copy with live resource data overlaid onto the capability view.
    pub fn with_live_snapshot(&self, snapshot: ResourceSnapshot) -> Self {
        let mut metrics = self.clone();
        let mut capabilities = detect_capabilities(self);
        metrics.resource = snapshot;
        capabilities.memory_available_mb = snapshot.available_mem_mb.unwrap_or_default() as u64;
        capabilities.memory_total_mb = snapshot.total_mem_mb.unwrap_or_default() as u64;
        capabilities.cpu_usage_percent = snapshot.cpu_pct.unwrap_or(0.0);
        capabilities.thermal_state = snapshot.thermal_state;
        capabilities.battery_level = snapshot.battery_pct.unwrap_or(self.battery);
        metrics.capabilities = capabilities;
        metrics
    }
}

impl Default for DeviceMetrics {
    fn default() -> Self {
        Self {
            network_rtt: 100,
            battery: 100,
            temperature: 25.0,
            capabilities: HardwareCapabilities::default(),
            resource: ResourceSnapshot::default(),
        }
    }
}

/// Metadata descriptor for a pipeline stage.
#[derive(Debug, Clone)]
pub struct StageDescriptor {
    pub name: String,
    /// Path to the downloaded bundle file (.xyb).
    /// Set by the SDK's `RegistryClient` after downloading the model.
    pub bundle_path: Option<String>,
    /// Execution target (device, server, integration, auto).
    /// If None, defaults to device/local execution.
    pub target: Option<ExecutionTarget>,
    /// Integration provider (required for integration target).
    /// E.g., OpenAI, Anthropic, Google, ElevenLabs.
    pub provider: Option<IntegrationProvider>,
    /// Model identifier for integration targets (e.g., "gpt-4o-mini").
    pub model: Option<String>,
    /// Stage-specific options (temperature, max_tokens, system_prompt, etc.).
    pub options: Option<StageOptions>,
}

impl StageDescriptor {
    /// Create a new stage descriptor with just a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bundle_path: None,
            target: None,
            provider: None,
            model: None,
            options: None,
        }
    }

    /// Set the bundle path (path to downloaded .xyb file).
    pub fn with_bundle_path(mut self, path: impl Into<String>) -> Self {
        self.bundle_path = Some(path.into());
        self
    }

    /// Set the execution target.
    pub fn with_target(mut self, target: ExecutionTarget) -> Self {
        self.target = Some(target);
        self
    }

    /// Set the cloud provider.
    pub fn with_provider(mut self, provider: IntegrationProvider) -> Self {
        self.provider = Some(provider);
        self.target = Some(ExecutionTarget::Cloud);
        self
    }

    /// Set the model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set stage options.
    pub fn with_options(mut self, options: StageOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Check if this stage is a cloud stage (uses third-party cloud API).
    pub fn is_cloud(&self) -> bool {
        matches!(self.target, Some(ExecutionTarget::Cloud)) || self.provider.is_some()
    }

    /// Check if this stage is a device/local stage.
    pub fn is_device(&self) -> bool {
        matches!(self.target, Some(ExecutionTarget::Device) | None) && self.provider.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MemoryPressure;

    #[test]
    fn default_device_metrics_carries_unknown_memory_pressure() {
        let metrics = DeviceMetrics::default();

        assert_eq!(metrics.resource.memory_pressure, MemoryPressure::Unknown);
    }
}
