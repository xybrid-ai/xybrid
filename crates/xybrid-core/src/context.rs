//! Context module - Shared data structures and types used across the orchestrator.
//!
//! This module defines common types such as `DeviceMetrics` and `StageDescriptor`
//! that are used throughout the orchestrator components. Data envelopes have
//! graduated into the IR layer (`crate::ir::Envelope`); we re-export them here
//! to maintain backwards compatibility while downstream code migrates.

use crate::device::{detect_capabilities, HardwareCapabilities, ResourceSnapshot};
pub use crate::ir::{Envelope, EnvelopeKind};
use crate::pipeline::{ExecutionTarget, IntegrationProvider, StageOptions};

/// Live device signals consumed by the orchestrator.
///
/// Two members:
/// - `capabilities` — static hardware view (memory total, GPU/Metal/NNAPI
///   flags, CPU cores) captured at construction time.
/// - `resource` — live resource snapshot (CPU, memory, thermal, battery)
///   sampled by `ResourceMonitor`.
///
/// Battery and thermal state on `capabilities` are populated by overlaying a
/// fresh `ResourceSnapshot` via [`Self::with_live_snapshot`]; the snapshot is
/// the only authoritative source.
#[derive(Debug, Clone)]
pub struct DeviceMetrics {
    /// Hardware capability snapshot used by the routing engine.
    pub capabilities: HardwareCapabilities,
    /// Live resource snapshot sampled by `ResourceMonitor`.
    pub resource: ResourceSnapshot,
}

impl DeviceMetrics {
    /// Return a copy with the snapshot's live values overlaid onto the
    /// capability view.
    ///
    /// `available_mem_mb`, `total_mem_mb`, `cpu_pct`, `battery_pct`, and
    /// `thermal_state` flow from the snapshot into `capabilities` so callers
    /// of `should_throttle()` and the routing ladder see live readings. When
    /// a snapshot field is `None` the existing capability value is preserved.
    pub fn with_live_snapshot(&self, snapshot: ResourceSnapshot) -> Self {
        let mut metrics = self.clone();
        metrics.resource = snapshot;
        if let Some(available_mb) = snapshot.available_mem_mb {
            metrics.capabilities.memory_available_mb = available_mb as u64;
        }
        if let Some(total_mb) = snapshot.total_mem_mb {
            metrics.capabilities.memory_total_mb = total_mb as u64;
        }
        if let Some(cpu_pct) = snapshot.cpu_pct {
            metrics.capabilities.cpu_usage_percent = cpu_pct;
        }
        if let Some(battery_pct) = snapshot.battery_pct {
            metrics.capabilities.battery_level = battery_pct;
        }
        metrics.capabilities.thermal_state = snapshot.thermal_state;
        metrics
    }
}

impl Default for DeviceMetrics {
    fn default() -> Self {
        Self {
            capabilities: detect_capabilities(),
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
    use crate::device::{MemoryPressure, ThermalState};

    #[test]
    fn default_device_metrics_carries_unknown_memory_pressure() {
        let metrics = DeviceMetrics::default();

        assert_eq!(metrics.resource.memory_pressure, MemoryPressure::Unknown);
    }

    #[test]
    fn with_live_snapshot_preserves_capability_memory_when_snapshot_missing() {
        let metrics = DeviceMetrics::default();
        let snapshot = ResourceSnapshot::unknown();

        let merged = metrics.with_live_snapshot(snapshot);

        assert!(
            merged.capabilities.memory_total_mb > 0,
            "memory_total_mb must not be zeroed by an unknown snapshot"
        );
    }

    #[test]
    fn with_live_snapshot_uses_snapshot_memory_when_present() {
        let metrics = DeviceMetrics::default();
        let mut snapshot = ResourceSnapshot::unknown();
        snapshot.memory_pressure = MemoryPressure::Normal;
        snapshot.available_mem_mb = Some(2048);
        snapshot.total_mem_mb = Some(8192);
        snapshot.cpu_pct = Some(42.5);

        let merged = metrics.with_live_snapshot(snapshot);

        assert_eq!(merged.capabilities.memory_available_mb, 2048);
        assert_eq!(merged.capabilities.memory_total_mb, 8192);
        assert_eq!(merged.capabilities.cpu_usage_percent, 42.5);
    }

    #[test]
    fn with_live_snapshot_overlays_battery_and_thermal_when_snapshot_carries_them() {
        let metrics = DeviceMetrics::default();
        let mut snapshot = ResourceSnapshot::unknown();
        snapshot.battery_pct = Some(42);
        snapshot.thermal_state = ThermalState::Hot;

        let merged = metrics.with_live_snapshot(snapshot);

        assert_eq!(merged.capabilities.battery_level, 42);
        assert_eq!(merged.capabilities.thermal_state, ThermalState::Hot);
    }
}
