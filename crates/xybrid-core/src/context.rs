//! Context module - Shared data structures and types used across the orchestrator.
//!
//! This module defines common types such as `DeviceMetrics` and `StageDescriptor`
//! that are used throughout the orchestrator components. Data envelopes have
//! graduated into the IR layer (`crate::ir::Envelope`); we re-export them here
//! to maintain backwards compatibility while downstream code migrates.

pub use crate::ir::{Envelope, EnvelopeKind};
use crate::pipeline::{ExecutionTarget, IntegrationProvider, StageOptions};

/// Live device metrics (network, battery, temperature, etc.).
#[derive(Debug, Clone)]
pub struct DeviceMetrics {
    pub network_rtt: u32,
    pub battery: u8,
    pub temperature: f32,
    // TODO: Add more fields
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
