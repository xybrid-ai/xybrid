//! Execution target types for pipeline stages.
//!
//! Each stage can execute on one of four targets:
//! - `device`: On-device inference using .xyb bundles
//! - `server`: Xybrid-hosted inference (e.g., vLLM)
//! - `cloud`: Third-party cloud API (OpenAI, Anthropic, etc.)
//! - `auto`: Framework decides based on availability and device capabilities

use serde::{Deserialize, Serialize};
use std::fmt;

/// Execution target for a pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ExecutionTarget {
    /// On-device inference using .xyb bundle from registry.
    /// Requires model to be available for the current platform.
    Device,

    /// Xybrid-hosted inference endpoint.
    /// Uses Xybrid's cloud infrastructure (e.g., vLLM for LLMs).
    Server,

    /// Third-party cloud API.
    /// Requires provider configuration (OpenAI, Anthropic, etc.).
    /// Uses CloudClient for gateway or direct API calls.
    Cloud,

    /// Framework decides at runtime based on:
    /// - Model availability (device bundle, server endpoint, integration)
    /// - Device capabilities (GPU, NPU, memory)
    /// - Network conditions (latency, bandwidth)
    /// - User preferences (privacy mode, performance mode)
    #[default]
    Auto,
}

impl ExecutionTarget {
    /// Returns true if this target requires network access.
    pub fn requires_network(&self) -> bool {
        matches!(self, ExecutionTarget::Server | ExecutionTarget::Cloud)
    }

    /// Returns true if this target can work offline.
    pub fn supports_offline(&self) -> bool {
        matches!(self, ExecutionTarget::Device)
    }

    /// Returns the string representation for YAML/JSON.
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionTarget::Device => "device",
            ExecutionTarget::Server => "server",
            ExecutionTarget::Cloud => "cloud",
            ExecutionTarget::Auto => "auto",
        }
    }
}

impl fmt::Display for ExecutionTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for ExecutionTarget {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "device" | "local" => Ok(ExecutionTarget::Device),
            "server" | "xybrid" => Ok(ExecutionTarget::Server),
            "cloud" | "integration" | "api" => Ok(ExecutionTarget::Cloud),
            "auto" | "default" => Ok(ExecutionTarget::Auto),
            _ => Err(format!(
                "Unknown execution target: '{}'. Valid values: device, server, cloud, auto",
                s
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_target_from_str() {
        assert_eq!(
            "device".parse::<ExecutionTarget>().unwrap(),
            ExecutionTarget::Device
        );
        assert_eq!(
            "server".parse::<ExecutionTarget>().unwrap(),
            ExecutionTarget::Server
        );
        assert_eq!(
            "cloud".parse::<ExecutionTarget>().unwrap(),
            ExecutionTarget::Cloud
        );
        assert_eq!(
            "auto".parse::<ExecutionTarget>().unwrap(),
            ExecutionTarget::Auto
        );

        // Aliases
        assert_eq!(
            "local".parse::<ExecutionTarget>().unwrap(),
            ExecutionTarget::Device
        );
        assert_eq!(
            "xybrid".parse::<ExecutionTarget>().unwrap(),
            ExecutionTarget::Server
        );
        assert_eq!(
            "integration".parse::<ExecutionTarget>().unwrap(),
            ExecutionTarget::Cloud
        ); // backward compat
        assert_eq!(
            "api".parse::<ExecutionTarget>().unwrap(),
            ExecutionTarget::Cloud
        );
        assert_eq!(
            "default".parse::<ExecutionTarget>().unwrap(),
            ExecutionTarget::Auto
        );
    }

    #[test]
    fn test_execution_target_display() {
        assert_eq!(ExecutionTarget::Device.to_string(), "device");
        assert_eq!(ExecutionTarget::Server.to_string(), "server");
        assert_eq!(ExecutionTarget::Cloud.to_string(), "cloud");
        assert_eq!(ExecutionTarget::Auto.to_string(), "auto");
    }

    #[test]
    fn test_requires_network() {
        assert!(!ExecutionTarget::Device.requires_network());
        assert!(ExecutionTarget::Server.requires_network());
        assert!(ExecutionTarget::Cloud.requires_network());
        assert!(!ExecutionTarget::Auto.requires_network()); // Auto doesn't inherently require network
    }

    #[test]
    fn test_supports_offline() {
        assert!(ExecutionTarget::Device.supports_offline());
        assert!(!ExecutionTarget::Server.supports_offline());
        assert!(!ExecutionTarget::Cloud.supports_offline());
        assert!(!ExecutionTarget::Auto.supports_offline()); // Auto may or may not work offline
    }

    #[test]
    fn test_serde_roundtrip() {
        let targets = vec![
            ExecutionTarget::Device,
            ExecutionTarget::Server,
            ExecutionTarget::Cloud,
            ExecutionTarget::Auto,
        ];

        for target in targets {
            let json = serde_json::to_string(&target).unwrap();
            let parsed: ExecutionTarget = serde_json::from_str(&json).unwrap();
            assert_eq!(target, parsed);
        }
    }
}
