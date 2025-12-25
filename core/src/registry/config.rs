//! Registry configuration primitives shared across local and remote backends.
//!
//! These types allow the orchestrator and CLI to describe registry sources in a
//! backend-agnostic way while still supporting backend-specific configuration
//! such as HTTP endpoints or authentication strategies.

use serde::{Deserialize, Serialize};

/// Identifies the origin of a bundle fetch so telemetry can capture cache hits
/// versus remote downloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleSource {
    /// Bundle was resolved from the local filesystem cache.
    LocalCache,
    /// Bundle was fetched from a remote registry backend.
    Remote,
}

/// Describes a bundle that can be resolved by either a local or remote registry.
///
/// This mirrors the schema used by both the local `RegistryIndex` and remote
/// discovery endpoints so that higher-level orchestrator components can operate
/// over a single descriptor model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleDescriptor {
    /// Logical identifier for the bundle (e.g., model or policy name).
    pub id: String,
    /// Semantic version or build identifier.
    pub version: String,
    /// Execution target (cpu, coreml, webgpu, etc.).
    #[serde(default)]
    pub target: Option<String>,
    /// Hash of the bundle contents for integrity validation.
    #[serde(default)]
    pub hash: Option<String>,
    /// Size of the bundle in bytes.
    pub size_bytes: u64,
    /// Absolute or backend-specific reference to the bundle location.
    pub location: BundleLocation,
}

/// Location for a bundle descriptor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BundleLocation {
    /// Local filesystem path.
    Local { path: String },
    /// Remote HTTP/S endpoint (supports templated downloads).
    Remote { url: String },
}

/// Authentication strategies for remote registry access.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RegistryAuth {
    /// No authentication—primarily for local development or unsecured endpoints.
    None,
    /// Static bearer token added to the `Authorization` header.
    Bearer { token: String },
    /// Custom header/value pair (e.g., API key, signed token).
    ApiKey { header: String, value: String },
}

impl Default for RegistryAuth {
    fn default() -> Self {
        RegistryAuth::None
    }
}

/// Configuration for connecting to a remote registry backend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteRegistryConfig {
    /// Base URL for API interactions (e.g., https://registry.xybrid.dev).
    pub base_url: String,
    /// Optional path override for the bundle index endpoint.
    #[serde(default)]
    pub index_path: Option<String>,
    /// Optional path override for bundle download endpoint.
    #[serde(default)]
    pub bundle_path: Option<String>,
    /// Authentication configuration.
    #[serde(default)]
    pub auth: RegistryAuth,
    /// Timeout in milliseconds for network operations.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts for transient failures.
    #[serde(default)]
    pub retry_attempts: Option<u32>,
}

/// Top-level registry configuration that can represent purely local, purely
/// remote, or hybrid (local cache + remote source) setups.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RegistryConfig {
    /// Optional explicit path for the local registry cache directory.
    #[serde(default)]
    pub local_path: Option<String>,
    /// Optional remote registry configuration. When present, the orchestrator
    /// should hydrate the local cache from the remote source on demand.
    #[serde(default)]
    pub remote: Option<RemoteRegistryConfig>,
}

impl RegistryConfig {
    /// Returns `true` if the configuration has a remote component.
    pub fn has_remote(&self) -> bool {
        self.remote.is_some()
    }
}
