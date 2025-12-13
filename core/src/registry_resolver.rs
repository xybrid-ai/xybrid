//! Registry Resolver - Hierarchical registry resolution with fallback chain.
//!
//! Provides seamless default behavior (registry.xybrid.dev) with support for
//! stage-level, pipeline-level, project-level, and environment variable overrides.
//!
//! # Fallback Chain
//!
//! 1. Stage-level registry (if specified)
//! 2. Pipeline-level registry (if specified)
//! 3. Project-level registry (if specified)
//! 4. Environment variable (`XYBRID_REGISTRY`)
//! 5. Default remote registry (`registry.xybrid.dev`) ← Primary default
//! 6. Local fallback registry (`file://test-registry`) ← Last resort
//!
//! # Example
//!
//! ```rust,no_run
//! use xybrid_core::registry_resolver::RegistryResolver;
//!
//! let resolver = RegistryResolver::default();
//! let registry = resolver.resolve_registry(None, None);
//! // Returns registry pointing to registry.xybrid.dev
//! ```

use crate::registry::{LocalRegistry, Registry, RegistryError, RegistryResult};
use crate::registry_config::{RegistryAuth, RegistryConfig, RemoteRegistryConfig};
use crate::registry_index::RegistryIndex;
use crate::registry_remote::{HttpRegistryTransport, RemoteRegistry};
use std::sync::Arc;

/// Default remote registry URL
const DEFAULT_REGISTRY_URL: &str = "https://registry.xybrid.dev";

/// Local fallback registry path (for testing/development)
const FALLBACK_LOCAL_REGISTRY: &str = "file://test-registry";

/// Registry resolver that implements hierarchical fallback chain.
pub struct RegistryResolver {
    /// Project-level registry config (loaded from ~/.xybrid/config.yaml)
    project_registry: Option<RegistryConfig>,
    /// Environment variable registry (from XYBRID_REGISTRY)
    env_registry: Option<String>,
    /// Default remote registry (registry.xybrid.dev)
    default_remote_registry: Arc<dyn Registry>,
    /// Local fallback registry (for testing)
    fallback_local_registry: Option<Arc<dyn Registry>>,
}

impl RegistryResolver {
    /// Creates a new registry resolver with default configuration.
    ///
    /// Initializes:
    /// - Default remote registry (registry.xybrid.dev)
    /// - Environment variable registry (if set)
    /// - Project config (if available)
    /// - Local fallback registry (for testing)
    pub fn new() -> RegistryResult<Self> {
        // Initialize default remote registry
        let default_config = RemoteRegistryConfig {
            base_url: DEFAULT_REGISTRY_URL.to_string(),
            index_path: None,
            bundle_path: None,
            auth: RegistryAuth::None,
            timeout_ms: Some(30000), // 30 seconds
            retry_attempts: Some(3),
        };

        // Create default remote registry with cache and index
        let default_cache = LocalRegistry::default()
            .map_err(|e| RegistryError::IOError(format!("Failed to create default cache: {}", e)))?;
        let default_index = RegistryIndex::load_or_create()
            .map_err(|e| RegistryError::IOError(format!("Failed to create default index: {}", e)))?;
        let default_transport = HttpRegistryTransport::new(default_config)
            .map_err(|e| RegistryError::RemoteError(format!("Failed to create default registry transport: {}", e)))?;
        let default_registry: Arc<dyn Registry> = Arc::new(RemoteRegistry::new(
            default_transport,
            default_cache,
            default_index,
        ));

        // Try to load environment variable registry
        let env_registry = std::env::var("XYBRID_REGISTRY").ok();

        // Try to load project config (optional, don't fail if not found)
        let project_registry = Self::load_project_config().ok().flatten();

        // Initialize local fallback registry (optional, for testing)
        let fallback_local_registry = LocalRegistry::new(FALLBACK_LOCAL_REGISTRY)
            .ok()
            .map(|r| Arc::new(r) as Arc<dyn Registry>);

        Ok(Self {
            project_registry,
            env_registry,
            default_remote_registry: default_registry,
            fallback_local_registry,
        })
    }

    /// Resolves a registry using the hierarchical fallback chain.
    ///
    /// # Arguments
    ///
    /// * `stage_registry` - Optional stage-level registry config
    /// * `pipeline_registry` - Optional pipeline-level registry config
    ///
    /// # Returns
    ///
    /// Resolved registry following the fallback chain
    pub fn resolve_registry(
        &self,
        stage_registry: Option<&RegistryConfig>,
        pipeline_registry: Option<&RegistryConfig>,
    ) -> Arc<dyn Registry> {
        // 1. Try stage-level registry
        if let Some(config) = stage_registry {
            if let Some(registry) = self.create_registry_from_config(config) {
                return registry;
            }
        }

        // 2. Try pipeline-level registry
        if let Some(config) = pipeline_registry {
            if let Some(registry) = self.create_registry_from_config(config) {
                return registry;
            }
        }

        // 3. Try project-level registry
        if let Some(config) = &self.project_registry {
            if let Some(registry) = self.create_registry_from_config(config) {
                return registry;
            }
        }

        // 4. Try environment variable registry
        if let Some(url) = &self.env_registry {
            if let Some(registry) = self.create_registry_from_url(url) {
                return registry;
            }
        }

        // 5. Use default remote registry (registry.xybrid.dev)
        Arc::clone(&self.default_remote_registry)
    }

    /// Creates a registry from a RegistryConfig.
    fn create_registry_from_config(&self, config: &RegistryConfig) -> Option<Arc<dyn Registry>> {
        // If remote config exists, prefer remote
        if let Some(remote_config) = &config.remote {
            return self.create_remote_registry(remote_config).ok();
        }

        // Otherwise, try local path
        if let Some(local_path) = &config.local_path {
            return LocalRegistry::new(local_path)
                .ok()
                .map(|r| Arc::new(r) as Arc<dyn Registry>);
        }

        None
    }

    /// Creates a registry from a URL string.
    ///
    /// Supports:
    /// - `https://registry.example.com` (remote)
    /// - `file:///path/to/registry` (local)
    /// - `file://test-registry` (local, relative)
    fn create_registry_from_url(&self, url: &str) -> Option<Arc<dyn Registry>> {
        if url.starts_with("file://") {
            // Local registry
            let path = url.strip_prefix("file://").unwrap();
            return LocalRegistry::new(path)
                .ok()
                .map(|r| Arc::new(r) as Arc<dyn Registry>);
        }

        // Remote registry
        let config = RemoteRegistryConfig {
            base_url: url.to_string(),
            index_path: None,
            bundle_path: None,
            auth: RegistryAuth::None,
            timeout_ms: Some(30000),
            retry_attempts: Some(3),
        };

        self.create_remote_registry(&config).ok()
    }

    /// Creates a remote registry from config.
    fn create_remote_registry(
        &self,
        config: &RemoteRegistryConfig,
    ) -> RegistryResult<Arc<dyn Registry>> {
        let transport = HttpRegistryTransport::new(config.clone())
            .map_err(|e| RegistryError::RemoteError(format!("Failed to create transport: {}", e)))?;
        
        // Create cache and index for remote registry
        let cache = LocalRegistry::default()
            .map_err(|e| RegistryError::IOError(format!("Failed to create cache: {}", e)))?;
        let index = RegistryIndex::load_or_create()
            .map_err(|e| RegistryError::IOError(format!("Failed to create index: {}", e)))?;
        
        Ok(Arc::new(RemoteRegistry::new(transport, cache, index)))
    }

    /// Loads project-level config from standard locations.
    ///
    /// Checks:
    /// - `~/.xybrid/config.yaml`
    /// - `./xybrid.yaml`
    /// - `./.xybrid.yaml`
    fn load_project_config() -> RegistryResult<Option<RegistryConfig>> {

        // Try user home config
        if let Some(mut home) = dirs::home_dir() {
            home.push(".xybrid");
            home.push("config.yaml");
            if home.exists() {
                return Self::load_config_file(&home);
            }
        }

        // Try project root configs
        let current_dir = std::env::current_dir()
            .map_err(|e| RegistryError::IOError(format!("Failed to get current dir: {}", e)))?;

        for filename in &["xybrid.yaml", ".xybrid.yaml"] {
            let config_path = current_dir.join(filename);
            if config_path.exists() {
                return Self::load_config_file(&config_path);
            }
        }

        Ok(None)
    }

    /// Loads a config file from path.
    fn load_config_file(path: &std::path::Path) -> RegistryResult<Option<RegistryConfig>> {
        use std::fs;

        let content = fs::read_to_string(path)
            .map_err(|e| RegistryError::IOError(format!("Failed to read config: {}", e)))?;

        let config: RegistryConfig = serde_yaml::from_str(&content)
            .map_err(|e| RegistryError::InvalidBundle(format!("Failed to parse config: {}", e)))?;

        Ok(Some(config))
    }
}

impl Default for RegistryResolver {
    fn default() -> Self {
        Self::new().expect("Failed to create default registry resolver")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolver_creation() {
        let resolver = RegistryResolver::new();
        assert!(resolver.is_ok());
    }

    #[test]
    fn test_resolve_default_registry() {
        let resolver = RegistryResolver::default();
        let registry = resolver.resolve_registry(None, None);
        // Should return default remote registry
        assert!(Arc::ptr_eq(&registry, &resolver.default_remote_registry));
    }
}

