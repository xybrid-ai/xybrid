//! Registry module - Bundle storage, retrieval, and resolution.
//!
//! The registry subsystem provides a unified interface for managing model bundles
//! across local and remote storage backends.
//!
//! ## Module Organization
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`config`] | Configuration types (RegistryConfig, BundleDescriptor) |
//! | [`index`] | JSON index for fast bundle discovery |
//! | [`local`] | Local filesystem registry implementation |
//! | [`remote`] | HTTP transport and caching registry |
//! | [`resolver`] | Hierarchical fallback chain resolution |
//!
//! ## Usage
//!
//! For most use cases, use the [`RegistryResolver`] which handles the fallback chain:
//!
//! ```rust,no_run
//! use xybrid_core::registry::RegistryResolver;
//!
//! let resolver = RegistryResolver::default();
//! let registry = resolver.resolve_registry(None, None);
//! // Returns registry pointing to registry.xybrid.dev
//! ```
//!
//! For direct local registry access:
//!
//! ```rust,no_run
//! use xybrid_core::registry::{LocalRegistry, Registry};
//!
//! let mut registry = LocalRegistry::default()?;
//! let data = registry.get_bundle("model-id", Some("1.0.0"))?;
//! ```

// ============================================================================
// Submodules
// ============================================================================

mod config;
mod index;
mod local;
mod remote;
mod resolver;

// ============================================================================
// Re-exports
// ============================================================================

// Config types
pub use config::{
    BundleDescriptor, BundleLocation, BundleSource, RegistryAuth, RegistryConfig,
    RemoteRegistryConfig,
};

// Index types
pub use index::{IndexEntry, IndexError, IndexResult, RegistryIndex};

// Local registry
pub use local::{BundleMetadata, LocalRegistry, Registry, RegistryError, RegistryResult};

// Remote registry
pub use remote::{HttpRegistryTransport, HttpRemoteRegistry, RegistryTransport, RemoteRegistry};

// Resolver
pub use resolver::RegistryResolver;
