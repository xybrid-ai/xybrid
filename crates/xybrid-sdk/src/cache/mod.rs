//! Cache module - Model bundle caching and provider interface.
//!
//! This module provides:
//! - `CacheManager`: Platform-specific bundle storage and cache management
//! - `SdkCacheProvider`: Implements Core's `CacheProvider` trait for orchestrator integration
//!
//! # Architecture
//!
//! ```text
//! Orchestrator
//!     └─► CacheProvider (trait from xybrid-core)
//!         └─► SdkCacheProvider (this module)
//!             └─► CacheManager (bundle storage)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use xybrid_sdk::{CacheManager, SdkCacheProvider};
//!
//! // Direct cache management
//! let cache = CacheManager::new()?;
//! let status = cache.status()?;
//! println!("Cached models: {}", status.total_models);
//!
//! // As a provider for the orchestrator
//! let provider = SdkCacheProvider::new()?;
//! if provider.is_model_cached("kokoro-82m") {
//!     println!("Model available locally");
//! }
//! ```

mod cache_manager;
mod cache_provider;

pub use cache_manager::{CacheManager, CacheStatus};
pub use cache_provider::SdkCacheProvider;
