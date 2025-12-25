//! Remote registry implementation and supporting HTTP transport.
//!
//! The remote registry acts as a read-through cache in front of a remote bundle
//! store. It uses the existing `LocalRegistry` for persistence while delegating
//! discovery and downloads to a pluggable `RegistryTransport`.

use std::collections::HashMap;
use std::io::Read;
use std::sync::Mutex;
use std::time::Duration;

use super::config::{BundleDescriptor, BundleLocation, RegistryAuth, RemoteRegistryConfig};
use super::index::{IndexEntry, RegistryIndex};
use super::local::{BundleMetadata, LocalRegistry, Registry, RegistryError, RegistryResult};
use url::Url;

/// Describes a transport capable of communicating with a remote registry
/// backend.
pub trait RegistryTransport: Send + Sync {
    /// Fetches the full bundle index from the remote backend.
    fn fetch_index(&self) -> RegistryResult<Vec<BundleDescriptor>>;

    /// Fetches bundle bytes for the provided descriptor.
    fn fetch_bundle(&self, descriptor: &BundleDescriptor) -> RegistryResult<Vec<u8>>;

    /// Resolves a descriptor for the given bundle id/version. Implementations
    /// may optimize this with targeted API calls; the default implementation
    /// falls back to scanning the fetched index.
    fn fetch_descriptor(
        &self,
        id: &str,
        version: Option<&str>,
    ) -> RegistryResult<BundleDescriptor> {
        let mut descriptors = self.fetch_index()?;
        if descriptors.is_empty() {
            return Err(RegistryError::BundleNotFound(id.to_string()));
        }

        descriptors.sort_by(|a, b| a.version.cmp(&b.version));

        if let Some(requested) = version {
            descriptors
                .into_iter()
                .rev()
                .find(|d| d.id == id && d.version == requested)
                .ok_or_else(|| RegistryError::BundleNotFound(format!("{}@{}", id, requested)))
        } else {
            descriptors
                .into_iter()
                .rev()
                .find(|d| d.id == id)
                .ok_or_else(|| RegistryError::BundleNotFound(id.to_string()))
        }
    }
}

/// HTTP transport backed by a REST-style registry API.
pub struct HttpRegistryTransport {
    config: RemoteRegistryConfig,
    agent: ureq::Agent,
    base_url: Url,
}

/// Convenience alias for the default HTTP-backed remote registry.
pub type HttpRemoteRegistry = RemoteRegistry<HttpRegistryTransport>;

impl HttpRegistryTransport {
    /// Constructs a new HTTP transport from the provided configuration.
    pub fn new(config: RemoteRegistryConfig) -> RegistryResult<Self> {
        let base_url = Url::parse(config.base_url.as_str())
            .map_err(|err| RegistryError::RemoteError(format!("Invalid registry URL: {}", err)))?;

        let mut builder = ureq::AgentBuilder::new();

        if let Some(timeout_ms) = config.timeout_ms {
            let timeout = Duration::from_millis(timeout_ms);
            builder = builder
                .timeout_connect(timeout)
                .timeout_read(timeout)
                .timeout_write(timeout);
        }

        let agent = builder.build();

        Ok(Self {
            config,
            agent,
            base_url,
        })
    }

    fn apply_auth(&self, request: ureq::Request) -> ureq::Request {
        match &self.config.auth {
            RegistryAuth::None => request,
            RegistryAuth::Bearer { token } => {
                request.set("Authorization", format!("Bearer {}", token).as_str())
            }
            RegistryAuth::ApiKey { header, value } => request.set(header, value),
        }
    }

    fn index_url(&self) -> RegistryResult<Url> {
        if let Some(path) = &self.config.index_path {
            self.base_url.join(path).map_err(|err| {
                RegistryError::RemoteError(format!("Failed to build index URL: {}", err))
            })
        } else {
            self.base_url.join("index").map_err(|err| {
                RegistryError::RemoteError(format!("Failed to build default index URL: {}", err))
            })
        }
    }

    fn bundle_url(&self, id: &str, version: &str) -> RegistryResult<Url> {
        let template = self
            .config
            .bundle_path
            .as_deref()
            .unwrap_or("bundles/{id}/{version}");
        let interpolated = template.replace("{id}", id).replace("{version}", version);

        self.base_url.join(interpolated.as_str()).map_err(|err| {
            RegistryError::RemoteError(format!(
                "Failed to build bundle URL for {}@{}: {}",
                id, version, err
            ))
        })
    }

    fn retry_attempts(&self) -> u32 {
        self.config.retry_attempts.unwrap_or(2)
    }
}

impl RegistryTransport for HttpRegistryTransport {
    fn fetch_index(&self) -> RegistryResult<Vec<BundleDescriptor>> {
        let url = self.index_url()?;
        let mut attempt = 0;
        let max_attempts = self.retry_attempts().max(1);

        loop {
            let request = self.agent.get(url.as_str());
            let request = self.apply_auth(request);

            match request.call() {
                Ok(response) => {
                    let descriptors: Vec<BundleDescriptor> =
                        response.into_json().map_err(|err| {
                            RegistryError::RemoteError(format!(
                                "Failed to parse registry index: {}",
                                err
                            ))
                        })?;
                    return Ok(descriptors);
                }
                Err(ureq::Error::Status(code, resp)) => {
                    if attempt + 1 >= max_attempts {
                        return Err(RegistryError::RemoteError(format!(
                            "Registry index request failed with status {}: {}",
                            code,
                            resp.into_string().unwrap_or_default()
                        )));
                    }
                }
                Err(err) => {
                    if attempt + 1 >= max_attempts {
                        return Err(RegistryError::RemoteError(format!(
                            "Registry index request failed: {}",
                            err
                        )));
                    }
                }
            }

            attempt += 1;
        }
    }

    fn fetch_bundle(&self, descriptor: &BundleDescriptor) -> RegistryResult<Vec<u8>> {
        let url = match &descriptor.location {
            BundleLocation::Remote { url } => {
                // Handle both absolute URLs and relative paths
                if url.starts_with("http://") || url.starts_with("https://") {
                    Url::parse(url).map_err(|err| {
                        RegistryError::RemoteError(format!(
                            "Descriptor contained invalid remote URL {}: {}",
                            url, err
                        ))
                    })?
                } else {
                    // Relative path - join with base URL
                    // Use strip_prefix to remove only a single leading slash, preserving paths like '//host/path'
                    let path = url.strip_prefix('/').unwrap_or(url);
                    self.base_url.join(path).map_err(|err| {
                        RegistryError::RemoteError(format!(
                            "Failed to build bundle URL from relative path {}: {}",
                            url, err
                        ))
                    })?
                }
            }
            BundleLocation::Local { .. } => self.bundle_url(&descriptor.id, &descriptor.version)?,
        };

        let mut attempt = 0;
        let max_attempts = self.retry_attempts().max(1);

        loop {
            let request = self.agent.get(url.as_str());
            let request = self.apply_auth(request);

            match request.call() {
                Ok(response) => {
                    let mut bytes = Vec::new();
                    response
                        .into_reader()
                        .read_to_end(&mut bytes)
                        .map_err(|err| {
                            RegistryError::RemoteError(format!(
                                "Failed reading bundle stream: {}",
                                err
                            ))
                        })?;
                    return Ok(bytes);
                }
                Err(ureq::Error::Status(code, resp)) => {
                    if attempt + 1 >= max_attempts {
                        return Err(RegistryError::RemoteError(format!(
                            "Bundle download failed with status {}: {}",
                            code,
                            resp.into_string().unwrap_or_default()
                        )));
                    }
                }
                Err(err) => {
                    if attempt + 1 >= max_attempts {
                        return Err(RegistryError::RemoteError(format!(
                            "Bundle download failed: {}",
                            err
                        )));
                    }
                }
            }

            attempt += 1;
        }
    }

    fn fetch_descriptor(
        &self,
        id: &str,
        version: Option<&str>,
    ) -> RegistryResult<BundleDescriptor> {
        let mut descriptors = self.fetch_index()?;
        if descriptors.is_empty() {
            return Err(RegistryError::BundleNotFound(id.to_string()));
        }

        descriptors.sort_by(|a, b| a.version.cmp(&b.version));

        let mut descriptor = if let Some(requested) = version {
            descriptors
                .into_iter()
                .rev()
                .find(|d| d.id == id && d.version == requested)
                .ok_or_else(|| RegistryError::BundleNotFound(format!("{}@{}", id, requested)))?
        } else {
            descriptors
                .into_iter()
                .rev()
                .find(|d| d.id == id)
                .ok_or_else(|| RegistryError::BundleNotFound(id.to_string()))?
        };

        if !matches!(descriptor.location, BundleLocation::Remote { .. }) {
            let url = self.bundle_url(id, descriptor.version.as_str())?;
            descriptor.location = BundleLocation::Remote {
                url: url.to_string(),
            };
        }

        Ok(descriptor)
    }
}

/// Remote registry implementation that hydrates bundles into a local cache.
pub struct RemoteRegistry<T: RegistryTransport> {
    transport: T,
    cache: Mutex<LocalRegistry>,
    index: Mutex<RegistryIndex>,
}

impl<T: RegistryTransport> RemoteRegistry<T> {
    /// Builds a remote registry from the provided transport and cache/index instances.
    pub fn new(transport: T, cache: LocalRegistry, index: RegistryIndex) -> Self {
        Self {
            transport,
            cache: Mutex::new(cache),
            index: Mutex::new(index),
        }
    }

    fn cache(&self) -> RegistryResult<std::sync::MutexGuard<'_, LocalRegistry>> {
        self.cache
            .lock()
            .map_err(|_| RegistryError::IOError("Local registry cache mutex was poisoned".into()))
    }

    fn index(&self) -> RegistryResult<std::sync::MutexGuard<'_, RegistryIndex>> {
        self.index
            .lock()
            .map_err(|_| RegistryError::IOError("Registry index mutex was poisoned".into()))
    }

    fn try_cache_bundle(&self, id: &str, version: Option<&str>) -> RegistryResult<Option<Vec<u8>>> {
        let cache = self.cache()?;
        match cache.get_bundle(id, version) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(RegistryError::BundleNotFound(_)) => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn try_cache_metadata(
        &self,
        id: &str,
        version: Option<&str>,
    ) -> RegistryResult<Option<BundleMetadata>> {
        let cache = self.cache()?;
        match cache.get_metadata(id, version) {
            Ok(meta) => Ok(Some(meta)),
            Err(RegistryError::BundleNotFound(_)) => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn hydrate_bundle(
        &self,
        descriptor: &BundleDescriptor,
    ) -> RegistryResult<(BundleMetadata, Vec<u8>)> {
        let bytes = self.transport.fetch_bundle(descriptor)?;
        let mut cache = self.cache()?;
        let metadata = cache.store_bundle(
            descriptor.id.as_str(),
            descriptor.version.as_str(),
            bytes.clone(),
        )?;
        drop(cache);

        let mut index = self.index()?;
        index
            .add_entry(IndexEntry::new(
                descriptor.id.clone(),
                descriptor.version.clone(),
                descriptor
                    .target
                    .clone()
                    .unwrap_or_else(|| "unspecified".to_string()),
                descriptor.hash.clone().unwrap_or_default(),
                descriptor.size_bytes,
                metadata.path.clone(),
            ))
            .map_err(|err| RegistryError::RemoteError(format!("Failed updating index: {}", err)))?;
        index.save().map_err(|err| {
            RegistryError::RemoteError(format!("Failed persisting index: {}", err))
        })?;

        Ok((metadata, bytes))
    }

    fn to_metadata_from_descriptor(&self, descriptor: BundleDescriptor) -> BundleMetadata {
        let path = match descriptor.location {
            BundleLocation::Local { path } => path,
            BundleLocation::Remote { url } => url,
        };

        BundleMetadata {
            id: descriptor.id,
            version: descriptor.version,
            path,
            size_bytes: descriptor.size_bytes,
        }
    }
}

impl RemoteRegistry<HttpRegistryTransport> {
    /// Convenience helper that constructs a remote registry from a high-level configuration.
    pub fn from_config(config: &super::config::RegistryConfig) -> RegistryResult<Self> {
        let local = if let Some(path) = &config.local_path {
            LocalRegistry::new(path)?
        } else {
            LocalRegistry::default()?
        };

        let index = RegistryIndex::load_or_create().map_err(|err| {
            RegistryError::RemoteError(format!("Failed to load registry index: {}", err))
        })?;

        let remote_config = config.remote.clone().ok_or_else(|| {
            RegistryError::RemoteError("Remote registry configuration is required".to_string())
        })?;

        let transport = HttpRegistryTransport::new(remote_config)?;

        Ok(Self::new(transport, local, index))
    }
}

impl<T: RegistryTransport> Registry for RemoteRegistry<T> {
    fn store_bundle(
        &mut self,
        id: &str,
        version: &str,
        bundle_data: Vec<u8>,
    ) -> RegistryResult<BundleMetadata> {
        // For the initial implementation, treat `store_bundle` as a cache write.
        let metadata = {
            let mut cache = self.cache()?;
            cache.store_bundle(id, version, bundle_data)?
        };

        let mut index = self.index()?;
        index
            .add_entry(IndexEntry::new(
                metadata.id.clone(),
                metadata.version.clone(),
                "unspecified",
                String::new(),
                metadata.size_bytes,
                metadata.path.clone(),
            ))
            .map_err(|err| RegistryError::RemoteError(format!("Failed updating index: {}", err)))?;
        index.save().map_err(|err| {
            RegistryError::RemoteError(format!("Failed persisting index: {}", err))
        })?;

        Ok(metadata)
    }

    fn get_bundle(&self, id: &str, version: Option<&str>) -> RegistryResult<Vec<u8>> {
        if let Some(bytes) = self.try_cache_bundle(id, version)? {
            return Ok(bytes);
        }

        let descriptor = self.transport.fetch_descriptor(id, version)?;
        let (_metadata, bytes) = self.hydrate_bundle(&descriptor)?;
        Ok(bytes)
    }

    fn get_metadata(&self, id: &str, version: Option<&str>) -> RegistryResult<BundleMetadata> {
        if let Some(meta) = self.try_cache_metadata(id, version)? {
            return Ok(meta);
        }

        let descriptor = self.transport.fetch_descriptor(id, version)?;
        Ok(self.to_metadata_from_descriptor(descriptor))
    }

    fn list_bundles(&self) -> RegistryResult<Vec<BundleMetadata>> {
        let mut combined: HashMap<String, BundleMetadata> = HashMap::new();

        {
            let cache = self.cache()?;
            for metadata in cache.list_bundles()? {
                combined.insert(format!("{}@{}", metadata.id, metadata.version), metadata);
            }
        }

        for descriptor in self.transport.fetch_index()? {
            let key = format!("{}@{}", descriptor.id, descriptor.version);
            combined
                .entry(key)
                .or_insert_with(|| self.to_metadata_from_descriptor(descriptor));
        }

        Ok(combined.into_values().collect())
    }

    fn remove_bundle(&mut self, id: &str, version: Option<&str>) -> RegistryResult<()> {
        {
            let mut cache = self.cache()?;
            cache.remove_bundle(id, version)?;
        }

        let mut index = self.index()?;
        match version {
            Some(v) => {
                let _ = index.remove_entry(id, v);
            }
            None => {
                index
                    .bundle_descriptors()
                    .into_iter()
                    .filter(|descriptor| descriptor.id == id)
                    .collect::<Vec<_>>()
                    .iter()
                    .for_each(|descriptor| {
                        let _ = index.remove_entry(id, descriptor.version.as_str());
                    });
            }
        }
        let _ = index.save();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    #[derive(Clone)]
    struct MockTransport {
        descriptors: Vec<BundleDescriptor>,
        bundle_bytes: Vec<u8>,
        fetch_count: Arc<Mutex<u32>>,
    }

    impl MockTransport {
        fn new(descriptor: BundleDescriptor, bundle_bytes: Vec<u8>) -> Self {
            Self {
                descriptors: vec![descriptor],
                bundle_bytes,
                fetch_count: Arc::new(Mutex::new(0)),
            }
        }
    }

    impl RegistryTransport for MockTransport {
        fn fetch_index(&self) -> RegistryResult<Vec<BundleDescriptor>> {
            let mut guard = self
                .fetch_count
                .lock()
                .map_err(|_| RegistryError::RemoteError("Mock counter poisoned".into()))?;
            *guard += 1;
            Ok(self.descriptors.clone())
        }

        fn fetch_bundle(&self, _descriptor: &BundleDescriptor) -> RegistryResult<Vec<u8>> {
            Ok(self.bundle_bytes.clone())
        }
    }

    #[test]
    fn remote_registry_hydrates_and_caches() -> RegistryResult<()> {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let cache_path = temp_dir.path().join("cache");
        let local = LocalRegistry::new(&cache_path)?;
        let index_path = temp_dir.path().join("index.json");
        let index = RegistryIndex::with_path(&index_path).expect("failed to init index");

        let descriptor = BundleDescriptor {
            id: "test-model".to_string(),
            version: "1.0.0".to_string(),
            target: Some("generic".to_string()),
            hash: Some("hash123".to_string()),
            size_bytes: 4,
            location: BundleLocation::Remote {
                url: "https://example.com/bundles/test-model/1.0.0".to_string(),
            },
        };

        let transport = MockTransport::new(descriptor, b"data".to_vec());
        let fetch_counter = transport.fetch_count.clone();
        let registry = RemoteRegistry::new(transport, local, index);

        let bundle = registry.get_bundle("test-model", Some("1.0.0"))?;
        assert_eq!(bundle, b"data");

        // Second call should use the cached copy and avoid extra fetch_index calls.
        let bundle_again = registry.get_bundle("test-model", Some("1.0.0"))?;
        assert_eq!(bundle_again, b"data");

        let counter = fetch_counter.lock().unwrap();
        assert_eq!(*counter, 1);

        Ok(())
    }
}
