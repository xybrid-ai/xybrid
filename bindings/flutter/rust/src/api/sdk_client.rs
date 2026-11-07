use flutter_rust_bridge::frb;
use xybrid_ffi_facade as facade;
use xybrid_sdk::ResourceTelemetryMode;

use super::{ensure_native_logging, FLUTTER_BINDING};

/// Install native logging (logcat / os_log) and the panic logger, and
/// register the `flutter` binding.
///
/// Runs automatically from `XybridRustLib.init()`, on every platform and
/// before any other call. Nothing else is guaranteed to run: without an API
/// key, `Xybrid.init` reaches no other Rust entry point on iOS, macOS, Linux
/// or Windows, so logging and registry attribution cannot wait for one.
#[frb(init)]
pub fn init_app() {
    ensure_native_logging();
    facade::set_binding(FLUTTER_BINDING.to_string());
}

/// Logical storage area containing a cached model entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiCacheEntryLocation {
    Registry,
    Extracted,
    HuggingFace,
    HuggingFaceHub,
}

impl From<facade::CacheEntryLocation> for FfiCacheEntryLocation {
    fn from(location: facade::CacheEntryLocation) -> Self {
        match location {
            facade::CacheEntryLocation::Registry => Self::Registry,
            facade::CacheEntryLocation::Extracted => Self::Extracted,
            facade::CacheEntryLocation::HuggingFace => Self::HuggingFace,
            facade::CacheEntryLocation::HuggingFaceHub => Self::HuggingFaceHub,
        }
    }
}

/// One physical model entry occupying managed cache storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FfiCacheEntry {
    pub model_id: String,
    pub location: FfiCacheEntryLocation,
    pub path: String,
    pub size_bytes: u64,
}

impl From<facade::CacheEntry> for FfiCacheEntry {
    fn from(entry: facade::CacheEntry) -> Self {
        Self {
            model_id: entry.model_id,
            location: entry.location.into(),
            path: entry.path,
            size_bytes: entry.size_bytes,
        }
    }
}

/// Aggregate model-cache storage status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FfiCacheStatus {
    pub total_size_bytes: u64,
    pub entry_count: u32,
    pub model_count: u32,
    pub extracted_model_count: u32,
    pub cache_root: String,
}

impl From<facade::CacheStatus> for FfiCacheStatus {
    fn from(status: facade::CacheStatus) -> Self {
        Self {
            total_size_bytes: status.total_size_bytes,
            entry_count: status.entry_count,
            model_count: status.model_count,
            extracted_model_count: status.extracted_model_count,
            cache_root: status.cache_root,
        }
    }
}

#[frb(opaque)]
pub struct XybridSdkClient;

/// Map a legacy ingest base URL to the OTLP trace route. Explicit paths
/// remain full endpoints so Collector and self-hosted routes work too.
fn resolve_ingest_endpoint(ingest_url: Option<&str>) -> Result<String, String> {
    let endpoint = ingest_url.map(str::trim).filter(|v| !v.is_empty());
    let Some(endpoint) = endpoint else {
        return Ok(facade::observability_default_traces_endpoint());
    };
    let mut url = url::Url::parse(endpoint).map_err(|e| format!("Invalid ingest URL: {e}"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err("Ingest endpoint must be an absolute HTTP(S) URL".into());
    }
    if url.path().is_empty() || url.path() == "/" {
        url.set_path("/v1/traces");
    }
    Ok(url.to_string())
}

fn parse_resource_telemetry_mode(value: Option<&str>) -> Option<ResourceTelemetryMode> {
    let raw = value?.trim().to_ascii_lowercase();
    if raw.is_empty() {
        return None;
    }

    let (head, interval) = match raw.split_once(':') {
        Some((head, tail)) => (head, Some(tail.parse::<u32>().ok()?)),
        None => (raw.as_str(), None),
    };
    let default_interval = ResourceTelemetryMode::DEFAULT_SUMMARY_INTERVAL_MS;
    let mode = match head {
        "off" if interval.is_none() => ResourceTelemetryMode::Off,
        "boundary" if interval.is_none() => ResourceTelemetryMode::Boundary,
        "summary" => ResourceTelemetryMode::Summary {
            interval_ms: interval.unwrap_or(default_interval),
        },
        "debug_local" | "debuglocal" | "debug-local" => ResourceTelemetryMode::DebugLocal {
            interval_ms: interval.unwrap_or(default_interval),
        },
        _ => return None,
    };
    Some(mode.normalized())
}

fn validated_resource_telemetry_mode(
    mode: Option<&str>,
) -> Result<Option<ResourceTelemetryMode>, String> {
    mode.map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|value| {
            parse_resource_telemetry_mode(Some(value))
                .ok_or_else(|| format!("Invalid resource telemetry mode: {value}"))
        })
        .transpose()
}

fn init_legacy_trace_exporter(api_key: String, endpoint: String) -> Result<(), String> {
    let config = facade::ObservabilityConfigHandle::xybrid(api_key);
    config.set_traces_endpoint(endpoint);
    config.set_metadata_only(true);
    config.init().map_err(|error| error.to_string())
}

impl XybridSdkClient {
    #[frb(sync)]
    pub fn init_sdk_cache_dir(cache_dir: String) {
        facade::init_sdk_cache_dir(cache_dir);
    }

    /// Returns aggregate storage usage across every managed model-cache area.
    // Default FRB dispatch runs these filesystem operations on its worker pool.
    // Do not mark them sync: large caches must not block the Dart UI isolate.
    pub fn cache_status() -> Result<FfiCacheStatus, String> {
        facade::cache_status()
            .map(Into::into)
            .map_err(|error| error.to_string())
    }

    /// Lists every physical model entry occupying managed cache storage.
    pub fn cache_entries() -> Result<Vec<FfiCacheEntry>, String> {
        facade::cache_entries()
            .map(|entries| entries.into_iter().map(Into::into).collect())
            .map_err(|error| error.to_string())
    }

    /// Returns whether a model occupies any managed model-cache entry.
    pub fn has_cached_model_data(model_id: String) -> Result<bool, String> {
        facade::cache_is_model_cached(model_id).map_err(|error| error.to_string())
    }

    /// Resolves the preferred local cache path for a model, if present.
    pub fn cached_model_path(model_id: String) -> Result<Option<String>, String> {
        facade::cache_model_path(model_id).map_err(|error| error.to_string())
    }

    /// Lists model IDs extracted, validated, and ready to run offline.
    pub fn list_extracted_model_ids() -> Result<Vec<String>, String> {
        facade::cache_list_extracted_model_ids().map_err(|error| error.to_string())
    }

    /// Removes every managed cache entry for one model.
    ///
    /// Do not call concurrently with a load of the same model.
    pub fn remove_cached_model(model_id: String) -> Result<u32, String> {
        facade::cache_remove_model(model_id).map_err(|error| error.to_string())
    }

    /// Clears all managed model-cache storage.
    ///
    /// Do not call concurrently with any model load.
    pub fn clear_model_cache() -> Result<u32, String> {
        facade::cache_clear().map_err(|error| error.to_string())
    }

    #[frb(sync)]
    pub fn set_api_key(api_key: &str) {
        facade::set_api_key(api_key.to_string());
    }

    /// Point the cloud gateway at a platform base URL (staging, self-hosted).
    ///
    /// Held in process memory rather than the environment, so it is safe to
    /// call after telemetry threads have started. Pass a bare base URL — the
    /// `/v1` suffix is applied internally.
    #[frb(sync)]
    pub fn set_platform_url(url: String) {
        facade::set_platform_url(url);
    }

    /// Enable speculative cloud fallback globally: a registry chat model that
    /// isn't downloaded yet is served from the gateway while the weights
    /// download.
    ///
    /// Only takes effect when an API key resolves. An uncached load asks the
    /// registry first, so other models (ASR, TTS, `.xyb` bundles) download and
    /// load as usual. `FfiModelLoader.fromRegistrySpeculative` opts a single
    /// load in instead.
    #[frb(sync)]
    pub fn set_speculative_cloud(enabled: bool) {
        facade::set_speculative_cloud(enabled);
    }

    /// Whether the global speculative-cloud default is on.
    #[frb(sync)]
    pub fn is_speculative_cloud_enabled() -> bool {
        facade::is_speculative_cloud_enabled()
    }

    /// Release every idle loaded model's memory; returns how many were released.
    ///
    /// Wire this to the app's low-memory signal
    /// (`WidgetsBindingObserver.didHaveMemoryPressure`). Models with a run in
    /// flight are skipped, and a released model reloads itself the next time
    /// it is used — Dart callers get no new error to handle.
    #[frb(sync)]
    pub fn release_memory() -> u32 {
        facade::release_memory()
    }

    /// Enable or disable automatic model release for subsequent loads.
    ///
    /// When enabled, loading a model under device memory pressure first
    /// releases least-recently-used idle models. Off by default;
    /// [`Self::release_memory`] works either way.
    #[frb(sync)]
    pub fn set_auto_release(enabled: bool) {
        facade::set_auto_release(enabled);
    }

    /// Whether automatic model release is enabled process-wide.
    #[frb(sync)]
    pub fn is_auto_release_enabled() -> bool {
        facade::is_auto_release_enabled()
    }

    #[frb(sync)]
    pub fn set_gateway_url(gateway_url: String) {
        // `set_gateway_url` is gateway-routing-specific and not part of the
        // facade's init surface (would bloat it for one platform). Route
        // straight to the SDK.
        xybrid_sdk::set_gateway_url(gateway_url);
    }

    /// Compatibility entry point: start the OTLP trace exporter. A base URL
    /// is mapped to `/v1/traces`; a full trace endpoint is used as supplied.
    /// Export stays metadata-only, matching the legacy telemetry payload.
    /// The export key does not change cloud authentication.
    /// Replaces a previous configuration and reports initialization errors.
    #[frb(sync)]
    pub fn init_telemetry(endpoint: String, api_key: String) -> Result<(), String> {
        let endpoint = resolve_ingest_endpoint(Some(&endpoint))?;
        init_legacy_trace_exporter(api_key, endpoint)
    }

    /// Compatibility path for explicit legacy telemetry configuration.
    #[frb(sync)]
    pub fn configure_platform_telemetry(
        api_key: String,
        ingest_url: Option<String>,
        resource_telemetry: Option<String>,
    ) -> Result<(), String> {
        let resource_mode = validated_resource_telemetry_mode(resource_telemetry.as_deref())?;
        let endpoint = resolve_ingest_endpoint(ingest_url.as_deref())?;
        init_legacy_trace_exporter(api_key.clone(), endpoint)?;
        facade::set_api_key(api_key);
        if let Some(mode) = resource_mode {
            xybrid_sdk::init().resource_telemetry(mode).run();
        }
        Ok(())
    }

    /// Validate resource sampling without changing native state.
    #[frb(sync)]
    pub fn validate_resource_telemetry(mode: Option<String>) -> Result<(), String> {
        validated_resource_telemetry_mode(mode.as_deref()).map(|_| ())
    }

    /// Configure resource sampling independently of authentication/export.
    #[frb(sync)]
    pub fn configure_resource_telemetry(mode: Option<String>) -> Result<(), String> {
        if let Some(mode) = validated_resource_telemetry_mode(mode.as_deref())? {
            xybrid_sdk::init().resource_telemetry(mode).run();
        }
        Ok(())
    }

    /// Authoritative native exporter state, including shutdown/reconfigure.
    #[frb(sync)]
    pub fn is_telemetry_initialized() -> bool {
        facade::observability_stats().enabled
    }

    /// Return the xybrid runtime features compiled into this native library.
    ///
    /// Studio uses this to decide whether image upload should be enabled for
    /// VisionLanguage models. Keeping the answer in Rust avoids stale Dart-side
    /// assumptions about Cargo features.
    #[frb(sync)]
    pub fn runtime_features() -> Vec<String> {
        xybrid_sdk::features::enabled()
            .iter()
            .map(|feature| (*feature).to_string())
            .collect()
    }

    #[frb(sync)]
    pub fn flush_platform_telemetry() -> Result<(), String> {
        facade::observability_flush(5000).map_err(|error| error.to_string())
    }

    /// Check if a model is cached locally (extracted and ready to use).
    ///
    /// This is a pure filesystem check — no network access required.
    /// Returns `true` if the model has been downloaded and extracted
    /// at `~/.xybrid/cache/extracted/{model_id}/model_metadata.json`.
    #[frb(sync)]
    pub fn is_model_cached(model_id: &str) -> bool {
        if let Ok(client) = xybrid_sdk::RegistryClient::from_env() {
            return client.is_extracted(model_id);
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_validation_rejects_malformed_intervals_before_configuration() {
        for mode in ["summary:invalid", "summary:", "off:100", "boundary:100"] {
            assert!(XybridSdkClient::validate_resource_telemetry(Some(mode.into())).is_err());
        }
        for mode in [
            "off",
            "boundary",
            "summary",
            "summary:200",
            "debug_local:200",
        ] {
            assert!(XybridSdkClient::validate_resource_telemetry(Some(mode.into())).is_ok());
        }
        assert!(XybridSdkClient::validate_resource_telemetry(None).is_ok());
    }

    #[test]
    fn cache_records_preserve_facade_storage_fields() {
        let entry = FfiCacheEntry::from(facade::CacheEntry {
            model_id: "owner/repo".into(),
            location: facade::CacheEntryLocation::HuggingFaceHub,
            path: "/cache/hf-hub/repo".into(),
            size_bytes: 99,
        });
        let status = FfiCacheStatus::from(facade::CacheStatus {
            total_size_bytes: 99,
            entry_count: 1,
            model_count: 1,
            extracted_model_count: 0,
            cache_root: "/cache".into(),
        });

        assert_eq!(entry.model_id, "owner/repo");
        assert_eq!(entry.location, FfiCacheEntryLocation::HuggingFaceHub);
        assert_eq!(entry.path, "/cache/hf-hub/repo");
        assert_eq!(entry.size_bytes, 99);
        assert_eq!(status.total_size_bytes, 99);
        assert_eq!(status.entry_count, 1);
        assert_eq!(status.cache_root, "/cache");
    }

    #[test]
    fn init_app_registers_flutter_binding() {
        // Single combined test (the binding is process-global via OnceLock —
        // splitting into multiple tests would race on which one observes the
        // first set_binding).
        init_app();
        XybridSdkClient::init_sdk_cache_dir(
            std::env::temp_dir()
                .join("xybrid-flutter-test-cache")
                .to_string_lossy()
                .into_owned(),
        );

        // Process-global binding now resolves to "flutter".
        assert_eq!(xybrid_sdk::get_binding(), FLUTTER_BINDING);

        // RegistryClient default constructors pick up the configured binding,
        // so the X-Xybrid-Client header on every metadata call will report
        // binding=flutter for Flutter apps.
        let client = xybrid_sdk::RegistryClient::default_client()
            .expect("default_client should succeed in tests");
        assert_eq!(client.binding(), FLUTTER_BINDING);
    }

    #[test]
    fn ingest_endpoint_defaults_when_absent() {
        assert_eq!(
            resolve_ingest_endpoint(None).unwrap(),
            facade::observability_default_traces_endpoint()
        );
    }

    #[test]
    fn ingest_endpoint_defaults_when_blank() {
        assert_eq!(
            resolve_ingest_endpoint(Some("   ")).unwrap(),
            facade::observability_default_traces_endpoint()
        );
    }

    #[test]
    fn ingest_endpoint_uses_supplied_value() {
        assert_eq!(
            resolve_ingest_endpoint(Some("http://192.168.1.78:8081")).unwrap(),
            "http://192.168.1.78:8081/v1/traces"
        );
    }

    #[test]
    fn ingest_endpoint_trims_surrounding_whitespace() {
        assert_eq!(
            resolve_ingest_endpoint(Some("  https://ingest.example/v1/traces  ")).unwrap(),
            "https://ingest.example/v1/traces"
        );
    }

    #[test]
    fn ingest_endpoint_preserves_custom_routes_and_rejects_invalid_urls() {
        assert_eq!(
            resolve_ingest_endpoint(Some("http://localhost:4318/custom/traces")).unwrap(),
            "http://localhost:4318/custom/traces"
        );
        assert!(resolve_ingest_endpoint(Some("file:///tmp/traces")).is_err());
        assert!(resolve_ingest_endpoint(Some("invalid")).is_err());
    }

    #[test]
    fn runtime_features_mirror_core_feature_introspection() {
        let expected: Vec<String> = xybrid_sdk::features::enabled()
            .iter()
            .map(|feature| (*feature).to_string())
            .collect();

        assert_eq!(XybridSdkClient::runtime_features(), expected);
    }

    #[cfg(feature = "llm-llamacpp-vision")]
    #[test]
    fn runtime_features_report_llama_cpp_vision_when_compiled() {
        assert!(XybridSdkClient::runtime_features().contains(&"llm-llamacpp-vision".to_string()));
    }
}
