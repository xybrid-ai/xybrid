use std::sync::atomic::{AtomicBool, Ordering};

use flutter_rust_bridge::frb;
use xybrid_sdk::ResourceTelemetryMode;

use super::FLUTTER_BINDING;

/// Process-wide once-guard for [`XybridSdkClient::init_telemetry`]. Set on
/// the first successful entry. Re-entry — whether from a duplicate Dart
/// caller, a second Dart isolate, or a Flutter hot-restart — observes
/// `true` and returns without spinning up a second exporter.
///
/// Telemetry init is non-trivial: it spawns a background HTTP sender,
/// registers a process-global execution listener, and activates the
/// resource-telemetry sampler. None of those have an unregister path
/// today, so a second call would burn duplicate senders and emit each
/// event twice. The guard trades runtime flexibility (no in-process
/// reconfigure) for safety; apps that need to change endpoints must
/// restart the process.
static TELEMETRY_INITIALIZED: AtomicBool = AtomicBool::new(false);

#[frb(opaque)]
pub struct XybridSdkClient;

fn initialize_telemetry_once(config: xybrid_sdk::TelemetryConfig) {
    if TELEMETRY_INITIALIZED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    xybrid_sdk::telemetry::init_platform_telemetry(config);
}

/// Resolve the telemetry ingest endpoint for the bundled init path: use the
/// caller-supplied URL when present and non-blank, otherwise fall back to
/// [`xybrid_sdk::telemetry::DEFAULT_INGEST_URL`]. Keeping this a pure free
/// function lets the defaulting rule be unit-tested without touching the
/// process-wide telemetry once-guard.
fn resolve_ingest_endpoint(ingest_url: Option<&str>) -> &str {
    ingest_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(xybrid_sdk::telemetry::DEFAULT_INGEST_URL)
}

fn parse_resource_telemetry_mode(value: Option<&str>) -> Option<ResourceTelemetryMode> {
    let raw = value?.trim().to_ascii_lowercase();
    if raw.is_empty() {
        return None;
    }

    let (head, interval) = match raw.split_once(':') {
        Some((head, tail)) => (head, tail.parse::<u32>().ok()),
        None => (raw.as_str(), None),
    };
    let default_interval = ResourceTelemetryMode::DEFAULT_SUMMARY_INTERVAL_MS;
    let mode = match head {
        "off" => ResourceTelemetryMode::Off,
        "boundary" => ResourceTelemetryMode::Boundary,
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

impl XybridSdkClient {
    #[frb(sync)]
    pub fn init_sdk_cache_dir(cache_dir: String) {
        xybrid_sdk::set_binding(FLUTTER_BINDING);
        xybrid_sdk::init_sdk_cache_dir(cache_dir);
    }

    #[frb(sync)]
    pub fn set_api_key(api_key: &str) {
        xybrid_sdk::set_binding(FLUTTER_BINDING);
        xybrid_sdk::set_api_key(api_key);
    }

    #[frb(sync)]
    pub fn set_gateway_url(gateway_url: String) {
        xybrid_sdk::set_binding(FLUTTER_BINDING);
        xybrid_sdk::set_gateway_url(gateway_url);
    }

    /// Initialize the platform telemetry exporter for this process.
    ///
    /// Starts the HTTP telemetry sender targeting `endpoint`, authenticated
    /// with `api_key`. Once initialized, the normal inference paths
    /// (`Xybrid.model().run()`, `Xybrid.pipeline().run()`, conversation
    /// turns, etc.) automatically emit `ExecutionStarted` /
    /// `ExecutionCompleted` / `ExecutionFailed` events through it — no
    /// per-call wiring required.
    ///
    /// Process-wide idempotent via [`TELEMETRY_INITIALIZED`]: only the
    /// first successful call enters `init_platform_telemetry`; subsequent
    /// calls (duplicate caller, second Dart isolate, Flutter hot-restart
    /// inside a surviving process) observe the guard and return without
    /// spawning a second exporter. No reconfigure path — restart the
    /// process to change endpoint/key.
    ///
    /// Sync because `init_platform_telemetry` is sync; the HTTP exporter
    /// spins up its own background thread for batched sends.
    #[frb(sync)]
    pub fn init_telemetry(endpoint: String, api_key: String) {
        xybrid_sdk::set_binding(FLUTTER_BINDING);
        let config = xybrid_sdk::TelemetryConfig::new(endpoint, api_key);
        initialize_telemetry_once(config);
    }

    /// Start the platform telemetry exporter from the bundled
    /// `Xybrid.init(apiKey: ...)` path.
    ///
    /// When `ingest_url` is absent or blank the exporter targets
    /// [`xybrid_sdk::telemetry::DEFAULT_INGEST_URL`], so providing only an
    /// API key is enough to light up the dashboard — the caller does not
    /// need to know the ingest endpoint. Shares the process-wide once-guard
    /// with [`Self::init_telemetry`]; whichever path runs first wins.
    #[frb(sync)]
    pub fn configure_platform_telemetry(
        api_key: String,
        ingest_url: Option<String>,
        resource_telemetry: Option<String>,
    ) {
        xybrid_sdk::set_binding(FLUTTER_BINDING);
        xybrid_sdk::set_api_key(&api_key);

        let endpoint = resolve_ingest_endpoint(ingest_url.as_deref());
        let mut config = xybrid_sdk::TelemetryConfig::new(endpoint, api_key);
        if let Some(mode) = parse_resource_telemetry_mode(resource_telemetry.as_deref()) {
            config = config.with_resource_telemetry(mode);
        }
        initialize_telemetry_once(config);
    }

    /// Whether [`Self::init_telemetry`] has run at least once in this
    /// process. Reflects the authoritative process-wide state, not any
    /// Dart-side flag — survives hot-restart, multiple isolates, etc.
    #[frb(sync)]
    pub fn is_telemetry_initialized() -> bool {
        TELEMETRY_INITIALIZED.load(Ordering::Acquire)
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
    pub fn flush_platform_telemetry() {
        xybrid_sdk::telemetry::flush_platform_telemetry();
    }

    /// Check if a model is cached locally (extracted and ready to use).
    ///
    /// This is a pure filesystem check — no network access required.
    /// Returns `true` if the model has been downloaded and extracted
    /// at `~/.xybrid/cache/extracted/{model_id}/model_metadata.json`.
    #[frb(sync)]
    pub fn is_model_cached(model_id: &str) -> bool {
        xybrid_sdk::set_binding(FLUTTER_BINDING);
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
    fn flutter_init_registers_flutter_binding() {
        // Single combined test (the binding is process-global via OnceLock —
        // splitting into multiple tests would race on which one observes the
        // first set_binding).
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
            resolve_ingest_endpoint(None),
            xybrid_sdk::telemetry::DEFAULT_INGEST_URL
        );
    }

    #[test]
    fn ingest_endpoint_defaults_when_blank() {
        assert_eq!(
            resolve_ingest_endpoint(Some("   ")),
            xybrid_sdk::telemetry::DEFAULT_INGEST_URL
        );
    }

    #[test]
    fn ingest_endpoint_uses_supplied_value() {
        assert_eq!(
            resolve_ingest_endpoint(Some("http://192.168.1.78:8081")),
            "http://192.168.1.78:8081"
        );
    }

    #[test]
    fn ingest_endpoint_trims_surrounding_whitespace() {
        assert_eq!(
            resolve_ingest_endpoint(Some("  https://ingest.example  ")),
            "https://ingest.example"
        );
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
