use std::sync::atomic::{AtomicBool, Ordering};

use flutter_rust_bridge::frb;

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
        if TELEMETRY_INITIALIZED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let config = xybrid_sdk::TelemetryConfig::new(endpoint, api_key);
        xybrid_sdk::telemetry::init_platform_telemetry(config);
    }

    /// Whether [`Self::init_telemetry`] has run at least once in this
    /// process. Reflects the authoritative process-wide state, not any
    /// Dart-side flag — survives hot-restart, multiple isolates, etc.
    #[frb(sync)]
    pub fn is_telemetry_initialized() -> bool {
        TELEMETRY_INITIALIZED.load(Ordering::Acquire)
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
}
