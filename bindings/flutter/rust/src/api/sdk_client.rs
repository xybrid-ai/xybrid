use flutter_rust_bridge::frb;

/// Binding identifier reported in the `X-Xybrid-Client` registry header for
/// Flutter apps. Routed through `xybrid_sdk::set_binding` at every FFI entry
/// so registry calls are attributed correctly even on platforms that skip
/// `init_sdk_cache_dir` (iOS/macOS).
const FLUTTER_BINDING: &str = "flutter";

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
