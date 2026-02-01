use flutter_rust_bridge::frb;

#[frb(opaque)]
pub struct XybridSdkClient;

impl XybridSdkClient {
    #[frb(sync)]
    pub fn init_sdk_cache_dir(cache_dir: impl Into<std::path::PathBuf>) {
        xybrid_sdk::init_sdk_cache_dir(cache_dir);
    }

    #[frb(sync)]
    pub fn set_api_key(api_key: &str) {
        xybrid_sdk::set_api_key(api_key);
    }
}