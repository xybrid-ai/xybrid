// Flutter FFI API module
// Thin wrappers with #[frb] attributes that delegate to xybrid-sdk

/// Binding identifier reported in the `X-Xybrid-Client` registry header
/// for Flutter apps. Routed through `xybrid_sdk::set_binding` at every
/// FFI entry so registry calls are attributed correctly even on
/// platforms that skip `init_sdk_cache_dir` (iOS/macOS), and on entry
/// points the host hits before `init_sdk_cache_dir` is called (the
/// push-state setters in [`device`]).
pub(crate) const FLUTTER_BINDING: &str = "flutter";

/// Initialize the platform-native `log` backend exactly once per process.
///
/// `android_logger` / `oslog` were declared as dependencies but never
/// initialized, so every `log::warn!` in the SDK (telemetry send failures
/// in particular) was silently discarded on device. Called from the
/// [`sdk_client`] entry points the Dart layer hits during `Xybrid.init`,
/// so logs flow before any exporter or model work starts. No-op on
/// desktop targets, where the host process owns logger setup.
pub(crate) fn ensure_native_logging() {
    static LOGGING_INIT: std::sync::Once = std::sync::Once::new();
    LOGGING_INIT.call_once(|| {
        #[cfg(target_os = "android")]
        android_logger::init_once(
            android_logger::Config::default()
                .with_max_level(log::LevelFilter::Info)
                .with_tag("xybrid"),
        );
        #[cfg(target_os = "ios")]
        {
            // Errors only if a logger is already registered — fine to ignore.
            let _ = oslog::OsLogger::new("dev.xybrid.sdk")
                .level_filter(log::LevelFilter::Info)
                .init();
        }
    });
}

pub mod context;
pub mod device;
pub mod envelope;
pub mod model;
pub mod pipeline;
pub mod result;
pub mod sdk_client;

// Re-export all public types for convenient access
pub use context::{FfiConversationContext, FfiMessageRole};
pub use device::{FfiThermalState, XybridDevice};
pub use envelope::FfiEnvelope;
pub use model::{
    FfiGenerationConfig, FfiModel, FfiModelLoader, FfiRunOptions, FfiStreamEvent, FfiStreamToken,
};
pub use pipeline::FfiPipeline;
pub use result::FfiResult;
