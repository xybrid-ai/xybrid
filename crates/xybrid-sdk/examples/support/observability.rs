//! Optional, explicitly configured observability for command-line demos.
use std::time::Duration;
use xybrid_sdk::observability::{self, ObservabilityConfig};
use xybrid_sdk::SdkResult;

pub struct ExportSession;

impl ExportSession {
    pub fn finish(self) -> SdkResult<()> {
        observability::shutdown_observability(Duration::from_secs(5))
    }
}

impl Drop for ExportSession {
    fn drop(&mut self) {
        // Also flush when a demo returns early with a model or input error.
        let _ = observability::shutdown_observability(Duration::from_secs(5));
    }
}

pub fn configure(service: &str) -> SdkResult<ExportSession> {
    let disabled = std::env::var("OTEL_SDK_DISABLED")
        .is_ok_and(|value| value.trim().eq_ignore_ascii_case("true"));
    let config = if disabled {
        None
    } else {
        ObservabilityConfig::from_env().or_else(|| {
            std::env::var("XYBRID_API_KEY")
                .ok()
                .filter(|key| !key.trim().is_empty())
                .map(|key| ObservabilityConfig::xybrid(&key))
        })
    };
    if let Some(config) = config {
        observability::init_observability(
            config.service(service, Some(env!("CARGO_PKG_VERSION").to_owned())),
        )?;
        println!("Observability export enabled");
    } else {
        println!(
            "Running locally; set OTEL_EXPORTER_OTLP_ENDPOINT or XYBRID_API_KEY to export traces"
        );
    }
    Ok(ExportSession)
}
