//! Explicit observability control plane. Exported data stays in Rust.
use flutter_rust_bridge::frb;
use xybrid_ffi_facade as facade;

/// Parent and content-correlation controls shared by load and ASR APIs.
#[derive(Clone, Debug, Default)]
pub struct FfiObservationOptions {
    pub traceparent: Option<String>,
    pub tracestate: Option<String>,
    pub conversation_id: Option<String>,
    pub suppress_trace: bool,
}

impl FfiObservationOptions {
    pub(crate) fn to_sdk(&self) -> Result<Option<xybrid_sdk::ObservationOptions>, String> {
        facade::ObservationControls {
            traceparent: self.traceparent.clone(),
            tracestate: self.tracestate.clone(),
            conversation_id: self.conversation_id.clone(),
            suppress_trace: self.suppress_trace,
        }
        .to_sdk()
        .map_err(|e| e.to_string())
    }
}

/// Export configuration. Set exactly one of `api_key` (Xybrid traces) or
/// `endpoint` (Collector traces, metrics and logs). Content is captured by
/// default; set `metadata_only` to exclude prompts, outputs and tool payloads.
pub struct FfiObservabilityConfig {
    pub api_key: Option<String>,
    pub endpoint: Option<String>,
    pub traces_endpoint: Option<String>,
    pub headers: Vec<(String, String)>,
    pub json_protocol: bool,
    pub service_name: Option<String>,
    pub service_version: Option<String>,
    pub resource_attributes: Vec<(String, String)>,
    pub metadata_only: bool,
    pub max_content_bytes: Option<u32>,
    pub live_trace_interval_ms: Option<u32>,
    pub device_id: Option<String>,
    pub redaction_rules: Vec<FfiRedactionRule>,
    pub provider_origins: Vec<String>,
}

/// Ordered literal replacement, or exact governed attribute omission.
pub enum FfiRedactionRule {
    ReplaceLiteral { find: String, replacement: String },
    OmitAttribute { key: String },
}

impl FfiObservabilityConfig {
    fn to_facade(&self) -> Result<std::sync::Arc<facade::ObservabilityConfigHandle>, String> {
        for endpoint in [self.endpoint.as_deref(), self.traces_endpoint.as_deref()]
            .into_iter()
            .flatten()
        {
            let url =
                url::Url::parse(endpoint).map_err(|e| format!("Invalid export endpoint: {e}"))?;
            if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
                return Err("Export endpoint must be an absolute HTTP(S) URL".into());
            }
        }
        let handle = match (&self.api_key, &self.endpoint) {
            (Some(key), None) if !key.trim().is_empty() => {
                facade::ObservabilityConfigHandle::xybrid(key.clone())
            }
            (None, Some(endpoint)) if !endpoint.trim().is_empty() => {
                facade::ObservabilityConfigHandle::otlp(endpoint.clone())
            }
            _ => return Err("Set exactly one non-blank api_key or endpoint".into()),
        };
        if let Some(endpoint) = &self.traces_endpoint {
            handle.set_traces_endpoint(endpoint.clone());
        }
        for (name, value) in &self.headers {
            handle.set_header(name.clone(), value.clone());
        }
        handle.set_json_protocol(self.json_protocol);
        if let Some(name) = &self.service_name {
            handle.set_service(name.clone(), self.service_version.clone());
        }
        for (key, value) in &self.resource_attributes {
            handle.set_resource_attribute(key.clone(), value.clone());
        }
        if let Some(id) = &self.device_id {
            handle.set_device_id(id.clone());
        }
        for origin in &self.provider_origins {
            handle
                .allow_provider_context(origin.clone())
                .map_err(|e| e.to_string())?;
        }
        for rule in &self.redaction_rules {
            match rule {
                FfiRedactionRule::ReplaceLiteral { find, replacement } => {
                    handle.add_literal_redaction(find.clone(), replacement.clone())
                }
                FfiRedactionRule::OmitAttribute { key } => {
                    handle.omit_content_attribute(key.clone())
                }
            }
            .map_err(|e| e.to_string())?;
        }
        handle.set_metadata_only(self.metadata_only);
        if let Some(bytes) = self.max_content_bytes {
            handle.set_max_content_bytes(bytes);
        }
        if let Some(ms) = self.live_trace_interval_ms {
            handle.set_live_trace_interval_ms(ms);
        }
        Ok(handle)
    }
}

/// Trace and span ids of a run, for feedback and evaluation correlation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FfiObservationRef {
    pub trace_id: String,
    pub span_id: String,
}

impl FfiObservationRef {
    pub(crate) fn from_sdk(reference: &xybrid_sdk::ObservationRef) -> Self {
        Self {
            trace_id: reference.trace_id.clone(),
            span_id: reference.span_id.clone(),
        }
    }

    fn to_facade(&self) -> facade::ObservationRef {
        facade::ObservationRef {
            trace_id: self.trace_id.clone(),
            span_id: self.span_id.clone(),
        }
    }
}

/// Delivery counts for a single signal. Pending includes queued and in-flight items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfiSignalStats {
    pub accepted: u64,
    pub exported: u64,
    pub dropped: u64,
    pub failed: u64,
    pub pending: u64,
}

impl From<facade::SignalStats> for FfiSignalStats {
    fn from(s: facade::SignalStats) -> Self {
        Self {
            accepted: s.accepted,
            exported: s.exported,
            dropped: s.dropped,
            failed: s.failed,
            pending: s.pending,
        }
    }
}

/// Delivery counters (zero when no packaged exporter is active).
pub struct FfiObservabilityStats {
    pub enabled: bool,
    pub traces: FfiSignalStats,
    pub logs: FfiSignalStats,
    pub metrics: FfiSignalStats,
}

/// Start exporting, replacing the previous configuration. Configuration
/// errors cross the bridge; failed initialization can be retried.
#[frb(sync)]
pub fn init_observability(config: FfiObservabilityConfig) -> Result<(), String> {
    config
        .to_facade()?
        .init()
        .map_err(|error| error.to_string())
}

#[frb(sync)]
pub fn observability_stats() -> FfiObservabilityStats {
    let stats = facade::observability_stats();
    FfiObservabilityStats {
        enabled: stats.enabled,
        traces: stats.traces.into(),
        logs: stats.logs.into(),
        metrics: stats.metrics.into(),
    }
}

/// Flush queued exports on a worker, within the supplied timeout.
pub fn flush_observability(timeout_ms: u32) -> Result<(), String> {
    facade::observability_flush(timeout_ms).map_err(|error| error.to_string())
}

/// Flush, stop exporting and disable observability. Safe to call repeatedly.
pub fn shutdown_observability(timeout_ms: u32) -> Result<(), String> {
    facade::observability_shutdown(timeout_ms).map_err(|error| error.to_string())
}

/// Returns false if logs export is disabled (including Xybrid traces-only).
#[frb(sync)]
pub fn record_feedback(
    target: FfiObservationRef,
    score: Option<f64>,
    label: Option<String>,
    comment: Option<String>,
) -> Result<bool, String> {
    facade::observability_record_feedback(&target.to_facade(), score, label, comment)
        .map_err(|error| error.to_string())
}

/// Returns false if logs export is disabled (including Xybrid traces-only).
#[frb(sync)]
pub fn record_evaluation(
    target: FfiObservationRef,
    name: String,
    score_value: Option<f64>,
    score_label: Option<String>,
    explanation: Option<String>,
) -> Result<bool, String> {
    facade::observability_record_evaluation(
        &target.to_facade(),
        name,
        score_value,
        score_label,
        explanation,
    )
    .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collector() -> FfiObservabilityConfig {
        FfiObservabilityConfig {
            api_key: None,
            endpoint: Some("http://127.0.0.1:4318".into()),
            traces_endpoint: None,
            headers: vec![],
            json_protocol: false,
            service_name: None,
            service_version: None,
            resource_attributes: vec![],
            metadata_only: false,
            max_content_bytes: None,
            live_trace_interval_ms: None,
            device_id: None,
            redaction_rules: Vec::new(),
            provider_origins: Vec::new(),
        }
    }

    #[test]
    fn rejects_ambiguous_or_missing_export_target() {
        let mut config = collector();
        config.api_key = Some("xy_test".into());
        assert!(config.to_facade().is_err());
        config.endpoint = None;
        assert!(config.to_facade().is_ok());
        config.api_key = None;
        assert!(config.to_facade().is_err());
        config.api_key = Some("  ".into());
        assert!(config.to_facade().is_err());
    }

    #[test]
    fn validates_collector_and_trace_override_urls() {
        let mut config = collector();
        config.endpoint = Some("file:///tmp/collector".into());
        assert!(config.to_facade().is_err());
        config = collector();
        config.traces_endpoint = Some("ftp://example.test/traces".into());
        assert!(config.to_facade().is_err());
        config.traces_endpoint = Some("https://example.test/custom/traces".into());
        assert!(config.to_facade().is_ok());
    }

    #[test]
    fn lifecycle_retries_replaces_and_disables_native_state() {
        // The only test installing an observer in this crate owns the whole
        // lifecycle; no parallel test depends on the process-global exporter.
        use super::super::sdk_client::XybridSdkClient;
        assert!(XybridSdkClient::configure_platform_telemetry(
            "xy_test".into(),
            None,
            Some("invalid".into())
        )
        .is_err());
        assert!(!observability_stats().enabled);
        let mut invalid = collector();
        invalid.endpoint = Some("file:///not-an-http-endpoint".into());
        assert!(init_observability(invalid).is_err());
        init_observability(collector()).unwrap();
        assert!(observability_stats().enabled);
        assert!(super::super::sdk_client::XybridSdkClient::is_telemetry_initialized());
        init_observability(collector()).unwrap();
        XybridSdkClient::configure_platform_telemetry(
            "xy_test".into(),
            Some("http://127.0.0.1:4318".into()),
            Some("boundary".into()),
        )
        .unwrap();
        assert_eq!(
            xybrid_sdk::observability::observer()
                .content_policy()
                .capture(),
            xybrid_sdk::observability::ContentCapture::NoContent
        );
        // Invalid replacement settings preserve the installed observer.
        assert!(XybridSdkClient::configure_platform_telemetry(
            "xy_test".into(),
            Some("file:///invalid".into()),
            None,
        )
        .is_err());
        assert!(observability_stats().enabled);
        XybridSdkClient::set_api_key("cloud-key");
        XybridSdkClient::init_telemetry(
            "http://127.0.0.1:4318/v1/traces".into(),
            "ingest-key".into(),
        )
        .unwrap();
        assert_eq!(xybrid_sdk::get_api_key().as_deref(), Some("cloud-key"));
        flush_observability(1000).unwrap();
        shutdown_observability(1000).unwrap();
        shutdown_observability(1000).unwrap();
        assert!(!observability_stats().enabled);
        assert!(!super::super::sdk_client::XybridSdkClient::is_telemetry_initialized());
    }

    #[test]
    fn rejects_invalid_feedback_reference() {
        let target = FfiObservationRef {
            trace_id: "invalid".into(),
            span_id: "invalid".into(),
        };
        assert!(record_feedback(target, Some(1.0), None, None).is_err());
    }
}
