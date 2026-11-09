//! Command-owned OpenTelemetry capture. No process-wide collector or reset.
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context as _, Result};
use opentelemetry::trace::TraceId;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider, Span, SpanData, SpanProcessor};
use opentelemetry_sdk::Resource;
use xybrid_observability::otlp::{trace_json, ObservabilityPipeline, OtlpConfig};
use xybrid_sdk::observability::{self, ObservabilitySession, Observer};

pub use xybrid_core::tracing::{add_metadata, SpanGuard};

#[derive(Debug)]
struct Captured {
    trace: Option<TraceId>,
    spans: Vec<SpanData>,
    bytes: usize,
    dropped: usize,
    resource: Resource,
}

#[derive(Clone, Debug)]
struct CaptureProcessor(Arc<Mutex<Captured>>);

impl SpanProcessor for CaptureProcessor {
    fn on_start(&self, _: &mut Span, _: &opentelemetry::Context) {}
    fn on_end(&self, span: SpanData) {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.trace != Some(span.span_context.trace_id()) {
            return;
        }
        // Include SDK allocations and encoded representations, with headroom for
        // capacities and grouping. These limits bound data we own, not process RSS.
        let bytes = trace_json(vec![span.clone()], &state.resource)
            .map(|body| body.len().saturating_mul(4).saturating_add(2048))
            .unwrap_or(usize::MAX);
        if state.spans.len() >= 10_000
            || bytes > (32 * 1024 * 1024usize).saturating_sub(state.bytes)
        {
            state.dropped += 1;
        } else {
            state.bytes += bytes;
            state.spans.push(span);
        }
    }
    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }
    fn shutdown_with_timeout(&self, _: Duration) -> OTelSdkResult {
        Ok(())
    }
    fn set_resource(&mut self, resource: &Resource) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .resource = resource.clone();
    }
}

/// Owns the providers and a bounded capture of one command's trace.
pub struct LocalCapture {
    state: Arc<Mutex<Captured>>,
    provider: Option<SdkTracerProvider>,
    pipeline: Option<ObservabilityPipeline>,
    session: ObservabilitySession,
}

impl LocalCapture {
    /// Builds the command's providers: a bounded in-memory capture, plus the
    /// OTLP pipeline when `config` is set. Both describe the process with the
    /// SDK's resource attributes, so a captured or exported trace carries the
    /// same `telemetry.sdk.*`, `xybrid.sdk.*` and device attributes as an SDK
    /// session.
    ///
    /// # Errors
    /// Invalid `device` id or OTLP configuration.
    pub fn configure(config: Option<OtlpConfig>, device: Option<&str>) -> Result<Self> {
        let attributes = observability::resource_attributes(true, device)?;
        let state = Arc::new(Mutex::new(Captured {
            trace: None,
            spans: Vec::new(),
            bytes: 0,
            dropped: 0,
            resource: Resource::builder_empty().build(),
        }));
        let processor = CaptureProcessor(state.clone());
        let (provider, pipeline, observer) = match config {
            Some(mut config) => {
                // The validated id replaces any copy the flags already added.
                config
                    .resource
                    .attributes
                    .retain(|kv| kv.key.as_str() != "device.id");
                config.resource.attributes.extend(attributes);
                let pipeline = ObservabilityPipeline::build_with_span_processor(config, processor)?;
                let observer = pipeline.observer();
                (None, Some(pipeline), observer)
            }
            None => {
                let resource = Resource::builder_empty()
                    .with_service_name("xybrid-cli")
                    .with_attribute(opentelemetry::KeyValue::new(
                        "service.version",
                        env!("CARGO_PKG_VERSION"),
                    ))
                    .with_attributes(attributes);
                let provider = SdkTracerProvider::builder()
                    .with_span_processor(processor)
                    .with_sampler(Sampler::ParentBased(Box::new(Sampler::AlwaysOn)))
                    .with_resource(resource.build())
                    .build();
                let observer = Observer::builder().tracer_provider(&provider).build();
                (Some(provider), None, observer)
            }
        };
        let session = ObservabilitySession::borrowed(observer);
        observability::install_default_session(session.clone());
        Ok(Self {
            state,
            provider,
            pipeline,
            session,
        })
    }

    /// Whether the captured trace is also exported over OTLP.
    pub fn exports(&self) -> bool {
        self.pipeline.is_some()
    }

    /// Spans captured so far (sorted by start time on [`Self::finish`]).
    #[cfg(test)]
    pub(crate) fn captured(&self) -> Vec<SpanData> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .spans
            .clone()
    }

    pub fn begin(&self) -> observability::ManualObservation<observability::WorkflowOperation> {
        let root = observability::start_workflow(
            "xybrid-cli run",
            &observability::ObservationOptions::default().with_session(&self.session),
        );
        if let Some(context) = root.context() {
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .trace = TraceId::from_hex(&context.trace_id()).ok();
        }
        root
    }

    pub fn finish(&self, render: bool, path: Option<&std::path::Path>) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.spans.sort_by_key(|span| span.start_time);
        if render {
            println!("Execution trace ({} spans)", state.spans.len());
            for span in &state.spans {
                let parent = span.parent_span_id.to_string();
                let duration = span
                    .end_time
                    .duration_since(span.start_time)
                    .unwrap_or_default();
                println!(
                    "  {}  {}  parent={}  {:8.3} ms  {}",
                    span.span_context.span_id(),
                    span.name,
                    parent,
                    duration.as_secs_f64() * 1000.0,
                    if matches!(span.status, opentelemetry::trace::Status::Error { .. }) {
                        "error"
                    } else {
                        ""
                    }
                );
            }
            if state.dropped > 0 {
                println!("{} spans not captured (capture budget)", state.dropped);
            }
        }
        if let Some(path) = path {
            let body = trace_json(std::mem::take(&mut state.spans), &state.resource)?;
            let directory = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(std::path::Path::new("."));
            let mut file = tempfile::NamedTempFile::new_in(directory)?;
            std::io::Write::write_all(&mut file, &body)?;
            file.persist(path)
                .with_context(|| format!("Failed to export trace to {}", path.display()))?;
        }
        Ok(())
    }

    pub fn shutdown(&self) -> Result<()> {
        if let Some(pipeline) = &self.pipeline {
            pipeline.shutdown(Duration::from_secs(5))?;
        }
        if let Some(provider) = &self.provider {
            provider.shutdown_with_timeout(Duration::from_secs(5))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xybrid_sdk::observability::{start_tool, ObservationOptions, ToolCallInfo};

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::OBSERVABILITY_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn local_capture_renders_exports_otlp_json_and_feeds_the_analyzer() {
        let _lock = lock();
        assert!(LocalCapture::configure(None, Some("bad\ndevice")).is_err());
        let capture = LocalCapture::configure(None, Some("cli-device")).unwrap();
        assert!(!capture.exports(), "local-only capture exports nothing");
        let root = capture.begin();
        {
            let _scope = root.enter();
            let _stage = SpanGuard::new("inference:local-model");
            start_tool(ToolCallInfo::new("lookup"), &ObservationOptions::default()).end();
        }
        root.end();
        let captured = capture.captured();
        let root_span = captured
            .iter()
            .find(|s| s.name == "invoke_workflow xybrid-cli run")
            .unwrap();
        let stage = captured
            .iter()
            .find(|s| s.name == "inference:local-model")
            .unwrap();
        let tool = captured
            .iter()
            .find(|s| s.name == "execute_tool lookup")
            .unwrap();
        assert_eq!(stage.parent_span_id, root_span.span_context.span_id());
        assert_eq!(
            tool.parent_span_id,
            stage.span_context.span_id(),
            "operations inside an open CLI stage are its children"
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.json");
        capture.finish(true, Some(&path)).unwrap();
        capture.shutdown().unwrap();
        let trace: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let resource = trace["resourceSpans"][0]["resource"]["attributes"]
            .as_array()
            .unwrap()
            .clone();
        let value = |key: &str| {
            resource
                .iter()
                .find(|a| a["key"] == key)
                .map(|a| a["value"]["stringValue"].clone())
        };
        assert_eq!(
            value("telemetry.sdk.name"),
            Some(serde_json::json!("xybrid"))
        );
        assert_eq!(value("device.id"), Some(serde_json::json!("cli-device")));
        assert_eq!(value("service.name"), Some(serde_json::json!("xybrid-cli")));
        assert!(value("xybrid.sdk.version").is_some());

        let summary = crate::commands::trace::summarize(&trace).unwrap();
        assert_eq!(summary["operation_counts"]["execute_tool"], 1);
        assert_eq!(summary["operation_counts"]["invoke_workflow"], 1);
        let names: Vec<&str> = summary["operations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"execute_tool lookup"));
    }
}
