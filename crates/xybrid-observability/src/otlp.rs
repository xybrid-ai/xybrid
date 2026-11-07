//! Optional OTLP/HTTP pipeline (feature `otlp`).
//!
//! Builds tracer, meter, and logger providers that the application owns.
//! Nothing is installed globally. Delivery is bounded in memory: spans beyond
//! the queue bound are dropped and counted, exports retry a bounded number of
//! times, request bodies are capped, and nothing is spooled to disk.

use std::collections::HashMap;
use std::fmt;
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry_http::{Bytes, HttpClient, HttpError, Request, Response};
use opentelemetry_otlp::{
    LogExporter, MetricExporter, RetryPolicy, SpanExporter, WithExportConfig, WithHttpConfig,
};
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::logs::{BatchLogProcessor, SdkLoggerProvider};
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{
    BatchConfigBuilder, BatchSpanProcessor, Sampler, SdkTracerProvider, SpanData,
    SpanExporter as SdkSpanExporter, SpanProcessor,
};
use opentelemetry_sdk::Resource;

use crate::content::ContentPolicy;
use crate::error::{Error, Result};
use crate::observer::Observer;

/// Default Xybrid trace ingest endpoint.
pub const XYBRID_TRACES_ENDPOINT: &str = "https://ingest.xybrid.dev/v1/traces";

/// OTLP/HTTP encoding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Protocol {
    /// `http/protobuf` (default).
    #[default]
    HttpProtobuf,
    /// `http/json`.
    HttpJson,
}

impl Protocol {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "http/protobuf" => Some(Self::HttpProtobuf),
            "http/json" => Some(Self::HttpJson),
            _ => None,
        }
    }

    fn otlp(self) -> opentelemetry_otlp::Protocol {
        match self {
            Self::HttpProtobuf => opentelemetry_otlp::Protocol::HttpBinary,
            Self::HttpJson => opentelemetry_otlp::Protocol::HttpJson,
        }
    }
}

/// Destination of one signal.
#[derive(Clone, PartialEq, Eq)]
pub struct SignalEndpoint {
    /// Full signal URL, e.g. `http://localhost:4318/v1/traces`.
    pub endpoint: String,
    /// Extra request headers (authentication, routing).
    pub headers: Vec<(String, String)>,
    /// Encoding.
    pub protocol: Protocol,
    /// Per-request timeout.
    pub timeout: Duration,
}

impl SignalEndpoint {
    /// An endpoint with default protocol and a 10 s timeout.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            headers: Vec::new(),
            protocol: Protocol::default(),
            timeout: Duration::from_secs(10),
        }
    }

    /// Adds a header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Sets the encoding.
    #[must_use]
    pub fn with_protocol(mut self, protocol: Protocol) -> Self {
        self.protocol = protocol;
        self
    }

    /// Sets the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl fmt::Debug for SignalEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header values carry credentials; never print them.
        f.debug_struct("SignalEndpoint")
            .field("endpoint", &self.endpoint)
            .field(
                "headers",
                &self
                    .headers
                    .iter()
                    .map(|(k, _)| k.as_str())
                    .collect::<Vec<_>>(),
            )
            .field("protocol", &self.protocol)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// In-memory delivery bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchLimits {
    /// Spans/log records held in memory before new ones are dropped.
    pub max_queue_size: usize,
    /// Items per export request.
    pub max_export_batch_size: usize,
    /// Delay between scheduled exports.
    pub scheduled_delay: Duration,
    /// Upper bound for one export request; caps each endpoint's timeout.
    pub export_timeout: Duration,
}

impl Default for BatchLimits {
    fn default() -> Self {
        Self {
            max_queue_size: 2048,
            max_export_batch_size: 256,
            scheduled_delay: Duration::from_millis(2000),
            export_timeout: Duration::from_millis(10_000),
        }
    }
}

/// Bounded retry policy for exports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryLimits {
    /// Retries after the first attempt.
    pub max_retries: usize,
    /// First backoff.
    pub initial_delay: Duration,
    /// Backoff cap (a server `Retry-After` may raise it).
    pub max_delay: Duration,
}

impl Default for RetryLimits {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_delay: Duration::from_millis(200),
            max_delay: Duration::from_millis(2000),
        }
    }
}

/// Resource identity attached to every signal.
#[derive(Clone, Debug, Default)]
pub struct ResourceConfig {
    /// `service.name` (falls back to `OTEL_SERVICE_NAME`, then `unknown_service`).
    pub service_name: Option<String>,
    /// `service.version`.
    pub service_version: Option<String>,
    /// Additional resource attributes.
    pub attributes: Vec<KeyValue>,
}

/// Pipeline configuration. Each signal is exported only when its endpoint is set.
#[derive(Clone, Debug)]
pub struct OtlpConfig {
    /// Trace destination.
    pub traces: Option<SignalEndpoint>,
    /// Metric destination.
    pub metrics: Option<SignalEndpoint>,
    /// Log/event destination.
    pub logs: Option<SignalEndpoint>,
    /// Resource identity.
    pub resource: ResourceConfig,
    /// Delivery bounds for spans and log records.
    pub batch: BatchLimits,
    /// Retry bounds.
    pub retry: RetryLimits,
    /// Metric export interval.
    pub metric_export_interval: Duration,
    /// Maximum request body size; larger requests are discarded, not sent.
    pub max_request_body_bytes: usize,
}

impl Default for OtlpConfig {
    fn default() -> Self {
        Self {
            traces: None,
            metrics: None,
            logs: None,
            resource: ResourceConfig::default(),
            batch: BatchLimits::default(),
            retry: RetryLimits::default(),
            metric_export_interval: Duration::from_millis(30_000),
            max_request_body_bytes: 491_520,
        }
    }
}

impl OtlpConfig {
    /// Sends all three signals to an OTLP/HTTP base URL such as an
    /// OpenTelemetry Collector at `http://localhost:4318`.
    pub fn collector(base_url: &str) -> Self {
        let base = base_url.trim_end_matches('/');
        Self {
            traces: Some(SignalEndpoint::new(format!("{base}/v1/traces"))),
            metrics: Some(SignalEndpoint::new(format!("{base}/v1/metrics"))),
            logs: Some(SignalEndpoint::new(format!("{base}/v1/logs"))),
            ..Self::default()
        }
    }

    /// Xybrid trace export: traces only, bearer-authenticated, to
    /// [`XYBRID_TRACES_ENDPOINT`]. Metrics and logs are never sent; route
    /// them through a Collector.
    pub fn xybrid_traces(api_key: &str) -> Self {
        Self::xybrid_traces_to(XYBRID_TRACES_ENDPOINT, api_key)
    }

    /// Xybrid trace export to a custom endpoint (staging, self-hosted).
    pub fn xybrid_traces_to(endpoint: &str, api_key: &str) -> Self {
        Self {
            traces: Some(
                SignalEndpoint::new(endpoint)
                    .with_header("Authorization", format!("Bearer {api_key}")),
            ),
            ..Self::default()
        }
    }

    /// Configuration from standard `OTEL_EXPORTER_OTLP_*` variables, or
    /// `None` when no endpoint is configured or `OTEL_SDK_DISABLED=true`.
    ///
    /// Headers (`OTEL_EXPORTER_OTLP_HEADERS`, per-signal variants) are applied
    /// by the exporter itself.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    fn from_vars(var: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let get = |k: &str| {
            var(k)
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
        };
        if get("OTEL_SDK_DISABLED").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
            return None;
        }
        let base = get("OTEL_EXPORTER_OTLP_ENDPOINT");
        let default_protocol = get("OTEL_EXPORTER_OTLP_PROTOCOL")
            .and_then(|p| Protocol::parse(&p))
            .unwrap_or_default();
        let signal = |name: &str, path: &str| -> Option<SignalEndpoint> {
            let upper = name.to_ascii_uppercase();
            let endpoint = get(&format!("OTEL_EXPORTER_OTLP_{upper}_ENDPOINT")).or_else(|| {
                base.as_ref()
                    .map(|b| format!("{}/{path}", b.trim_end_matches('/')))
            })?;
            let protocol = get(&format!("OTEL_EXPORTER_OTLP_{upper}_PROTOCOL"))
                .and_then(|p| Protocol::parse(&p))
                .unwrap_or(default_protocol);
            Some(SignalEndpoint::new(endpoint).with_protocol(protocol))
        };
        let config = Self {
            traces: signal("traces", "v1/traces"),
            metrics: signal("metrics", "v1/metrics"),
            logs: signal("logs", "v1/logs"),
            ..Self::default()
        };
        (config.traces.is_some() || config.metrics.is_some() || config.logs.is_some())
            .then_some(config)
    }

    /// Sets the service name.
    #[must_use]
    pub fn with_service(mut self, name: impl Into<String>, version: Option<String>) -> Self {
        self.resource.service_name = Some(name.into());
        self.resource.service_version = version;
        self
    }

    /// Adds a resource attribute.
    #[must_use]
    pub fn with_resource_attribute(mut self, kv: KeyValue) -> Self {
        self.resource.attributes.push(kv);
        self
    }

    /// Sets the trace destination.
    #[must_use]
    pub fn with_traces(mut self, endpoint: Option<SignalEndpoint>) -> Self {
        self.traces = endpoint;
        self
    }

    /// Sets the metric destination.
    #[must_use]
    pub fn with_metrics(mut self, endpoint: Option<SignalEndpoint>) -> Self {
        self.metrics = endpoint;
        self
    }

    /// Sets the log destination.
    #[must_use]
    pub fn with_logs(mut self, endpoint: Option<SignalEndpoint>) -> Self {
        self.logs = endpoint;
        self
    }

    /// Sets delivery bounds.
    #[must_use]
    pub fn with_batch(mut self, batch: BatchLimits) -> Self {
        self.batch = batch;
        self
    }

    /// Sets retry bounds.
    #[must_use]
    pub fn with_retry(mut self, retry: RetryLimits) -> Self {
        self.retry = retry;
        self
    }
}

/// Delivery counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExportStats {
    /// Spans accepted into the bounded queue.
    pub spans_queued: u64,
    /// Spans the destination accepted.
    pub spans_exported: u64,
    /// Spans dropped because the queue was full.
    pub spans_dropped: u64,
    /// Export requests that failed after retries.
    pub export_failures: u64,
    /// Spans lost in failed exports.
    pub spans_failed: u64,
}

#[derive(Debug, Default)]
struct Counters {
    queued: AtomicU64,
    exported: AtomicU64,
    dropped: AtomicU64,
    failures: AtomicU64,
    failed_spans: AtomicU64,
    pending: AtomicU64,
}

/// An application-owned OTLP/HTTP pipeline.
pub struct ObservabilityPipeline {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
    logger_provider: Option<SdkLoggerProvider>,
    counters: Arc<Counters>,
    shut_down: AtomicBool,
}

impl ObservabilityPipeline {
    /// Builds providers for every configured signal. Starts background export
    /// threads; no network I/O happens until the first export.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when no signal is configured, or
    /// [`Error::ExporterBuild`] when an exporter rejects its configuration.
    pub fn build(config: OtlpConfig) -> Result<Self> {
        if config.traces.is_none() && config.metrics.is_none() && config.logs.is_none() {
            return Err(Error::Config("no signal endpoint configured".into()));
        }
        let resource = build_resource(&config.resource);
        let retry = RetryPolicy::recommended()
            .with_max_retries(config.retry.max_retries)
            .with_initial_delay(config.retry.initial_delay)
            .with_max_delay(config.retry.max_delay);
        let counters = Arc::new(Counters::default());
        let cap = config.batch.export_timeout;
        let clamp = |e: &SignalEndpoint| SignalEndpoint {
            timeout: e.timeout.min(cap),
            ..e.clone()
        };
        let (traces, metrics, logs) = (
            config.traces.as_ref().map(clamp),
            config.metrics.as_ref().map(clamp),
            config.logs.as_ref().map(clamp),
        );

        let tracer_provider = match &traces {
            Some(endpoint) => {
                let exporter = SpanExporter::builder()
                    .with_http()
                    .with_endpoint(endpoint.endpoint.clone())
                    .with_protocol(endpoint.protocol.otlp())
                    .with_timeout(endpoint.timeout)
                    .with_headers(headers(endpoint))
                    .with_http_client(UreqClient::new(endpoint.timeout))
                    .with_retry_policy(retry.clone())
                    .with_max_request_body_size(config.max_request_body_bytes)
                    .build()
                    .map_err(|e| Error::ExporterBuild {
                        signal: "traces",
                        message: e.to_string(),
                    })?;
                let limit = config.batch.max_queue_size.max(1);
                let processor = BatchSpanProcessor::builder(CountingExporter {
                    inner: exporter,
                    counters: counters.clone(),
                })
                .with_batch_config(
                    BatchConfigBuilder::default()
                        // The gate below enforces `limit`; the inner queue is
                        // larger so it never drops uncounted.
                        .with_max_queue_size(limit.saturating_mul(2))
                        .with_max_export_batch_size(config.batch.max_export_batch_size.max(1))
                        .with_scheduled_delay(config.batch.scheduled_delay)
                        .build(),
                )
                .build();
                Some(
                    SdkTracerProvider::builder()
                        .with_span_processor(GatedProcessor {
                            inner: processor,
                            counters: counters.clone(),
                            limit: limit as u64,
                        })
                        .with_sampler(Sampler::ParentBased(Box::new(Sampler::AlwaysOn)))
                        .with_resource(resource.clone())
                        .build(),
                )
            }
            None => None,
        };

        let meter_provider = match &metrics {
            Some(endpoint) => {
                let exporter = MetricExporter::builder()
                    .with_http()
                    .with_endpoint(endpoint.endpoint.clone())
                    .with_protocol(endpoint.protocol.otlp())
                    .with_timeout(endpoint.timeout)
                    .with_headers(headers(endpoint))
                    .with_http_client(UreqClient::new(endpoint.timeout))
                    .with_retry_policy(retry.clone())
                    .with_max_request_body_size(config.max_request_body_bytes)
                    .build()
                    .map_err(|e| Error::ExporterBuild {
                        signal: "metrics",
                        message: e.to_string(),
                    })?;
                let reader = PeriodicReader::builder(exporter)
                    .with_interval(config.metric_export_interval)
                    .build();
                Some(
                    SdkMeterProvider::builder()
                        .with_reader(reader)
                        .with_resource(resource.clone())
                        .build(),
                )
            }
            None => None,
        };

        let logger_provider = match &logs {
            Some(endpoint) => {
                let exporter = LogExporter::builder()
                    .with_http()
                    .with_endpoint(endpoint.endpoint.clone())
                    .with_protocol(endpoint.protocol.otlp())
                    .with_timeout(endpoint.timeout)
                    .with_headers(headers(endpoint))
                    .with_http_client(UreqClient::new(endpoint.timeout))
                    .with_retry_policy(retry)
                    .with_max_request_body_size(config.max_request_body_bytes)
                    .build()
                    .map_err(|e| Error::ExporterBuild {
                        signal: "logs",
                        message: e.to_string(),
                    })?;
                let processor = BatchLogProcessor::builder(exporter)
                    .with_batch_config(
                        opentelemetry_sdk::logs::BatchConfigBuilder::default()
                            .with_max_queue_size(config.batch.max_queue_size.max(1))
                            .with_max_export_batch_size(config.batch.max_export_batch_size.max(1))
                            .with_scheduled_delay(config.batch.scheduled_delay)
                            .build(),
                    )
                    .build();
                Some(
                    SdkLoggerProvider::builder()
                        .with_log_processor(processor)
                        .with_resource(resource)
                        .build(),
                )
            }
            None => None,
        };

        Ok(Self {
            tracer_provider,
            meter_provider,
            logger_provider,
            counters,
            shut_down: AtomicBool::new(false),
        })
    }

    /// An observer wired to this pipeline's providers with the default
    /// content policy (environment override honored).
    #[must_use]
    pub fn observer(&self) -> Observer {
        self.observer_with(ContentPolicy::from_env())
    }

    /// An observer wired to this pipeline's providers.
    #[must_use]
    pub fn observer_with(&self, content: ContentPolicy) -> Observer {
        let mut builder = Observer::builder().content_policy(content);
        if let Some(p) = &self.tracer_provider {
            builder = builder.tracer_provider(p);
        }
        if let Some(p) = &self.meter_provider {
            builder = builder.meter_provider(p);
        }
        if let Some(p) = &self.logger_provider {
            builder = builder.logger_provider(p);
        }
        builder.build()
    }

    /// The tracer provider, for applications that also create their own spans.
    #[must_use]
    pub fn tracer_provider(&self) -> Option<&SdkTracerProvider> {
        self.tracer_provider.as_ref()
    }

    /// The meter provider.
    #[must_use]
    pub fn meter_provider(&self) -> Option<&SdkMeterProvider> {
        self.meter_provider.as_ref()
    }

    /// The logger provider.
    #[must_use]
    pub fn logger_provider(&self) -> Option<&SdkLoggerProvider> {
        self.logger_provider.as_ref()
    }

    /// Delivery counters.
    #[must_use]
    pub fn stats(&self) -> ExportStats {
        let c = &self.counters;
        ExportStats {
            spans_queued: c.queued.load(Ordering::Relaxed),
            spans_exported: c.exported.load(Ordering::Relaxed),
            spans_dropped: c.dropped.load(Ordering::Relaxed),
            export_failures: c.failures.load(Ordering::Relaxed),
            spans_failed: c.failed_spans.load(Ordering::Relaxed),
        }
    }

    /// Exports everything queued, waiting at most `timeout`. A no-op after
    /// shutdown.
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`] when the bound is exceeded (export continues in the
    /// background), [`Error::Lifecycle`] when a provider reports failure.
    pub fn force_flush(&self, timeout: Duration) -> Result<()> {
        if self.shut_down.load(Ordering::SeqCst) {
            return Ok(());
        }
        let (t, m, l) = (
            self.tracer_provider.clone(),
            self.meter_provider.clone(),
            self.logger_provider.clone(),
        );
        bounded("force_flush", timeout, move || {
            let mut errors = Vec::new();
            if let Some(p) = t {
                errors.extend(p.force_flush().err().map(|e| e.to_string()));
            }
            if let Some(p) = m {
                errors.extend(p.force_flush().err().map(|e| e.to_string()));
            }
            if let Some(p) = l {
                errors.extend(p.force_flush().err().map(|e| e.to_string()));
            }
            errors
        })
    }

    /// Flushes and stops every provider, waiting at most `timeout`.
    /// Idempotent: later calls return `Ok(())` immediately.
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`] or [`Error::Lifecycle`] as for
    /// [`ObservabilityPipeline::force_flush`].
    pub fn shutdown(&self, timeout: Duration) -> Result<()> {
        if self.shut_down.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let (t, m, l) = (
            self.tracer_provider.clone(),
            self.meter_provider.clone(),
            self.logger_provider.clone(),
        );
        bounded("shutdown", timeout, move || {
            let mut errors = Vec::new();
            if let Some(p) = t {
                errors.extend(
                    p.shutdown_with_timeout(timeout)
                        .err()
                        .map(|e| e.to_string()),
                );
            }
            if let Some(p) = m {
                errors.extend(
                    p.shutdown_with_timeout(timeout)
                        .err()
                        .map(|e| e.to_string()),
                );
            }
            if let Some(p) = l {
                errors.extend(
                    p.shutdown_with_timeout(timeout)
                        .err()
                        .map(|e| e.to_string()),
                );
            }
            errors
        })
    }

    /// Whether [`ObservabilityPipeline::shutdown`] has been called.
    #[must_use]
    pub fn is_shut_down(&self) -> bool {
        self.shut_down.load(Ordering::SeqCst)
    }
}

impl fmt::Debug for ObservabilityPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObservabilityPipeline")
            .field("traces", &self.tracer_provider.is_some())
            .field("metrics", &self.meter_provider.is_some())
            .field("logs", &self.logger_provider.is_some())
            .field("stats", &self.stats())
            .field("shut_down", &self.is_shut_down())
            .finish()
    }
}

fn build_resource(config: &ResourceConfig) -> Resource {
    let mut builder = Resource::builder();
    if let Some(name) = &config.service_name {
        builder = builder.with_service_name(name.clone());
    }
    let mut attributes = config.attributes.clone();
    if let Some(version) = &config.service_version {
        attributes.push(KeyValue::new("service.version", version.clone()));
    }
    builder.with_attributes(attributes).build()
}

fn headers(endpoint: &SignalEndpoint) -> HashMap<String, String> {
    endpoint.headers.iter().cloned().collect()
}

/// Runs `f` on a helper thread and waits at most `timeout`.
fn bounded(
    operation: &'static str,
    timeout: Duration,
    f: impl FnOnce() -> Vec<String> + Send + 'static,
) -> Result<()> {
    let (tx, rx) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name(format!("xybrid-observability-{operation}"))
        .spawn(move || {
            let _ = tx.send(f());
        });
    if spawned.is_err() {
        return Err(Error::Lifecycle {
            operation,
            message: "could not spawn lifecycle thread".into(),
        });
    }
    match rx.recv_timeout(timeout) {
        Ok(errors) if errors.is_empty() => Ok(()),
        Ok(errors) => Err(Error::Lifecycle {
            operation,
            message: errors.join("; "),
        }),
        Err(_) => Err(Error::Timeout {
            operation,
            timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
        }),
    }
}

/// Enforces the queue bound and counts drops precisely.
#[derive(Debug)]
struct GatedProcessor {
    inner: BatchSpanProcessor,
    counters: Arc<Counters>,
    limit: u64,
}

impl SpanProcessor for GatedProcessor {
    fn on_start(&self, span: &mut opentelemetry_sdk::trace::Span, cx: &opentelemetry::Context) {
        self.inner.on_start(span, cx);
    }

    fn on_end(&self, span: SpanData) {
        if !span.span_context.is_sampled() {
            return;
        }
        let c = &self.counters;
        if c.pending.fetch_add(1, Ordering::AcqRel) >= self.limit {
            c.pending.fetch_sub(1, Ordering::AcqRel);
            c.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        c.queued.fetch_add(1, Ordering::Relaxed);
        self.inner.on_end(span);
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

/// Counts exported and failed spans; releases queue slots.
#[derive(Debug)]
struct CountingExporter<E> {
    inner: E,
    counters: Arc<Counters>,
}

impl<E: SdkSpanExporter> SdkSpanExporter for CountingExporter<E> {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        let n = batch.len() as u64;
        let result = self.inner.export(batch).await;
        let c = &self.counters;
        c.pending
            .fetch_sub(n.min(c.pending.load(Ordering::Acquire)), Ordering::AcqRel);
        match &result {
            Ok(()) => {
                c.exported.fetch_add(n, Ordering::Relaxed);
            }
            Err(_) => {
                c.failures.fetch_add(1, Ordering::Relaxed);
                c.failed_spans.fetch_add(n, Ordering::Relaxed);
            }
        }
        result
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

/// Maximum response body read from a destination.
const MAX_RESPONSE_BYTES: u64 = 64 * 1024;

/// Blocking ureq transport. Exports run on the SDK's background threads, so
/// blocking here never touches application threads.
#[derive(Clone)]
struct UreqClient {
    agent: ureq::Agent,
}

impl UreqClient {
    fn new(timeout: Duration) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout_connect(timeout.min(Duration::from_secs(5)))
                .timeout(timeout)
                .redirects(0)
                .build(),
        }
    }
}

impl fmt::Debug for UreqClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("UreqClient")
    }
}

#[async_trait::async_trait]
impl HttpClient for UreqClient {
    async fn send_bytes(
        &self,
        request: Request<Bytes>,
    ) -> std::result::Result<Response<Bytes>, HttpError> {
        let (parts, body) = request.into_parts();
        let mut req = self
            .agent
            .request(parts.method.as_str(), &parts.uri.to_string());
        for (name, value) in &parts.headers {
            if let Ok(v) = value.to_str() {
                req = req.set(name.as_str(), v);
            }
        }
        let response = match req.send_bytes(&body) {
            Ok(r) | Err(ureq::Error::Status(_, r)) => r,
            Err(ureq::Error::Transport(t)) => {
                // Transport errors never include request payloads.
                return Err(format!("transport error: {}", t.kind()).into());
            }
        };
        let mut builder = Response::builder().status(response.status());
        for name in response.headers_names() {
            if let Some(v) = response.header(&name) {
                builder = builder.header(name.as_str(), v);
            }
        }
        let mut buf = Vec::new();
        response
            .into_reader()
            .take(MAX_RESPONSE_BYTES)
            .read_to_end(&mut buf)?;
        Ok(builder.body(Bytes::from(buf))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_config_derives_signal_paths_and_protocols() {
        let vars: HashMap<&str, &str> = [
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/"),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            (
                "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
                "http://logs:4318/custom",
            ),
            ("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "http/protobuf"),
        ]
        .into_iter()
        .collect();
        let config = OtlpConfig::from_vars(|k| vars.get(k).map(|v| (*v).to_owned())).unwrap();
        let traces = config.traces.unwrap();
        assert_eq!(traces.endpoint, "http://collector:4318/v1/traces");
        assert_eq!(traces.protocol, Protocol::HttpJson);
        let logs = config.logs.unwrap();
        assert_eq!(logs.endpoint, "http://logs:4318/custom");
        assert_eq!(logs.protocol, Protocol::HttpProtobuf);
    }

    #[test]
    fn env_config_absent_or_disabled() {
        assert!(OtlpConfig::from_vars(|_| None).is_none());
        let disabled = |k: &str| match k {
            "OTEL_SDK_DISABLED" => Some("TRUE".to_owned()),
            "OTEL_EXPORTER_OTLP_ENDPOINT" => Some("http://x".to_owned()),
            _ => None,
        };
        assert!(OtlpConfig::from_vars(disabled).is_none());
    }

    #[test]
    fn xybrid_preset_is_traces_only_and_hides_credentials_in_debug() {
        let config = OtlpConfig::xybrid_traces("sk-secret");
        assert!(config.metrics.is_none() && config.logs.is_none());
        let traces = config.traces.unwrap();
        assert_eq!(traces.endpoint, XYBRID_TRACES_ENDPOINT);
        assert!(!format!("{traces:?}").contains("sk-secret"));
    }

    #[test]
    fn empty_config_is_rejected() {
        assert!(ObservabilityPipeline::build(OtlpConfig::default()).is_err());
    }
}
