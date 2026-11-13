//! OTLP configuration, environment sources and construction-time validation
//! (CONTRACT.md "Configuration sources").

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry_sdk::resource::TelemetryResourceDetector;
use opentelemetry_sdk::Resource;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use reqwest::Url;

use super::defaults;
use super::encode::Signal;
use super::hook::ResponseHook;
use crate::content::SpanContentEncoding;
use crate::error::{Error, Result};

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
    /// Parses `http/protobuf` or `http/json` (trimmed, case-insensitive).
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "http/protobuf" => Some(Self::HttpProtobuf),
            "http/json" => Some(Self::HttpJson),
            _ => None,
        }
    }

    /// The OTLP spelling (`http/protobuf`, `http/json`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HttpProtobuf => "http/protobuf",
            Self::HttpJson => "http/json",
        }
    }

    /// The request `Content-Type`.
    #[must_use]
    pub fn content_type(self) -> &'static str {
        match self {
            Self::HttpProtobuf => "application/x-protobuf",
            Self::HttpJson => "application/json",
        }
    }

    /// How this protocol writes span content, which the content budgets of
    /// an observer are fitted to.
    #[must_use]
    pub fn span_content_encoding(self) -> SpanContentEncoding {
        match self {
            Self::HttpProtobuf => SpanContentEncoding::Protobuf,
            Self::HttpJson => SpanContentEncoding::Json,
        }
    }
}

/// Whether the environment may contribute request headers
/// (`catalog.enums.header_policy`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeaderPolicy {
    /// `OTEL_EXPORTER_OTLP_HEADERS`, overlaid by
    /// `OTEL_EXPORTER_OTLP_{SIGNAL}_HEADERS`, overlaid by programmatic headers
    /// (case-insensitive keys), read once when the pipeline or exporter is
    /// built. The default for generic endpoints.
    #[default]
    MergeEnvironment,
    /// Only programmatic headers are sent; the environment never adds,
    /// replaces or removes a header. Used by the Xybrid preset.
    Isolated,
}

impl HeaderPolicy {
    /// The catalog spelling (`merge_environment`, `isolated`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MergeEnvironment => "merge_environment",
            Self::Isolated => "isolated",
        }
    }
}

/// Where `OTEL_*` variables are read from: environment-derived endpoints
/// ([`OtlpConfig::from_environment`]), `merge_environment` headers and
/// resource detection (`OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`).
#[derive(Clone, Default, PartialEq, Eq)]
pub enum EnvironmentSource {
    /// The process environment (default). Non-UTF-8 values are unset.
    #[default]
    Process,
    /// No environment at all.
    Disabled,
    /// An injected map (tests, embedders with their own configuration).
    Vars(BTreeMap<String, String>),
}

impl EnvironmentSource {
    /// An injected environment.
    pub fn vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self::Vars(
            vars.into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        )
    }

    /// The value of `key`; empty values are unset.
    pub(crate) fn get(&self, key: &str) -> Option<String> {
        let value = match self {
            Self::Process => std::env::var(key).ok(),
            Self::Disabled => None,
            Self::Vars(vars) => vars.get(key).cloned(),
        }?;
        (!value.trim().is_empty()).then_some(value)
    }
}

impl fmt::Debug for EnvironmentSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Process => f.write_str("Process"),
            Self::Disabled => f.write_str("Disabled"),
            // Values may carry credentials (`*_HEADERS`); print keys only.
            Self::Vars(vars) => f
                .debug_tuple("Vars")
                .field(&vars.keys().collect::<Vec<_>>())
                .finish(),
        }
    }
}

/// Destination of one signal.
#[derive(Clone, PartialEq, Eq)]
pub struct SignalEndpoint {
    /// Full signal URL, e.g. `http://localhost:4318/v1/traces`.
    pub endpoint: String,
    /// Programmatic request headers (authentication, routing).
    pub headers: Vec<(String, String)>,
    /// Encoding.
    pub protocol: Protocol,
    /// Per-attempt timeout; each attempt also ends at the export deadline.
    pub timeout: Duration,
    /// Whether environment headers are merged under the programmatic ones.
    pub header_policy: HeaderPolicy,
    /// Called once for every HTTP response of this destination; it may
    /// disable the destination (see [`super::ResponseHook`]).
    pub response_hook: Option<ResponseHook>,
    /// Built by the Xybrid preset ([`OtlpConfig::xybrid_traces_to`]).
    xybrid_preset: bool,
}

impl SignalEndpoint {
    /// An endpoint with the default protocol, a 10 s attempt timeout,
    /// `merge_environment` headers and no response hook.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            headers: Vec::new(),
            protocol: Protocol::default(),
            timeout: defaults::SIGNAL_TIMEOUT,
            header_policy: HeaderPolicy::default(),
            response_hook: None,
            xybrid_preset: false,
        }
    }

    /// Whether this destination was configured through the Xybrid preset
    /// ([`OtlpConfig::xybrid_traces`] / [`OtlpConfig::xybrid_traces_to`]),
    /// as opposed to a generic endpoint or one read from the environment.
    /// Adapters use it to decide which destinations' vendor responses to
    /// trust; the transport itself treats every destination alike.
    #[must_use]
    pub fn is_xybrid_preset(&self) -> bool {
        self.xybrid_preset
    }

    /// Sets the response hook, replacing any previous one.
    #[must_use]
    pub fn with_response_hook(mut self, hook: ResponseHook) -> Self {
        self.response_hook = Some(hook);
        self
    }

    /// Adds a programmatic header.
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

    /// Sets the per-attempt timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the header policy.
    #[must_use]
    pub fn with_header_policy(mut self, policy: HeaderPolicy) -> Self {
        self.header_policy = policy;
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
            .field("header_policy", &self.header_policy)
            .field("response_hook", &self.response_hook.is_some())
            .field("xybrid_preset", &self.xybrid_preset)
            .finish()
    }
}

/// In-memory delivery bounds for spans and log records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchLimits {
    /// Queued plus in-flight items per signal; items beyond are dropped.
    pub max_queue_size: usize,
    /// Items per export batch (effective value `min(batch, queue)`).
    pub max_export_batch_size: usize,
    /// Delay between scheduled exports.
    pub scheduled_delay: Duration,
    /// One absolute deadline per export: every split part, attempt, backoff
    /// and response read of a batch shares it.
    pub export_timeout: Duration,
    /// Queued plus in-flight bytes per signal (a conservative per-item
    /// estimate; library-owned data, not process memory).
    pub max_buffered_bytes: usize,
}

impl Default for BatchLimits {
    fn default() -> Self {
        Self {
            max_queue_size: defaults::MAX_QUEUE_SIZE,
            max_export_batch_size: defaults::MAX_EXPORT_BATCH_SIZE,
            scheduled_delay: defaults::SCHEDULED_DELAY,
            export_timeout: defaults::EXPORT_TIMEOUT,
            max_buffered_bytes: defaults::MAX_BUFFERED_BYTES_PER_SIGNAL,
        }
    }
}

/// Bounded retry policy for exports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryLimits {
    /// Retries after the first attempt.
    pub max_retries: usize,
    /// First backoff; retry `n` waits `min(initial_delay × 2^n, max_delay)`.
    pub initial_delay: Duration,
    /// Backoff cap (a server `Retry-After` may exceed it, never the deadline).
    pub max_delay: Duration,
}

impl Default for RetryLimits {
    fn default() -> Self {
        Self {
            max_retries: defaults::MAX_RETRIES,
            initial_delay: defaults::INITIAL_DELAY,
            max_delay: defaults::MAX_DELAY,
        }
    }
}

/// Resource identity attached to every signal.
#[derive(Clone, Debug, Default)]
pub struct ResourceConfig {
    /// `service.name`; wins over `OTEL_SERVICE_NAME` and
    /// `OTEL_RESOURCE_ATTRIBUTES` (default `unknown_service`).
    pub service_name: Option<String>,
    /// `service.version`.
    pub service_version: Option<String>,
    /// Additional resource attributes; win over `OTEL_RESOURCE_ATTRIBUTES`.
    pub attributes: Vec<KeyValue>,
}

/// Pipeline or exporter configuration. Each signal is exported only when its
/// endpoint is set; building validates every option first.
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
    /// Metric export interval; the metric export deadline is
    /// `min(export_timeout, metric_export_interval)`.
    pub metric_export_interval: Duration,
    /// Maximum request body size. Trace and log batches are split to fit; a
    /// single item over the bound is dropped; a metrics collection over the
    /// bound fails.
    pub max_request_body_bytes: usize,
    /// Source of `OTEL_*` variables for headers and resource detection.
    pub environment: EnvironmentSource,
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
            metric_export_interval: defaults::METRIC_EXPORT_INTERVAL,
            max_request_body_bytes: defaults::MAX_REQUEST_BODY_BYTES,
            environment: EnvironmentSource::default(),
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

    /// Xybrid trace export: traces only, `http/protobuf`, exactly
    /// `Authorization: Bearer <key>`, to [`XYBRID_TRACES_ENDPOINT`]. No
    /// `OTEL_EXPORTER_OTLP_*` variable is read (`header_policy = isolated`).
    /// Metrics and logs are never sent; route them through a Collector.
    pub fn xybrid_traces(api_key: &str) -> Self {
        Self::xybrid_traces_to(XYBRID_TRACES_ENDPOINT, api_key)
    }

    /// Xybrid trace export to a custom endpoint (staging, self-hosted).
    ///
    /// The traces endpoint reports [`SignalEndpoint::is_xybrid_preset`].
    pub fn xybrid_traces_to(endpoint: &str, api_key: &str) -> Self {
        let mut traces = SignalEndpoint::new(endpoint)
            .with_protocol(Protocol::HttpProtobuf)
            .with_header_policy(HeaderPolicy::Isolated)
            .with_header("Authorization", format!("Bearer {api_key}"));
        traces.xybrid_preset = true;
        Self {
            traces: Some(traces),
            ..Self::default()
        }
    }

    /// Configuration from the standard `OTEL_*` variables of the process, or
    /// `None` when no endpoint is configured or `OTEL_SDK_DISABLED=true`.
    /// Headers (`OTEL_EXPORTER_OTLP_[{SIGNAL}_]HEADERS`) are merged when the
    /// pipeline is built.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        Self::from_environment(EnvironmentSource::Process)
    }

    /// [`OtlpConfig::from_env`] over an explicit environment source, which
    /// the configuration keeps for header merging and resource detection.
    #[must_use]
    pub fn from_environment(environment: EnvironmentSource) -> Option<Self> {
        let get = |k: &str| environment.get(k).map(|v| v.trim().to_owned());
        if get("OTEL_SDK_DISABLED").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
            return None;
        }
        let base = get("OTEL_EXPORTER_OTLP_ENDPOINT");
        let default_protocol = get("OTEL_EXPORTER_OTLP_PROTOCOL").map(|p| parse_protocol(&p));
        let default_timeout = get("OTEL_EXPORTER_OTLP_TIMEOUT")
            .and_then(|v| parse_positive(&v, "OTEL_EXPORTER_OTLP_TIMEOUT"));
        let signal = |signal: Signal| -> Option<SignalEndpoint> {
            let upper = signal.as_str().to_ascii_uppercase();
            if get(&format!("OTEL_{upper}_EXPORTER")).is_some_and(|v| {
                v.split(',')
                    .any(|item| item.trim().eq_ignore_ascii_case("none"))
            }) {
                return None;
            }
            let endpoint = get(&format!("OTEL_EXPORTER_OTLP_{upper}_ENDPOINT")).or_else(|| {
                base.as_ref()
                    .map(|b| format!("{}/v1/{}", b.trim_end_matches('/'), signal.as_str()))
            })?;
            let protocol = get(&format!("OTEL_EXPORTER_OTLP_{upper}_PROTOCOL"))
                .map(|p| parse_protocol(&p))
                .or(default_protocol)
                .unwrap_or_default();
            let timeout_var = format!("OTEL_EXPORTER_OTLP_{upper}_TIMEOUT");
            let timeout = get(&timeout_var)
                .and_then(|v| parse_positive(&v, &timeout_var))
                .or(default_timeout)
                .map_or(defaults::SIGNAL_TIMEOUT, Duration::from_millis);
            Some(
                SignalEndpoint::new(endpoint)
                    .with_protocol(protocol)
                    .with_timeout(timeout),
            )
        };
        let mut batch = BatchLimits::default();
        let positive = |name: &str| get(name).and_then(|v| parse_positive(&v, name));
        if let Some(v) = positive("OTEL_BSP_MAX_QUEUE_SIZE") {
            batch.max_queue_size = usize::try_from(v).unwrap_or(usize::MAX);
        }
        if let Some(v) = positive("OTEL_BSP_MAX_EXPORT_BATCH_SIZE") {
            batch.max_export_batch_size = usize::try_from(v).unwrap_or(usize::MAX);
        }
        if let Some(v) = positive("OTEL_BSP_SCHEDULE_DELAY") {
            batch.scheduled_delay = Duration::from_millis(v);
        }
        if let Some(v) = positive("OTEL_BSP_EXPORT_TIMEOUT") {
            batch.export_timeout = Duration::from_millis(v);
        }
        let metric_export_interval = positive("OTEL_METRIC_EXPORT_INTERVAL")
            .map_or(defaults::METRIC_EXPORT_INTERVAL, Duration::from_millis);
        let config = Self {
            traces: signal(Signal::Traces),
            metrics: signal(Signal::Metrics),
            logs: signal(Signal::Logs),
            batch,
            metric_export_interval,
            environment: environment.clone(),
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

    /// Sets the metric export interval.
    #[must_use]
    pub fn with_metric_export_interval(mut self, interval: Duration) -> Self {
        self.metric_export_interval = interval;
        self
    }

    /// Sets the request body bound.
    #[must_use]
    pub fn with_max_request_body_bytes(mut self, bytes: usize) -> Self {
        self.max_request_body_bytes = bytes;
        self
    }

    /// Sets the environment source.
    #[must_use]
    pub fn with_environment(mut self, environment: EnvironmentSource) -> Self {
        self.environment = environment;
        self
    }
}

fn parse_protocol(value: &str) -> Protocol {
    Protocol::parse(value).unwrap_or_else(|| {
        log::warn!("xybrid-observability: unsupported OTLP protocol; using http/protobuf");
        Protocol::HttpProtobuf
    })
}

fn parse_positive(value: &str, name: &str) -> Option<u64> {
    match value.trim().parse::<u64>() {
        Ok(n) if n >= 1 => Some(n),
        _ => {
            log::warn!("xybrid-observability: {name} is not a positive integer; ignored");
            None
        }
    }
}

/// Parses a comma-separated, percent-encoded `key=value` list
/// (`OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_RESOURCE_ATTRIBUTES`). Malformed pairs
/// are skipped with a diagnostic that never contains the value.
pub(crate) fn parse_pairs(value: &str, variable: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for item in value.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let parsed = item.split_once('=').and_then(|(k, v)| {
            let key = percent_decode(k.trim())?;
            let value = percent_decode(v.trim())?;
            (!key.is_empty()).then_some((key, value))
        });
        match parsed {
            Some(pair) => pairs.push(pair),
            None => log::warn!("xybrid-observability: skipped a malformed entry in {variable}"),
        }
    }
    pairs
}

fn percent_decode(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = bytes.get(i + 1..i + 3)?;
            let hex = std::str::from_utf8(hex).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// One validated signal destination.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedSignal {
    pub signal: Signal,
    pub url: Url,
    pub headers: HeaderMap,
    pub protocol: Protocol,
    pub timeout: Duration,
    pub hook: Option<ResponseHook>,
}

/// Effective delivery limits after validation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub max_queue_size: usize,
    pub batch_size: usize,
    pub scheduled_delay: Duration,
    pub export_timeout: Duration,
    pub max_buffered_bytes: u64,
    pub max_request_body_bytes: usize,
    pub metric_export_interval: Duration,
    pub retry: RetryLimits,
}

impl Limits {
    /// The deadline budget of one metrics export.
    pub(crate) fn metric_export_timeout(&self) -> Duration {
        self.export_timeout.min(self.metric_export_interval)
    }
}

/// A configuration that passed validation, with environment headers merged
/// and the resource detected.
#[derive(Debug)]
pub(crate) struct Resolved {
    pub traces: Option<ResolvedSignal>,
    pub metrics: Option<ResolvedSignal>,
    pub logs: Option<ResolvedSignal>,
    pub limits: Limits,
    pub resource: Resource,
}

fn at_least_one_ms(value: Duration, name: &str) -> Result<()> {
    if value < Duration::from_millis(1) {
        return Err(Error::Config(format!("{name} must be at least 1 ms")));
    }
    Ok(())
}

/// Validates every option (D25) and resolves headers (D24) before anything
/// is started. Pure: no thread, runtime, timer or connection is created.
pub(crate) fn resolve(config: &OtlpConfig) -> Result<Resolved> {
    if config.traces.is_none() && config.metrics.is_none() && config.logs.is_none() {
        return Err(Error::Config("no signal endpoint configured".into()));
    }
    let batch = &config.batch;
    if batch.max_queue_size < 1 {
        return Err(Error::Config("batch.max_queue_size must be >= 1".into()));
    }
    if batch.max_export_batch_size < 1 {
        return Err(Error::Config(
            "batch.max_export_batch_size must be >= 1".into(),
        ));
    }
    at_least_one_ms(batch.scheduled_delay, "batch.scheduled_delay")?;
    at_least_one_ms(batch.export_timeout, "batch.export_timeout")?;
    if batch.max_buffered_bytes < 1 {
        return Err(Error::Config(
            "batch.max_buffered_bytes must be >= 1".into(),
        ));
    }
    if config.retry.max_delay < config.retry.initial_delay {
        return Err(Error::Config(
            "retry.max_delay must be >= retry.initial_delay".into(),
        ));
    }
    if config.max_request_body_bytes < 1 {
        return Err(Error::Config("max_request_body_bytes must be >= 1".into()));
    }
    at_least_one_ms(config.metric_export_interval, "metric_export_interval")?;

    let resolve_signal = |signal: Signal, endpoint: &Option<SignalEndpoint>| {
        endpoint
            .as_ref()
            .map(|e| resolve_endpoint(signal, e, &config.environment))
            .transpose()
    };
    Ok(Resolved {
        traces: resolve_signal(Signal::Traces, &config.traces)?,
        metrics: resolve_signal(Signal::Metrics, &config.metrics)?,
        logs: resolve_signal(Signal::Logs, &config.logs)?,
        limits: Limits {
            max_queue_size: batch.max_queue_size,
            batch_size: batch.max_export_batch_size.min(batch.max_queue_size),
            scheduled_delay: batch.scheduled_delay,
            export_timeout: batch.export_timeout,
            max_buffered_bytes: u64::try_from(batch.max_buffered_bytes).unwrap_or(u64::MAX),
            max_request_body_bytes: config.max_request_body_bytes,
            metric_export_interval: config.metric_export_interval,
            retry: config.retry,
        },
        resource: build_resource(&config.resource, &config.environment),
    })
}

fn resolve_endpoint(
    signal: Signal,
    endpoint: &SignalEndpoint,
    environment: &EnvironmentSource,
) -> Result<ResolvedSignal> {
    let name = signal.as_str();
    let url = Url::parse(endpoint.endpoint.trim())
        .ok()
        .filter(|u| matches!(u.scheme(), "http" | "https") && u.host().is_some())
        .ok_or_else(|| Error::Config(format!("{name} endpoint must be an absolute http(s) URL")))?;
    at_least_one_ms(endpoint.timeout, &format!("{name} timeout"))?;

    let mut headers = HeaderMap::new();
    if endpoint.header_policy == HeaderPolicy::MergeEnvironment {
        let upper = name.to_ascii_uppercase();
        for variable in [
            "OTEL_EXPORTER_OTLP_HEADERS".to_owned(),
            format!("OTEL_EXPORTER_OTLP_{upper}_HEADERS"),
        ] {
            let Some(value) = environment.get(&variable) else {
                continue;
            };
            for (key, value) in parse_pairs(&value, &variable) {
                match header(&key, &value) {
                    Some((k, v)) => {
                        headers.insert(k, v);
                    }
                    None => {
                        log::warn!("xybrid-observability: skipped an invalid header in {variable}")
                    }
                }
            }
        }
    }
    for (key, value) in &endpoint.headers {
        let (k, v) = header(key, value).ok_or_else(|| {
            Error::Config(format!("{name} endpoint has an invalid header {key:?}"))
        })?;
        headers.insert(k, v);
    }
    // The transport-owned header always wins; the transport sets it per
    // request (D24). Every other configured header is sent as resolved.
    headers.remove(CONTENT_TYPE);
    Ok(ResolvedSignal {
        signal,
        url,
        headers,
        protocol: endpoint.protocol,
        timeout: endpoint.timeout,
        hook: endpoint.response_hook.clone(),
    })
}

fn header(key: &str, value: &str) -> Option<(HeaderName, HeaderValue)> {
    let name = HeaderName::from_bytes(key.trim().as_bytes()).ok()?;
    let mut value = HeaderValue::from_str(value).ok()?;
    value.set_sensitive(true);
    Some((name, value))
}

/// The resource: programmatic service name and attributes win over
/// `OTEL_SERVICE_NAME`, which wins over `service.name` in
/// `OTEL_RESOURCE_ATTRIBUTES`; the default name is `unknown_service`.
pub(crate) fn build_resource(config: &ResourceConfig, environment: &EnvironmentSource) -> Resource {
    let mut attributes: Vec<KeyValue> = Vec::new();
    let mut service_name: Option<String> = None;
    if let Some(value) = environment.get("OTEL_RESOURCE_ATTRIBUTES") {
        for (key, value) in parse_pairs(&value, "OTEL_RESOURCE_ATTRIBUTES") {
            if key == "service.name" {
                service_name = Some(value.clone());
            }
            attributes.push(KeyValue::new(key, value));
        }
    }
    if let Some(name) = environment.get("OTEL_SERVICE_NAME") {
        service_name = Some(name.trim().to_owned());
    }
    for kv in &config.attributes {
        if kv.key.as_str() == "service.name" {
            service_name = Some(kv.value.to_string());
        }
        attributes.push(kv.clone());
    }
    if let Some(version) = &config.service_version {
        attributes.push(KeyValue::new("service.version", version.clone()));
    }
    if let Some(name) = &config.service_name {
        service_name = Some(name.clone());
    }
    Resource::builder_empty()
        .with_detector(Box::new(TelemetryResourceDetector))
        .with_attributes(attributes)
        .with_service_name(service_name.unwrap_or_else(|| "unknown_service".to_owned()))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> EnvironmentSource {
        EnvironmentSource::vars(pairs.iter().copied())
    }

    #[test]
    fn env_config_derives_signal_paths_protocols_and_timeouts() {
        let config = OtlpConfig::from_environment(env(&[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/"),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", " HTTP/JSON "),
            (
                "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
                "http://logs:4318/custom",
            ),
            ("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "http/protobuf"),
            ("OTEL_EXPORTER_OTLP_TIMEOUT", "3000"),
            ("OTEL_EXPORTER_OTLP_METRICS_TIMEOUT", "1500"),
            ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", ""),
        ]))
        .unwrap();
        let traces = config.traces.unwrap();
        assert_eq!(traces.endpoint, "http://collector:4318/v1/traces");
        assert_eq!(traces.protocol, Protocol::HttpJson, "empty is unset");
        assert_eq!(traces.timeout, Duration::from_millis(3000));
        assert_eq!(traces.header_policy, HeaderPolicy::MergeEnvironment);
        let logs = config.logs.unwrap();
        assert_eq!(logs.endpoint, "http://logs:4318/custom");
        assert_eq!(logs.protocol, Protocol::HttpProtobuf);
        let metrics = config.metrics.unwrap();
        assert_eq!(metrics.timeout, Duration::from_millis(1500));
    }

    #[test]
    fn env_config_absent_disabled_and_per_signal_none() {
        assert!(OtlpConfig::from_environment(EnvironmentSource::Disabled).is_none());
        assert!(OtlpConfig::from_environment(env(&[])).is_none());
        for disabled in ["true", "TRUE", " true "] {
            assert!(OtlpConfig::from_environment(env(&[
                ("OTEL_SDK_DISABLED", disabled),
                ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x"),
            ]))
            .is_none());
        }
        assert!(OtlpConfig::from_environment(env(&[
            ("OTEL_SDK_DISABLED", "false"),
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x"),
        ]))
        .is_some());
        let config = OtlpConfig::from_environment(env(&[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x"),
            ("OTEL_METRICS_EXPORTER", "otlp, NONE"),
            ("OTEL_LOGS_EXPORTER", "none"),
        ]))
        .unwrap();
        assert!(config.traces.is_some());
        assert!(config.metrics.is_none() && config.logs.is_none());
    }

    #[test]
    fn env_config_reads_batch_and_metric_variables() {
        let config = OtlpConfig::from_environment(env(&[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x"),
            ("OTEL_BSP_MAX_QUEUE_SIZE", "64"),
            ("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "16"),
            ("OTEL_BSP_SCHEDULE_DELAY", "250"),
            ("OTEL_BSP_EXPORT_TIMEOUT", "4000"),
            ("OTEL_METRIC_EXPORT_INTERVAL", "1000"),
        ]))
        .unwrap();
        assert_eq!(config.batch.max_queue_size, 64);
        assert_eq!(config.batch.max_export_batch_size, 16);
        assert_eq!(config.batch.scheduled_delay, Duration::from_millis(250));
        assert_eq!(config.batch.export_timeout, Duration::from_millis(4000));
        assert_eq!(config.metric_export_interval, Duration::from_millis(1000));
        // Invalid values are ignored, keeping the defaults.
        let config = OtlpConfig::from_environment(env(&[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x"),
            ("OTEL_BSP_MAX_QUEUE_SIZE", "0"),
            ("OTEL_BSP_SCHEDULE_DELAY", "-5"),
            ("OTEL_EXPORTER_OTLP_TIMEOUT", "1.5"),
        ]))
        .unwrap();
        assert_eq!(config.batch, BatchLimits::default());
        assert_eq!(config.traces.unwrap().timeout, defaults::SIGNAL_TIMEOUT);
    }

    #[test]
    fn header_lists_are_percent_decoded_and_malformed_pairs_skipped() {
        assert_eq!(
            parse_pairs(
                " a=1 , b = x%20y ,bad, =novalue, c=%ZZ, d=%E2%9C%93,,",
                "TEST"
            ),
            vec![
                ("a".to_owned(), "1".to_owned()),
                ("b".to_owned(), "x y".to_owned()),
                ("d".to_owned(), "\u{2713}".to_owned()),
            ]
        );
    }

    #[test]
    fn merge_environment_headers_follow_precedence_and_isolated_ignores_env() {
        let environment = env(&[
            (
                "OTEL_EXPORTER_OTLP_HEADERS",
                "Authorization=Basic%20x,x-honeycomb-team=k,x-general=g",
            ),
            (
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
                "X-General=signal,x-other=y",
            ),
        ]);
        let generic =
            SignalEndpoint::new("http://h/v1/traces").with_header("AUTHORIZATION", "mine");
        let resolved = resolve_endpoint(Signal::Traces, &generic, &environment).unwrap();
        assert_eq!(resolved.headers.len(), 4);
        assert_eq!(resolved.headers["authorization"], "mine");
        assert_eq!(resolved.headers["x-honeycomb-team"], "k");
        assert_eq!(resolved.headers["x-general"], "signal");
        assert_eq!(resolved.headers["x-other"], "y");

        let preset = OtlpConfig::xybrid_traces_to("http://h/v1/traces", "sk");
        let resolved = resolve_endpoint(
            Signal::Traces,
            preset.traces.as_ref().unwrap(),
            &environment,
        )
        .unwrap();
        assert_eq!(resolved.headers.len(), 1);
        assert_eq!(resolved.headers["authorization"], "Bearer sk");

        // Content-Type is transport-owned; nothing else is (an explicit
        // Accept is the caller's header and is sent as configured).
        let typed = SignalEndpoint::new("http://h/v1/logs")
            .with_header("content-type", "text/plain")
            .with_header("accept", "application/json");
        let resolved =
            resolve_endpoint(Signal::Logs, &typed, &EnvironmentSource::Disabled).unwrap();
        assert_eq!(resolved.headers.len(), 1);
        assert_eq!(resolved.headers["accept"], "application/json");
    }

    #[test]
    fn invalid_options_are_rejected_by_name() {
        let base = OtlpConfig::collector("http://localhost:4318");
        let cases: Vec<(OtlpConfig, &str)> = vec![
            (OtlpConfig::default(), "no signal"),
            (
                base.clone().with_batch(BatchLimits {
                    max_queue_size: 0,
                    ..BatchLimits::default()
                }),
                "max_queue_size",
            ),
            (
                base.clone().with_batch(BatchLimits {
                    max_export_batch_size: 0,
                    ..BatchLimits::default()
                }),
                "max_export_batch_size",
            ),
            (
                base.clone().with_batch(BatchLimits {
                    scheduled_delay: Duration::ZERO,
                    ..BatchLimits::default()
                }),
                "scheduled_delay",
            ),
            (
                base.clone().with_batch(BatchLimits {
                    export_timeout: Duration::from_micros(10),
                    ..BatchLimits::default()
                }),
                "export_timeout",
            ),
            (
                base.clone().with_batch(BatchLimits {
                    max_buffered_bytes: 0,
                    ..BatchLimits::default()
                }),
                "max_buffered_bytes",
            ),
            (
                base.clone().with_retry(RetryLimits {
                    initial_delay: Duration::from_millis(10),
                    max_delay: Duration::from_millis(5),
                    ..RetryLimits::default()
                }),
                "max_delay",
            ),
            (
                base.clone().with_max_request_body_bytes(0),
                "max_request_body_bytes",
            ),
            (
                base.clone().with_metric_export_interval(Duration::ZERO),
                "metric_export_interval",
            ),
            (
                base.clone()
                    .with_traces(Some(SignalEndpoint::new("/v1/traces"))),
                "traces endpoint",
            ),
            (
                base.clone()
                    .with_logs(Some(SignalEndpoint::new("ftp://h/v1/logs"))),
                "logs endpoint",
            ),
            (
                base.clone().with_metrics(Some(
                    SignalEndpoint::new("http://h/v1/metrics").with_timeout(Duration::ZERO),
                )),
                "metrics timeout",
            ),
            (
                base.clone().with_traces(Some(
                    SignalEndpoint::new("http://h/v1/traces").with_header("bad name", "v"),
                )),
                "invalid header",
            ),
        ];
        for (config, needle) in cases {
            let error = resolve(&config).unwrap_err();
            assert!(
                matches!(&error, Error::Config(m) if m.contains(needle)),
                "{needle}: {error}"
            );
        }
        // Very large durations are valid limits (they mean "practically
        // never"); they must be accepted and must never overflow later.
        let huge = resolve(
            &base
                .clone()
                .with_batch(BatchLimits {
                    scheduled_delay: Duration::MAX,
                    export_timeout: Duration::MAX,
                    ..BatchLimits::default()
                })
                .with_retry(RetryLimits {
                    max_retries: usize::MAX,
                    initial_delay: Duration::MAX,
                    max_delay: Duration::MAX,
                })
                .with_metric_export_interval(Duration::MAX)
                .with_traces(Some(
                    SignalEndpoint::new("http://h/v1/traces").with_timeout(Duration::MAX),
                )),
        )
        .unwrap();
        assert_eq!(huge.limits.metric_export_timeout(), Duration::MAX);
        let resolved = resolve(&base.clone().with_batch(BatchLimits {
            max_queue_size: 4,
            max_export_batch_size: 16,
            ..BatchLimits::default()
        }))
        .unwrap();
        assert_eq!(
            resolved.limits.batch_size, 4,
            "batch is clamped to the queue"
        );
    }

    #[test]
    fn resource_precedence() {
        let environment = env(&[
            (
                "OTEL_RESOURCE_ATTRIBUTES",
                "service.name=from-attrs,deployment.environment=prod,team=a",
            ),
            ("OTEL_SERVICE_NAME", "from-env"),
        ]);
        let value = |r: &Resource, k: &str| {
            r.get(&opentelemetry::Key::from(k.to_owned()))
                .map(|v| v.to_string())
        };
        let r = build_resource(&ResourceConfig::default(), &environment);
        assert_eq!(value(&r, "service.name").as_deref(), Some("from-env"));
        assert_eq!(value(&r, "deployment.environment").as_deref(), Some("prod"));
        assert_eq!(value(&r, "telemetry.sdk.language").as_deref(), Some("rust"));
        let r = build_resource(
            &ResourceConfig {
                service_name: Some("mine".into()),
                service_version: Some("1.2".into()),
                attributes: vec![KeyValue::new("team", "b")],
            },
            &environment,
        );
        assert_eq!(value(&r, "service.name").as_deref(), Some("mine"));
        assert_eq!(value(&r, "service.version").as_deref(), Some("1.2"));
        assert_eq!(value(&r, "team").as_deref(), Some("b"));
        let r = build_resource(
            &ResourceConfig::default(),
            &env(&[("OTEL_RESOURCE_ATTRIBUTES", "service.name=from-attrs")]),
        );
        assert_eq!(value(&r, "service.name").as_deref(), Some("from-attrs"));
        let r = build_resource(&ResourceConfig::default(), &EnvironmentSource::Disabled);
        assert_eq!(
            value(&r, "service.name").as_deref(),
            Some("unknown_service")
        );
    }

    #[test]
    fn only_the_xybrid_preset_is_marked_and_hooks_resolve_per_destination() {
        assert!(OtlpConfig::xybrid_traces("sk")
            .traces
            .unwrap()
            .is_xybrid_preset());
        assert!(OtlpConfig::xybrid_traces_to("http://h/v1/traces", "sk")
            .traces
            .unwrap()
            .is_xybrid_preset());
        let collector = OtlpConfig::collector("http://h");
        for endpoint in [&collector.traces, &collector.metrics, &collector.logs] {
            assert!(!endpoint.as_ref().unwrap().is_xybrid_preset());
        }
        let env = OtlpConfig::from_environment(env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x")]))
            .unwrap();
        assert!(!env.traces.unwrap().is_xybrid_preset());

        let hook = ResponseHook::new(|_| super::super::ResponseAction::Continue);
        let config = collector.with_traces(Some(
            SignalEndpoint::new("http://h/v1/traces").with_response_hook(hook.clone()),
        ));
        let resolved = resolve(&config).unwrap();
        assert_eq!(resolved.traces.unwrap().hook, Some(hook));
        assert!(resolved.logs.unwrap().hook.is_none());
    }

    #[test]
    fn debug_output_hides_credentials() {
        let config = OtlpConfig::xybrid_traces("sk-secret")
            .with_environment(env(&[("OTEL_EXPORTER_OTLP_HEADERS", "k=hidden-value")]));
        let printed = format!("{config:?}");
        assert!(!printed.contains("sk-secret") && !printed.contains("hidden-value"));
    }
}
