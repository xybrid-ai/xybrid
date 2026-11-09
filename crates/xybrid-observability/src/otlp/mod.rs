//! Optional OTLP/HTTP pipeline and paired exporters (feature `otlp`).
//!
//! Two ways to export:
//!
//! - [`ObservabilityPipeline`]: application-owned tracer, meter and logger
//!   providers over the package's bounded processors and exporters. Build it
//!   from an [`OtlpConfig`], wire an observer with
//!   [`ObservabilityPipeline::observer`], and shut it down.
//! - [`exporters`]: the paired [`OtlpSpanExporter`], [`OtlpLogExporter`] and
//!   [`OtlpMetricExporter`] for providers and processors the application
//!   builds itself (pair them with `Observer::builder()` over those
//!   providers).
//!
//! Both encode OTLP with the package's wire rules (catalog `int`/`double`
//! types and real JSON nulls for scope `xybrid-observability`), split trace
//! and log batches by final request size, give every export one absolute
//! deadline shared by attempts, backoff and response reads, never follow
//! redirects, and send preset credentials only at the final transport
//! boundary. Transport work runs on a private Tokio runtime with reqwest
//! (rustls); caller threads never do network I/O and the caller's runtime is
//! never used. Nothing is installed globally and nothing is spooled to disk.
//!
//! # The null carrier
//!
//! The OpenTelemetry Rust logs API has no null value. Records of scope
//! `xybrid-observability` carry JSON `null` as `AnyValue::Bytes(empty)`
//! between the observer and the paired log exporter, which encodes a real
//! OTLP null (`{}` in OTLP/JSON). Log processors placed in front of the
//! exporter must keep such values; see [`crate::wire`].
//!
//! # Examples
//!
//! ```no_run
//! use std::time::Duration;
//! use xybrid_observability::otlp::{ObservabilityPipeline, OtlpConfig};
//!
//! # fn main() -> xybrid_observability::Result<()> {
//! let pipeline = ObservabilityPipeline::build(OtlpConfig::collector("http://localhost:4318"))?;
//! let observer = pipeline.observer();
//! // ... instrument work with `observer` ...
//! # drop(observer);
//! pipeline.shutdown(Duration::from_secs(5))?;
//! println!("{:?}", pipeline.stats());
//! # Ok(())
//! # }
//! ```

mod batch;
mod config;
mod encode;
mod export;
mod pipeline;
mod runtime;
mod stats;
mod transport;

pub use config::{
    BatchLimits, EnvironmentSource, HeaderPolicy, OtlpConfig, Protocol, ResourceConfig,
    RetryLimits, SignalEndpoint, XYBRID_TRACES_ENDPOINT,
};
pub use export::{exporters, OtlpExporters, OtlpLogExporter, OtlpMetricExporter, OtlpSpanExporter};
pub use pipeline::ObservabilityPipeline;
pub use stats::{PipelineStats, SignalStats};

/// Catalog defaults (`catalog.defaults`) of the OTLP transport.
pub mod defaults {
    use std::time::Duration;

    /// `batch.max_queue_size`.
    pub const MAX_QUEUE_SIZE: usize = 2048;
    /// `batch.max_export_batch_size`.
    pub const MAX_EXPORT_BATCH_SIZE: usize = 256;
    /// `batch.scheduled_delay_ms`.
    pub const SCHEDULED_DELAY: Duration = Duration::from_millis(2000);
    /// `batch.export_timeout_ms`.
    pub const EXPORT_TIMEOUT: Duration = Duration::from_millis(10_000);
    /// `retry.max_retries`.
    pub const MAX_RETRIES: usize = 2;
    /// `retry.initial_delay_ms`.
    pub const INITIAL_DELAY: Duration = Duration::from_millis(200);
    /// `retry.max_delay_ms`.
    pub const MAX_DELAY: Duration = Duration::from_millis(2000);
    /// `retry.retryable_status`.
    pub const RETRYABLE_STATUS: [u16; 4] = [429, 502, 503, 504];
    /// `max_request_body_bytes` (480 KiB).
    pub const MAX_REQUEST_BODY_BYTES: usize = 491_520;
    /// `max_buffered_bytes_per_signal` (32 MiB).
    pub const MAX_BUFFERED_BYTES_PER_SIGNAL: usize = 33_554_432;
    /// `max_response_body_bytes` (64 KiB).
    pub const MAX_RESPONSE_BODY_BYTES: usize = 65_536;
    /// `shutdown_timeout_ms`.
    pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(5000);
    /// `flush_timeout_ms` (also used when the SDK provider flushes a
    /// pipeline processor).
    pub const FLUSH_TIMEOUT: Duration = Duration::from_millis(5000);
    /// `metric_export_interval_ms`.
    pub const METRIC_EXPORT_INTERVAL: Duration = Duration::from_millis(30_000);
    /// Per-attempt timeout of a [`super::SignalEndpoint`] (not in the
    /// catalog; every attempt also ends at the export deadline).
    pub const SIGNAL_TIMEOUT: Duration = Duration::from_secs(10);
}

/// Serializes locally captured spans with the same grouping and catalog types as OTLP export.
///
/// # Errors
/// Returns a serialization error if the OTLP request cannot be represented as JSON.
pub fn trace_json(
    spans: Vec<opentelemetry_sdk::trace::SpanData>,
    resource: &opentelemetry_sdk::Resource,
) -> std::result::Result<Vec<u8>, serde_json::Error> {
    let prepared = encode::prepare_spans(spans);
    let request = encode::build(
        &encode::ResourceInfo::from_sdk(resource),
        &prepared.scopes,
        prepared.items,
    );
    serde_json::to_vec(&request.request)
}
