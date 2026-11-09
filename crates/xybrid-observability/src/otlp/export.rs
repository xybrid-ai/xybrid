//! Exporters: the shared split-and-send logic (D27, D28) and the paired
//! standalone exporters for application-owned providers (D15).

use std::fmt;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use opentelemetry::logs::Severity;
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::logs::{LogBatch, LogExporter};
use opentelemetry_sdk::metrics::data::ResourceMetrics;
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use opentelemetry_sdk::metrics::Temporality;
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
use opentelemetry_sdk::Resource;
use tokio::runtime::Handle;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::config::{self, Limits, OtlpConfig, Protocol, ResolvedSignal};
use super::encode::{self, Estimator, Grouped, ResourceInfo, Scoped, Signal};
use super::runtime::{self, after, lock, Completion, TransportRuntime};
use super::stats::{Outcome, SignalStats};
use super::transport::{self, Delivery, Sender};
use crate::error::Result;

/// Time allowed for cancelled work to settle its counters.
pub(crate) const SETTLE_GRACE: Duration = Duration::from_secs(1);

/// Encodes, splits and delivers one signal's exports.
#[derive(Debug)]
pub(crate) struct ExportCore {
    sender: Sender,
    max_body: usize,
}

impl ExportCore {
    pub(crate) fn new(sender: Sender, max_body: usize) -> Self {
        Self { sender, max_body }
    }

    pub(crate) fn protocol(&self) -> Protocol {
        self.sender.protocol()
    }

    pub(crate) fn signal(&self) -> Signal {
        self.sender.signal()
    }

    /// Sends a trace or log batch: each part is encoded with its own
    /// resource/scope grouping and halved while its final body exceeds the
    /// bound; a single oversized item is dropped. Every part shares
    /// `deadline`; parts not sent by then fail.
    pub(crate) async fn send_split<M: Grouped>(
        &self,
        resource: &ResourceInfo,
        scoped: Scoped<M>,
        deadline: tokio::time::Instant,
        cancel: &CancellationToken,
    ) -> Outcome {
        let signal = self.signal().as_str();
        let Scoped { scopes, items } = scoped;
        let mut outcome = Outcome::default();
        let mut work: Vec<Vec<(usize, M)>> = vec![items];
        while let Some(items) = work.pop() {
            let n = items.len() as u64;
            if n == 0 {
                continue;
            }
            if cancel.is_cancelled() || tokio::time::Instant::now() >= deadline {
                outcome.add(Outcome::failed(n));
                for rest in work.drain(..) {
                    outcome.add(Outcome::failed(rest.len() as u64));
                }
                break;
            }
            let part = encode::build::<M>(resource, &scopes, items);
            match encode::encode(&part.request, self.protocol()) {
                Some(body) if body.len() <= self.max_body => {
                    drop(part);
                    outcome.add(delivered(
                        self.sender.deliver(body, deadline, cancel).await,
                        n,
                    ));
                }
                body => {
                    if part.len() > 1 {
                        let mut first = encode::unbuild(part, scopes.len());
                        let second = first.split_off(first.len().div_ceil(2));
                        work.push(second);
                        work.push(first);
                    } else if body.is_some() {
                        log::warn!(
                            "xybrid-observability: a single {signal} item exceeds max_request_body_bytes; dropped"
                        );
                        outcome.add(Outcome::dropped(1));
                    } else {
                        log::warn!("xybrid-observability: a {signal} item could not be encoded");
                        outcome.add(Outcome::failed(1));
                    }
                }
            }
        }
        outcome
    }

    /// Sends one metrics collection, never split: a body over the bound (or
    /// an encoding failure) fails every data point without a request.
    pub(crate) async fn send_whole(
        &self,
        body: Option<Vec<u8>>,
        points: u64,
        deadline: tokio::time::Instant,
        cancel: &CancellationToken,
    ) -> Outcome {
        match body {
            Some(body) if body.len() <= self.max_body => {
                delivered(self.sender.deliver(body, deadline, cancel).await, points)
            }
            Some(_) => {
                log::warn!(
                    "xybrid-observability: a metrics collection exceeds max_request_body_bytes; not sent"
                );
                Outcome::failed(points)
            }
            None => {
                log::warn!("xybrid-observability: a metrics collection could not be encoded");
                Outcome::failed(points)
            }
        }
    }
}

fn delivered(delivery: Delivery, n: u64) -> Outcome {
    match delivery {
        Delivery::Acknowledged { rejected } => {
            let rejected = rejected.min(n);
            Outcome {
                exported: n - rejected,
                failed: rejected,
                dropped: 0,
            }
        }
        Delivery::Failed => Outcome::failed(n),
    }
}

pub(crate) fn core_for(
    client: &reqwest::Client,
    resolved: &ResolvedSignal,
    limits: &Limits,
) -> Arc<ExportCore> {
    Arc::new(ExportCore::new(
        Sender::new(client.clone(), resolved, limits.retry),
        limits.max_request_body_bytes,
    ))
}

#[derive(Debug, Default)]
struct ExporterState {
    stats: SignalStats,
    buffered: u64,
    in_flight: usize,
    closed: bool,
    shutdown: Option<Option<Result<(), Duration>>>,
    /// An export capped by the pipeline's shutdown deadline failed when it
    /// was reached (cut off by the shutdown, D30.4).
    cut_at_deadline: bool,
}

/// State shared by an exporter's handle and its in-flight exports.
pub(crate) struct ExporterShared {
    core: Arc<ExportCore>,
    handle: Handle,
    cancel: CancellationToken,
    /// Owned transport (standalone exporters); `None` inside a pipeline.
    runtime: Mutex<Option<Arc<TransportRuntime>>>,
    /// Inside a pipeline, items offered after admission closed are `dropped`
    /// (offered after shutdown began, D31); a standalone exporter shut down
    /// by its provider rejects them as `failed` (D30.6).
    refuse_as_dropped: bool,
    state: Mutex<ExporterState>,
    changed: Condvar,
    /// Notified whenever an export settles (async waiters).
    idle: Notify,
    /// Completed once [`ExporterShared::shutdown`] ran: for the pipeline's
    /// metrics this is the periodic reader finishing its final collection
    /// (the reader shuts its exporter down right after it).
    finished: Arc<Completion<()>>,
    budget: Duration,
    max_buffered_bytes: u64,
    estimator: Estimator,
    /// Serializes metrics exports (D28.5).
    serial: Arc<tokio::sync::Mutex<()>>,
    /// Pipeline shutdown deadline capping every export deadline.
    deadline_cap: Mutex<Option<Instant>>,
}

impl ExporterShared {
    pub(crate) fn new(
        core: Arc<ExportCore>,
        handle: Handle,
        cancel: CancellationToken,
        runtime: Option<Arc<TransportRuntime>>,
        budget: Duration,
        max_buffered_bytes: u64,
    ) -> Arc<Self> {
        let estimator = Estimator::new(core.protocol());
        let refuse_as_dropped = runtime.is_none();
        Arc::new(Self {
            core,
            handle,
            cancel,
            runtime: Mutex::new(runtime),
            refuse_as_dropped,
            state: Mutex::new(ExporterState::default()),
            changed: Condvar::new(),
            idle: Notify::new(),
            finished: Completion::new(),
            budget,
            max_buffered_bytes,
            estimator,
            serial: Arc::new(tokio::sync::Mutex::new(())),
            deadline_cap: Mutex::new(None),
        })
    }

    pub(crate) fn stats(&self) -> SignalStats {
        lock(&self.state).stats
    }

    pub(crate) fn is_closed(&self) -> bool {
        lock(&self.state).closed
    }

    /// Whether an export was cut off at the shutdown deadline.
    pub(crate) fn cut_at_deadline(&self) -> bool {
        lock(&self.state).cut_at_deadline
    }

    /// Closes admission; later exports fail.
    pub(crate) fn close(&self) {
        lock(&self.state).closed = true;
    }

    pub(crate) fn cap_deadline(&self, deadline: Instant) {
        *lock(&self.deadline_cap) = Some(deadline);
    }

    pub(crate) fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Completed when this exporter has been shut down (see `finished`).
    pub(crate) fn finished(&self) -> Arc<Completion<()>> {
        self.finished.clone()
    }

    /// Waits until no export is in flight (counters settled); any executor.
    pub(crate) async fn wait_idle(&self) {
        loop {
            let mut notified = std::pin::pin!(self.idle.notified());
            notified.as_mut().enable();
            if lock(&self.state).in_flight == 0 {
                return;
            }
            notified.await;
        }
    }

    fn admit(&self, n: u64, bytes: u64) -> std::result::Result<(), Outcome> {
        let mut st = lock(&self.state);
        st.stats.accepted += n;
        let refused = if st.closed && self.refuse_as_dropped {
            Outcome::dropped(n)
        } else if st.closed {
            Outcome::failed(n)
        } else if st.buffered.saturating_add(bytes) > self.max_buffered_bytes {
            Outcome::dropped(n)
        } else {
            st.stats.pending += n;
            st.buffered += bytes;
            st.in_flight += 1;
            return Ok(());
        };
        st.stats.exported += refused.exported;
        st.stats.failed += refused.failed;
        st.stats.dropped += refused.dropped;
        Err(refused)
    }

    /// Runs `work` on the transport runtime with this export's deadline and
    /// token, then settles the counters.
    async fn dispatch<F, Fut>(
        self: &Arc<Self>,
        n: u64,
        bytes: u64,
        serial: bool,
        work: F,
    ) -> Outcome
    where
        F: FnOnce(Arc<ExportCore>, tokio::time::Instant, CancellationToken) -> Fut,
        Fut: std::future::Future<Output = Outcome> + Send + 'static,
    {
        let own = after(Instant::now(), self.budget);
        let cap = *lock(&self.deadline_cap);
        let capped = cap.is_some_and(|cap| cap < own);
        let deadline = cap.map_or(own, |cap| own.min(cap));
        let deadline = tokio::time::Instant::from_std(deadline);
        let token = self.cancel.child_token();
        let task = work(self.core.clone(), deadline, token.clone());
        // Settlement belongs to the transport task. Dropping the application's
        // export future must not strand admission bytes or pending counters.
        let lease = ExportLease {
            shared: self.clone(),
            n,
            bytes,
            settled: false,
            capped_at: capped.then_some(deadline),
        };
        let gate = self.serial.clone();
        runtime::spawn(&self.handle, async move {
            let outcome = if serial {
                let wait =
                    token.run_until_cancelled(tokio::time::timeout_at(deadline, gate.lock_owned()));
                match wait.await {
                    Some(Ok(_guard)) => task.await,
                    _ => Outcome::failed(n),
                }
            } else {
                task.await
            };
            lease.finish(outcome);
            outcome
        })
        .await
        .unwrap_or_else(|_| Outcome::failed(n))
    }

    /// D30.6: closes admission, waits for in-flight exports (each bounded by
    /// its own deadline) for at most `timeout`, cancels whatever remains,
    /// then releases the transport. Later calls return the first result.
    pub(crate) fn shutdown(&self, timeout: Duration) -> std::result::Result<(), Duration> {
        let deadline = after(Instant::now(), timeout);
        let mut st = lock(&self.state);
        loop {
            match st.shutdown {
                Some(Some(result)) => return result,
                Some(None) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return Err(timeout);
                    }
                    st = self
                        .changed
                        .wait_timeout(st, deadline - now)
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .0;
                }
                None => break,
            }
        }
        st.shutdown = Some(None);
        st.closed = true;
        while st.in_flight > 0 {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            st = self
                .changed
                .wait_timeout(st, deadline - now)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        let timed_out = st.in_flight > 0;
        if timed_out {
            drop(st);
            self.cancel.cancel();
            st = lock(&self.state);
            let grace = Instant::now() + SETTLE_GRACE;
            while st.in_flight > 0 && Instant::now() < grace {
                st = self
                    .changed
                    .wait_timeout(st, grace.saturating_duration_since(Instant::now()))
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
        }
        let result = if timed_out { Err(timeout) } else { Ok(()) };
        st.shutdown = Some(Some(result));
        drop(st);
        self.changed.notify_all();
        self.finished.complete(());
        let runtime = lock(&self.runtime).take();
        drop(runtime);
        result
    }
}

/// Returns an admitted export's resources even if its task is cancelled.
struct ExportLease {
    shared: Arc<ExporterShared>,
    n: u64,
    bytes: u64,
    settled: bool,
    /// The export's deadline when the shutdown deadline capped it.
    capped_at: Option<tokio::time::Instant>,
}

impl ExportLease {
    fn finish(mut self, outcome: Outcome) {
        debug_assert_eq!(outcome.total(), self.n, "every item reaches one outcome");
        self.settle(outcome);
    }

    fn settle(&mut self, outcome: Outcome) {
        let mut st = lock(&self.shared.state);
        if outcome.failed > 0
            && self
                .capped_at
                .is_some_and(|at| tokio::time::Instant::now() >= at)
        {
            st.cut_at_deadline = true;
        }
        st.stats.settle(outcome);
        st.buffered = st.buffered.saturating_sub(self.bytes);
        st.in_flight -= 1;
        self.settled = true;
        drop(st);
        self.shared.changed.notify_all();
        self.shared.idle.notify_waiters();
    }
}

impl Drop for ExportLease {
    fn drop(&mut self) {
        if !self.settled {
            self.settle(Outcome::failed(self.n));
        }
    }
}

impl fmt::Debug for ExporterShared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExporterShared")
            .field("signal", &self.core.signal())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

fn sdk_result(signal: Signal, outcome: Outcome, n: u64) -> OTelSdkResult {
    if outcome.exported == n {
        Ok(())
    } else {
        Err(OTelSdkError::InternalFailure(format!(
            "{} export: {} failed, {} dropped of {n}",
            signal.as_str(),
            outcome.failed,
            outcome.dropped
        )))
    }
}

fn shutdown_result(result: std::result::Result<(), Duration>) -> OTelSdkResult {
    result.map_err(OTelSdkError::Timeout)
}

/// Paired OTLP span exporter for an application-owned
/// `SdkTracerProvider` (feature `otlp`). Built by [`exporters`].
///
/// Encodes with the package's wire rules (catalog numeric types for scope
/// `xybrid-observability`), splits batches by final body size, and delivers
/// on its own transport runtime within one deadline per `export()` call.
pub struct OtlpSpanExporter {
    shared: Arc<ExporterShared>,
    resource: Arc<ResourceInfo>,
}

impl OtlpSpanExporter {
    /// Counters of the items handed to `export()`.
    #[must_use]
    pub fn stats(&self) -> SignalStats {
        self.shared.stats()
    }
}

impl fmt::Debug for OtlpSpanExporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OtlpSpanExporter")
            .field("stats", &self.stats())
            .finish()
    }
}

impl SpanExporter for OtlpSpanExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        let n = batch.len() as u64;
        if n == 0 {
            return Ok(());
        }
        let shared = &self.shared;
        let bytes = batch.iter().map(|s| shared.estimator.span(s)).sum();
        if let Err(refused) = shared.admit(n, bytes) {
            return sdk_result(Signal::Traces, refused, n);
        }
        let scoped = encode::prepare_spans(batch);
        let resource = self.resource.clone();
        let outcome = shared
            .dispatch(n, bytes, false, move |core, deadline, token| async move {
                core.send_split(&resource, scoped, deadline, &token).await
            })
            .await;
        sdk_result(Signal::Traces, outcome, n)
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        shutdown_result(self.shared.shutdown(timeout))
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.resource = Arc::new(ResourceInfo::from_sdk(resource));
    }
}

/// Paired OTLP log exporter for an application-owned `SdkLoggerProvider`
/// (feature `otlp`). Built by [`exporters`].
///
/// Decodes the null carrier (`AnyValue::Bytes(empty)`) of records of scope
/// `xybrid-observability` into OTLP nulls and enforces catalog numeric
/// types; processors in front of it must preserve such values.
pub struct OtlpLogExporter {
    shared: Arc<ExporterShared>,
    resource: Arc<ResourceInfo>,
}

impl OtlpLogExporter {
    /// Counters of the records handed to `export()`.
    #[must_use]
    pub fn stats(&self) -> SignalStats {
        self.shared.stats()
    }
}

impl fmt::Debug for OtlpLogExporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OtlpLogExporter")
            .field("stats", &self.stats())
            .finish()
    }
}

impl LogExporter for OtlpLogExporter {
    async fn export(&self, batch: LogBatch<'_>) -> OTelSdkResult {
        let shared = &self.shared;
        let n = batch.iter().count() as u64;
        if n == 0 {
            return Ok(());
        }
        let bytes = batch.iter().map(|(r, s)| shared.estimator.log(r, s)).sum();
        if let Err(refused) = shared.admit(n, bytes) {
            return sdk_result(Signal::Logs, refused, n);
        }
        let scoped = encode::prepare_logs(batch.iter());
        let resource = self.resource.clone();
        let outcome = shared
            .dispatch(n, bytes, false, move |core, deadline, token| async move {
                core.send_split(&resource, scoped, deadline, &token).await
            })
            .await;
        sdk_result(Signal::Logs, outcome, n)
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        shutdown_result(self.shared.shutdown(timeout))
    }

    fn event_enabled(&self, _level: Severity, _target: &str, _name: Option<&str>) -> bool {
        !self.shared.is_closed()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.resource = Arc::new(ResourceInfo::from_sdk(resource));
    }
}

/// Paired OTLP metric exporter for an application-owned `SdkMeterProvider`
/// reader (feature `otlp`). Built by [`exporters`]; cumulative temporality.
///
/// Collections are never split: one over `max_request_body_bytes` fails.
/// Exports run one at a time, each within
/// `min(export_timeout, metric_export_interval)`.
///
/// With the SDK's `PeriodicReader`, `SdkMeterProvider::shutdown` stops
/// waiting after a fixed 5 s while the reader finishes its final export in
/// the background (still bounded by that export's deadline). The packaged
/// [`super::ObservabilityPipeline`] does not rely on that call: it waits for
/// the reader itself, up to its own shutdown timeout.
pub struct OtlpMetricExporter {
    shared: Arc<ExporterShared>,
}

impl OtlpMetricExporter {
    pub(crate) fn new(shared: Arc<ExporterShared>) -> Self {
        Self { shared }
    }

    /// Counters of the data points handed to `export()`.
    #[must_use]
    pub fn stats(&self) -> SignalStats {
        self.shared.stats()
    }
}

impl fmt::Debug for OtlpMetricExporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OtlpMetricExporter")
            .field("stats", &self.stats())
            .finish()
    }
}

impl PushMetricExporter for OtlpMetricExporter {
    async fn export(&self, metrics: &ResourceMetrics) -> OTelSdkResult {
        let shared = &self.shared;
        let (request, points) = encode::prepare_metrics(metrics);
        if points == 0 {
            return Ok(());
        }
        let body = encode::encode(&request, shared.core.protocol());
        drop(request);
        let bytes = body.as_ref().map_or(0, |b| b.len() as u64);
        if let Err(refused) = shared.admit(points, bytes) {
            return sdk_result(Signal::Metrics, refused, points);
        }
        let outcome = shared
            .dispatch(
                points,
                bytes,
                true,
                move |core, deadline, token| async move {
                    core.send_whole(body, points, deadline, &token).await
                },
            )
            .await;
        sdk_result(Signal::Metrics, outcome, points)
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        shutdown_result(self.shared.shutdown(timeout))
    }

    fn temporality(&self) -> Temporality {
        Temporality::Cumulative
    }
}

/// Paired exporters for application-owned providers, one per configured
/// signal. They share one private transport runtime, released when the last
/// of them is shut down or dropped.
#[derive(Debug)]
pub struct OtlpExporters {
    /// For `SdkTracerProvider::builder().with_batch_exporter(..)` or a span
    /// processor of the application's choice.
    pub traces: Option<OtlpSpanExporter>,
    /// For `SdkLoggerProvider::builder().with_batch_exporter(..)`.
    pub logs: Option<OtlpLogExporter>,
    /// For `PeriodicReader::builder(..)`.
    pub metrics: Option<OtlpMetricExporter>,
}

/// Builds the paired exporters of every configured signal.
///
/// Construction requirements (CONTRACT.md "Paired observer and exporter
/// adapters"): each exporter is the final stage of one provider's pipeline
/// and is shut down by that provider; its `export_timeout` should not exceed
/// the calling processor's or reader's timeout (it enforces its own deadline
/// regardless); it never uses the caller's Tokio runtime and may be called
/// from any thread. Log processors in front of the log exporter see JSON
/// null as `AnyValue::Bytes(empty)` in records of scope
/// `xybrid-observability` and must preserve it. Only that scope's telemetry
/// is type-enforced or carrier-decoded. `stats()` counts items handed to
/// `export()`; the byte budget covers in-flight bodies.
///
/// # Errors
///
/// [`crate::Error::Config`] when an option is invalid (checked before any
/// thread or connection exists), [`crate::Error::Lifecycle`] when the
/// transport cannot start.
pub fn exporters(config: &OtlpConfig) -> Result<OtlpExporters> {
    let resolved = config::resolve(config)?;
    let client = transport::client()?;
    let runtime = TransportRuntime::start()?;
    let limits = resolved.limits;
    let resource = Arc::new(ResourceInfo::from_sdk(&resolved.resource));
    let shared = |signal: &ResolvedSignal, budget: Duration| {
        ExporterShared::new(
            core_for(&client, signal, &limits),
            runtime.handle().clone(),
            runtime.token(),
            Some(runtime.clone()),
            budget,
            limits.max_buffered_bytes,
        )
    };
    Ok(OtlpExporters {
        traces: resolved.traces.as_ref().map(|s| OtlpSpanExporter {
            shared: shared(s, limits.export_timeout),
            resource: resource.clone(),
        }),
        logs: resolved.logs.as_ref().map(|s| OtlpLogExporter {
            shared: shared(s, limits.export_timeout),
            resource: resource.clone(),
        }),
        metrics: resolved
            .metrics
            .as_ref()
            .map(|s| OtlpMetricExporter::new(shared(s, limits.metric_export_timeout()))),
    })
}
